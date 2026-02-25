#!/usr/bin/env python3
"""Simple GUI to generate, transmit and receive signals."""
import logging
import io
import threading
import queue
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import customtkinter as ctk
import time
import multiprocessing
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import json
import math
import contextlib
import tempfile
import os
from multiprocessing import shared_memory, Pipe, Process
from pathlib import Path
from datetime import datetime
from queue import Empty as ThreadQueueEmpty

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph import exporters as pg_exporters

import sys
from .helpers.tx_generator import apply_frequency_domain_zeroing, generate_waveform
from .helpers.iq_utils import save_interleaved
from .helpers import rx_convert
from .helpers.correlation_utils import (
    apply_manual_lags as _apply_manual_lags,
    autocorr_fft as _autocorr_fft,
    classify_peak_group_from_mag as _classify_peak_group_from_mag,
    find_los_echo_from_mag as _find_los_echo_from_mag,
    filter_peak_indices_to_period_group as _filter_peak_indices_to_period_group,
    lag_overlap as _lag_overlap,
    resolve_manual_los_idx as _resolve_manual_los_idx,
    xcorr_fft as _xcorr_fft,
)
from .helpers.path_cancellation import apply_path_cancellation
from .helpers.continuous_processing import continuous_processing_worker
from .helpers.echo_aoa import _find_peaks_simple
from .helpers.number_parser import parse_number_expr
from .helpers.plot_colors import PLOT_COLORS
from .tx_controller import TxController

# --- suggestion helper -------------------------------------------------------

# --- suggestion helper ---
SUGGESTIONS_FILE = Path(__file__).with_name("suggestions.json")


def _load_suggestions() -> dict:
    try:
        with open(SUGGESTIONS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_suggestions(data: dict) -> None:
    try:
        with open(SUGGESTIONS_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


_SUGGESTIONS = _load_suggestions()

# --- preset helper -----------------------------------------------------------
PRESETS_FILE = Path(__file__).with_name("presets.json")


def _load_presets() -> dict:
    try:
        with open(PRESETS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_presets(data: dict) -> None:
    try:
        with open(PRESETS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


_PRESETS = _load_presets()

# --- state persistence helper ----------------------------------------------
STATE_FILE = Path(__file__).with_name("state.json")


def _load_state() -> dict:
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(data: dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


_STATE = _load_state()

AUTOSAVE_INTERVAL = 5  # seconds
# Arrays larger than this skip shared memory and use .npy + mmap in plot worker.
USE_SHARED_MEMORY = False
SHM_SIZE_THRESHOLD_BYTES = 25 * 1024 * 1024  # 25 MB
XCORR_EXTRA_PEAKS_BEFORE = 4
XCORR_EXTRA_PEAKS_AFTER = 4
XCORR_EXTRA_PEAK_MIN_REL_HEIGHT = 0.1
XCORR_EXTRA_PEAK_COLORS = (
    "#FFB300",
    "#8E24AA",
    "#00ACC1",
    "#F4511E",
    "#43A047",
    "#5E35B1",
    "#FB8C00",
    "#3949AB",
)
CONTINUOUS_INPUT_SLOT_COUNT = 4
CONTINUOUS_INPUT_SLOT_MIN_BYTES = 4 * 1024 * 1024
CONTINUOUS_INPUT_SLOT_MAX_BYTES = 64 * 1024 * 1024
CONTINUOUS_INPUT_SLOT_HEADROOM = 1.35


def _repetition_period_samples_from_tx(tx_length_samples: int, lag_step: int = 1) -> int:
    """Return TX repetition period in the active xcorr sample domain."""
    return max(1, int(tx_length_samples) * max(1, int(lag_step)))


def _classify_visible_xcorr_peaks(
    mag: np.ndarray,
    *,
    repetition_period_samples: int,
    peaks_before: int = XCORR_EXTRA_PEAKS_BEFORE,
    peaks_after: int = XCORR_EXTRA_PEAKS_AFTER,
    min_rel_height: float = XCORR_EXTRA_PEAK_MIN_REL_HEIGHT,
) -> tuple[int | None, int | None, list[int]]:
    """Return (highest_idx, los_idx, echo_indices) from visible local maxima."""
    highest_idx, los_idx, echo_indices, _group_indices = _classify_peak_group_from_mag(
        mag,
        peaks_before=peaks_before,
        peaks_after=peaks_after,
        min_rel_height=min_rel_height,
        repetition_period_samples=repetition_period_samples,
    )
    return highest_idx, los_idx, echo_indices


def _current_peak_group_indices(
    lags: np.ndarray,
    peak_source_los_idx: int | None,
    peak_source_echo_indices: list[int],
    peak_source_highest_idx: int | None,
    period_samples: int | None,
) -> list[int]:
    indices = [
        idx
        for idx in [peak_source_los_idx, *peak_source_echo_indices, peak_source_highest_idx]
        if idx is not None
    ]
    return _filter_peak_indices_to_period_group(
        lags,
        indices,
        peak_source_highest_idx,
        period_samples,
    )


_SHM_REGISTRY: set[str] = set()


def _create_shared_memory(size: int) -> shared_memory.SharedMemory:
    """Create shared memory with tracking disabled when available."""
    # Ownership rule: the creator unlinks shared memory; consumers only close it.
    try:
        shm = shared_memory.SharedMemory(create=True, size=size, track=False)  # type: ignore[call-arg]
    except TypeError:
        shm = shared_memory.SharedMemory(create=True, size=size)
    _SHM_REGISTRY.add(shm.name)
    return shm


def _cleanup_shared_memory() -> None:
    """Unlink any shared memory segments we created."""
    if not _SHM_REGISTRY:
        return
    for shm_name in sorted(_SHM_REGISTRY):
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except (FileNotFoundError, OSError, ValueError):
            continue
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        finally:
            with contextlib.suppress(Exception):
                shm.close()
    _SHM_REGISTRY.clear()


def _configure_multiprocessing() -> None:
    """Prefer spawn to avoid resource tracker issues with shared memory."""
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass


def _make_section(
    parent: tk.Misc, title: str
) -> tuple[ctk.CTkFrame, ctk.CTkFrame]:
    frame = ctk.CTkFrame(parent, corner_radius=10)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)
    header = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(weight="bold"))
    header.grid(row=0, column=0, sticky="w", padx=12, pady=(5, 0))
    body = ctk.CTkFrame(frame, fg_color="transparent")
    body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(3, 5))
    body.columnconfigure(1, weight=1)
    return frame, body


def _make_group(
    parent: tk.Misc,
    title: str,
    toggle_var: tk.BooleanVar | None = None,
    toggle_command=None,
) -> tuple[ctk.CTkFrame, ctk.CTkFrame, ctk.CTkCheckBox | None]:
    frame = ctk.CTkFrame(parent, corner_radius=10)
    frame.columnconfigure(0, weight=1)
    header = ctk.CTkFrame(frame, fg_color="transparent")
    header.grid(row=0, column=0, sticky="w", padx=10, pady=(4, 0))
    toggle = None
    if toggle_var is not None:
        toggle = ctk.CTkCheckBox(
            header, text=title, variable=toggle_var, command=toggle_command, width=24
        )
        toggle.grid(row=0, column=0, sticky="w")
    else:
        label = ctk.CTkLabel(header, text=title)
        label.grid(row=0, column=0, sticky="w")
    body = ctk.CTkFrame(frame, fg_color="transparent")
    body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(2, 4))
    body.columnconfigure(1, weight=1)
    frame.rowconfigure(1, weight=1)
    return frame, body, toggle


def _resolve_theme_color(color: str | tuple[str, str]) -> str:
    if isinstance(color, (tuple, list)):
        if ctk.get_appearance_mode() == "Light":
            return color[0]
        return color[1]
    return color


BUTTON_CURSOR = "hand2"


def _apply_button_cursor(widget: tk.Misc) -> None:
    for child in widget.winfo_children():
        if isinstance(child, ctk.CTkButton):
            child.configure(cursor=BUTTON_CURSOR)
        _apply_button_cursor(child)


INPUT_FIELD_MARGIN = 4
INPUT_WIDGET_TYPES = (
    ctk.CTkEntry,
    ctk.CTkComboBox,
    ctk.CTkOptionMenu,
    ctk.CTkTextbox,
)


def _parse_padding(value) -> tuple[int, int]:
    if value in (None, "", ()):
        return 0, 0
    if isinstance(value, (int, float)):
        pad = int(value)
        return pad, pad
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            pad = int(value[0])
            return pad, pad
        if len(value) >= 2:
            return int(value[0]), int(value[1])
    if isinstance(value, str):
        cleaned = value.strip().strip("(){}")
        if not cleaned:
            return 0, 0
        parts = cleaned.replace(",", " ").split()
        if len(parts) == 1:
            pad = int(float(parts[0]))
            return pad, pad
        if len(parts) >= 2:
            return int(float(parts[0])), int(float(parts[1]))
    return 0, 0


def _apply_input_margins(widget: tk.Misc) -> None:
    for child in widget.winfo_children():
        _apply_input_margins(child)
    if not isinstance(widget, INPUT_WIDGET_TYPES):
        return
    manager = widget.winfo_manager()
    if manager == "grid":
        info = widget.grid_info()
        top, bottom = _parse_padding(info.get("pady"))
        widget.grid_configure(
            pady=(max(top, INPUT_FIELD_MARGIN), max(bottom, INPUT_FIELD_MARGIN))
        )
    elif manager == "pack":
        info = widget.pack_info()
        top, bottom = _parse_padding(info.get("pady"))
        widget.pack_configure(
            pady=(max(top, INPUT_FIELD_MARGIN), max(bottom, INPUT_FIELD_MARGIN))
        )


def _resolve_ctk_frame_bg(widget: tk.Misc) -> str:
    parent = widget
    while isinstance(parent, ctk.CTkBaseClass):
        resolved = _resolve_theme_color(parent.cget("fg_color"))
        if resolved != "transparent":
            return resolved
        parent = parent.master
    return _resolve_theme_color(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])


def _tk_color_to_rgb(widget: tk.Misc, color: str):
    if not color or color == "transparent":
        return color
    try:
        red, green, blue = widget.winfo_rgb(color)
    except tk.TclError:
        return color
    return red // 257, green // 257, blue // 257


def _apply_mpl_transparent(fig: Figure, ax) -> None:
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)


def _apply_mpl_gray_style(ax, color: str = "#9E9E9E") -> None:
    ax.tick_params(axis="both", colors=color)
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    ax.title.set_color(color)
    for spine in ax.spines.values():
        spine.set_color(color)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(color)
        legend.get_frame().set_edgecolor(color)
        legend.get_frame().set_facecolor("none")


def _apply_mpl_preview_layout(ax) -> None:
    """Ensure preview plots reserve enough space for axis labels."""
    fig = ax.figure
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.88)


def _make_bordered_group(
    parent: tk.Misc,
    title: str,
    toggle_var: tk.BooleanVar | None = None,
    toggle_command=None,
) -> tuple[ctk.CTkFrame, ctk.CTkFrame, ctk.CTkCheckBox | None]:
    frame = ctk.CTkFrame(
        parent, corner_radius=12, border_width=1, border_color="#3b3b3b"
    )
    frame.columnconfigure(0, weight=1)
    header = ctk.CTkFrame(frame, fg_color="transparent")
    header.grid(row=0, column=0, sticky="w", padx=12, pady=(5, 0))
    toggle = None
    if toggle_var is not None:
        toggle = ctk.CTkCheckBox(
            header, text=title, variable=toggle_var, command=toggle_command, width=24
        )
        toggle.grid(row=0, column=0, sticky="w")
    else:
        label = ctk.CTkLabel(header, text=title)
        label.grid(row=0, column=0, sticky="w")
    body = ctk.CTkFrame(frame, fg_color="transparent")
    body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(3, 6))
    body.columnconfigure(1, weight=1)
    frame.rowconfigure(1, weight=1)
    return frame, body, toggle


def _make_side_bordered_group(
    parent: tk.Misc,
    title: str,
    toggle_var: tk.BooleanVar | None = None,
    toggle_command=None,
) -> tuple[ctk.CTkFrame, ctk.CTkFrame, ctk.CTkCheckBox | ctk.CTkLabel]:
    frame = ctk.CTkFrame(
        parent, corner_radius=12, border_width=1, border_color="#3b3b3b"
    )
    frame.columnconfigure(0, weight=1)
    body = ctk.CTkFrame(frame, fg_color="transparent")
    body.grid(row=0, column=0, sticky="nsew", padx=12, pady=6)
    body.columnconfigure(1, weight=1)
    body.columnconfigure(2, weight=1)
    if toggle_var is not None:
        header = ctk.CTkCheckBox(
            body, text=title, variable=toggle_var, command=toggle_command, width=24
        )
    else:
        header = ctk.CTkLabel(body, text=title, font=ctk.CTkFont(weight="bold"))
    header.grid(row=0, column=0, sticky="nw", padx=(6, 12), pady=1)
    frame.rowconfigure(0, weight=1)
    return frame, body, header


def _make_inline_toggle_row(
    parent: tk.Misc,
    title: str,
    toggle_var: tk.BooleanVar,
    toggle_command=None,
) -> tuple[ctk.CTkFrame, ctk.CTkCheckBox]:
    frame = ctk.CTkFrame(parent, corner_radius=10)
    frame.columnconfigure(1, weight=1)
    toggle = ctk.CTkCheckBox(
        frame, text=title, variable=toggle_var, command=toggle_command
    )
    toggle.grid(row=0, column=0, sticky="w", padx=(10, 6), pady=3)
    return frame, toggle


class _QueueLogHandler(logging.Handler):
    def __init__(self, output_queue: "queue.Queue[str]") -> None:
        super().__init__(level=logging.INFO)
        self._output_queue = output_queue

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        if not message.endswith("\n"):
            message += "\n"
        try:
            self._output_queue.put(message)
        except Exception:
            pass


class _FDCapture:
    def __init__(self, output_queue: "queue.Queue[str]") -> None:
        self._output_queue = output_queue
        self._reader_thread: threading.Thread | None = None
        self._read_fd: int | None = None
        self._orig_stdout_fd: int | None = None
        self._orig_stderr_fd: int | None = None

    def __enter__(self) -> "_FDCapture":
        if self._reader_thread is not None:
            return self
        self._orig_stdout_fd = os.dup(1)
        self._orig_stderr_fd = os.dup(2)
        read_fd, write_fd = os.pipe()
        self._read_fd = read_fd
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="tx-fd-reader", daemon=True
        )
        self._reader_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._restore_fds()
        return None

    def _reader_loop(self) -> None:
        import codecs

        if self._read_fd is None:
            return
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                try:
                    chunk = os.read(self._read_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    self._output_queue.put(text)
            tail = decoder.decode(b"", final=True)
            if tail:
                self._output_queue.put(tail)
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                if self._read_fd is not None:
                    os.close(self._read_fd)
                    self._read_fd = None

    def _restore_fds(self) -> None:
        if self._orig_stdout_fd is None and self._orig_stderr_fd is None:
            return
        with contextlib.suppress(Exception):
            sys.stdout.flush()
            sys.stderr.flush()
        if self._orig_stdout_fd is not None:
            with contextlib.suppress(Exception):
                os.dup2(self._orig_stdout_fd, 1)
            with contextlib.suppress(Exception):
                os.close(self._orig_stdout_fd)
            self._orig_stdout_fd = None
        if self._orig_stderr_fd is not None:
            with contextlib.suppress(Exception):
                os.dup2(self._orig_stderr_fd, 2)
            with contextlib.suppress(Exception):
                os.close(self._orig_stderr_fd)
            self._orig_stderr_fd = None
        if self._read_fd is not None:
            with contextlib.suppress(Exception):
                os.close(self._read_fd)
            self._read_fd = None
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None


class RangeSlider(ctk.CTkFrame):
    """Horizontal slider with two handles and optional signal preview."""

    def __init__(
        self,
        parent,
        start_var: tk.DoubleVar,
        end_var: tk.DoubleVar,
        command=None,
        width: int = 200,
        height: int = 40,
    ) -> None:
        super().__init__(parent)
        self.start_var = start_var
        self.end_var = end_var
        self.command = command
        self.enabled = True
        self.width = width
        self.height = height
        self.signal_color = "#9aa0a6"
        self.region_color = "#6c7eb8"
        self.handle_color = "#d18282"
        self.region_radius = 6
        self.canvas = tk.Canvas(
            self,
            width=width,
            height=height,
            highlightthickness=0,
            bg=self._get_canvas_bg_color(),
        )
        self.canvas.grid(row=0, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", self._on_resize)
        self.data = np.array([], dtype=np.float32)
        self.region = self._create_rounded_rect(
            0, 0, 0, height, radius=self.region_radius, fill=self.region_color, outline=""
        )
        self.handle_start = self.canvas.create_line(
            0, 0, 0, height, fill=self.handle_color, width=2
        )
        self.handle_end = self.canvas.create_line(
            width, 0, width, height, fill=self.handle_color, width=2
        )
        self.active = None
        self.range_offset = 0.0
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        start_var.trace_add("write", self._update_from_vars)
        end_var.trace_add("write", self._update_from_vars)
        self._update_from_vars()

    def set_data(self, data: np.ndarray) -> None:
        self.data = np.asarray(data)
        self._draw_signal()

    def configure_state(self, state: str) -> None:
        self.enabled = state == "normal"

    # Internal helpers -------------------------------------------------
    def _get_canvas_bg_color(self) -> str:
        fg_color = self.cget("fg_color")
        resolved = self._resolve_color(fg_color)
        if resolved == "transparent":
            parent = self.master
            if isinstance(parent, ctk.CTkBaseClass):
                resolved = self._resolve_color(parent.cget("fg_color"))
            if resolved == "transparent":
                resolved = self._resolve_color(
                    ctk.ThemeManager.theme["CTkFrame"]["fg_color"]
                )
        return resolved

    def _resolve_color(self, color) -> str:
        if isinstance(color, (tuple, list)):
            return color[0] if ctk.get_appearance_mode() == "Light" else color[1]
        return color

    def _create_rounded_rect(
        self, x1: float, y1: float, x2: float, y2: float, radius: float, **kwargs
    ):
        points = self._rounded_rect_points(x1, y1, x2, y2, radius)
        return self.canvas.create_polygon(points, smooth=True, **kwargs)

    def _rounded_rect_points(
        self, x1: float, y1: float, x2: float, y2: float, radius: float
    ) -> list[float]:
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        r = max(0.0, min(radius, width / 2, height / 2))
        return [
            x1 + r,
            y1,
            x2 - r,
            y1,
            x2,
            y1,
            x2,
            y1 + r,
            x2,
            y2 - r,
            x2,
            y2,
            x2 - r,
            y2,
            x1 + r,
            y2,
            x1,
            y2,
            x1,
            y2 - r,
            x1,
            y1 + r,
            x1,
            y1,
        ]

    def _draw_signal(self) -> None:
        self.canvas.delete("signal")
        if self.data.size:
            y = np.asarray(self.data)
            if np.iscomplexobj(y):
                y = np.abs(y)
            step = max(1, len(y) // self.width)
            y = y[::step]
            max_val = float(np.max(np.abs(y))) if y.size else 0.0
            if max_val > 0:
                y = y / max_val
            prev_x = 0
            prev_y = self.height / 2
            for i, val in enumerate(y):
                x = int(i * self.width / (len(y) - 1)) if len(y) > 1 else 0
                yv = self.height / 2 - val * (self.height / 2 - 2)
                self.canvas.create_line(
                    prev_x, prev_y, x, yv, fill=self.signal_color, tags="signal"
                )
                prev_x, prev_y = x, yv
        self._update_from_vars()

    def _update_from_vars(self, *_args) -> None:
        start = max(0.0, min(100.0, self.start_var.get()))
        end = max(0.0, min(100.0, self.end_var.get()))
        if end < start:
            end = start
        x1 = start / 100 * self.width
        x2 = end / 100 * self.width
        radius = min(self.region_radius, (x2 - x1) / 2, self.height / 2)
        self.canvas.coords(self.handle_start, x1, 0, x1, self.height)
        self.canvas.coords(self.handle_end, x2, 0, x2, self.height)
        self.canvas.coords(
            self.region, *self._rounded_rect_points(x1, 0, x2, self.height, radius)
        )

    def _on_resize(self, event) -> None:
        if event.width == self.width and event.height == self.height:
            return
        self.width = event.width
        self.height = event.height
        self.canvas.configure(width=self.width, height=self.height)
        self._draw_signal()

    def _on_press(self, event) -> None:
        if not self.enabled:
            return
        x = max(0, min(self.width, event.x))
        x1 = self.canvas.coords(self.handle_start)[0]
        x2 = self.canvas.coords(self.handle_end)[0]
        handle_grab = 6
        if abs(x - x1) <= handle_grab or abs(x - x2) <= handle_grab:
            self.active = "start" if abs(x - x1) <= abs(x - x2) else "end"
        elif x1 <= x <= x2:
            self.active = "range"
            self.range_offset = x - x1
        else:
            self.active = "start" if abs(x - x1) <= abs(x - x2) else "end"
        self._move(x)

    def _on_drag(self, event) -> None:
        if not self.enabled or self.active is None:
            return
        self._move(event.x)

    def _move(self, x: float) -> None:
        x = max(0, min(self.width, x))
        if self.active == "range":
            range_width = (self.end_var.get() - self.start_var.get()) / 100 * self.width
            if range_width < 0:
                range_width = 0
            new_x1 = x - self.range_offset
            new_x1 = max(0, min(self.width - range_width, new_x1))
            new_x2 = new_x1 + range_width
            self.start_var.set(100 * new_x1 / self.width)
            self.end_var.set(100 * new_x2 / self.width)
        else:
            pct = 100 * x / self.width
            if self.active == "start":
                if pct > self.end_var.get():
                    self.end_var.set(pct)
                self.start_var.set(pct)
            else:
                if pct < self.start_var.get():
                    self.start_var.set(pct)
                self.end_var.set(pct)
        if self.command:
            self.command(None)


class ConsoleWindow(ctk.CTkToplevel):
    """Simple window to display text output."""

    def __init__(self, parent, title: str = "Console") -> None:
        super().__init__(parent)
        self.title(title)
        self.text = ctk.CTkTextbox(self, wrap="none")
        self.text.pack(fill="both", expand=True)

    def append(self, text: str) -> None:
        self.text.insert(tk.END, text)
        self.text.see(tk.END)


class SignalViewer(ctk.CTkToplevel):
    """Window to display a previously recorded signal."""

    def __init__(self, parent, data: np.ndarray, fs: float, title: str) -> None:
        super().__init__(parent)
        self.parent = parent
        self.title(Path(title).name)
        self.raw_data = data
        self.latest_data = data
        self.latest_fs = fs

        self.trim_var = tk.BooleanVar(value=False)
        self.trim_start = tk.DoubleVar(value=0.0)
        self.trim_end = tk.DoubleVar(value=100.0)
        self.trim_dirty = False

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        trim_frame = ctk.CTkFrame(self)
        trim_frame.grid(row=0, column=0, sticky="ew")
        trim_frame.columnconfigure(2, weight=1)

        ctk.CTkCheckBox(
            trim_frame,
            text="Trim",
            variable=self.trim_var,
            command=self._on_trim_change,
        ).grid(row=0, column=0, sticky="w")

        self.range_slider = RangeSlider(
            trim_frame,
            self.trim_start,
            self.trim_end,
            command=self._on_trim_change,
        )
        self.range_slider.grid(row=0, column=2, sticky="ew", padx=2)

        self.apply_trim_btn = ctk.CTkButton(
            trim_frame,
            text="Apply",
            command=self.update_trim,
        )
        self.apply_trim_btn.grid(row=0, column=4, padx=2)
        self.apply_trim_btn.configure(state="disabled")

        self.trim_start_label = ctk.CTkLabel(trim_frame, width=50, text="")
        self.trim_start_label.grid(row=0, column=1, sticky="w")
        self.trim_end_label = ctk.CTkLabel(trim_frame, width=50, text="")
        self.trim_end_label.grid(row=0, column=3, sticky="e")

        btn_frame = ctk.CTkFrame(self)
        btn_frame.grid(row=1, column=0, pady=5)
        btn_frame.columnconfigure(0, weight=1)

        ctk.CTkButton(btn_frame, text="Save Trim", command=self.save_trimmed).grid(
            row=0, column=0, padx=2
        )

        scroll = ctk.CTkFrame(self)
        scroll.grid(row=2, column=0, sticky="nsew")
        scroll.columnconfigure(0, weight=1)
        scroll.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(scroll)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vscroll = ctk.CTkScrollbar(scroll, orientation="vertical", command=self.canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=vscroll.set)

        self.plots_frame = ctk.CTkFrame(self.canvas)
        self.plots_frame.columnconfigure(0, weight=1)
        self.canvas.create_window((0, 0), window=self.plots_frame, anchor="nw")
        self.plots_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvases: list[FigureCanvasTkAgg] = []

        self.stats_label = ctk.CTkLabel(self.plots_frame, justify="left", anchor="w", text="")

        _apply_button_cursor(self)
        self._display_plots(data, fs)

    def _trim_data(self, data: np.ndarray) -> np.ndarray:
        if data.size == 0:
            return data
        start_pct = max(0.0, min(100.0, self.trim_start.get()))
        end_pct = max(0.0, min(100.0, self.trim_end.get()))
        if end_pct <= start_pct:
            end_pct = min(100.0, start_pct + 1.0)
        s = int(round(len(data) * start_pct / 100))
        e = int(round(len(data) * end_pct / 100))
        e = max(s + 1, min(len(data), e))
        return data[s:e]

    def _on_trim_change(self, *_args) -> None:
        state = "normal" if self.trim_var.get() else "disabled"
        self.range_slider.configure_state(state)
        self.trim_start_label.configure(text=f"{self.trim_start.get():.0f}%")
        self.trim_end_label.configure(text=f"{self.trim_end.get():.0f}%")
        self.trim_dirty = True
        self.apply_trim_btn.configure(state="normal")
        if hasattr(self.parent, "_reset_manual_xcorr_lags"):
            self.parent._reset_manual_xcorr_lags("Trim geändert")

    def update_trim(self, *_args) -> None:
        self._on_trim_change()
        self.apply_trim_btn.configure(state="disabled")
        self.trim_dirty = False
        if self.raw_data is not None:
            self._display_plots(self.raw_data, self.latest_fs)

    def save_trimmed(self) -> None:
        if self.latest_data is None:
            messagebox.showerror("Save Trim", "No data available", parent=self)
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".bin",
            initialfile="rx_trimmed.bin",
            filetypes=[("Binary files", "*.bin"), ("All files", "*.*")],
            parent=self,
        )
        if not filename:
            return
        try:
            save_interleaved(filename, self.latest_data)
        except Exception as exc:
            messagebox.showerror("Save Trim", str(exc), parent=self)

    def _display_plots(self, data: np.ndarray, fs: float) -> None:
        self.latest_fs = fs
        if hasattr(self.parent, "_reset_manual_xcorr_lags"):
            self.parent._reset_manual_xcorr_lags("Neue RX-Daten")
        if data.ndim != 1:
            data = np.asarray(data)
            if data.ndim >= 2:
                data = data[0]
        self.range_slider.set_data(data)
        if self.trim_var.get():
            data = self._trim_data(data)
        self.latest_data = data

        try:
            raw = np.fromfile(self.parent.tx_file.get(), dtype=np.int16)
            if raw.size % 2:
                raw = raw[:-1]
            raw = raw.reshape(-1, 2).astype(np.float32)
            self.tx_data = raw[:, 0] + 1j * raw[:, 1]
        except Exception:
            self.tx_data = np.array([], dtype=np.complex64)

        for c in self.canvases:
            c.get_tk_widget().destroy()
        self.canvases.clear()

        modes = ["Signal", "Freq", "Crosscorr"]
        for idx, mode in enumerate(modes):
            fig = Figure(figsize=(5, 2), dpi=100)
            ax = fig.add_subplot(111)
            _plot_on_mpl(
                ax,
                data,
                fs,
                mode,
                f"RX {mode}",
                self.tx_data,
                manual_lags=getattr(self.parent, "manual_xcorr_lags", None),
            )
            canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.grid(row=idx, column=0, sticky="nsew", pady=2)
            widget.bind(
                "<Button-1>",
                lambda _e, m=mode, d=data, s=fs, r=self.tx_data: self.parent._show_fullscreen(
                    d, s, m, f"RX {m}", r
                ),
            )
            self.canvases.append(canvas)

        stats = _calc_stats(
            data,
            fs,
            self.tx_data,
            manual_lags=getattr(self.parent, "manual_xcorr_lags", None),
            xcorr_reduce=True,
        )
        text = _format_stats_text(stats)
        self.stats_label.grid(row=len(modes), column=0, sticky="ew", pady=2)
        self.stats_label.configure(text=text)


class OpenSignalDialog(ctk.CTkToplevel):
    """Custom file dialog listing signals sortable by modification date."""

    def __init__(self, parent, initialdir: str | Path) -> None:
        super().__init__(parent)
        self.title("Open Signal")
        self.initialdir = Path(initialdir)
        self.result: str | None = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(self, columns=("name", "mtime"), show="headings")
        self.tree.heading("name", text="Name", command=lambda: self._sort("name", False))
        self.tree.heading("mtime", text="Modified", command=lambda: self._sort("mtime", False))
        self.tree.column("name", width=200)
        self.tree.column("mtime", width=150)
        self.tree.grid(row=0, column=0, sticky="nsew")

        vsb = ctk.CTkScrollbar(self, orientation="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        btn_frame = ctk.CTkFrame(self)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=5)
        ctk.CTkButton(btn_frame, text="Open", command=self._on_open).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", command=self.destroy).pack(side="left", padx=5)

        self._populate()
        self._sort("mtime", True)
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"{w * 2}x{h * 2}")
        self.tree.bind("<Double-1>", lambda _e: self._on_open())
        self.grab_set()
        self.transient(parent)
        _apply_button_cursor(self)

    def _populate(self) -> None:
        files = sorted(
            self.initialdir.glob("*.bin"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in files:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            self.tree.insert("", "end", iid=str(p), values=(p.name, mtime))

    def _sort(self, col: str, reverse: bool) -> None:
        data = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]
        if col == "mtime":
            data.sort(
                key=lambda t: datetime.strptime(t[0], "%Y-%m-%d %H:%M:%S"),
                reverse=reverse,
            )
        else:
            data.sort(reverse=reverse)
        for idx, (_val, k) in enumerate(data):
            self.tree.move(k, "", idx)
        self.tree.heading(col, command=lambda: self._sort(col, not reverse))

    def _on_open(self) -> None:
        sel = self.tree.selection()
        if sel:
            self.result = sel[0]
        self.destroy()

    def show(self) -> str | None:
        self.wait_window()
        return self.result


class SignalColumn(ctk.CTkFrame):
    """Frame to load and display a single signal."""

    def __init__(self, parent, main_parent) -> None:
        super().__init__(parent)
        self.main_parent = main_parent

        self.trim_var = tk.BooleanVar(value=False)
        self.trim_start = tk.DoubleVar(value=0.0)
        self.trim_end = tk.DoubleVar(value=100.0)
        self.trim_dirty = False

        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        ctk.CTkButton(self, text="Open Signal", command=self.open_signal).grid(
            row=0, column=0, pady=2
        )

        trim_frame = ctk.CTkFrame(self)
        trim_frame.grid(row=1, column=0, sticky="ew")
        trim_frame.columnconfigure(2, weight=1)

        ctk.CTkCheckBox(
            trim_frame,
            text="Trim",
            variable=self.trim_var,
            command=self._on_trim_change,
        ).grid(row=0, column=0, sticky="w")

        self.range_slider = RangeSlider(
            trim_frame,
            self.trim_start,
            self.trim_end,
            command=self._on_trim_change,
        )
        self.range_slider.grid(row=0, column=2, sticky="ew", padx=2)

        self.apply_trim_btn = ctk.CTkButton(
            trim_frame,
            text="Apply",
            command=self.update_trim,
        )
        self.apply_trim_btn.grid(row=0, column=4, padx=2)
        self.apply_trim_btn.configure(state="disabled")

        self.trim_start_label = ctk.CTkLabel(trim_frame, width=50, text="")
        self.trim_start_label.grid(row=0, column=1, sticky="w")
        self.trim_end_label = ctk.CTkLabel(trim_frame, width=50, text="")
        self.trim_end_label.grid(row=0, column=3, sticky="e")

        btn_frame = ctk.CTkFrame(self)
        btn_frame.grid(row=2, column=0, pady=5)
        btn_frame.columnconfigure(0, weight=1)

        ctk.CTkButton(btn_frame, text="Save Trim", command=self.save_trimmed).grid(
            row=0, column=0, padx=2
        )

        scroll = ctk.CTkFrame(self)
        scroll.grid(row=3, column=0, sticky="nsew")
        scroll.columnconfigure(0, weight=1)
        scroll.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(scroll)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vscroll = ctk.CTkScrollbar(scroll, orientation="vertical", command=self.canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=vscroll.set)

        self.plots_frame = ctk.CTkFrame(self.canvas)
        self.plots_frame.columnconfigure(0, weight=1)
        self.canvas.create_window((0, 0), window=self.plots_frame, anchor="nw")
        self.plots_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvases: list[FigureCanvasTkAgg] = []
        self.latest_data = None
        self.latest_fs = None
        self.raw_data = None
        self.latest_title = ""
        self.stats_label = ctk.CTkLabel(self.plots_frame, justify="left", anchor="w", text="")
        _apply_button_cursor(self)

    def open_signal(self) -> None:
        """Open a signal and display it inside this column."""
        dialog = OpenSignalDialog(self, "signals/rx")
        filename = dialog.show()
        if not filename:
            return
        try:
            raw = np.fromfile(filename, dtype=np.int16)
            if raw.size % 2:
                raw = raw[:-1]
            raw = raw.reshape(-1, 2).astype(np.float32)
            data = raw[:, 0] + 1j * raw[:, 1]
        except Exception as exc:
            messagebox.showerror("Open Signal", str(exc))
            return

        try:
            fs = float(
                simpledialog.askstring(
                    "Sample Rate",
                    "Sample rate [Hz]",
                    initialvalue=self.main_parent.rx_rate.get(),
                    parent=self,
                )
            )
        except Exception:
            fs = None
        if not fs:
            return

        self._display(data, fs, Path(filename).name)

    def _display(self, data: np.ndarray, fs: float, title: str) -> None:
        self.raw_data = data
        self.latest_fs = fs
        self.latest_title = title
        self.range_slider.set_data(data)
        if self.trim_var.get():
            data = self._trim_data(data)
        self.latest_data = data

        try:
            raw = np.fromfile(self.main_parent.tx_file.get(), dtype=np.int16)
            if raw.size % 2:
                raw = raw[:-1]
            raw = raw.reshape(-1, 2).astype(np.float32)
            tx_data = raw[:, 0] + 1j * raw[:, 1]
        except Exception:
            tx_data = np.array([], dtype=np.complex64)

        for c in self.canvases:
            c.get_tk_widget().destroy()
        self.canvases.clear()

        modes = ["Signal", "Freq", "Crosscorr"]
        for idx, mode in enumerate(modes):
            fig = Figure(figsize=(4, 2), dpi=100)
            ax = fig.add_subplot(111)
            _plot_on_mpl(ax, data, fs, mode, f"{title} {mode}", tx_data)
            canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.grid(row=idx, column=0, sticky="nsew", pady=2)
            widget.bind(
                "<Button-1>",
                lambda _e, m=mode, d=data, s=fs, r=tx_data: self.main_parent._show_fullscreen(
                    d, s, m, f"{title} {m}", r
                ),
            )
            self.canvases.append(canvas)

        stats = _calc_stats(data, fs, tx_data, xcorr_reduce=True)
        text = _format_stats_text(stats)
        self.stats_label.grid(row=len(modes), column=0, sticky="ew", pady=2)
        self.stats_label.configure(text=text)

    def _trim_data(self, data: np.ndarray) -> np.ndarray:
        if data.size == 0:
            return data
        start_pct = max(0.0, min(100.0, self.trim_start.get()))
        end_pct = max(0.0, min(100.0, self.trim_end.get()))
        if end_pct <= start_pct:
            end_pct = min(100.0, start_pct + 1.0)
        s = int(round(len(data) * start_pct / 100))
        e = int(round(len(data) * end_pct / 100))
        e = max(s + 1, min(len(data), e))
        return data[s:e]

    def _on_trim_change(self, *_args) -> None:
        state = "normal" if self.trim_var.get() else "disabled"
        self.range_slider.configure_state(state)
        self.trim_start_label.configure(text=f"{self.trim_start.get():.0f}%")
        self.trim_end_label.configure(text=f"{self.trim_end.get():.0f}%")
        self.trim_dirty = True
        self.apply_trim_btn.configure(state="normal")

    def update_trim(self, *_args) -> None:
        self._on_trim_change()
        self.apply_trim_btn.configure(state="disabled")
        self.trim_dirty = False
        if self.raw_data is not None:
            self._display(self.raw_data, self.latest_fs, self.latest_title)

    def save_trimmed(self) -> None:
        if self.latest_data is None:
            messagebox.showerror("Save Trim", "No data available", parent=self)
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".bin",
            initialfile="rx_trimmed.bin",
            filetypes=[("Binary files", "*.bin"), ("All files", "*.*")],
            parent=self,
        )
        if not filename:
            return
        try:
            save_interleaved(filename, self.latest_data)
        except Exception as exc:
            messagebox.showerror("Save Trim", str(exc), parent=self)


class CompareWindow(ctk.CTkToplevel):
    """Window with four columns to compare signals."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.parent = parent
        self.title("Compare Signals")
        try:
            self.state("zoomed")
        except Exception:
            try:
                self.attributes("-zoomed", True)
            except Exception:
                pass
        for i in range(4):
            self.columnconfigure(i, weight=1)
        self.rowconfigure(0, weight=1)
        self.columns: list[SignalColumn] = []
        for i in range(4):
            col = SignalColumn(self, parent)
            col.grid(row=0, column=i, sticky="nsew", padx=5, pady=5)
            self.columns.append(col)
        _apply_button_cursor(self)


class SuggestEntry(ctk.CTkFrame):
    """Entry widget with removable suggestion buttons.

    Parameters
    ----------
    parent : widget
        Parent widget.
    name : str
        Identifier used to store/load suggestions.
    width : int | None, optional
        Entry width.
    textvariable : tk.StringVar | None, optional
        Shared variable for the underlying entry.
    """

    def __init__(
        self,
        parent,
        name: str,
        width: int | None = None,
        textvariable: tk.StringVar | None = None,
    ) -> None:
        super().__init__(parent)
        self.name = name
        if width is None:
            self.entry = ctk.CTkEntry(self, textvariable=textvariable)
        else:
            self.entry = ctk.CTkEntry(
                self, width=width, textvariable=textvariable
            )
        self.var = textvariable
        self.entry.grid(row=0, column=0, sticky="ew")
        self.sugg_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.sugg_frame.grid(row=1, column=0, sticky="w")
        self.columnconfigure(0, weight=1)
        self.suggestions = _SUGGESTIONS.get(name, [])
        self._render()
        self.entry.bind("<Return>", self._on_return)

    # Proxy common methods
    def get(self):
        return self.entry.get()

    def insert(self, *args):
        return self.entry.insert(*args)

    def delete(self, *args):
        return self.entry.delete(*args)

    def _on_return(self, _event):
        self.add_suggestion(self.entry.get())

    def add_suggestion(self, text: str) -> None:
        text = text.strip()
        if not text or text in self.suggestions:
            return
        self.suggestions.append(text)
        _SUGGESTIONS[self.name] = self.suggestions
        _save_suggestions(_SUGGESTIONS)
        self._render()

    def _fill_entry(self, text: str) -> None:
        self.entry.delete(0, tk.END)
        self.entry.insert(0, text)

    def _remove_suggestion(self, text: str) -> None:
        if text in self.suggestions:
            self.suggestions.remove(text)
            _SUGGESTIONS[self.name] = self.suggestions
            _save_suggestions(_SUGGESTIONS)
            self._render()

    def _render(self) -> None:
        for w in self.sugg_frame.winfo_children():
            w.destroy()
        if not self.suggestions:
            self.sugg_frame.grid_remove()
            return
        self.sugg_frame.grid()
        for val in self.suggestions:
            frame = ctk.CTkFrame(self.sugg_frame, corner_radius=6)
            lbl = ctk.CTkLabel(frame, text=val)
            lbl.pack(side="left")
            rm = ctk.CTkButton(
                frame,
                text="x",
                command=lambda v=val: self._remove_suggestion(v),
                width=24,
            )
            rm.configure(cursor=BUTTON_CURSOR)
            rm.pack(side="right")
            frame.pack(side="left", padx=2, pady=1)
            frame.bind("<Button-1>", lambda _e, v=val: self._fill_entry(v))
            lbl.bind("<Button-1>", lambda _e, v=val: self._fill_entry(v))
        _apply_button_cursor(self.sugg_frame)


def _reduce_data(
    data: np.ndarray, max_bytes: int = 1_000_000
) -> tuple[np.ndarray, int]:
    """Return a downsampled view of *data* and the step used."""
    step = 1
    if data.nbytes > max_bytes:
        step = int(np.ceil(data.nbytes / max_bytes))
        data = data[::step]
    return data, step


def _reduce_pair(
    a: np.ndarray, b: np.ndarray, max_bytes: int = 1_000_000
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return *a* and *b* unchanged with step 1 (no downsampling)."""
    return a, b, 1


class DraggableLagMarker(pg.ScatterPlotItem):
    """Draggable marker for cross-correlation lag points."""

    def __init__(
        self,
        view_box: pg.ViewBox,
        lags: np.ndarray,
        magnitudes: np.ndarray,
        index: int,
        color: str,
        size: int = 10,
        on_drag=None,
        on_drag_end=None,
    ) -> None:
        self._view_box = view_box
        self._lags = np.asarray(lags)
        self._magnitudes = np.asarray(magnitudes)
        self._index = int(index)
        self._on_drag = on_drag
        self._on_drag_end = on_drag_end
        self._dragging = False
        pen = pg.mkPen(color)
        brush = pg.mkBrush(color)
        super().__init__(symbol="o", size=size, pen=pen, brush=brush)
        self.setZValue(10)
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        self._update_position(self._index)

    def _update_position(self, index: int) -> None:
        index = int(np.clip(index, 0, len(self._lags) - 1))
        self._index = index
        self.setData([self._lags[index]], [self._magnitudes[index]])

    def set_index(self, index: int) -> None:
        self._update_position(index)

    def mouseDragEvent(self, ev) -> None:
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        if ev.isStart():
            self._dragging = True
        if not self._dragging:
            ev.ignore()
            return
        pos = self._view_box.mapSceneToView(ev.scenePos())
        idx = int(np.abs(self._lags - pos.x()).argmin())
        self._update_position(idx)
        if self._on_drag is not None:
            self._on_drag(self._index, float(self._lags[self._index]))
        if ev.isFinish():
            self._dragging = False
            if self._on_drag_end is not None:
                self._on_drag_end(self._index, float(self._lags[self._index]))
        ev.accept()


def _add_draggable_markers(
    plot: pg.PlotItem,
    lags: np.ndarray,
    magnitudes: np.ndarray,
    los_idx: int | None,
    echo_idx: int | None,
    on_los_drag=None,
    on_echo_drag=None,
    on_los_drag_end=None,
    on_echo_drag_end=None,
    *,
    los_lags: np.ndarray | None = None,
    los_magnitudes: np.ndarray | None = None,
    echo_lags: np.ndarray | None = None,
    echo_magnitudes: np.ndarray | None = None,
) -> tuple[DraggableLagMarker | None, DraggableLagMarker | None]:
    """Attach draggable LOS/echo markers to a plot."""
    view_box = plot.getViewBox()
    los_lags = np.asarray(los_lags) if los_lags is not None else np.asarray(lags)
    echo_lags = np.asarray(echo_lags) if echo_lags is not None else np.asarray(lags)
    los_magnitudes = (
        np.asarray(los_magnitudes)
        if los_magnitudes is not None
        else np.asarray(magnitudes)
    )
    echo_magnitudes = (
        np.asarray(echo_magnitudes)
        if echo_magnitudes is not None
        else np.asarray(magnitudes)
    )
    los_marker = None
    echo_marker = None
    if los_idx is not None:
        los_marker = DraggableLagMarker(
            view_box,
            los_lags,
            los_magnitudes,
            los_idx,
            "r",
            on_drag=on_los_drag,
            on_drag_end=on_los_drag_end,
        )
        plot.addItem(los_marker)
    if echo_idx is not None:
        echo_marker = DraggableLagMarker(
            view_box,
            echo_lags,
            echo_magnitudes,
            echo_idx,
            "g",
            on_drag=on_echo_drag,
            on_drag_end=on_echo_drag_end,
        )
        plot.addItem(echo_marker)
    return los_marker, echo_marker


def _crosscorr_peak_labels(group_indices: list[int]) -> dict[int, str]:
    """Return labels for visible cross-correlation peak markers.

    ``group_indices`` is expected to be lag-sorted with LOS at index 0 and the
    remaining entries representing Echo 1..N.
    """
    labels: dict[int, str] = {}
    if not group_indices:
        return labels

    labels[int(group_indices[0])] = "LOS"
    for number, idx in enumerate(group_indices[1:], start=1):
        labels[int(idx)] = str(number)
    return labels


def _echo_delay_samples(
    lags: np.ndarray, los_idx: int | None, echo_idx: int | None
) -> int | None:
    """Return the absolute LOS/echo lag distance in samples."""
    if los_idx is None or echo_idx is None:
        return None
    return int(abs(lags[echo_idx] - lags[los_idx]))


def _build_crosscorr_ctx(
    data: np.ndarray,
    ref_data: np.ndarray,
    *,
    crosscorr_compare: np.ndarray | None = None,
    manual_lags: dict[str, int | None] | None = None,
    lag_step: int = 1,
    normalized: bool = False,
    normalize: bool = False,
) -> dict[str, object]:
    """Return cross-correlation context for one frame.

    The returned dict contains at least ``cc``, ``lags``, ``mag``, ``los_idx``,
    ``echo_indices``, ``highest_idx`` and ``peak``. When a comparison trace is
    present, ``cc2``, ``lags2`` and ``mag2`` are included as well.
    """
    step = max(1, int(lag_step))

    def _to_magnitude(corr: np.ndarray, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        mag = np.abs(corr)
        if not normalized:
            return mag
        denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
        if denom <= 0.0 or not np.isfinite(denom):
            return mag
        return mag / denom

    def _normalize_magnitude(mag: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        if not normalize:
            return mag
        if mag.size == 0:
            return mag
        max_mag = float(np.max(mag))
        if not np.isfinite(max_mag):
            return mag
        scale = max(max_mag, eps)
        return mag / scale

    cc = _xcorr_fft(data, ref_data)
    lags = np.arange(-(len(ref_data) - 1), len(data)) * step
    mag = _normalize_magnitude(_to_magnitude(cc, data, ref_data))
    crosscorr_ctx: dict[str, object] = {
        "cc": cc,
        "lags": lags,
        "mag": mag,
        "magnitude_normalized": bool(normalize),
    }

    period_samples = _repetition_period_samples_from_tx(len(ref_data), step)
    highest_idx, base_los_idx, base_echo_indices = _classify_visible_xcorr_peaks(
        mag,
        repetition_period_samples=period_samples,
    )
    los_lags = lags

    compare_available = crosscorr_compare is not None and np.size(crosscorr_compare) != 0
    if compare_available:
        cc2 = _xcorr_fft(crosscorr_compare, ref_data)
        lags2 = np.arange(-(len(ref_data) - 1), len(crosscorr_compare)) * step
        mag2 = _normalize_magnitude(_to_magnitude(cc2, crosscorr_compare, ref_data))
        crosscorr_ctx.update({"cc2": cc2, "lags2": lags2, "mag2": mag2})
        highest_idx, base_los_idx, base_echo_indices = _classify_visible_xcorr_peaks(
            mag2,
            repetition_period_samples=period_samples,
        )
        los_lags = lags2

    current_peak_group = _current_peak_group_indices(
        los_lags,
        base_los_idx,
        list(base_echo_indices),
        highest_idx,
        period_samples,
    )
    los_idx, _ = _resolve_manual_los_idx(
        los_lags,
        base_los_idx,
        manual_lags,
        peak_group_indices=current_peak_group,
        highest_idx=highest_idx,
        period_samples=period_samples,
    )
    filtered_echo_indices = _filter_peak_indices_to_period_group(
        los_lags,
        [idx for idx in base_echo_indices if idx is not None],
        los_idx,
        period_samples,
    )
    echo_idx = filtered_echo_indices[0] if filtered_echo_indices else None
    _, manual_echo_idx = _apply_manual_lags(
        los_lags,
        los_idx,
        echo_idx,
        {
            "los": None,
            "echo": manual_lags.get("echo") if manual_lags else None,
        },
    )
    if manual_echo_idx is not None:
        filtered_echo_indices = [
            int(manual_echo_idx),
            *[int(idx) for idx in filtered_echo_indices if int(idx) != int(manual_echo_idx)],
        ]

    peak = float(np.max(mag)) if mag.size else 0.0
    if compare_available:
        mag2 = crosscorr_ctx.get("mag2")
        if isinstance(mag2, np.ndarray) and mag2.size:
            peak = max(peak, float(np.max(mag2)))

    crosscorr_ctx.update(
        {
            "highest_idx": int(highest_idx) if highest_idx is not None else None,
            "los_idx": int(los_idx) if los_idx is not None else None,
            "echo_indices": [int(idx) for idx in filtered_echo_indices],
            "peak": peak,
            "period_samples": period_samples,
            "current_peak_group": [int(idx) for idx in current_peak_group],
            "peak_source_los_idx": int(base_los_idx) if base_los_idx is not None else None,
            "peak_source_echo_indices": [int(idx) for idx in base_echo_indices],
            "peak_source_highest_idx": int(highest_idx) if highest_idx is not None else None,
        }
    )
    return crosscorr_ctx


def _estimate_los_lag(
    data: np.ndarray,
    ref_data: np.ndarray,
    manual_lags: dict[str, int | None] | None = None,
) -> int | None:
    if data.size == 0 or ref_data.size == 0:
        return None
    data_red, ref_red, step = _reduce_pair(data, ref_data)
    n = min(len(data_red), len(ref_red))
    if n == 0:
        return None
    cc = _xcorr_fft(data_red[:n], ref_red[:n])
    lags = np.arange(-n + 1, n) * step
    period_samples = _repetition_period_samples_from_tx(len(ref_red[:n]), step)
    mag = np.abs(cc)
    los_idx, _echo_idx = _find_los_echo_from_mag(
        mag,
        repetition_period_samples=period_samples,
    )
    current_peak_group = _current_peak_group_indices(
        lags,
        los_idx,
        [],
        los_idx,
        period_samples,
    )
    los_idx, _ = _resolve_manual_los_idx(
        lags,
        los_idx,
        manual_lags,
        peak_group_indices=current_peak_group,
        highest_idx=los_idx,
        period_samples=period_samples,
    )
    if los_idx is None:
        return None
    return int(lags[los_idx])


def _format_complex(value: complex | np.complexfloating | None) -> str:
    if value is None:
        return "--"
    return f"{value.real:+.4g}{value.imag:+.4g}j"


def _subtract_reference_at_lag(
    data: np.ndarray,
    ref_data: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, complex | None]:
    """Return *data* with a scaled reference path removed at *lag*."""
    if data.size == 0 or ref_data.size == 0:
        return data, None
    r_start, s_start, length = _lag_overlap(len(data), len(ref_data), lag)
    if length <= 0:
        return data, None
    r_seg = data[r_start : r_start + length]
    s_seg = ref_data[s_start : s_start + length]
    denom = np.vdot(s_seg, s_seg)
    if denom == 0:
        return data, None
    coeff = np.vdot(s_seg, r_seg) / (denom + 1e-12)
    residual = data.copy()
    residual[r_start : r_start + length] = r_seg - coeff * s_seg
    return residual, coeff


def _strip_trailing_zeros(
    data: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Return *data* without trailing zero padding."""
    if data.size == 0:
        return data
    nz = np.flatnonzero(np.abs(data) > eps)
    if nz.size == 0:
        return data
    return data[: nz[-1] + 1]


def _pretty(val: float) -> str:
    """Shorten numeric values for filenames."""
    abs_v = abs(val)
    if abs_v >= 1e6 and abs_v % 1e6 == 0:
        return f"{int(val/1e6)}M"
    if abs_v >= 1e3 and abs_v % 1e3 == 0:
        return f"{int(val/1e3)}k"
    return f"{int(val)}"


def _format_bandwidth_token(val_hz: float) -> str:
    """Return robust bandwidth token (e.g. bw2p5M, bw500k, bw125)."""
    if not math.isfinite(val_hz) or val_hz <= 0:
        return "bw0"

    abs_v = abs(val_hz)
    if abs_v >= 1e6:
        scaled = abs_v / 1e6
        suffix = "M"
    elif abs_v >= 1e3:
        scaled = abs_v / 1e3
        suffix = "k"
    else:
        scaled = abs_v
        suffix = ""

    if math.isclose(scaled, round(scaled), rel_tol=0.0, abs_tol=1e-9):
        number = str(int(round(scaled)))
    else:
        number = f"{scaled:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"bw{number}{suffix}"


def _format_hz(val: float) -> str:
    """Return *val* in a human friendly frequency unit."""
    abs_v = abs(val)
    if abs_v >= 1e9:
        return f"{val/1e9:.2f} GHz"
    if abs_v >= 1e6:
        return f"{val/1e6:.2f} MHz"
    if abs_v >= 1e3:
        return f"{val/1e3:.2f} kHz"
    if abs_v >= 1.0:
        return f"{val:.0f} Hz"
    return f"{val*1e3:.2f} mHz"


def _format_amp(val: float) -> str:
    """Return *val* in a compact, non-zero amplitude format."""
    if not math.isfinite(val):
        return "nan"
    abs_v = abs(val)
    if abs_v >= 1.0:
        return f"{val:.1f}"
    if abs_v >= 1e-6:
        return f"{val:.6f}"
    return f"{val:.2e}"


def _decimate_for_display(data: np.ndarray, max_points: int = 4096) -> np.ndarray:
    """Return *data* unchanged for UI rendering."""
    del max_points
    return np.asarray(data)


def _try_parse_number_expr(text: str, default: float = 0.0) -> float:
    try:
        return parse_number_expr(text)
    except ValueError:
        return default


def _parse_number_expr_or_error(
    text: str,
    *,
    allow_empty: bool = False,
    empty_value: float = 0.0,
) -> float:
    if allow_empty and (text is None or not text.strip()):
        return empty_value
    return parse_number_expr(text)


def _gen_tx_filename(app) -> str:
    """Generate TX filename based on current UI settings."""
    w = app.wave_var.get().lower()
    parts = [w]
    fs = _try_parse_number_expr(app.fs_entry.get(), default=0.0)
    try:
        samples = int(app.samples_entry.get())
    except Exception:
        samples = 0
    filter_mode = "frequency_domain_zeroing"
    if hasattr(app, "filter_mode_var"):
        filter_mode = str(app.filter_mode_var.get() or "frequency_domain_zeroing").strip().lower()
    fdz_var = getattr(app, "fdz_enable", None)
    filter_enabled = bool(fdz_var.get()) if fdz_var is not None else False
    filter_active = filter_enabled and w == "zadoffchu" and filter_mode == "frequency_domain_zeroing"
    filter_bandwidth = 0.0
    if hasattr(app, "filter_bandwidth_entry"):
        filter_bandwidth = _try_parse_number_expr(app.filter_bandwidth_entry.get(), default=0.0)

    if w == "sinus":
        f = _try_parse_number_expr(app.f_entry.get(), default=0.0)
        parts.append(f"f{_pretty(f)}")
    elif w == "doppelsinus":
        f = _try_parse_number_expr(app.f_entry.get(), default=0.0)
        f2 = _try_parse_number_expr(app.f1_entry.get(), default=0.0)
        parts.append(f"{_pretty(f)}_{_pretty(f2)}")
    elif w == "zadoffchu":
        q = app.q_entry.get() or "1"
        parts.append(f"q{q}")
    elif w == "chirp":
        f0 = _try_parse_number_expr(app.f_entry.get(), default=0.0)
        f1 = _try_parse_number_expr(app.f1_entry.get(), default=f0)
        parts.append(f"{_pretty(f0)}_{_pretty(f1)}")
    elif w == "ofdm_preamble":
        nfft = _try_parse_number_expr(app.ofdm_nfft_entry.get(), default=64.0)
        cp_len = _try_parse_number_expr(app.ofdm_cp_entry.get(), default=16.0)
        num_symbols = _try_parse_number_expr(app.ofdm_symbols_entry.get(), default=2.0)
        short_repeats = _try_parse_number_expr(
            app.ofdm_short_entry.get(), default=10.0
        )
        parts.append(f"nfft{int(nfft)}")
        parts.append(f"cp{int(cp_len)}")
        parts.append(f"sym{int(num_symbols)}")
        if int(short_repeats) > 0:
            parts.append(f"short{int(short_repeats)}")
    elif w == "pseudo_noise":
        pn_rate = _try_parse_number_expr(app.pn_chip_entry.get(), default=1e6)
        pn_seed = app.pn_seed_entry.get() or "1"
        parts.append(f"pn{_pretty(pn_rate)}")
        parts.append(f"seed{pn_seed}")

    if filter_active and filter_bandwidth > 0:
        parts.append(_format_bandwidth_token(filter_bandwidth))

    parts.append(f"fs{_pretty(fs)}")
    if w == "zadoffchu":
        parts.append(f"Nsym{samples}")
    elif w == "ofdm_preamble":
        parts.append(f"N{samples}")
    else:
        parts.append(f"N{samples}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "_".join(parts) + f"_{stamp}.bin"
    return str(Path("signals/tx") / name)


def _gen_filtered_tx_filename(filename: str) -> str:
    """Return an FDZ-filtered filename derived from *filename*."""
    path = Path(filename)
    stem = path.stem if path.suffix else path.name
    return str(path.with_name(f"{stem}_fdz{path.suffix}"))


def _gen_repeat_tx_filename(filename: str) -> str:
    """Return a repeated filename derived from *filename*."""
    path = Path(filename)
    stem = path.stem if path.suffix else path.name
    return str(path.with_name(f"{stem}_repeat{path.suffix}"))


def _gen_zeros_tx_filename(filename: str) -> str:
    """Return a zero-padded filename derived from *filename*."""
    path = Path(filename)
    stem = path.stem if path.suffix else path.name
    return str(path.with_name(f"{stem}_zeros{path.suffix}"))


def _strip_repeat_tx_filename(filename: str) -> str:
    """Return *filename* without a trailing ``_repeat`` suffix in the stem."""
    path = Path(filename)
    suffix = "_repeat"
    if path.stem.endswith(suffix):
        base_stem = path.stem[: -len(suffix)]
        return str(path.with_name(f"{base_stem}{path.suffix}"))
    return filename


def _strip_zeros_tx_filename(filename: str) -> str:
    """Return *filename* without a trailing ``_zeros`` suffix in the stem."""
    path = Path(filename)
    suffix = "_zeros"
    if path.stem.endswith(suffix):
        base_stem = path.stem[: -len(suffix)]
        return str(path.with_name(f"{base_stem}{path.suffix}"))
    return filename


def _gen_rx_filename(app) -> str:
    """Generate RX filename based on current UI settings."""
    freq = _try_parse_number_expr(app.rx_freq.get(), default=0.0)
    rate = _try_parse_number_expr(app.rx_rate.get(), default=0.0)
    dur = app.rx_dur.get() or "0"
    gain = app.rx_gain.get() or "0"
    parts = [f"f{_pretty(freq)}", f"r{_pretty(rate)}", f"d{dur}s", f"g{gain}"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "_".join(parts) + f"_{stamp}.bin"
    return str(Path("signals/rx") / name)


def _calc_stats(
    data: np.ndarray,
    fs: float,
    ref_data: np.ndarray | None = None,
    manual_lags: dict[str, int | None] | None = None,
    symbol_rate: float | None = None,
    xcorr_reduce: bool = False,
    path_cancel_info: dict[str, object] | None = None,
    include_spectrum: bool = True,
    include_amp: bool = True,
    include_echo: bool = True,
    precomputed_crosscorr: dict[str, object] | None = None,
    xcorr_normalized: bool = False,
) -> dict:
    """Return basic signal statistics for display.

    If ``ref_data`` is given, the delay between LOS peak and the first
    echo is added as ``echo_delay`` (in samples).
    """
    if data.ndim != 1:
        data = np.asarray(data)
        if data.ndim >= 2:
            data = data[0]

    stats = {
        "f_low": 0.0,
        "f_high": 0.0,
        "bw": 0.0,
        "bw_norm_nyq": None,
        "bw_rs": None,
        "amp": 0.0,
        "echo_delay": None,
    }

    if data.size == 0 or fs <= 0:
        return stats

    if include_amp:
        stats["amp"] = float(np.max(np.abs(data))) if np.any(data) else 0.0

    spectrum_data = data
    if include_spectrum:
        # Suppress DC components which would otherwise dominate the spectrum and
        # mask the actual signal.  This prevents the f_low/f_high detection from
        # always returning 0 Hz when a noticeable DC offset is present in the
        # received samples.
        spectrum_data = data - np.mean(data)

        spec = np.fft.fftshift(np.fft.fft(spectrum_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(spectrum_data), d=1 / fs))
        mag = 20 * np.log10(np.abs(spec) / len(spectrum_data) + 1e-12)
        max_mag = mag.max()
        mask = mag >= max_mag - 3.0
        if np.any(mask):
            stats["f_low"] = float(freqs[mask].min())
            stats["f_high"] = float(freqs[mask].max())
            stats["bw"] = stats["f_high"] - stats["f_low"]
            if fs > 0:
                stats["bw_norm_nyq"] = stats["bw"] / (fs / 2)
            if symbol_rate is not None and symbol_rate > 0:
                stats["bw_rs"] = stats["bw"] / symbol_rate

    if include_echo:
        xcorr_data = spectrum_data if include_spectrum else data
        if precomputed_crosscorr is not None:
            lags = precomputed_crosscorr.get("lags2")
            if not isinstance(lags, np.ndarray):
                lags = precomputed_crosscorr.get("lags")
            los_idx = precomputed_crosscorr.get("los_idx")
            echo_indices = precomputed_crosscorr.get("echo_indices")
            echo_idx = None
            if isinstance(echo_indices, list) and echo_indices:
                echo_idx = echo_indices[0]
            if isinstance(lags, np.ndarray):
                stats["echo_delay"] = _echo_delay_samples(
                    lags,
                    int(los_idx) if los_idx is not None else None,
                    int(echo_idx) if echo_idx is not None else None,
                )
        elif ref_data is not None and ref_data.size and xcorr_data.size:
            xcorr_ref = ref_data
            xcorr_step = 1
            if xcorr_reduce:
                xcorr_data, xcorr_ref, xcorr_step = _reduce_pair(
                    xcorr_data, xcorr_ref
                )
            crosscorr_ctx = _build_crosscorr_ctx(
                xcorr_data,
                xcorr_ref,
                manual_lags=manual_lags,
                lag_step=xcorr_step,
                normalize=xcorr_normalized,
            )
            stats["echo_delay"] = _echo_delay_samples(
                crosscorr_ctx["lags"],
                crosscorr_ctx["los_idx"],
                crosscorr_ctx["echo_indices"][0] if crosscorr_ctx["echo_indices"] else None,
            )
        elif path_cancel_info is not None:
            if (
                path_cancel_info.get("k0") is not None
                and path_cancel_info.get("k1") is not None
            ):
                stats["echo_delay"] = path_cancel_info.get("delta_k")

    return stats


def _format_stats_rows(
    stats: dict,
    *,
    include_bw_extras: bool = True,
    include_bw_nyq: bool = True,
    include_echo: bool = True,
) -> list[tuple[str, str]]:
    """Return rows of labels/values for signal statistics."""
    rows = [
        ("fmin", _format_hz(stats["f_low"])),
        ("fmax", _format_hz(stats["f_high"])),
        ("max Amp", _format_amp(stats["amp"])),
        ("BW (3dB)", _format_hz(stats["bw"])),
    ]
    if include_bw_extras:
        if include_bw_nyq and stats.get("bw_norm_nyq") is not None:
            rows.append(("BW (Nyq)", f"{stats['bw_norm_nyq']:.3f}"))
        if stats.get("bw_rs") is not None:
            rows.append(("BW (Rs)", f"{stats['bw_rs']:.3f}×Rs"))
    if include_echo and stats.get("echo_delay") is not None:
        meters = stats["echo_delay"] * 1.5
        rows.append(("LOS-Echo", f"{stats['echo_delay']} samp ({meters:.1f} m)"))
    return rows


def _format_rx_stats_rows(stats: dict) -> list[tuple[str, str]]:
    """Return rows for RX stats with a fixed layout order."""
    echo_value = "--"
    if stats.get("echo_delay") is not None:
        meters = stats["echo_delay"] * 1.5
        echo_value = f"{stats['echo_delay']} samp ({meters:.1f} m)"
    return [
        ("fmin", _format_hz(stats["f_low"])),
        ("fmax", _format_hz(stats["f_high"])),
        ("max Amp", _format_amp(stats["amp"])),
        ("LOS-Echo", echo_value),
        ("BW (3dB)", _format_hz(stats["bw"])),
    ]


def _format_stats_text(
    stats: dict,
    *,
    include_bw_extras: bool = True,
    include_bw_nyq: bool = True,
    include_echo: bool = True,
) -> str:
    """Return a formatted multi-line string for signal statistics."""
    rows = _format_stats_rows(
        stats,
        include_bw_extras=include_bw_extras,
        include_bw_nyq=include_bw_nyq,
        include_echo=include_echo,
    )
    return "\n".join(f"{label}: {value}" for label, value in rows)


class PlotWorkerManager:
    """Manage a persistent PyQtGraph plot worker process."""

    def __init__(self) -> None:
        self._process: Process | None = None
        self._conn = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._process and self._process.is_alive():
                return
            if self._conn:
                try:
                    self._conn.close()
                except OSError:
                    pass
                self._conn = None
            recv_conn, send_conn = Pipe(duplex=False)
            self._conn = send_conn
            from .helpers import plot_worker

            self._process = Process(
                target=plot_worker.worker_loop,
                args=(recv_conn,),
                daemon=True,
            )
            self._process.start()
            recv_conn.close()

    def send_payload(self, payload: dict[str, object]) -> None:
        if payload is None:
            return
        with self._lock:
            if not self._process or not self._process.is_alive():
                self.start()
            if not self._conn:
                return
            try:
                self._conn.send({"command": "plot", "payload": payload})
            except (BrokenPipeError, EOFError, OSError):
                self.start()
                if self._conn:
                    self._conn.send({"command": "plot", "payload": payload})

    def stop(self) -> None:
        with self._lock:
            if self._conn:
                try:
                    self._conn.send({"command": "shutdown"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
                try:
                    self._conn.close()
                except OSError:
                    pass
                self._conn = None
            if self._process:
                self._process.join(timeout=2)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=2)
                self._process = None


_plot_worker_manager: PlotWorkerManager | None = None


def _get_plot_worker_manager() -> PlotWorkerManager:
    global _plot_worker_manager
    if _plot_worker_manager is None:
        _plot_worker_manager = PlotWorkerManager()
    return _plot_worker_manager


def visualize(
    data: np.ndarray,
    fs: float,
    mode: str,
    title: str,
    ref_data: np.ndarray | None = None,
) -> None:
    """Visualize *data* using PyQtGraph."""
    if data.size == 0:
        messagebox.showerror("Error", "No data to visualize")
        return
    if mode == "Crosscorr" and (ref_data is None or ref_data.size == 0):
        messagebox.showinfo("Info", "Crosscorrelation requires TX data.")
        return

    _spawn_plot_worker(
        data,
        fs,
        mode,
        title,
        ref_data=ref_data,
        fullscreen=False,
    )

    return


def _spawn_plot_worker(
    data: np.ndarray,
    fs: float,
    mode: str,
    title: str,
    ref_data: np.ndarray | None = None,
    manual_lags: dict[str, int | None] | None = None,
    crosscorr_compare: np.ndarray | None = None,
    fullscreen: bool = False,
    xcorr_normalized: bool = False,
) -> str | None:
    """Launch the PyQtGraph plot worker in a separate process."""
    temp_dir = Path(tempfile.mkdtemp(prefix="transceiver_plot_"))
    data_path = None
    shm_name = None
    shm_shape = None
    shm_dtype = None
    data_contiguous = np.ascontiguousarray(data)
    reduction_step = 1
    ref_contiguous = None
    compare_contiguous = None
    if ref_data is not None and np.size(ref_data) != 0:
        ref_contiguous = np.ascontiguousarray(ref_data)
        if mode != "Crosscorr":
            data_contiguous, ref_contiguous, reduction_step = _reduce_pair(
                data_contiguous, ref_contiguous
            )
        if crosscorr_compare is not None and np.size(crosscorr_compare) != 0:
            compare_contiguous = np.ascontiguousarray(crosscorr_compare)[
                ::reduction_step
            ]
    else:
        if mode != "Crosscorr":
            data_contiguous, reduction_step = _reduce_data(data_contiguous)
    fs = float(fs) / reduction_step
    if (not USE_SHARED_MEMORY) or data_contiguous.nbytes >= SHM_SIZE_THRESHOLD_BYTES:
        data_path = temp_dir / "data.npy"
        np.save(data_path, data_contiguous)
    else:
        try:
            shm = _create_shared_memory(data_contiguous.nbytes)
            shm_view = np.ndarray(
                data_contiguous.shape, dtype=data_contiguous.dtype, buffer=shm.buf
            )
            shm_view[...] = data_contiguous
            shm_name = shm.name
            shm_shape = list(data_contiguous.shape)
            shm_dtype = data_contiguous.dtype.str
            shm.close()

        except (BufferError, FileNotFoundError, OSError, ValueError):
            data_path = temp_dir / "data.npy"
            np.save(data_path, data_contiguous)
    payload: dict[str, object] = {
        "mode": mode,
        "title": title,
        "fs": fs,
        "fullscreen": fullscreen,
        "reduction_step": reduction_step,
    }
    if shm_name:
        payload["shm_name"] = shm_name
        payload["shape"] = shm_shape
        payload["dtype"] = shm_dtype
    if data_path is not None:
        payload["data_file"] = str(data_path)
    if ref_contiguous is not None:
        ref_path = None
        ref_shm_name = None
        ref_shm_shape = None
        ref_shm_dtype = None
        if (not USE_SHARED_MEMORY) or  ref_contiguous.nbytes >= SHM_SIZE_THRESHOLD_BYTES:
            ref_path = temp_dir / "ref.npy"
            np.save(ref_path, ref_contiguous)
        else:
            try:
                ref_shm = _create_shared_memory(ref_contiguous.nbytes)
                ref_view = np.ndarray(
                    ref_contiguous.shape,
                    dtype=ref_contiguous.dtype,
                    buffer=ref_shm.buf,
                )
                ref_view[...] = ref_contiguous
                ref_shm_name = ref_shm.name
                ref_shm_shape = list(ref_contiguous.shape)
                ref_shm_dtype = ref_contiguous.dtype.str
                ref_shm.close()

            except (BufferError, FileNotFoundError, OSError, ValueError):
                ref_path = temp_dir / "ref.npy"
                np.save(ref_path, ref_contiguous)
        if ref_shm_name:
            payload["ref_shm_name"] = ref_shm_name
            payload["ref_shape"] = ref_shm_shape
            payload["ref_dtype"] = ref_shm_dtype
        if ref_path is not None:
            payload["ref_file"] = str(ref_path)
    if compare_contiguous is not None and np.size(compare_contiguous) != 0:
        compare_path = None
        compare_shm_name = None
        compare_shm_shape = None
        compare_shm_dtype = None
        if (not USE_SHARED_MEMORY) or compare_contiguous.nbytes >= SHM_SIZE_THRESHOLD_BYTES:
            compare_path = temp_dir / "compare.npy"
            np.save(compare_path, compare_contiguous)
        else:
            try:
                compare_shm = _create_shared_memory(compare_contiguous.nbytes)
                compare_view = np.ndarray(
                    compare_contiguous.shape,
                    dtype=compare_contiguous.dtype,
                    buffer=compare_shm.buf,
                )
                compare_view[...] = compare_contiguous
                compare_shm_name = compare_shm.name
                compare_shm_shape = list(compare_contiguous.shape)
                compare_shm_dtype = compare_contiguous.dtype.str
                compare_shm.close()
            except (BufferError, FileNotFoundError, OSError, ValueError):
                compare_path = temp_dir / "compare.npy"
                np.save(compare_path, compare_contiguous)
        if compare_shm_name:
            payload["compare_shm_name"] = compare_shm_name
            payload["compare_shape"] = compare_shm_shape
            payload["compare_dtype"] = compare_shm_dtype
        if compare_path is not None:
            payload["compare_file"] = str(compare_path)
    output_path = None
    payload["xcorr_normalized"] = bool(xcorr_normalized)
    if manual_lags is not None:
        payload["manual_lags"] = {
            key: (int(val) if val is not None else None)
            for key, val in manual_lags.items()
        }
        output_path = temp_dir / "manual_lags.json"
        payload["output_path"] = str(output_path)
    _get_plot_worker_manager().send_payload(payload)
    return str(output_path) if output_path is not None else None


def _plot_on_pg(
    plot: pg.PlotItem,
    data: np.ndarray,
    fs: float,
    mode: str,
    title: str,
    ref_data: np.ndarray | None = None,
    crosscorr_compare: np.ndarray | None = None,
    manual_lags: dict[str, int | None] | None = None,
    crosscorr_ctx: dict[str, object] | None = None,
    xcorr_normalized: bool = False,
    on_los_drag=None,
    on_echo_drag=None,
    on_los_drag_end=None,
    on_echo_drag_end=None,
    *,
    reduce_data: bool = True,
    reduction_step: int = 1,
) -> None:
    """Helper to draw the selected visualization on a PyQtGraph PlotItem."""
    colors = PLOT_COLORS
    step = max(1, int(reduction_step))
    scene = plot.scene()
    if scene is not None and hasattr(plot, "_xcorr_click_handler"):
        try:
            scene.sigMouseClicked.disconnect(plot._xcorr_click_handler)
        except (TypeError, RuntimeError):
            pass
        delattr(plot, "_xcorr_click_handler")
    if hasattr(plot, "_crosscorr_peak"):
        delattr(plot, "_crosscorr_peak")
    if reduce_data and mode != "Crosscorr":
        data, step = _reduce_data(data)
        fs /= step
    if mode == "Signal":
        legend = plot.addLegend()
        if legend is not None:
            legend.show()
        plot.plot(np.real(data), pen=pg.mkPen(colors["real"]), name="Real")
        plot.plot(
            np.imag(data),
            pen=pg.mkPen(colors["imag"], style=QtCore.Qt.DashLine),
            name="Imag",
        )
        plot.setTitle(title)
        plot.setLabel("bottom", "Sample Index")
        plot.setLabel("left", "Amplitude")
    elif mode in ("Freq", "Freq Analysis"):
        spec = np.fft.fftshift(np.fft.fft(data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data), d=1 / fs))
        plot.plot(
            freqs,
            20 * np.log10(np.abs(spec) + 1e-9),
            pen=pg.mkPen(colors["freq"]),
        )
        plot.setTitle(f"Spectrum: {title}")
        plot.setLabel("bottom", "Frequency [Hz]")
        plot.setLabel("left", "Magnitude [dB]")
    elif mode == "Autocorr":
        ac = _autocorr_fft(data)
        lags = np.arange(-len(data) + 1, len(data))
        plot.plot(lags, np.abs(ac), pen=pg.mkPen(colors["autocorr"]))
        plot.setTitle(f"Autocorrelation: {title}")
        plot.setLabel("bottom", "Lag")
        plot.setLabel("left", "Magnitude")
    elif mode == "Crosscorr":
        if ref_data is None or ref_data.size == 0:
            plot.setTitle("No TX data")
            plot.showGrid(x=True, y=True)
            return
        step_r = step
        compare_available = crosscorr_compare is not None and crosscorr_compare.size
        if reduce_data:
            data, ref_data, step_r = _reduce_pair(data, ref_data)
            fs /= step_r
            if compare_available:
                crosscorr_compare = crosscorr_compare[::step_r]
        if crosscorr_ctx is None:
            crosscorr_ctx = _build_crosscorr_ctx(
                data,
                ref_data,
                crosscorr_compare=crosscorr_compare,
                manual_lags=manual_lags,
                lag_step=step_r,
                normalize=xcorr_normalized,
            )
        lags = crosscorr_ctx["lags"]
        mag = crosscorr_ctx["mag"]
        los_lags = crosscorr_ctx["lags2"] if isinstance(crosscorr_ctx.get("lags2"), np.ndarray) else lags
        los_mag = crosscorr_ctx["mag2"] if isinstance(crosscorr_ctx.get("mag2"), np.ndarray) else mag
        peak_source_highest_idx = crosscorr_ctx.get("highest_idx")
        peak_source_los_idx = crosscorr_ctx.get("peak_source_los_idx", crosscorr_ctx.get("los_idx"))
        peak_source_echo_indices = list(crosscorr_ctx.get("peak_source_echo_indices", []))
        period_samples = int(crosscorr_ctx.get("period_samples", len(ref_data) * step_r))
        current_peak_group = list(crosscorr_ctx.get("current_peak_group", []))
        los_idx = crosscorr_ctx.get("los_idx")
        filtered_echo_indices = list(crosscorr_ctx.get("echo_indices", []))
        visible_group_indices = [int(los_idx), *filtered_echo_indices] if los_idx is not None else []
        echo_idx = visible_group_indices[1] if len(visible_group_indices) > 1 else None
        visible_peak_indices = list(visible_group_indices)
        legend = plot.addLegend()
        main_label = (
            "mit Pfad-Cancellation"
            if compare_available
            else "Kreuzkorrelation"
        )
        plot.plot(lags, mag, pen=pg.mkPen(colors["crosscorr"]), name=main_label)
        visible_peak_traces: list[tuple[np.ndarray, np.ndarray]] = [(lags, mag)]
        if compare_available and isinstance(crosscorr_ctx.get("lags2"), np.ndarray) and isinstance(crosscorr_ctx.get("mag2"), np.ndarray):
            lags2 = crosscorr_ctx["lags2"]
            mag2 = crosscorr_ctx["mag2"]
            visible_peak_traces.append((lags2, mag2))
            plot.plot(
                lags2,
                mag2,
                pen=pg.mkPen(colors["compare"], style=QtCore.Qt.DashLine),
                name="ohne Pfad-Cancellation",
            )
        if legend is None:
            legend = plot.addLegend()
        if legend is not None:
            legend.show()
        max_peak = float(crosscorr_ctx.get("peak", 0.0) or 0.0)

        echo_text = pg.TextItem(color=colors["text"], anchor=(0, 1))

        def _lag_value(source_lags: np.ndarray, idx: int | None) -> float | None:
            if idx is None or source_lags.size == 0:
                return None
            return float(source_lags[int(np.clip(idx, 0, len(source_lags) - 1))])

        def _position_echo_text() -> None:
            view_box = plot.getViewBox()
            x_range, y_range = view_box.viewRange()
            echo_text.setPos(x_range[0], y_range[0])

        def _echo_indices_for_los(anchor_idx: int | None) -> list[int]:
            indices = _filter_peak_indices_to_period_group(
                los_lags,
                [int(idx) for idx in peak_source_echo_indices if idx is not None],
                anchor_idx,
                period_samples,
            )
            echo_idx_local = indices[0] if indices else None
            _, echo_idx_local = _apply_manual_lags(
                los_lags,
                anchor_idx,
                echo_idx_local,
                {
                    "los": None,
                    "echo": manual_lags.get("echo") if manual_lags else None,
                },
            )
            if echo_idx_local is not None:
                return [int(echo_idx_local), *[int(idx) for idx in indices if int(idx) != int(echo_idx_local)]]
            return [int(idx) for idx in indices]

        def _update_echo_text() -> None:
            adj_los_idx, _ = _resolve_manual_los_idx(
                los_lags,
                peak_source_los_idx,
                manual_lags,
                peak_group_indices=current_peak_group,
                highest_idx=peak_source_highest_idx,
                period_samples=period_samples,
            )
            adj_echo_indices = _echo_indices_for_los(adj_los_idx)
            adj_group_indices = (
                [int(adj_los_idx), *adj_echo_indices]
                if adj_los_idx is not None
                else []
            )
            los_lag_value = _lag_value(los_lags, adj_los_idx)
            if los_lag_value is None or len(adj_group_indices) <= 1:
                echo_text.setText("LOS-Echos: --")
            else:
                rows = []
                for i, peak_idx in enumerate(adj_group_indices[1:], start=1):
                    echo_lag_value = _lag_value(los_lags, peak_idx)
                    if echo_lag_value is None:
                        continue
                    delay = int(round(abs(echo_lag_value - los_lag_value)))
                    meters = delay * 1.5
                    rows.append(f"Echo {i}: {delay} samp ({meters:.1f} m)")
                if rows:
                    echo_text.setText("LOS-Echos:\n" + "\n".join(rows))
                else:
                    echo_text.setText("LOS-Echos: --")
            _position_echo_text()

        def _wrap_drag(callback):
            def _handler(idx, lag):
                if callback is not None:
                    callback(idx, lag)
                _update_echo_text()

            return _handler

        plot.addItem(echo_text, ignoreBounds=True)
        plot.getViewBox().sigRangeChanged.connect(
            lambda *_args: _position_echo_text()
        )
        _update_echo_text()

        legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))
        los_color = pg.mkColor(colors["los"])
        echo_color = pg.mkColor(colors["echo"])
        los_legend = pg.PlotDataItem(
            [],
            [],
            pen=None,
            symbol="o",
            symbolBrush=pg.mkBrush(los_color),
            symbolPen=pg.mkPen(los_color),
        )
        echo_legend = pg.PlotDataItem(
            [],
            [],
            pen=None,
            symbol="o",
            symbolBrush=pg.mkBrush(echo_color),
            symbolPen=pg.mkPen(echo_color),
        )
        highest_legend = pg.PlotDataItem(
            [],
            [],
            pen=None,
            symbol="t",
            symbolBrush=pg.mkBrush(colors["text"]),
            symbolPen=pg.mkPen(colors["text"]),
        )
        los_legend.setData([0], [0])
        echo_legend.setData([0], [0])
        highest_legend.setData([0], [0])
        legend.addItem(highest_legend, "Höchster Peak")
        legend.addItem(los_legend, "LOS")
        legend.addItem(echo_legend, "Echo 1")
        if visible_peak_indices:
            extra_legend = pg.PlotDataItem(
                [],
                [],
                pen=None,
                symbol="o",
                symbolBrush=pg.mkBrush(XCORR_EXTRA_PEAK_COLORS[0]),
                symbolPen=pg.mkPen(XCORR_EXTRA_PEAK_COLORS[0]),
            )
            extra_legend.setData([0], [0])
            legend.addItem(extra_legend, "lokale Maxima")

        if peak_source_highest_idx is not None:
            plot.plot(
                [los_lags[peak_source_highest_idx]],
                [los_mag[peak_source_highest_idx]],
                pen=None,
                symbol="t",
                symbolSize=10,
                symbolBrush=pg.mkBrush(colors["text"]),
                symbolPen=pg.mkPen(colors["text"]),
            )

        peak_labels = _crosscorr_peak_labels(visible_group_indices)

        for color_idx, peak_idx in enumerate(visible_peak_indices):
            if peak_idx == los_idx:
                continue
            c = XCORR_EXTRA_PEAK_COLORS[color_idx % len(XCORR_EXTRA_PEAK_COLORS)]
            plot.plot(
                [los_lags[peak_idx]],
                [los_mag[peak_idx]],
                pen=None,
                symbol="o",
                symbolSize=8,
                symbolBrush=pg.mkBrush(c),
                symbolPen=pg.mkPen(c),
            )
            label_text = peak_labels.get(int(peak_idx))
            if label_text is not None:
                label_item = pg.TextItem(label_text, color=colors["text"], anchor=(0, 1))
                label_item.setPos(float(los_lags[peak_idx]), float(los_mag[peak_idx]))
                plot.addItem(label_item)

        if los_idx is not None:
            los_label_item = pg.TextItem("LOS", color=colors["text"], anchor=(0, 1))
            los_label_item.setPos(float(los_lags[los_idx]), float(los_mag[los_idx]))
            plot.addItem(los_label_item)

        los_drag_callback = _wrap_drag(on_los_drag)
        echo_drag_callback = _wrap_drag(on_echo_drag)
        los_end_callback = _wrap_drag(on_los_drag_end)
        echo_end_callback = _wrap_drag(on_echo_drag_end)
        los_marker, echo_marker = _add_draggable_markers(
            plot,
            lags,
            mag,
            los_idx,
            echo_idx,
            on_los_drag=los_drag_callback,
            on_echo_drag=echo_drag_callback,
            on_los_drag_end=los_end_callback,
            on_echo_drag_end=echo_end_callback,
            los_lags=los_lags,
            los_magnitudes=los_mag,
        )
        if scene is not None and manual_lags is not None:
            def _handle_click(ev) -> None:
                if ev.button() != QtCore.Qt.LeftButton:
                    return
                modifiers = ev.modifiers()
                if not (
                    modifiers & QtCore.Qt.ShiftModifier
                    or modifiers & QtCore.Qt.AltModifier
                ):
                    return
                pos = plot.getViewBox().mapSceneToView(ev.scenePos())
                if modifiers & QtCore.Qt.ShiftModifier:
                    idx = int(np.abs(los_lags - pos.x()).argmin())
                    lag_value = float(los_lags[idx])
                    manual_lags["los"] = int(round(lag_value))
                    if los_marker is not None:
                        los_marker.set_index(idx)
                    callback = los_drag_callback or los_end_callback
                    if callback is not None:
                        callback(idx, lag_value)
                if modifiers & QtCore.Qt.AltModifier:
                    idx = int(np.abs(lags - pos.x()).argmin())
                    lag_value = float(lags[idx])
                    manual_lags["echo"] = int(round(lag_value))
                    if echo_marker is not None:
                        echo_marker.set_index(idx)
                    callback = echo_drag_callback or echo_end_callback
                    if callback is not None:
                        callback(idx, lag_value)

            plot._xcorr_click_handler = _handle_click
            scene.sigMouseClicked.connect(_handle_click)
        plot.setTitle(f"Crosscorr. with TX: {title}")
        plot.setLabel("bottom", "Lag")
        mag_axis_label = "Magnitude (normiert)" if xcorr_normalized else "Magnitude"
        plot.setLabel("left", mag_axis_label)
        plot._crosscorr_peak = max_peak
        plot._crosscorr_peak_traces = visible_peak_traces
    plot.showGrid(x=True, y=True)


def _crosscorr_dynamic_y_range(
    current_range: tuple[float, float],
    peak: float,
    *,
    low_threshold: float = 0.2,
    headroom_ratio: float = 0.2,
) -> tuple[float, float] | None:
    """Return a new Y-range for cross-correlation previews when needed."""
    y_min, y_max = current_range
    if not np.isfinite(peak):
        return None
    peak = float(max(0.0, peak))
    if y_max <= y_min:
        return (0.0, max(1.0, peak * (1.0 + headroom_ratio)))
    # In previews we expect a non-negative magnitude axis. If the view box
    # drifts below zero (padding/manual zoom), evaluate the low-threshold trigger
    # against an effective baseline at 0.0 to keep the trigger meaningful.
    effective_min = max(0.0, y_min)
    visible_span = y_max - effective_min
    if visible_span <= 0.0:
        return (0.0, max(1.0, peak * (1.0 + headroom_ratio)))
    lower_trigger = effective_min + visible_span * low_threshold
    outside_visible = peak > y_max or peak < effective_min
    too_low = peak < lower_trigger
    if not outside_visible and not too_low:
        return None
    target_top = peak * (1.0 + headroom_ratio)
    return (0.0, max(target_top, 1e-9))


def _signal_dynamic_view_ranges(
    data: np.ndarray,
    *,
    y_padding_ratio: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return dynamic X/Y view ranges for continuous signal previews."""
    if data.size == 0:
        return None

    sample_count = int(len(data))
    x_max = float(max(1, sample_count - 1))

    real_part = np.real(data)
    imag_part = np.imag(data)
    y_min = float(min(np.min(real_part), np.min(imag_part)))
    y_max = float(max(np.max(real_part), np.max(imag_part)))

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None

    if y_max <= y_min:
        base = max(1e-6, abs(y_max))
        y_pad = base * max(y_padding_ratio, 0.1)
    else:
        y_pad = (y_max - y_min) * max(y_padding_ratio, 0.0)

    return (0.0, x_max), (y_min - y_pad, y_max + y_pad)


def _signal_dynamic_axis_labels(
    data: np.ndarray,
    y_range: tuple[float, float] | None = None,
) -> tuple[str, str]:
    """Return dynamic axis labels for signal plots based on current data/range."""
    sample_count = int(len(data))
    if sample_count <= 0:
        return "Sample Index", "Amplitude"

    x_label = f"Sample Index (0…{sample_count - 1})"

    if y_range is None:
        real_part = np.real(data)
        imag_part = np.imag(data)
        y_min = float(min(np.min(real_part), np.min(imag_part))) if data.size else 0.0
        y_max = float(max(np.max(real_part), np.max(imag_part))) if data.size else 0.0
    else:
        y_min, y_max = float(y_range[0]), float(y_range[1])

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return x_label, "Amplitude"

    max_abs = max(abs(y_min), abs(y_max))
    if max_abs <= 0.0:
        return x_label, "Amplitude (±0.0)"

    return x_label, f"Amplitude (±{max_abs:.3g})"


def _clear_pg_plot(plot_item: pg.PlotItem) -> None:
    if plot_item.legend is not None:
        legend = plot_item.legend
        try:
            legend.clear()
        except Exception:
            pass
        legend.hide()
    plot_item.clear()


def _style_pg_preview_axes(plot_item: pg.PlotItem, color: str) -> None:
    axis_pen = pg.mkPen(color)
    for axis_name in ("bottom", "left"):
        axis = plot_item.getAxis(axis_name)
        axis.setPen(axis_pen)
        axis.setTextPen(axis_pen)


def _export_pg_plot_image(
    plot_item: pg.PlotItem,
    width: int,
    height: int,
) -> ImageTk.PhotoImage:
    exporter = pg_exporters.ImageExporter(plot_item)
    params = exporter.parameters()
    if "width" in params:
        params["width"] = width
    if "height" in params:
        params["height"] = height
    png_bytes = exporter.export(toBytes=True)
    if isinstance(png_bytes, QtGui.QImage):
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.WriteOnly)
        png_bytes.save(buffer, "PNG")
        png_bytes = bytes(buffer.data())
        buffer.close()
    elif isinstance(png_bytes, QtCore.QByteArray):
        png_bytes = bytes(png_bytes)
    image = Image.open(io.BytesIO(png_bytes))
    return ImageTk.PhotoImage(image)


def _plot_on_mpl(
    ax,
    data: np.ndarray,
    fs: float,
    mode: str,
    title: str,
    ref_data: np.ndarray | None = None,
    crosscorr_compare: np.ndarray | None = None,
    manual_lags: dict[str, int | None] | None = None,
    xcorr_normalized: bool = False,
) -> None:
    """Helper to draw a small matplotlib preview plot."""
    mpl_colors = PLOT_COLORS
    if data.ndim != 1:
        data = np.asarray(data)
        if data.ndim >= 2:
            data = data[0]
    data, step = _reduce_data(data)
    fs /= step
    if mode == "Signal":
        ax.plot(np.real(data), color=mpl_colors["real"], label="Real")
        ax.plot(
            np.imag(data),
            color=mpl_colors["imag"],
            linestyle="--",
            label="Imag",
        )
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
    elif mode in ("Freq", "Freq Analysis"):
        spec = np.fft.fftshift(np.fft.fft(data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data), d=1 / fs))
        ax.plot(
            freqs,
            20 * np.log10(np.abs(spec) + 1e-9),
            color=mpl_colors["freq"],
        )
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
    elif mode == "Autocorr":
        ac = _autocorr_fft(data)
        lags = np.arange(-len(data) + 1, len(data))
        ax.plot(lags, np.abs(ac), color=mpl_colors["autocorr"])
        ax.set_xlabel("Lag")
        ax.set_ylabel("Magnitude")
    elif mode == "Crosscorr":
        if ref_data is None or ref_data.size == 0:
            ax.set_title("No TX data")
            ax.grid(True)
            _apply_mpl_gray_style(ax)
            _apply_mpl_preview_layout(ax)
            return
        data, ref_data, step_r = _reduce_pair(data, ref_data)
        if crosscorr_compare is not None and crosscorr_compare.size:
            crosscorr_compare = crosscorr_compare[::step_r]
        fs /= step_r
        cc = _xcorr_fft(data, ref_data)
        lags = np.arange(-(len(ref_data) - 1), len(data)) * step_r
        if xcorr_normalized:
            denom = float(np.linalg.norm(data) * np.linalg.norm(ref_data))
            mag = np.abs(cc) / denom if denom > 0.0 and np.isfinite(denom) else np.abs(cc)
        else:
            mag = np.abs(cc)
        ax.plot(lags, mag, color=mpl_colors["crosscorr"])
        compare_handles: list[Line2D] = []
        if crosscorr_compare is not None and crosscorr_compare.size:
            cc2 = _xcorr_fft(crosscorr_compare, ref_data)
            lags2 = np.arange(
                -(len(ref_data) - 1), len(crosscorr_compare)
            ) * step_r
            if xcorr_normalized:
                denom2 = float(np.linalg.norm(crosscorr_compare) * np.linalg.norm(ref_data))
                mag2 = np.abs(cc2) / denom2 if denom2 > 0.0 and np.isfinite(denom2) else np.abs(cc2)
            else:
                mag2 = np.abs(cc2)
            ax.plot(
                lags2,
                mag2,
                color=mpl_colors["compare"],
                linestyle="--",
                alpha=0.85,
            )
            compare_handles = [
                Line2D(
                    [0],
                    [0],
                    color=mpl_colors["crosscorr"],
                    label="mit Pfad-Cancellation",
                ),
                Line2D(
                    [0],
                    [0],
                    color=mpl_colors["compare"],
                    linestyle="--",
                    label="ohne Pfad-Cancellation",
                ),
            ]
        period_samples = _repetition_period_samples_from_tx(len(ref_data), step_r)
        highest_idx, base_los_idx, base_echo_indices = _classify_visible_xcorr_peaks(
            mag,
            repetition_period_samples=period_samples,
        )
        los_lags = lags
        los_mag = mag
        if crosscorr_compare is not None and crosscorr_compare.size:
            highest_idx2, base_los_idx2, base_echo_indices2 = _classify_visible_xcorr_peaks(
                mag2,
                repetition_period_samples=period_samples,
            )
            highest_idx = highest_idx2
            base_los_idx = base_los_idx2
            base_echo_indices = base_echo_indices2
            los_lags = lags2
            los_mag = mag2
        current_peak_group = _current_peak_group_indices(
            los_lags,
            base_los_idx,
            list(base_echo_indices),
            highest_idx,
            period_samples,
        )
        los_idx, _ = _resolve_manual_los_idx(
            los_lags,
            base_los_idx,
            manual_lags,
            peak_group_indices=current_peak_group,
            highest_idx=highest_idx,
            period_samples=period_samples,
        )
        filtered_echo_indices = _filter_peak_indices_to_period_group(
            los_lags,
            [idx for idx in base_echo_indices if idx is not None],
            los_idx,
            period_samples,
        )
        echo_idx = filtered_echo_indices[0] if filtered_echo_indices else None
        visible_group_indices = [int(los_idx), *filtered_echo_indices] if los_idx is not None else []
        visible_peak_indices = list(visible_group_indices)
        peak_labels = _crosscorr_peak_labels(visible_group_indices)

        for color_idx, peak_idx in enumerate(visible_peak_indices):
            if peak_idx == los_idx:
                continue
            c = XCORR_EXTRA_PEAK_COLORS[color_idx % len(XCORR_EXTRA_PEAK_COLORS)]
            ax.plot(
                los_lags[peak_idx],
                los_mag[peak_idx],
                marker="o",
                linestyle="",
                color=c,
                markersize=5,
            )
            label_text = peak_labels.get(int(peak_idx))
            if label_text is not None:
                ax.annotate(
                    label_text,
                    (los_lags[peak_idx], los_mag[peak_idx]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    color=mpl_colors["text"],
                    fontsize=8,
                )
        if highest_idx is not None:
            ax.plot(
                los_lags[highest_idx],
                los_mag[highest_idx],
                marker="^",
                linestyle="",
                color=mpl_colors["text"],
            )
        if los_idx is not None:
            ax.plot(
                los_lags[los_idx],
                los_mag[los_idx],
                marker="o",
                linestyle="",
                color=mpl_colors["los"],
            )
            ax.annotate(
                "LOS",
                (los_lags[los_idx], los_mag[los_idx]),
                textcoords="offset points",
                xytext=(5, 5),
                color=mpl_colors["text"],
                fontsize=8,
            )
        if echo_idx is not None:
            ax.plot(
                los_lags[echo_idx],
                los_mag[echo_idx],
                marker="o",
                linestyle="",
                color=mpl_colors["echo"],
            )
        ax.set_xlabel("Lag")
        ax.set_ylabel("Magnitude (normiert)" if xcorr_normalized else "Magnitude")
    ax.set_title(title)
    ax.grid(True)
    _apply_mpl_gray_style(ax)
    _apply_mpl_preview_layout(ax)


class TransceiverUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Signal Transceiver")
        Path("signals/tx").mkdir(parents=True, exist_ok=True)
        Path("signals/rx").mkdir(parents=True, exist_ok=True)
        # define view variables early so callbacks won't fail
        self.rx_view = tk.StringVar(value="Signal")
        self.rx_channel_view = tk.StringVar(value="Kanal 1")
        self.sync_var = tk.BooleanVar(value=True)
        self.rate_var = tk.StringVar(value="200e6")
        # individual variables used when rate sync is disabled
        self.fs_var = self.rate_var
        self.tx_rate_var = self.rate_var
        self.rx_rate_var = self.rate_var
        self.console = None
        self._out_queue = queue.Queue()
        self._tx_log_handler = _QueueLogHandler(self._out_queue)
        self._tx_log_handler.setFormatter(logging.Formatter("%(message)s"))
        self._tx_logger = logging.getLogger("transceiver.tx_controller")
        self._tx_logger.addHandler(self._tx_log_handler)
        self._tx_logger.setLevel(logging.INFO)
        self._tx_logger.propagate = False
        self._cmd_running = False
        self._proc = None
        self._stop_requested = False
        self._plot_win = None
        self.manual_xcorr_lags = {"los": None, "echo": None}
        self._tx_running = False
        self._last_tx_end = 0.0
        self._filtered_tx_file = None
        self._last_generated_tx_file: str | None = None
        self._active_tx_file: str | None = None
        self._cached_tx_path: str | None = None
        self._cached_tx_data = np.array([], dtype=np.complex64)
        self._cached_tx_load_error_path: str | None = None
        self._tx_controller = None
        self._tx_output_capture: _FDCapture | None = None
        self._closing = False
        self._plot_worker_manager = _get_plot_worker_manager()
        self._plot_worker_manager.start()
        self._xcorr_manual_file: Path | None = None
        self._xcorr_polling = False
        self._last_path_cancel_log: tuple[object, ...] | None = None
        self._last_path_cancel_info: dict[str, object] | None = None
        self._tx_indicator_state = "idle"
        self._tx_indicator_job: str | None = None
        self._tx_indicator_frame = 0
        self._tx_indicator_spinner_frames: list[ctk.CTkImage] = []
        self._tx_indicator_blink_frames: list[ctk.CTkImage] = []
        self._tx_indicator_blank: ctk.CTkImage | None = None
        self._cont_task_queue: multiprocessing.Queue | None = None
        self._cont_result_queue: multiprocessing.Queue | None = None
        self._cont_worker_process: Process | None = None
        self._cont_task_queue_drops = 0
        self._cont_rendered_frames = 0
        self._cont_last_processing_ms = 0.0
        self._cont_last_end_to_end_ms = 0.0
        self._cont_worker_result_drops = 0
        self._cont_runtime_config: dict[str, object] = {}
        self._cont_input_slots: list[shared_memory.SharedMemory] = []
        self._cont_input_slot_size = 0
        self._cont_input_free_slots: deque[int] = deque()
        self._build_tx_indicator_assets()
        self.create_widgets()
        try:
            self.state("zoomed")
        except tk.TclError:
            try:
                self.attributes("-zoomed", True)
            except tk.TclError:
                pass
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        if _STATE:
            self._apply_params(_STATE)
        self._last_saved_state = self._get_current_params()
        self.after(AUTOSAVE_INTERVAL * 1000, self._autosave_state)

    def _start_tx_output_capture(self) -> None:
        if self._tx_output_capture is not None:
            return
        self._tx_output_capture = _FDCapture(self._out_queue)
        self._tx_output_capture.__enter__()

    def _stop_tx_output_capture(self) -> None:
        if self._tx_output_capture is None:
            return
        self._tx_output_capture.__exit__(None, None, None)
        self._tx_output_capture = None

    def _build_tx_indicator_assets(self) -> None:
        size = 14
        blank = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        self._tx_indicator_blank = ctk.CTkImage(
            light_image=blank, dark_image=blank, size=(size, size)
        )

        blink_levels = [60, 100, 150, 200, 255, 200, 150, 100]
        blink_frames = []
        for alpha in blink_levels:
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.ellipse(
                (2, 2, size - 2, size - 2), fill=(0, 200, 0, alpha)
            )
            blink_frames.append(
                ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
            )
        self._tx_indicator_blink_frames = blink_frames

        spinner_frames = []
        spokes = 12
        center = (size - 1) / 2
        radius = size / 2 - 2
        for frame in range(spokes):
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            for i in range(spokes):
                delta = (i - frame) % spokes
                alpha = max(40, 255 - delta * 20)
                angle = 2 * math.pi * (i / spokes)
                x = center + math.cos(angle) * radius
                y = center + math.sin(angle) * radius
                draw.ellipse(
                    (x - 1.5, y - 1.5, x + 1.5, y + 1.5),
                    fill=(220, 220, 220, alpha),
                )
            spinner_frames.append(
                ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
            )
        self._tx_indicator_spinner_frames = spinner_frames

    def _set_tx_indicator_image(self, image: ctk.CTkImage | None) -> None:
        if hasattr(self, "tx_button") and self.tx_button.winfo_exists():
            self.tx_button.configure(image=image, compound="left")

    def _stop_tx_indicator_animation(self) -> None:
        if self._tx_indicator_job is not None:
            self.after_cancel(self._tx_indicator_job)
            self._tx_indicator_job = None

    def _set_tx_indicator_state(self, state: str) -> None:
        if state == self._tx_indicator_state:
            return
        self._tx_indicator_state = state
        self._tx_indicator_frame = 0
        self._stop_tx_indicator_animation()
        if state == "idle":
            self._set_tx_indicator_image(self._tx_indicator_blank)
            return
        if state == "pending":
            self._animate_tx_spinner()
        elif state == "active":
            self._animate_tx_blink()

    def _animate_tx_spinner(self) -> None:
        if self._tx_indicator_state != "pending":
            return
        frames = self._tx_indicator_spinner_frames
        if frames:
            frame = frames[self._tx_indicator_frame % len(frames)]
            self._set_tx_indicator_image(frame)
        self._tx_indicator_frame += 1
        self._tx_indicator_job = self.after(120, self._animate_tx_spinner)

    def _animate_tx_blink(self) -> None:
        if self._tx_indicator_state != "active":
            return
        frames = self._tx_indicator_blink_frames
        if frames:
            frame = frames[self._tx_indicator_frame % len(frames)]
            self._set_tx_indicator_image(frame)
        self._tx_indicator_frame += 1
        self._tx_indicator_job = self.after(180, self._animate_tx_blink)

    def create_widgets(self):
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1, uniform="cols")
        self.columnconfigure(1, weight=1, uniform="cols")
        self.columnconfigure(2, weight=1, uniform="cols")
        terminal_container_fg = ctk.ThemeManager.theme["CTkFrame"]["fg_color"]
        terminal_container_bg = _resolve_theme_color(terminal_container_fg)
        terminal_container_corner = 10

        # ----- Column 1: Generation -----
        gen_frame, gen_body = _make_section(self, "Signal Generation")
        gen_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        gen_body.columnconfigure(0, weight=0)
        gen_body.columnconfigure(1, weight=1)
        gen_body.columnconfigure(2, weight=1)
        label_padx = (6, 8)
        self._label_padx = label_padx

        waveform_frame, waveform_body, _ = _make_side_bordered_group(
            gen_body,
            "Waveform",
        )
        waveform_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        waveform_left = ctk.CTkFrame(waveform_body, fg_color="transparent")
        waveform_left.grid(row=0, column=1, sticky="nsew")
        waveform_left.columnconfigure(1, weight=1)
        waveform_right = ctk.CTkFrame(waveform_body, fg_color="transparent")
        waveform_right.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        waveform_right.columnconfigure(1, weight=1)

        ctk.CTkLabel(waveform_left, text="Waveform", anchor="e").grid(
            row=0, column=0, sticky="e", padx=label_padx
        )
        self.wave_var = tk.StringVar(value="sinus")
        wave_box = ctk.CTkComboBox(
            waveform_left,
            variable=self.wave_var,
            values=[
                "sinus",
                "doppelsinus",
                "zadoffchu",
                "chirp",
                "ofdm_preamble",
                "pseudo_noise",
            ],
            text_color="#f5f5f5",
            dropdown_text_color="#f5f5f5",
        )
        wave_box.grid(row=0, column=1, sticky="ew")
        wave_box.configure(
            command=lambda _value: (
                self.update_waveform_fields(),
                self.auto_update_tx_filename(),
            )
        )

        ctk.CTkLabel(waveform_right, text="fs", anchor="e").grid(
            row=0, column=0, sticky="e", padx=label_padx
        )
        self.fs_entry = SuggestEntry(
            waveform_right, "fs_entry", textvariable=self.fs_var
        )
        self.fs_entry.grid(row=0, column=1, sticky="ew")
        self.fs_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        self.f_label = ctk.CTkLabel(waveform_left, text="f", anchor="e")
        self.f_label.grid(row=1, column=0, sticky="e", padx=label_padx)
        self.f_entry = SuggestEntry(waveform_left, "f_entry")
        self.f_entry.insert(0, "25e3")
        self.f_entry.grid(row=1, column=1, sticky="ew")
        self.f_entry.entry.bind("<FocusOut>", lambda _e: self.auto_update_tx_filename())

        self.f1_label = ctk.CTkLabel(waveform_left, text="f1", anchor="e")
        self.f1_entry = SuggestEntry(waveform_left, "f1_entry")
        self.f1_label.grid(row=2, column=0, sticky="e", padx=label_padx)
        self.f1_entry.grid(row=2, column=1, sticky="ew")
        self.f1_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        self.q_label = ctk.CTkLabel(waveform_left, text="q", anchor="e")
        self.q_entry = SuggestEntry(waveform_left, "q_entry")
        self.q_entry.insert(0, "1")
        # row placement will be adjusted in update_waveform_fields
        self.q_label.grid(row=1, column=0, sticky="e", padx=label_padx)
        self.q_entry.grid(row=1, column=1, sticky="ew")
        self.q_entry.entry.bind("<FocusOut>", lambda _e: self.auto_update_tx_filename())

        self.pn_chip_label = ctk.CTkLabel(
            waveform_left, text="PN Chip-Rate", anchor="e"
        )
        self.pn_chip_entry = SuggestEntry(waveform_left, "pn_chip_entry")
        self.pn_chip_entry.insert(0, "1e6")
        self.pn_chip_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        self.pn_seed_label = ctk.CTkLabel(waveform_left, text="PN Seed", anchor="e")
        self.pn_seed_entry = SuggestEntry(waveform_left, "pn_seed_entry")
        self.pn_seed_entry.insert(0, "1")
        self.pn_seed_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        ctk.CTkLabel(waveform_right, text="Samples", anchor="e").grid(
            row=1, column=0, sticky="e", padx=label_padx
        )
        self.samples_entry = SuggestEntry(waveform_right, "samples_entry")
        self.samples_entry.insert(0, "40000")
        self.samples_entry.grid(row=1, column=1, sticky="ew")
        self.samples_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        ctk.CTkLabel(waveform_right, text="Amplitude", anchor="e").grid(
            row=2, column=0, sticky="e", padx=label_padx
        )
        self.amp_entry = SuggestEntry(waveform_right, "amp_entry")
        self.amp_entry.insert(0, "10000")
        self.amp_entry.grid(row=2, column=1, sticky="ew")

        self.ofdm_nfft_label = ctk.CTkLabel(waveform_right, text="NFFT", anchor="e")
        self.ofdm_nfft_entry = SuggestEntry(waveform_right, "ofdm_nfft_entry")
        self.ofdm_nfft_entry.insert(0, "64")
        self.ofdm_nfft_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        self.ofdm_cp_label = ctk.CTkLabel(waveform_right, text="CP", anchor="e")
        self.ofdm_cp_entry = SuggestEntry(waveform_right, "ofdm_cp_entry")
        self.ofdm_cp_entry.insert(0, "16")
        self.ofdm_cp_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        self.ofdm_symbols_label = ctk.CTkLabel(
            waveform_right, text="OFDM Symbols", anchor="e"
        )
        self.ofdm_symbols_entry = SuggestEntry(waveform_right, "ofdm_symbols_entry")
        self.ofdm_symbols_entry.insert(0, "2")
        self.ofdm_symbols_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        self.ofdm_short_label = ctk.CTkLabel(
            waveform_right, text="Short Repeats", anchor="e"
        )
        self.ofdm_short_entry = SuggestEntry(waveform_right, "ofdm_short_entry")
        self.ofdm_short_entry.insert(0, "10")
        self.ofdm_short_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        self.fdz_enable = tk.BooleanVar(value=True)
        self.filter_mode_var = tk.StringVar(value="frequency_domain_zeroing")
        filter_frame, filter_body, _ = _make_side_bordered_group(
            gen_body,
            "Filter",
            toggle_var=self.fdz_enable,
            toggle_command=self._on_filter_toggle,
        )
        filter_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        filter_left = ctk.CTkFrame(filter_body, fg_color="transparent")
        filter_left.grid(row=0, column=1, sticky="nsew")
        filter_left.columnconfigure(1, weight=1)
        filter_right = ctk.CTkFrame(filter_body, fg_color="transparent")
        filter_right.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        filter_right.columnconfigure(1, weight=1)

        self.fdz_label = ctk.CTkLabel(
            filter_left,
            text="frequency-domain zeroing (hard edge / harte Kante)",
            anchor="e",
        )
        self.fdz_label.grid(row=0, column=0, sticky="e", padx=label_padx)
        self.filter_bandwidth_label = ctk.CTkLabel(
            filter_right,
            text="Bandwidth [Hz]",
            anchor="e",
        )
        self.filter_bandwidth_label.grid(row=0, column=0, sticky="e", padx=label_padx)
        self.filter_bandwidth_entry = SuggestEntry(filter_right, "filter_bandwidth_entry")
        self.filter_bandwidth_entry.insert(0, "1e6")
        self.filter_bandwidth_entry.grid(row=0, column=1, sticky="ew")
        self.filter_bandwidth_entry.entry.bind(
            "<FocusOut>",
            lambda _e: self._on_filter_bandwidth_changed(),
        )

        if not self.fdz_enable.get():
            self.filter_bandwidth_entry.entry.configure(state="disabled")
        repeat_zero_row = ctk.CTkFrame(gen_body, fg_color="transparent")
        repeat_zero_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        repeat_zero_row.columnconfigure((0, 1), weight=1, uniform="repeat-zeros")

        self.repeat_enable = tk.BooleanVar(value=True)
        self._repeat_last_value = "1"
        repeat_frame, repeat_body, _ = _make_side_bordered_group(
            repeat_zero_row,
            "Repeats",
            toggle_var=self.repeat_enable,
            toggle_command=self._on_repeat_toggle,
        )
        repeat_frame.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.repeat_entry = SuggestEntry(repeat_body, "repeat_entry")
        self.repeat_entry.insert(0, "1")
        self.repeat_entry.entry.configure(height=28)
        self.repeat_entry.sugg_frame.grid_remove()
        self.repeat_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=6)

        self.zeros_enable = tk.BooleanVar(value=False)
        self._zeros_last_value = "same"
        zeros_frame, zeros_body, _ = _make_side_bordered_group(
            repeat_zero_row,
            "Zeros",
            toggle_var=self.zeros_enable,
            toggle_command=self._on_zeros_toggle,
        )
        zeros_frame.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.zeros_var = tk.StringVar(value="same")
        self.zeros_values = [
            "same",
            "half",
            "quarter",
            "double",
            "quadruple",
            "octuple",
        ]
        self.zeros_combo = ctk.CTkComboBox(
            zeros_body,
            variable=self.zeros_var,
            values=self.zeros_values,
        )
        self.zeros_combo.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=6)
        self.zeros_combo.configure(state="disabled")

        file_frame, file_body, _ = _make_side_bordered_group(gen_body, "File")
        file_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.file_entry = SuggestEntry(file_body, "file_entry")
        self.file_entry.insert(0, "tx_signal.bin")
        self.file_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(0, 10))

        gen_buttons = ctk.CTkFrame(gen_body, fg_color="transparent")
        gen_buttons.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)
        gen_buttons.columnconfigure((0, 1, 2, 3), weight=1)
        ctk.CTkButton(gen_buttons, text="Generate", command=self.generate).grid(
            row=0, column=0, padx=3, sticky="ew"
        )
        ctk.CTkButton(
            gen_buttons, text="Load Preset", command=self.open_load_preset_window
        ).grid(row=0, column=1, padx=3, sticky="ew")
        ctk.CTkButton(
            gen_buttons, text="Save Preset", command=self.open_save_preset_window
        ).grid(row=0, column=2, padx=3, sticky="ew")
        ctk.CTkCheckBox(
            gen_buttons,
            text="Sync Sample Rates",
            variable=self.sync_var,
            command=lambda: self.toggle_rate_sync(self.sync_var.get()),
        ).grid(row=0, column=3, padx=3, sticky="ew")

        scroll_container = ctk.CTkFrame(gen_body)
        scroll_container.grid(row=5, column=0, columnspan=3, sticky="nsew")
        scroll_container.columnconfigure(0, weight=1)
        scroll_container.rowconfigure(0, weight=1)

        self.gen_canvas = tk.Canvas(
            scroll_container,
            bg=terminal_container_bg,
            highlightthickness=0,
        )
        self.gen_canvas.grid(row=0, column=0, sticky="nsew")
        self.gen_scroll = ctk.CTkScrollbar(
            scroll_container, orientation="vertical", command=self.gen_canvas.yview
        )
        self.gen_scroll.grid(row=0, column=1, sticky="ns")
        self.gen_scroll.grid_remove()
        self.gen_canvas.configure(yscrollcommand=None)
        self._gen_scroll_active = False

        # enable mouse wheel scrolling
        self.gen_canvas.bind("<Enter>", self._bind_gen_mousewheel)
        self.gen_canvas.bind("<Leave>", self._unbind_gen_mousewheel)

        self.gen_plots_frame = ctk.CTkFrame(
            self.gen_canvas,
            fg_color="transparent",
            corner_radius=terminal_container_corner,
        )
        self.gen_plots_frame.columnconfigure(0, weight=1)
        self.gen_plots_window = self.gen_canvas.create_window(
            (0, 0), window=self.gen_plots_frame, anchor="n"
        )
        self.gen_plots_frame.bind(
            "<Configure>",
            lambda _e: (
                self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox("all")),
                self._update_gen_scrollbar(),
            ),
        )
        self.gen_canvas.bind(
            "<Configure>",
            lambda _e: (
                self._center_canvas_window(self.gen_canvas, self.gen_plots_window),
                self._update_gen_scrollbar(),
            ),
        )
        gen_body.rowconfigure(5, weight=1)
        self.gen_canvases = []
        self.latest_data = None
        self.latest_fs = 0.0

        # ----- Column 2: Transmit -----
        tx_frame, tx_body = _make_section(self, "Transmit")
        tx_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        tx_body.columnconfigure(0, weight=1)
        tx_body.columnconfigure(1, weight=0)

        tx_params_frame, tx_params_body, _ = _make_side_bordered_group(
            tx_body,
            "Parameters",
        )
        tx_params_frame.grid(row=0, column=0, sticky="ew")
        tx_params_left = ctk.CTkFrame(tx_params_body, fg_color="transparent")
        tx_params_left.grid(row=0, column=1, sticky="nsew")
        tx_params_left.columnconfigure(1, weight=1)
        tx_params_right = ctk.CTkFrame(tx_params_body, fg_color="transparent")
        tx_params_right.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        tx_params_right.columnconfigure(1, weight=1)

        ctk.CTkLabel(tx_params_left, text="Args", anchor="e").grid(
            row=0, column=0, sticky="e", padx=label_padx
        )
        self.tx_args = SuggestEntry(tx_params_left, "tx_args")
        self.tx_args.insert(0, "addr=192.168.10.2")
        self.tx_args.grid(row=0, column=1, sticky="ew")

        ctk.CTkLabel(tx_params_right, text="Rate", anchor="e").grid(
            row=0, column=0, sticky="e", padx=label_padx
        )
        self.tx_rate = SuggestEntry(
            tx_params_right, "tx_rate", textvariable=self.tx_rate_var
        )
        self.tx_rate.grid(row=0, column=1, sticky="ew")

        ctk.CTkLabel(tx_params_left, text="Freq", anchor="e").grid(
            row=1, column=0, sticky="e", padx=label_padx
        )
        self.tx_freq = SuggestEntry(tx_params_left, "tx_freq")
        self.tx_freq.insert(0, "5.18e9")
        self.tx_freq.grid(row=1, column=1, sticky="ew")

        ctk.CTkLabel(tx_params_right, text="Gain", anchor="e").grid(
            row=1, column=0, sticky="e", padx=label_padx
        )
        self.tx_gain = SuggestEntry(tx_params_right, "tx_gain")
        self.tx_gain.insert(0, "30")
        self.tx_gain.grid(row=1, column=1, sticky="ew")

        tx_file_frame, tx_file_body, _ = _make_side_bordered_group(tx_body, "File")
        tx_file_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.tx_file = SuggestEntry(tx_file_body, "tx_file")
        self.tx_file.insert(0, "tx_signal.bin")
        self.tx_file.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(0, 10))
        self.tx_file.entry.bind(
            "<FocusOut>",
            lambda _e: self._reset_manual_xcorr_lags("TX-Datei geändert"),
        )

        tx_entry_border = ctk.ThemeManager.theme["CTkEntry"].get("border_color")
        if isinstance(tx_entry_border, (list, tuple)):
            tx_entry_border = tx_entry_border[0]
        for entry in (self.tx_args, self.tx_rate, self.tx_freq, self.tx_gain, self.tx_file):
            entry.entry.configure(border_width=2, border_color=tx_entry_border)

        btn_frame = ctk.CTkFrame(tx_body)
        btn_frame.grid(row=2, column=0, pady=5)
        btn_frame.columnconfigure((0, 1, 2), weight=1)

        self.tx_button = ctk.CTkButton(btn_frame, text="Transmit", command=self.transmit)
        self.tx_button.grid(row=0, column=0, padx=2)
        self._set_tx_indicator_image(self._tx_indicator_blank)

        self.tx_retrans = ctk.CTkButton(
            btn_frame, text="Retransmit", command=self.retransmit, state="disabled"
        )
        self.tx_retrans.grid(row=0, column=1, padx=2)

        self.tx_stop = ctk.CTkButton(
            btn_frame, text="Stop", command=self.stop_transmit, state="disabled"
        )
        self.tx_stop.grid(row=0, column=2, padx=2)

        log_frame = ctk.CTkFrame(
            tx_body,
            fg_color=terminal_container_fg,
            corner_radius=terminal_container_corner,
        )
        log_frame.grid(row=3, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.tx_log = ctk.CTkTextbox(
            log_frame,
            height=150,
            wrap="word",
            activate_scrollbars=False,
        )
        self.tx_log.grid(row=0, column=0, sticky="nsew")
        self.tx_log_scroll = ctk.CTkScrollbar(
            log_frame, orientation="vertical", command=self.tx_log.yview
        )
        self.tx_log_scroll.grid(row=0, column=1, sticky="ns")
        self.tx_log_scroll.grid_remove()
        self.tx_log.configure(yscrollcommand=self._on_tx_log_scroll)
        self.tx_log.bind("<Configure>", lambda _e: self._update_tx_log_scrollbar())
        log_frame.bind("<Configure>", lambda _e: self._update_tx_log_scrollbar())
        tx_body.rowconfigure(3, weight=1)

        # ----- Column 3: Receive -----
        rx_frame, rx_body = _make_section(self, "Receive")
        rx_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        rx_body.columnconfigure(0, weight=1)
        rx_body.rowconfigure(0, weight=1)

        rx_tabs = ctk.CTkTabview(rx_body)
        rx_tabs.grid(row=0, column=0, sticky="nsew")
        rx_tabs.configure(command=self._on_rx_tab_change)
        self.rx_tabs = rx_tabs

        rx_single_tab = rx_tabs.add("Single")
        rx_single_tab.columnconfigure((0, 1), weight=1)

        rx_params_frame, rx_params_body, _ = _make_side_bordered_group(
            rx_single_tab,
            "Parameters",
        )
        rx_params_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        rx_params_left = ctk.CTkFrame(rx_params_body, fg_color="transparent")
        rx_params_left.grid(row=0, column=1, sticky="nsew")
        rx_params_left.columnconfigure(1, weight=1)
        rx_params_right = ctk.CTkFrame(rx_params_body, fg_color="transparent")
        rx_params_right.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        rx_params_right.columnconfigure(1, weight=1)

        ctk.CTkLabel(rx_params_left, text="Rate", anchor="e").grid(
            row=0, column=0, sticky="e", padx=label_padx
        )
        self.rx_rate = SuggestEntry(
            rx_params_left, "rx_rate", textvariable=self.rx_rate_var
        )
        self.rx_rate.grid(row=0, column=1, sticky="ew")
        self.rx_rate.entry.bind("<FocusOut>", lambda _e: self.auto_update_rx_filename())

        ctk.CTkLabel(rx_params_right, text="Freq", anchor="e").grid(
            row=0, column=0, sticky="e", padx=label_padx
        )
        self.rx_freq = SuggestEntry(rx_params_right, "rx_freq")
        self.rx_freq.insert(0, "5.18e9")
        self.rx_freq.grid(row=0, column=1, sticky="ew")
        self.rx_freq.entry.bind("<FocusOut>", lambda _e: self.auto_update_rx_filename())

        ctk.CTkLabel(rx_params_left, text="Duration", anchor="e").grid(
            row=1, column=0, sticky="e", padx=label_padx
        )
        self.rx_dur = SuggestEntry(rx_params_left, "rx_dur")
        self.rx_dur.insert(0, "0.01")
        self.rx_dur.grid(row=1, column=1, sticky="ew")
        self.rx_dur.entry.bind("<FocusOut>", lambda _e: self.auto_update_rx_filename())

        ctk.CTkLabel(rx_params_right, text="Gain", anchor="e").grid(
            row=1, column=0, sticky="e", padx=label_padx
        )
        self.rx_gain = SuggestEntry(rx_params_right, "rx_gain")
        self.rx_gain.insert(0, "80")
        self.rx_gain.grid(row=1, column=1, sticky="ew")
        self.rx_gain.entry.bind("<FocusOut>", lambda _e: self.auto_update_rx_filename())

        ctk.CTkLabel(rx_params_body, text="Args", anchor="e").grid(
            row=1, column=0, sticky="e", padx=label_padx
        )
        self.rx_args = SuggestEntry(rx_params_body, "rx_args")
        self.rx_args.insert(0, "addr=192.168.20.2,clock_source=external")
        self.rx_args.grid(row=1, column=1, columnspan=2, sticky="ew")

        self.rx_channel_2 = tk.BooleanVar(value=False)
        rx_ant_frame, rx_ant_body, _ = _make_side_bordered_group(
            rx_single_tab,
            "Antenne 2",
            toggle_var=self.rx_channel_2,
        )
        rx_ant_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        rx_ant_content = ctk.CTkFrame(rx_ant_body, fg_color="transparent")
        rx_ant_content.grid(row=0, column=1, columnspan=2, sticky="nsew")
        rx_ant_content.columnconfigure(1, weight=1)
        rx_ant_content.columnconfigure(3, weight=1)

        self.rx_channel_view_label = ctk.CTkLabel(
            rx_ant_content, text="RX Ansicht", anchor="e"
        )
        self.rx_channel_view_label.grid(
            row=0, column=0, sticky="e", padx=label_padx
        )
        self.rx_channel_view_box = ctk.CTkComboBox(
            rx_ant_content,
            variable=self.rx_channel_view,
            values=["Kanal 1", "Kanal 2", "Differenz"],
            width=140,
            command=lambda _value: self.update_trim(),
        )
        self.rx_channel_view_box.grid(row=0, column=1, sticky="w", padx=(0, 8))
        self.rx_channel_view_box.configure(state="disabled")

        ctk.CTkLabel(rx_ant_content, text="Wellenlänge [m]", anchor="e").grid(
            row=0, column=2, sticky="e", padx=label_padx
        )
        self.rx_wavelength = SuggestEntry(rx_ant_content, "rx_wavelength")
        self.rx_wavelength.insert(0, "3e8/5.18e9")
        self.rx_wavelength.grid(row=0, column=3, sticky="ew", padx=(0, 8))

        self.rx_aoa_label = ctk.CTkLabel(rx_ant_content, text="AoA (ESPRIT): deaktiviert")
        self.rx_aoa_label.grid(
            row=1, column=0, sticky="w", padx=label_padx, pady=(4, 0)
        )
        self.rx_echo_aoa_label = ctk.CTkLabel(rx_ant_content, text="Echo AoA: deaktiviert")
        self.rx_echo_aoa_label.grid(
            row=1, column=1, sticky="w", padx=label_padx, pady=(4, 0)
        )

        ctk.CTkLabel(rx_ant_content, text="Antennenabstand [m]", anchor="e").grid(
            row=1, column=2, sticky="e", padx=label_padx, pady=(4, 0)
        )
        self.rx_ant_spacing = SuggestEntry(rx_ant_content, "rx_ant_spacing")
        self.rx_ant_spacing.insert(0, "0.03")
        self.rx_ant_spacing.grid(row=1, column=3, sticky="ew", padx=(0, 8), pady=(4, 0))

        rx_output_frame, rx_output_body, _ = _make_side_bordered_group(
            rx_single_tab,
            "Output",
        )
        rx_output_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.rx_file = SuggestEntry(rx_output_body, "rx_file")
        self.rx_file.insert(0, "rx_signal.bin")
        self.rx_file.grid(row=0, column=1, columnspan=2, sticky="ew")

        # --- Trim controls -------------------------------------------------
        self.trim_var = tk.BooleanVar(value=False)
        self.trim_start = tk.DoubleVar(value=0.0)
        self.trim_end = tk.DoubleVar(value=100.0)
        self.trim_dirty = False

        trim_frame, trim_body, _ = _make_side_bordered_group(
            rx_single_tab,
            "Trim",
            toggle_var=self.trim_var,
            toggle_command=self._on_trim_change,
        )
        trim_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        trim_body.grid_configure(padx=8, pady=4)
        trim_body.columnconfigure(2, weight=1)

        self.range_slider = RangeSlider(
            trim_body,
            self.trim_start,
            self.trim_end,
            command=self._on_trim_change,
        )
        self.range_slider.grid(row=0, column=2, sticky="ew", padx=4)

        self.apply_trim_btn = ctk.CTkButton(
            trim_body,
            text="Apply",
            command=self.update_trim,
        )
        self.apply_trim_btn.grid(row=0, column=4, padx=2)
        self.apply_trim_btn.configure(state="disabled")

        self.trim_start_label = ctk.CTkLabel(trim_body, width=50, text="")
        self.trim_start_label.grid(row=0, column=1, sticky="w", padx=(0, 4))
        self.trim_end_label = ctk.CTkLabel(trim_body, width=50, text="")
        self.trim_end_label.grid(row=0, column=3, sticky="e", padx=(4, 2))

        self.rx_magnitude_enable = tk.BooleanVar(value=False)
        (
            self.rx_magnitude_frame,
            self.rx_magnitude_body,
            self.rx_magnitude_check,
        ) = _make_side_bordered_group(
            rx_single_tab,
            "Betrag",
            toggle_var=self.rx_magnitude_enable,
            toggle_command=self._on_rx_magnitude_toggle,
        )
        self.rx_magnitude_frame.grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )

        self.rx_xcorr_normalized_enable = tk.BooleanVar(value=False)
        (
            self.rx_xcorr_normalized_frame,
            self.rx_xcorr_normalized_body,
            self.rx_xcorr_normalized_check,
        ) = _make_side_bordered_group(
            rx_single_tab,
            "Kreuzkorrelation normiert",
            toggle_var=self.rx_xcorr_normalized_enable,
            toggle_command=self._on_rx_xcorr_normalized_toggle,
        )
        self.rx_xcorr_normalized_frame.grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )

        self.rx_path_cancel_enable = tk.BooleanVar(value=False)
        (
            self.rx_path_cancel_frame,
            self.rx_path_cancel_body,
            self.rx_path_cancel_check,
        ) = _make_side_bordered_group(
            rx_single_tab,
            "LOS Cancellation",
            toggle_var=self.rx_path_cancel_enable,
            toggle_command=self._on_rx_path_cancel_toggle,
        )
        self.rx_path_cancel_frame.grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )

        rx_btn_frame = ctk.CTkFrame(rx_single_tab)
        rx_btn_frame.grid(row=7, column=0, columnspan=2, pady=(0, 5))
        rx_btn_frame.columnconfigure((0, 1, 2, 3), weight=1)

        self.rx_button = ctk.CTkButton(rx_btn_frame, text="Receive", command=self.receive)
        self.rx_button.grid(row=0, column=0, padx=2)
        self.rx_stop = ctk.CTkButton(
            rx_btn_frame, text="Stop", command=self.stop_receive, state="disabled"
        )
        self.rx_stop.grid(row=0, column=1, padx=2)
        self.rx_save_trim = ctk.CTkButton(
            rx_btn_frame, text="Save Trim", command=self.save_trimmed
        )
        self.rx_save_trim.grid(row=0, column=2, padx=2)
        ctk.CTkButton(rx_btn_frame, text="Compare", command=self.open_signal).grid(
            row=0, column=3, padx=2
        )

        rx_scroll_container = ctk.CTkFrame(
            rx_single_tab,
            fg_color=terminal_container_fg,
            corner_radius=terminal_container_corner,
        )
        rx_scroll_container.grid(row=8, column=0, columnspan=2, sticky="nsew")
        rx_scroll_container.columnconfigure(0, weight=1)
        rx_scroll_container.rowconfigure(0, weight=1)

        self.rx_canvas = tk.Canvas(
            rx_scroll_container,
            bg=terminal_container_bg,
            highlightthickness=0,
        )
        self.rx_canvas.grid(row=0, column=0, sticky="nsew")
        self.rx_vscroll = ctk.CTkScrollbar(
            rx_scroll_container, orientation="vertical", command=self.rx_canvas.yview
        )
        self.rx_vscroll.grid(row=0, column=1, sticky="ns")
        self.rx_vscroll.grid_remove()
        self.rx_canvas.configure(yscrollcommand=None)
        self.rx_canvas.bind(
            "<Enter>",
            lambda event, tab_name="Single": self._bind_rx_mousewheel(
                event, tab_name
            ),
        )
        self.rx_canvas.bind("<Leave>", self._unbind_rx_mousewheel)

        self.rx_plots_frame = ctk.CTkFrame(
            self.rx_canvas,
            fg_color=terminal_container_fg,
            corner_radius=terminal_container_corner,
        )
        self.rx_plots_frame.columnconfigure(0, weight=1)
        self.rx_plots_window = self.rx_canvas.create_window(
            (0, 0), window=self.rx_plots_frame, anchor="n"
        )
        self.rx_plots_frame.bind(
            "<Configure>",
            lambda _e, tab_name="Single": (
                self.rx_canvas.configure(scrollregion=self.rx_canvas.bbox("all")),
                self._update_rx_scrollbar(tab_name),
            ),
        )
        self.rx_canvas.bind(
            "<Configure>",
            lambda _e, tab_name="Single": (
                self._center_canvas_window(self.rx_canvas, self.rx_plots_window),
                self._update_rx_scrollbar(tab_name),
            ),
        )
        rx_single_tab.rowconfigure(8, weight=1)

        rx_continuous_tab = rx_tabs.add("Continuous")
        rx_continuous_tab.columnconfigure((0, 1), weight=1)

        rx_cont_params_frame, rx_cont_params_body, _ = _make_side_bordered_group(
            rx_continuous_tab,
            "Parameters",
        )
        rx_cont_params_frame.grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )
        rx_cont_params_left = ctk.CTkFrame(rx_cont_params_body, fg_color="transparent")
        rx_cont_params_left.grid(row=0, column=1, sticky="nsew")
        rx_cont_params_left.columnconfigure(1, weight=1)
        rx_cont_params_right = ctk.CTkFrame(rx_cont_params_body, fg_color="transparent")
        rx_cont_params_right.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        rx_cont_params_right.columnconfigure(1, weight=1)

        ctk.CTkLabel(rx_cont_params_left, text="Rate", anchor="e").grid(
            row=0, column=0, sticky="e", padx=label_padx
        )
        self.rx_cont_rate = SuggestEntry(rx_cont_params_left, "rx_cont_rate")
        self.rx_cont_rate.insert(0, "200e6")
        self.rx_cont_rate.grid(row=0, column=1, sticky="ew")

        ctk.CTkLabel(rx_cont_params_right, text="Freq", anchor="e").grid(
            row=0, column=0, sticky="e", padx=label_padx
        )
        self.rx_cont_freq = SuggestEntry(rx_cont_params_right, "rx_cont_freq")
        self.rx_cont_freq.insert(0, "5.18e9")
        self.rx_cont_freq.grid(row=0, column=1, sticky="ew")

        ctk.CTkLabel(rx_cont_params_left, text="Ring (s)", anchor="e").grid(
            row=1, column=0, sticky="e", padx=label_padx
        )
        self.rx_cont_ring_seconds = SuggestEntry(
            rx_cont_params_left, "rx_cont_ring_seconds"
        )
        self.rx_cont_ring_seconds.insert(0, "4.0")
        self.rx_cont_ring_seconds.grid(row=1, column=1, sticky="ew")

        ctk.CTkLabel(rx_cont_params_right, text="Gain", anchor="e").grid(
            row=1, column=0, sticky="e", padx=label_padx
        )
        self.rx_cont_gain = SuggestEntry(rx_cont_params_right, "rx_cont_gain")
        self.rx_cont_gain.insert(0, "80")
        self.rx_cont_gain.grid(row=1, column=1, sticky="ew")

        ctk.CTkLabel(rx_cont_params_left, text="Restart margin (s)", anchor="e").grid(
            row=2, column=0, sticky="e", padx=label_padx
        )
        self.rx_cont_restart_margin = SuggestEntry(
            rx_cont_params_left, "rx_cont_restart_margin"
        )
        self.rx_cont_restart_margin.insert(0, "1.5")
        self.rx_cont_restart_margin.grid(row=2, column=1, sticky="ew")

        ctk.CTkLabel(rx_cont_params_body, text="Args", anchor="e").grid(
            row=1, column=0, sticky="e", padx=label_padx
        )
        self.rx_cont_args = SuggestEntry(rx_cont_params_body, "rx_cont_args")
        self.rx_cont_args.insert(0, "addr=192.168.20.2,clock_source=external")
        self.rx_cont_args.grid(row=1, column=1, columnspan=2, sticky="ew")

        rx_cont_snippet_frame, rx_cont_snippet_body, _ = _make_side_bordered_group(
            rx_continuous_tab,
            "Snippet",
        )
        rx_cont_snippet_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )
        rx_cont_snippet_body.columnconfigure(1, weight=1)
        rx_cont_snippet_body.columnconfigure(3, weight=1)

        ctk.CTkLabel(rx_cont_snippet_body, text="Snippet (s)", anchor="e").grid(
            row=0, column=0, sticky="e", padx=label_padx
        )
        self.rx_cont_snippet_seconds = SuggestEntry(
            rx_cont_snippet_body, "rx_cont_snippet_seconds"
        )
        self.rx_cont_snippet_seconds.insert(0, "0.05")
        self.rx_cont_snippet_seconds.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        ctk.CTkLabel(rx_cont_snippet_body, text="Interval (s)", anchor="e").grid(
            row=0, column=2, sticky="e", padx=label_padx
        )
        self.rx_cont_snippet_interval = SuggestEntry(
            rx_cont_snippet_body, "rx_cont_snippet_interval"
        )
        self.rx_cont_snippet_interval.insert(0, "1.0")
        self.rx_cont_snippet_interval.grid(
            row=0, column=3, sticky="ew", padx=(0, 8)
        )

        rx_cont_output_frame, rx_cont_output_body, _ = _make_side_bordered_group(
            rx_continuous_tab,
            "Output",
        )
        rx_cont_output_frame.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )
        self.rx_cont_output_prefix = SuggestEntry(
            rx_cont_output_body, "rx_cont_output_prefix"
        )
        self.rx_cont_output_prefix.insert(0, "signals/rx/snippet")
        self.rx_cont_output_prefix.grid(row=0, column=1, columnspan=2, sticky="ew")

        rx_cont_btn_frame = ctk.CTkFrame(rx_continuous_tab)
        rx_cont_btn_frame.grid(row=3, column=0, columnspan=2, pady=(0, 5))
        rx_cont_btn_frame.columnconfigure((0, 1), weight=1)
        self.rx_cont_start = ctk.CTkButton(
            rx_cont_btn_frame,
            text="Start",
            command=self.start_continuous,
        )
        self.rx_cont_start.grid(row=0, column=0, padx=2)
        self.rx_cont_stop = ctk.CTkButton(
            rx_cont_btn_frame,
            text="Stop",
            command=self.stop_continuous,
            state="disabled",
        )
        self.rx_cont_stop.grid(row=0, column=1, padx=2)
        rx_cont_scroll_container = ctk.CTkFrame(
            rx_continuous_tab,
            fg_color=terminal_container_fg,
            corner_radius=terminal_container_corner,
        )
        rx_cont_scroll_container.grid(row=4, column=0, columnspan=2, sticky="nsew")
        rx_cont_scroll_container.columnconfigure(0, weight=1)
        rx_cont_scroll_container.rowconfigure(0, weight=1)

        self.rx_cont_canvas = tk.Canvas(
            rx_cont_scroll_container,
            bg=terminal_container_bg,
            highlightthickness=0,
        )
        self.rx_cont_canvas.grid(row=0, column=0, sticky="nsew")
        self.rx_cont_vscroll = ctk.CTkScrollbar(
            rx_cont_scroll_container,
            orientation="vertical",
            command=self.rx_cont_canvas.yview,
        )
        self.rx_cont_vscroll.grid(row=0, column=1, sticky="ns")
        self.rx_cont_vscroll.grid_remove()
        self.rx_cont_canvas.configure(yscrollcommand=None)
        self.rx_cont_canvas.bind(
            "<Enter>",
            lambda event, tab_name="Continuous": self._bind_rx_mousewheel(
                event, tab_name
            ),
        )
        self.rx_cont_canvas.bind("<Leave>", self._unbind_rx_mousewheel)

        self.rx_cont_plots_frame = ctk.CTkFrame(
            self.rx_cont_canvas,
            fg_color=terminal_container_fg,
            corner_radius=terminal_container_corner,
        )
        self.rx_cont_plots_frame.columnconfigure(0, weight=1)
        self.rx_cont_plots_window = self.rx_cont_canvas.create_window(
            (0, 0), window=self.rx_cont_plots_frame, anchor="n"
        )
        self.rx_cont_plots_frame.bind(
            "<Configure>",
            lambda _e, tab_name="Continuous": (
                self.rx_cont_canvas.configure(
                    scrollregion=self.rx_cont_canvas.bbox("all")
                ),
                self._update_rx_scrollbar(tab_name),
            ),
        )
        self.rx_cont_canvas.bind(
            "<Configure>",
            lambda _e, tab_name="Continuous": (
                self._center_canvas_window(
                    self.rx_cont_canvas, self.rx_cont_plots_window
                ),
                self._update_rx_scrollbar(tab_name),
            ),
        )
        rx_continuous_tab.rowconfigure(4, weight=1)

        self._rx_scroll_active: dict[str, bool] = {
            "Single": False,
            "Continuous": False,
        }
        self._rx_plot_containers = {
            "Single": {
                "name": "Single",
                "canvas": self.rx_canvas,
                "scrollbar": self.rx_vscroll,
                "frame": self.rx_plots_frame,
                "window": self.rx_plots_window,
            },
            "Continuous": {
                "name": "Continuous",
                "canvas": self.rx_cont_canvas,
                "scrollbar": self.rx_cont_vscroll,
                "frame": self.rx_cont_plots_frame,
                "window": self.rx_cont_plots_window,
            },
        }
        self.rx_canvases: dict[str, list[object]] = {
            "Single": [],
            "Continuous": [],
        }
        self.update_waveform_fields()
        self.auto_update_tx_filename()
        self.auto_update_rx_filename()
        self.toggle_rate_sync(self.sync_var.get())
        self.update_trim()
        _apply_input_margins(self)
        _apply_button_cursor(self)

    def update_waveform_fields(self) -> None:
        """Show or hide waveform specific parameters."""
        w = self.wave_var.get().lower()

        # hide all optional fields first
        self.f_label.grid_remove()
        self.f_entry.grid_remove()
        self.f1_label.grid_remove()
        self.f1_entry.grid_remove()
        self.q_label.grid_remove()
        self.q_entry.grid_remove()
        self.pn_chip_label.grid_remove()
        self.pn_chip_entry.grid_remove()
        self.pn_seed_label.grid_remove()
        self.pn_seed_entry.grid_remove()
        self.ofdm_nfft_label.grid_remove()
        self.ofdm_nfft_entry.grid_remove()
        self.ofdm_cp_label.grid_remove()
        self.ofdm_cp_entry.grid_remove()
        self.ofdm_symbols_label.grid_remove()
        self.ofdm_symbols_entry.grid_remove()
        self.ofdm_short_label.grid_remove()
        self.ofdm_short_entry.grid_remove()

        filter_visible = w == "zadoffchu"
        if filter_visible:
            self.fdz_label.grid(row=0, column=0, sticky="e", padx=self._label_padx)
            self.filter_bandwidth_label.grid(
                row=0, column=0, sticky="e", padx=self._label_padx
            )
            self.filter_bandwidth_entry.grid(row=0, column=1, sticky="ew")
            state = "normal" if self.fdz_enable.get() else "disabled"
            self.filter_bandwidth_entry.entry.configure(state=state)
        else:
            self.fdz_label.grid_remove()
            self.filter_bandwidth_label.grid_remove()
            self.filter_bandwidth_entry.grid_remove()
            self.filter_bandwidth_entry.entry.configure(state="disabled")

        if w == "sinus":
            self.f_label.configure(text="f")
            self.f_label.grid(row=1, column=0, sticky="e", padx=self._label_padx)
            self.f_entry.grid(row=1, column=1, sticky="ew")
        elif w == "doppelsinus":
            self.f_label.configure(text="f1")
            self.f_label.grid(row=1, column=0, sticky="e", padx=self._label_padx)
            self.f_entry.grid(row=1, column=1, sticky="ew")
            self.f1_label.configure(text="f2")
            self.f1_label.grid(row=2, column=0, sticky="e", padx=self._label_padx)
            self.f1_entry.grid(row=2, column=1, sticky="ew")
        elif w == "zadoffchu":
            self.q_label.grid(row=1, column=0, sticky="e", padx=self._label_padx)
            self.q_entry.grid(row=1, column=1, sticky="ew")
        elif w == "chirp":
            self.f_label.configure(text="f0")
            self.f_label.grid(row=1, column=0, sticky="e", padx=self._label_padx)
            self.f_entry.grid(row=1, column=1, sticky="ew")
            self.f1_label.grid(row=2, column=0, sticky="e", padx=self._label_padx)
            self.f1_entry.grid(row=2, column=1, sticky="ew")
        elif w == "ofdm_preamble":
            self.ofdm_nfft_label.grid(row=3, column=0, sticky="e", padx=self._label_padx)
            self.ofdm_nfft_entry.grid(row=3, column=1, sticky="ew")
            self.ofdm_cp_label.grid(row=4, column=0, sticky="e", padx=self._label_padx)
            self.ofdm_cp_entry.grid(row=4, column=1, sticky="ew")
            self.ofdm_symbols_label.grid(
                row=5, column=0, sticky="e", padx=self._label_padx
            )
            self.ofdm_symbols_entry.grid(row=5, column=1, sticky="ew")
            self.ofdm_short_label.grid(
                row=6, column=0, sticky="e", padx=self._label_padx
            )
            self.ofdm_short_entry.grid(row=6, column=1, sticky="ew")
        elif w == "pseudo_noise":
            self.pn_chip_label.grid(row=1, column=0, sticky="e", padx=self._label_padx)
            self.pn_chip_entry.grid(row=1, column=1, sticky="ew")
            self.pn_seed_label.grid(row=2, column=0, sticky="e", padx=self._label_padx)
            self.pn_seed_entry.grid(row=2, column=1, sticky="ew")

        self.auto_update_tx_filename()
        _apply_input_margins(self)

    def _is_filter_active(self) -> bool:
        mode = (self.filter_mode_var.get() or "frequency_domain_zeroing").strip().lower()
        return (
            self.fdz_enable.get()
            and self.wave_var.get().lower() == "zadoffchu"
            and mode == "frequency_domain_zeroing"
        )

    def _on_filter_bandwidth_changed(self) -> None:
        self.auto_update_tx_filename()
        self._reset_manual_xcorr_lags(
            "frequency-domain zeroing/Bandbreite geändert"
        )

    def _on_filter_toggle(self) -> None:
        state = "normal" if self.fdz_enable.get() else "disabled"
        self.filter_bandwidth_entry.entry.configure(state=state)
        self.auto_update_tx_filename()
        self._reset_manual_xcorr_lags(
            "frequency-domain zeroing/Bandbreite geändert"
        )

    def _get_repeat_count(self) -> int:
        try:
            return int(self.repeat_entry.get())
        except Exception:
            return 1

    def _tx_transmit_file(self) -> str:
        if getattr(self, "_zeros_tx_file", None):
            return self._zeros_tx_file
        if getattr(self, "_repeat_tx_file", None):
            return self._repeat_tx_file
        if self._is_filter_active():
            return self._filtered_tx_file or self.tx_file.get()
        return self.tx_file.get()

    def _tx_transmit_file_for_start(self) -> str:
        """Return the newest generated TX file when starting a transmission."""
        preferred = self._last_generated_tx_file
        if preferred:
            return preferred
        return self._tx_transmit_file()

    def _on_repeat_toggle(self) -> None:
        if self.repeat_enable.get():
            self.repeat_entry.entry.configure(state="normal")
            if self.repeat_entry.get() in ("", "0"):
                self.repeat_entry.delete(0, tk.END)
                self.repeat_entry.insert(0, self._repeat_last_value or "1")
        else:
            current = self.repeat_entry.get()
            if current and current != "0":
                self._repeat_last_value = current
            self.repeat_entry.delete(0, tk.END)
            self.repeat_entry.insert(0, "0")
            self.repeat_entry.entry.configure(state="disabled")
        self.auto_update_tx_filename()

    def _on_zeros_toggle(self) -> None:
        if self.zeros_enable.get():
            self.zeros_combo.configure(state="normal")
            if self.zeros_var.get() not in self.zeros_values:
                self.zeros_var.set(self._zeros_last_value or "same")
        else:
            current = self.zeros_var.get()
            if current:
                self._zeros_last_value = current
            self.zeros_combo.configure(state="disabled")
        self.auto_update_tx_filename()

    def _on_rx_magnitude_toggle(self) -> None:
        self._reset_manual_xcorr_lags("Betrag geändert")
        self.update_trim()

    def _on_rx_xcorr_normalized_toggle(self) -> None:
        normalized_enabled = bool(self.rx_xcorr_normalized_enable.get())
        if self._cont_runtime_config:
            self._cont_runtime_config["xcorr_normalized_enabled"] = normalized_enabled
            self._cont_runtime_config["normalize_enabled"] = normalized_enabled
        self._reset_manual_xcorr_lags("XCorr-Normierung geändert")
        self.update_trim()

    def _on_rx_path_cancel_toggle(self) -> None:
        self._reset_manual_xcorr_lags("Pfad-Cancellation geändert")
        self.update_trim()

    def _apply_path_cancellation(
        self, data: np.ndarray, ref_data: np.ndarray
    ) -> tuple[np.ndarray, dict[str, object]]:
        return apply_path_cancellation(
            data, ref_data, manual_lags=self.manual_xcorr_lags
        )

    def _update_path_cancellation_status(self) -> None:
        """Placeholder to keep path cancellation controls in sync."""
        if hasattr(self, "rx_path_cancel_check"):
            self.rx_path_cancel_check.configure(state="normal")

    def _reset_manual_xcorr_lags(self, reason: str | None = None) -> None:
        if reason == "TX-Datei geändert":
            self._cached_tx_path = None
            self._cached_tx_data = np.array([], dtype=np.complex64)
            self._cached_tx_load_error_path = None
        if self.manual_xcorr_lags.get("los") is None and self.manual_xcorr_lags.get(
            "echo"
        ) is None:
            return
        self.manual_xcorr_lags = {"los": None, "echo": None}
        self._xcorr_manual_file = None
        if reason:
            text = f"Manuelle Marker zurückgesetzt ({reason})"
        else:
            text = "Manuelle Marker zurückgesetzt"
        self._show_toast(text)

    def auto_update_tx_filename(self) -> None:
        """Update TX filename entry based on current parameters."""
        previous = self.tx_file.get()
        name = _gen_tx_filename(self)
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, name)
        self.tx_file.delete(0, tk.END)
        if self._is_filter_active():
            filtered_name = _gen_filtered_tx_filename(name)
            self._filtered_tx_file = filtered_name
            base_name = filtered_name
        else:
            self._filtered_tx_file = None
            base_name = name
        repeats = self._get_repeat_count()
        if repeats > 1:
            repeat_name = _gen_repeat_tx_filename(base_name)
            self._repeat_tx_file = repeat_name
            base_name = repeat_name
        else:
            self._repeat_tx_file = None
        zeros_enabled = self.zeros_enable.get()
        if zeros_enabled:
            zeros_name = _gen_zeros_tx_filename(base_name)
            self._zeros_tx_file = zeros_name
            self.tx_file.insert(0, zeros_name)
        else:
            self._zeros_tx_file = None
            self.tx_file.insert(0, base_name)
        if previous != self.tx_file.get():
            self._reset_manual_xcorr_lags("TX-Datei geändert")

    def auto_update_rx_filename(self) -> None:
        """Update RX filename entry based on current parameters."""
        name = _gen_rx_filename(self)
        self.rx_file.delete(0, tk.END)
        self.rx_file.insert(0, name)

    def toggle_rate_sync(self, enable: bool) -> None:
        """Enable or disable rate synchronization between TX and RX."""
        if enable:
            self.rate_var.set(
                self.fs_entry.get() or self.tx_rate.get() or self.rx_rate.get()
            )
            self.fs_var = self.rate_var
            self.tx_rate_var = self.rate_var
            self.rx_rate_var = self.rate_var
        else:
            self.fs_var = tk.StringVar(value=self.fs_entry.get())
            self.tx_rate_var = tk.StringVar(value=self.tx_rate.get())
            self.rx_rate_var = tk.StringVar(value=self.rx_rate.get())
        self.fs_entry.entry.configure(textvariable=self.fs_var)
        self.fs_entry.var = self.fs_var
        self.tx_rate.entry.configure(textvariable=self.tx_rate_var)
        self.tx_rate.var = self.tx_rate_var
        self.rx_rate.entry.configure(textvariable=self.rx_rate_var)
        self.rx_rate.var = self.rx_rate_var

    def _render_gen_tab(
        self,
        frame: ctk.CTkFrame,
        data: np.ndarray,
        fs: float,
        symbol_rate: float | None = None,
        corr_mode: str = "Autocorr",
        corr_ref: np.ndarray | None = None,
    ) -> None:
        modes = ["Signal", "Freq", corr_mode]
        for idx, mode in enumerate(modes):
            fig = Figure(figsize=(5, 2), dpi=100)
            ax = fig.add_subplot(111)
            _apply_mpl_transparent(fig, ax)
            _plot_on_mpl(ax, data, fs, mode, f"TX {mode}", ref_data=corr_ref)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.configure(bg=_resolve_ctk_frame_bg(frame))
            widget.grid(row=idx, column=0, sticky="n", pady=2)
            widget.bind(
                "<Button-1>",
                lambda _e, m=mode, d=data, s=fs: self._show_fullscreen(
                    d, s, m, f"TX {m}"
                ),
            )
            self.gen_canvases.append(canvas)

        stats = _calc_stats(data, fs, symbol_rate=symbol_rate)
        stats_rows = _format_stats_rows(stats, include_bw_extras=False)
        stats_frame = ctk.CTkFrame(frame, fg_color="transparent")
        stats_frame.grid(row=len(modes), column=0, sticky="ew", pady=2)
        stats_frame.columnconfigure((0, 1, 2, 3), weight=1)
        for idx, (label, value) in enumerate(stats_rows):
            row = idx // 2
            col = (idx % 2) * 2
            ctk.CTkLabel(
                stats_frame,
                justify="right",
                anchor="e",
                text=f"{label}:",
            ).grid(
                row=row, column=col, sticky="e", padx=6
            )
            ctk.CTkLabel(
                stats_frame,
                justify="left",
                anchor="w",
                text=value,
            ).grid(
                row=row, column=col + 1, sticky="w", padx=6
            )

    def _display_gen_plots(
        self,
        data: np.ndarray,
        fs: float,
        filtered_data: np.ndarray | None = None,
        filtered_fs: float | None = None,
        repeated_data: np.ndarray | None = None,
        repeated_fs: float | None = None,
        zeros_data: np.ndarray | None = None,
        zeros_fs: float | None = None,
        symbol_rate: float | None = None,
        filtered_symbol_rate: float | None = None,
        repeated_symbol_rate: float | None = None,
        zeros_symbol_rate: float | None = None,
    ) -> None:
        """Render preview plots below the generation parameters."""
        if zeros_data is not None:
            self.latest_data = zeros_data
            self.latest_fs = zeros_fs if zeros_fs is not None else fs
        elif repeated_data is not None:
            self.latest_data = repeated_data
            self.latest_fs = repeated_fs if repeated_fs is not None else fs
        elif filtered_data is not None:
            self.latest_data = filtered_data
            self.latest_fs = filtered_fs if filtered_fs is not None else fs
        else:
            self.latest_data = data
            self.latest_fs = fs

        for child in self.gen_plots_frame.winfo_children():
            child.destroy()
        self.gen_canvases.clear()

        plot_top_pad = (8, 0)
        if filtered_data is None and repeated_data is None and zeros_data is None:
            tab_frame = ctk.CTkFrame(self.gen_plots_frame)
            tab_frame.grid(row=0, column=0, sticky="nsew", pady=plot_top_pad)
            tab_frame.columnconfigure(0, weight=1)
            self._render_gen_tab(tab_frame, data, fs, symbol_rate=symbol_rate)
            return

        notebook = ctk.CTkTabview(self.gen_plots_frame)
        notebook.grid(row=0, column=0, sticky="nsew", pady=plot_top_pad)
        self.gen_plots_frame.columnconfigure(0, weight=1)

        corr_reference = filtered_data if filtered_data is not None else data
        corr_reference = _strip_trailing_zeros(corr_reference)
        if corr_reference.size == 0:
            corr_reference = None

        tabs: list[tuple[str, np.ndarray, float, float | None, str, np.ndarray | None]] = []
        if filtered_data is None:
            tabs.append(("Signal", data, fs, symbol_rate, "Autocorr", None))
        else:
            tabs.append(("Ungefiltert", data, fs, symbol_rate, "Autocorr", None))
            tabs.append(
                (
                    "Gefiltert",
                    filtered_data,
                    filtered_fs if filtered_fs is not None else fs,
                    filtered_symbol_rate,
                    "Autocorr",
                    None,
                )
            )

        if repeated_data is not None:
            repeat_label = (
                "Gefiltert + Wiederholt" if filtered_data is not None else "Signal + Wiederholt"
            )
            tabs.append(
                (
                    repeat_label,
                    repeated_data,
                    repeated_fs if repeated_fs is not None else fs,
                    repeated_symbol_rate or filtered_symbol_rate or symbol_rate,
                    "Crosscorr",
                    corr_reference,
                )
            )
        if zeros_data is not None:
            zeros_label = "Signal + Nullen"
            if filtered_data is not None and repeated_data is not None:
                zeros_label = "Gefiltert + Wiederholt + Nullen"
            elif filtered_data is not None:
                zeros_label = "Gefiltert + Nullen"
            elif repeated_data is not None:
                zeros_label = "Signal + Wiederholt + Nullen"
            tabs.append(
                (
                    zeros_label,
                    zeros_data,
                    zeros_fs if zeros_fs is not None else fs,
                    zeros_symbol_rate
                    or repeated_symbol_rate
                    or filtered_symbol_rate
                    or symbol_rate,
                    "Crosscorr",
                    corr_reference,
                )
            )

        for label, tab_data, tab_fs, tab_symbol_rate, corr_mode, corr_ref in tabs:
            tab = notebook.add(label)
            tab.columnconfigure(0, weight=1)
            self._render_gen_tab(
                tab,
                tab_data,
                tab_fs,
                symbol_rate=tab_symbol_rate,
                corr_mode=corr_mode,
                corr_ref=corr_ref,
            )

        if tabs:
            notebook.set(tabs[-1][0])

    def _select_rx_display_data(self, data: np.ndarray) -> tuple[np.ndarray, str]:
        """Return the RX data according to the channel view selection."""
        if data.ndim == 1:
            return data, ""
        if data.ndim != 2 or data.shape[0] == 0:
            return np.array([], dtype=np.complex64), ""
        if data.shape[0] < 2:
            return data[0], ""
        selection = self.rx_channel_view.get()
        if selection == "Kanal 2":
            return data[1], "Kanal 2"
        if selection == "Differenz":
            return data[0] - data[1], "Differenz"
        return data[0], "Kanal 1"

    def _get_crosscorr_reference(self) -> tuple[np.ndarray, str]:
        """Return TX reference data and a label for cross-correlation."""
        return getattr(self, "tx_data", np.array([], dtype=np.complex64)), "TX"

    def _display_rx_plots(
        self,
        data: np.ndarray,
        fs: float,
        reset_manual: bool = True,
        target_tab: str | None = None,
    ) -> None:
        """Render preview plots below the receive parameters."""
        if reset_manual:
            self._reset_manual_xcorr_lags("Neue RX-Daten")
        self.rx_stats_labels = []
        self.raw_rx_data = data
        self.latest_fs_raw = fs
        if data.ndim == 2 and data.shape[0] >= 2:
            self.rx_channel_view_box.configure(state="normal")
            selection_label = self.rx_channel_view.get()
        else:
            self.rx_channel_view.set("Kanal 1")
            self.rx_channel_view_box.configure(state="disabled")
            selection_label = ""
        data, channel_label = self._select_rx_display_data(data)
        if selection_label and not channel_label:
            channel_label = selection_label
        self.range_slider.set_data(data)
        if self.trim_var.get():
            data = self._trim_data(data)

        def _load_tx_samples(path: str) -> np.ndarray:
            raw = np.fromfile(path, dtype=np.int16)
            if raw.size % 2:
                raw = raw[:-1]
            raw = raw.reshape(-1, 2).astype(np.float32)
            return raw[:, 0] + 1j * raw[:, 1]

        tx_reference_path = _strip_zeros_tx_filename(self.tx_file.get())
        if tx_reference_path == self._cached_tx_path:
            self.tx_data = self._cached_tx_data
        else:
            try:
                tx_data = _load_tx_samples(tx_reference_path)
            except Exception as exc:
                self.tx_data = np.array([], dtype=np.complex64)
                self._cached_tx_path = tx_reference_path
                self._cached_tx_data = self.tx_data
                if self._cached_tx_load_error_path != tx_reference_path:
                    logging.warning(
                        "TX-Referenzdatei konnte nicht geladen werden (%s): %s",
                        tx_reference_path,
                        exc,
                    )
                    self._cached_tx_load_error_path = tx_reference_path
            else:
                self.tx_data = tx_data
                self._cached_tx_path = tx_reference_path
                self._cached_tx_data = tx_data
                self._cached_tx_load_error_path = None
        ref_data, ref_label = self._get_crosscorr_reference()
        if self.rx_magnitude_enable.get():
            data = np.abs(data)
            ref_data = np.abs(ref_data)
        data_uncanceled = data
        aoa_text = "AoA (ESPRIT): deaktiviert"
        echo_aoa_text = "Echo AoA: deaktiviert"
        self.echo_aoa_results = []
        aoa_time = None
        aoa_series = None

        cancel_info: dict[str, object] | None = None
        if self.rx_path_cancel_enable.get():
            data, cancel_info = self._apply_path_cancellation(data, ref_data)
            if cancel_info is not None:
                self._log_path_cancellation(cancel_info, "RX")
            self._last_path_cancel_info = cancel_info
        else:
            self._last_path_cancel_info = None

        self.latest_fs = fs
        self.latest_data = data

        target_container = self._get_rx_plot_target(target_tab)
        target_name = target_container["name"]
        target_frame = target_container["frame"]
        target_canvas = target_container["canvas"]
        target_window = target_container["window"]

        if target_name != "Continuous":
            for c in self.rx_canvases[target_name]:
                if hasattr(c, "get_tk_widget"):
                    c.get_tk_widget().destroy()
                else:
                    c.destroy()
            self.rx_canvases[target_name].clear()

        modes = ["Signal", "Freq", "Crosscorr"]
        title_suffix = f" ({channel_label})" if channel_label else ""

        def _render_rx_preview(
            target_frame: ctk.CTkFrame,
            plot_data: np.ndarray,
            plot_fs: float,
            plot_ref_data: np.ndarray,
            plot_ref_label: str,
            aoa_plot_time: np.ndarray | None,
            aoa_plot_series: np.ndarray | None,
            path_cancel_info: dict[str, object] | None,
            crosscorr_compare: np.ndarray | None,
        ) -> None:
            target_frame.columnconfigure(0, weight=1)
            target_frame.rowconfigure(0, weight=1)
            for idx, mode in enumerate(modes):
                fig = Figure(figsize=(5, 2), dpi=100)
                ax = fig.add_subplot(111)
                _apply_mpl_transparent(fig, ax)
                ref = plot_ref_data if mode == "Crosscorr" else None
                crosscorr_title = (
                    f"RX {mode}{title_suffix} ({plot_ref_label})"
                    if mode == "Crosscorr" and plot_ref_label
                    else f"RX {mode}{title_suffix}"
                )
                _plot_on_mpl(
                    ax,
                    plot_data,
                    plot_fs,
                    mode,
                    crosscorr_title,
                    ref,
                    crosscorr_compare if mode == "Crosscorr" else None,
                    manual_lags=self.manual_xcorr_lags,
                    xcorr_normalized=self.rx_xcorr_normalized_enable.get(),
                )
                canvas = FigureCanvasTkAgg(fig, master=target_frame)
                canvas.draw()
                widget = canvas.get_tk_widget()
                widget.configure(bg=_resolve_ctk_frame_bg(target_frame))
                widget.grid(row=idx, column=0, sticky="n", pady=2)
                if mode == "Crosscorr":
                    handler = (
                        lambda _e,
                        m=mode,
                        d=plot_data,
                        s=plot_fs,
                        r=ref,
                        c=crosscorr_compare,
                        t=crosscorr_title: (
                            self._show_fullscreen(
                                d, s, m, t, ref_data=r, crosscorr_compare=c
                            )
                        )
                    )
                else:
                    handler = (
                        lambda _e,
                        m=mode,
                        d=plot_data,
                        s=plot_fs: self._show_fullscreen(
                            d, s, m, f"RX {m}{title_suffix}"
                        )
                    )
                widget.bind("<Button-1>", handler)
                rx_canvas_list.append(canvas)

            stats = _calc_stats(
                plot_data,
                plot_fs,
                plot_ref_data,
                manual_lags=self.manual_xcorr_lags,
                xcorr_reduce=True,
                path_cancel_info=path_cancel_info,
                xcorr_normalized=self.rx_xcorr_normalized_enable.get(),
            )
            stats_rows = _format_rx_stats_rows(stats)
            stats_frame = ctk.CTkFrame(target_frame, fg_color="transparent")
            stats_frame.grid(row=len(modes), column=0, sticky="ew", pady=2)
            stats_frame.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)
            value_labels: list[ctk.CTkLabel] = []
            for idx, (label, value) in enumerate(stats_rows):
                row = 0 if idx < 3 else 1
                col = (idx if idx < 3 else idx - 3) * 2
                ctk.CTkLabel(
                    stats_frame,
                    justify="right",
                    anchor="e",
                    text=f"{label}:",
                ).grid(row=row, column=col, sticky="e", padx=6)
                value_label = ctk.CTkLabel(
                    stats_frame,
                    justify="left",
                    anchor="w",
                    text=value,
                )
                value_label.grid(row=row, column=col + 1, sticky="w", padx=6)
                value_labels.append(value_label)
            self.rx_stats_labels.append(value_labels)

        def _render_rx_preview_pg(
            target_frame: ctk.CTkFrame,
            plot_data: np.ndarray,
            plot_fs: float,
            plot_ref_data: np.ndarray,
            plot_ref_label: str,
            aoa_plot_time: np.ndarray | None,
            aoa_plot_series: np.ndarray | None,
            path_cancel_info: dict[str, object] | None,
            crosscorr_compare: np.ndarray | None,
        ) -> None:
            pg.mkQApp()
            target_frame.columnconfigure(0, weight=1)
            preview_width = 500
            preview_height = 240
            bg_color = _resolve_ctk_frame_bg(target_frame)
            pg_bg_color = _tk_color_to_rgb(target_frame, bg_color)
            axis_color = "#9E9E9E"
            zoom_half_window = 50

            def _crosscorr_zoom_range(
                precomputed_ctx: dict[str, object] | None,
            ) -> tuple[float, float] | None:
                if not precomputed_ctx:
                    return None
                los_lags = precomputed_ctx.get("lags2")
                if not isinstance(los_lags, np.ndarray):
                    los_lags = precomputed_ctx.get("lags")
                if not isinstance(los_lags, np.ndarray) or los_lags.size == 0:
                    return None

                def _lag_value(source_lags: np.ndarray, idx: int | None) -> float | None:
                    if idx is None or source_lags.size == 0:
                        return None
                    return float(
                        source_lags[int(np.clip(idx, 0, len(source_lags) - 1))]
                    )

                focus_lags = []
                los_idx = precomputed_ctx.get("los_idx")
                echo_indices = precomputed_ctx.get("echo_indices")
                los_lag = _lag_value(los_lags, int(los_idx) if los_idx is not None else None)
                if los_lag is not None:
                    focus_lags.append(los_lag)
                if isinstance(echo_indices, list):
                    for idx in echo_indices:
                        echo_lag = _lag_value(los_lags, int(idx) if idx is not None else None)
                        if echo_lag is not None:
                            focus_lags.append(echo_lag)
                if not focus_lags:
                    return None
                center = float(sum(focus_lags) / len(focus_lags))
                return (
                    center - zoom_half_window,
                    center + zoom_half_window,
                )

            pg_state = getattr(self, "_rx_cont_pg_state", None)
            plots_state = pg_state.get("plots") if isinstance(pg_state, dict) else None
            tabview = pg_state.get("tabview") if isinstance(pg_state, dict) else None
            active_plot_tab = "Signal"
            if tabview is not None:
                try:
                    active_plot_tab = tabview.get()
                except Exception:
                    active_plot_tab = "Signal"
            if (
                pg_state is None
                or not isinstance(pg_state, dict)
                or not plots_state
                or tabview is None
                or any(
                    not plot_info["label"].winfo_exists()
                    for plot_info in plots_state.values()
                )
            ):
                pg_state = {
                    "plots": {},
                    "tabview": None,
                    "stats_labels": {},
                    "aoa_plot": None,
                }
                tabs_map = {
                    "Signal": ("Signal", "max Amp:"),
                    "Spectrum": ("Freq", "fmin/fmax:"),
                    "X-Corr": ("Crosscorr", "LOS-Echos:"),
                }
                tabview = ctk.CTkTabview(target_frame)
                tabview.grid(row=0, column=0, sticky="nsew", pady=(2, 2))
                tabview.configure(command=self._on_rx_cont_plot_tab_change)
                for tab_name in tabs_map:
                    tab = tabview.add(tab_name)
                    tab.columnconfigure(0, weight=1)
                    tab.rowconfigure(0, weight=1)
                tabview.set("Signal")
                pg_state["tabview"] = tabview
                for tab_name, (mode, metric_label_text) in tabs_map.items():
                    plot_widget = pg.GraphicsLayoutWidget()
                    plot_widget.setBackground(pg_bg_color)
                    plot_widget.setFixedSize(preview_width, preview_height)
                    plot_item = plot_widget.addPlot()
                    plot_item.setMenuEnabled(False)
                    _style_pg_preview_axes(plot_item, axis_color)
                    label = tk.Label(tabview.tab(tab_name), bg=bg_color)
                    label.grid(row=0, column=0, sticky="nsew", pady=(2, 2))
                    stats_frame = ctk.CTkFrame(tabview.tab(tab_name), fg_color="transparent")
                    stats_frame.grid(row=1, column=0, sticky="ew", pady=2)
                    stats_frame.columnconfigure((0, 1), weight=1)
                    ctk.CTkLabel(
                        stats_frame,
                        justify="right",
                        anchor="e",
                        text=metric_label_text,
                    ).grid(row=0, column=0, sticky="e", padx=6)
                    value_label = ctk.CTkLabel(
                        stats_frame,
                        justify="left",
                        anchor="w",
                        text="--",
                    )
                    value_label.grid(row=0, column=1, sticky="w", padx=6)
                    pg_state["stats_labels"][mode] = value_label
                    pg_state["plots"][mode] = {
                        "widget": plot_widget,
                        "plot": plot_item,
                        "label": label,
                        "initialized": False,
                    }
                self._rx_cont_pg_state = pg_state
                active_plot_tab = "Signal"

            mode_by_tab = {
                "Signal": "Signal",
                "Spectrum": "Freq",
                "X-Corr": "Crosscorr",
            }
            mode = mode_by_tab.get(active_plot_tab, "Signal")
            plot_info = pg_state["plots"][mode]
            plot_item = plot_info["plot"]
            _clear_pg_plot(plot_item)
            ref = plot_ref_data if mode == "Crosscorr" else None
            crosscorr_ctx = None
            if mode == "Crosscorr" and ref is not None and ref.size:
                reduced_data, reduced_ref, step_r = _reduce_pair(plot_data, ref)
                compare_reduced = None
                if crosscorr_compare is not None and crosscorr_compare.size:
                    compare_reduced = crosscorr_compare[::step_r]
                crosscorr_ctx = _build_crosscorr_ctx(
                    reduced_data,
                    reduced_ref,
                    crosscorr_compare=compare_reduced,
                    manual_lags=self.manual_xcorr_lags,
                    lag_step=step_r,
                    normalize=self.rx_xcorr_normalized_enable.get(),
                )
            crosscorr_title = (
                f"RX {mode}{title_suffix} ({plot_ref_label})"
                if mode == "Crosscorr" and plot_ref_label
                else f"RX {mode}{title_suffix}"
            )
            _plot_on_pg(
                plot_item,
                plot_data,
                plot_fs,
                mode,
                crosscorr_title,
                ref_data=ref,
                crosscorr_compare=crosscorr_compare if mode == "Crosscorr" else None,
                manual_lags=self.manual_xcorr_lags if mode == "Crosscorr" else None,
                crosscorr_ctx=crosscorr_ctx,
                xcorr_normalized=self.rx_xcorr_normalized_enable.get(),
            )
            if mode == "Signal":
                signal_ranges = _signal_dynamic_view_ranges(plot_data)
                y_range = None
                if signal_ranges is not None:
                    (x_min, x_max), (y_min, y_max) = signal_ranges
                    plot_item.setXRange(x_min, x_max, padding=0.0)
                    plot_item.setYRange(y_min, y_max, padding=0.0)
                    y_range = (y_min, y_max)
                    plot_info["initialized"] = True
                elif not plot_info["initialized"]:
                    plot_item.enableAutoRange(axis="xy", enable=True)
                    plot_item.autoRange()
                    plot_item.enableAutoRange(axis="xy", enable=False)
                    plot_info["initialized"] = True
                x_label, y_label = _signal_dynamic_axis_labels(plot_data, y_range)
                plot_item.setLabel("bottom", x_label)
                plot_item.setLabel("left", y_label)
            elif not plot_info["initialized"]:
                plot_item.enableAutoRange(axis="xy", enable=True)
                plot_item.autoRange()
                plot_item.enableAutoRange(axis="xy", enable=False)
                plot_info["initialized"] = True
            if mode == "Crosscorr":
                zoom_range = _crosscorr_zoom_range(crosscorr_ctx)
                if zoom_range is not None:
                    plot_item.setXRange(*zoom_range, padding=0.0)
                peak = getattr(plot_item, "_crosscorr_peak", None)
                trace_data = getattr(plot_item, "_crosscorr_peak_traces", None)
                if trace_data:
                    current_x = plot_item.getViewBox().viewRange()[0]
                    x_min, x_max = float(current_x[0]), float(current_x[1])
                    visible_peak = None
                    for lags_trace, mag_trace in trace_data:
                        if lags_trace.size == 0 or mag_trace.size == 0:
                            continue
                        visible_mask = (lags_trace >= x_min) & (lags_trace <= x_max)
                        if not np.any(visible_mask):
                            continue
                        trace_peak = float(np.max(mag_trace[visible_mask]))
                        visible_peak = trace_peak if visible_peak is None else max(visible_peak, trace_peak)
                    if visible_peak is not None:
                        peak = visible_peak
                if peak is not None:
                    current_y = plot_item.getViewBox().viewRange()[1]
                    y_range = _crosscorr_dynamic_y_range(
                        (float(current_y[0]), float(current_y[1])),
                        float(peak),
                    )
                    if y_range is not None:
                        plot_item.setYRange(*y_range, padding=0.0)
            image = _export_pg_plot_image(
                plot_item,
                preview_width,
                preview_height,
            )
            label = plot_info["label"]
            label.configure(image=image)
            label.image = image
            if mode == "Crosscorr":
                handler = (
                    lambda _e,
                    m=mode,
                    d=plot_data,
                    s=plot_fs,
                    r=ref,
                    c=crosscorr_compare,
                    t=crosscorr_title: (
                        self._show_fullscreen(d, s, m, t, ref_data=r, crosscorr_compare=c)
                    )
                )
            else:
                handler = (
                    lambda _e,
                    m=mode,
                    d=plot_data,
                    s=plot_fs: self._show_fullscreen(d, s, m, f"RX {m}{title_suffix}")
                )
            label.bind("<Button-1>", handler)

            stats = _calc_stats(
                plot_data,
                plot_fs,
                plot_ref_data,
                manual_lags=self.manual_xcorr_lags,
                xcorr_reduce=True,
                path_cancel_info=path_cancel_info,
                include_spectrum=(mode == "Freq"),
                include_amp=(mode == "Signal"),
                include_echo=(mode == "Crosscorr"),
                precomputed_crosscorr=crosscorr_ctx,
                xcorr_normalized=self.rx_xcorr_normalized_enable.get(),
            )
            stats_labels = pg_state.get("stats_labels", {})
            if mode == "Freq":
                value = f"{_format_hz(stats['f_low'])} | {_format_hz(stats['f_high'])}"
            elif mode == "Signal":
                value = _format_amp(stats['amp'])
            else:
                echo_value = "--"
                if stats.get("echo_delay") is not None:
                    meters = stats["echo_delay"] * 1.5
                    echo_value = f"{stats['echo_delay']} samp ({meters:.1f} m)"
                value = echo_value
            value_label = stats_labels.get(mode)
            if value_label is not None:
                value_label.configure(text=value)
                self.rx_stats_labels.append([value_label])


        if target_name == "Continuous":
            _render_rx_preview_pg(
                target_frame,
                data,
                fs,
                ref_data,
                ref_label,
                aoa_time,
                aoa_series,
                cancel_info if cancel_info and cancel_info.get("applied") else None,
                data_uncanceled if self.rx_path_cancel_enable.get() else None,
            )
        else:
            rx_canvas_list = self.rx_canvases[target_name]
            _render_rx_preview(
                target_frame,
                data,
                fs,
                ref_data,
                ref_label,
                aoa_time,
                aoa_series,
                cancel_info if cancel_info and cancel_info.get("applied") else None,
                data_uncanceled if self.rx_path_cancel_enable.get() else None,
            )
        self._center_canvas_window(target_canvas, target_window)
        self._update_rx_scrollbar(target_name)

        if hasattr(self, "rx_aoa_label"):
            self.rx_aoa_label.configure(text=aoa_text)
        if hasattr(self, "rx_echo_aoa_label"):
            self.rx_echo_aoa_label.configure(text=echo_aoa_text)

    def _get_rx_active_tab(self) -> str:
        if hasattr(self, "rx_tabs"):
            try:
                return self.rx_tabs.get()
            except Exception:
                pass
        return "Single"

    def _get_rx_plot_target(self, tab_name: str | None = None) -> dict[str, object]:
        name = tab_name or self._get_rx_active_tab()
        containers = getattr(self, "_rx_plot_containers", {})
        if name in containers:
            return containers[name]
        if "Single" in containers:
            return containers["Single"]
        raise KeyError("RX plot containers not initialized")

    def _on_rx_tab_change(self, *_args) -> None:
        tab_name = self._get_rx_active_tab()
        has_rx_data = hasattr(self, "raw_rx_data") and self.raw_rx_data is not None
        if has_rx_data:
            fs = getattr(self, "latest_fs_raw", self.latest_fs)
            self._display_rx_plots(
                self.raw_rx_data,
                fs,
                reset_manual=False,
                target_tab=tab_name,
            )

    def _on_rx_cont_plot_tab_change(self, *_args) -> None:
        has_rx_data = hasattr(self, "raw_rx_data") and self.raw_rx_data is not None
        if not has_rx_data:
            return
        runtime_normalize = self._cont_runtime_config.get("normalize_enabled")
        if runtime_normalize is None:
            runtime_normalize = self._cont_runtime_config.get("xcorr_normalized_enabled")
        if runtime_normalize is not None:
            self.rx_xcorr_normalized_enable.set(bool(runtime_normalize))
        fs = getattr(self, "latest_fs_raw", self.latest_fs)
        self._display_rx_plots(
            self.raw_rx_data,
            fs,
            reset_manual=False,
            target_tab="Continuous",
        )

    def _trim_data(self, data: np.ndarray) -> np.ndarray:
        """Return trimmed view of *data* based on slider settings."""
        if data.size == 0:
            return data
        start_pct = max(0.0, min(100.0, self.trim_start.get()))
        end_pct = max(0.0, min(100.0, self.trim_end.get()))
        if end_pct <= start_pct:
            end_pct = min(100.0, start_pct + 1.0)
        s = int(round(len(data) * start_pct / 100))
        e = int(round(len(data) * end_pct / 100))
        e = max(s + 1, min(len(data), e))
        return data[s:e]

    def _trim_data_multichannel(self, data: np.ndarray) -> np.ndarray:
        """Return trimmed view of multi-channel *data* based on slider settings."""
        if data.size == 0:
            return data
        start_pct = max(0.0, min(100.0, self.trim_start.get()))
        end_pct = max(0.0, min(100.0, self.trim_end.get()))
        if end_pct <= start_pct:
            end_pct = min(100.0, start_pct + 1.0)
        s = int(round(data.shape[1] * start_pct / 100))
        e = int(round(data.shape[1] * end_pct / 100))
        e = max(s + 1, min(data.shape[1], e))
        return data[:, s:e]

    def _on_trim_change(self, *_args, reset_manual: bool = True) -> None:
        state = "normal" if self.trim_var.get() else "disabled"
        self.range_slider.configure_state(state)
        try:
            self.trim_start_label.configure(text=f"{self.trim_start.get():.0f}%")
            self.trim_end_label.configure(text=f"{self.trim_end.get():.0f}%")
        except Exception:
            pass
        self.trim_dirty = True
        self.apply_trim_btn.configure(state="normal")
        if reset_manual:
            self._reset_manual_xcorr_lags("Trim geändert")

    def update_trim(self, *_args) -> None:
        """Re-apply trimming and refresh RX plots."""
        self._on_trim_change(reset_manual=False)
        self.apply_trim_btn.configure(state="disabled")
        self.trim_dirty = False
        has_rx_data = hasattr(self, "raw_rx_data") and self.raw_rx_data is not None
        if has_rx_data:
            fs = getattr(self, "latest_fs_raw", self.latest_fs)
            self._display_rx_plots(self.raw_rx_data, fs, reset_manual=False)
        if (
            has_rx_data
            and hasattr(self, "latest_data")
            and self.latest_data is not None
            and self.trim_var.get()
        ):
            self.crosscorr_full()
        else:
            self.full_xcorr_lags = None
            self.full_xcorr_mag = None

    def save_trimmed(self) -> None:
        """Save the currently trimmed RX data to a file."""
        if not hasattr(self, "latest_data") or self.latest_data is None:
            messagebox.showerror("Save Trim", "No RX data available")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".bin",
            initialfile="rx_trimmed.bin",
            filetypes=[("Binary files", "*.bin"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            save_interleaved(filename, self.latest_data)
        except Exception as exc:
            messagebox.showerror("Save Trim", str(exc))

    def open_signal(self) -> None:
        """Open a window to compare up to four signals."""
        CompareWindow(self)

    def _emit_rx_log(self, text: str) -> None:
        message = text.rstrip() + "\n"
        logged = False
        if self.console and self.console.winfo_exists():
            self.console.append(message)
            logged = True
        if hasattr(self, "tx_log") and self.tx_log.winfo_exists():
            self.tx_log.insert(tk.END, message)
            self.tx_log.see(tk.END)
            logged = True
        if not logged:
            print(message, end="")

    def _log_path_cancellation(self, info: dict[str, object], label: str) -> None:
        if not self.rx_path_cancel_enable.get():
            return
        warning = info.get("warning")
        if not info.get("applied"):
            if warning:
                log_key = (label, "warn", warning)
                if self._last_path_cancel_log != log_key:
                    self._emit_rx_log(
                        f"Pfad-Cancellation ({label}): {warning}"
                    )
                    self._last_path_cancel_log = log_key
            return
        k0 = info.get("k0")
        a0 = info.get("a0")
        k1 = info.get("k1")
        corr2_peak = info.get("corr2_peak")
        delta_k = info.get("delta_k")
        corr2_text = (
            f"{corr2_peak:.4g}" if isinstance(corr2_peak, (int, float)) else "--"
        )
        log_key = (label, k0, a0, k1, corr2_text, delta_k, warning)
        if self._last_path_cancel_log == log_key:
            return
        self._last_path_cancel_log = log_key
        message = (
            "Pfad-Cancellation ({label}): "
            "k0={k0}, a0={a0}, k1={k1}, corr2_peak={corr2_peak}, delta_k={delta_k}"
        ).format(
            label=label,
            k0=k0 if k0 is not None else "--",
            a0=_format_complex(
                a0 if isinstance(a0, (complex, np.complexfloating)) else None
            ),
            k1=k1 if k1 is not None else "--",
            corr2_peak=corr2_text,
            delta_k=delta_k if delta_k is not None else "--",
        )
        self._emit_rx_log(message)
        if warning:
            self._emit_rx_log(f"Pfad-Cancellation ({label}): {warning}")

    def _open_console(self, title: str) -> None:
        if self.console is None or not self.console.winfo_exists():
            self.console = ConsoleWindow(self, title)
        else:
            self.console.title(title)
            self.console.text.delete("1.0", tk.END)
        while not self._out_queue.empty():
            self._out_queue.get_nowait()

    def _center_canvas_window(self, canvas: tk.Canvas, window_id: int) -> None:
        canvas_width = max(1, canvas.winfo_width())
        canvas.itemconfigure(window_id, width=canvas_width)
        canvas.coords(window_id, canvas_width / 2, 0)

    def _on_tx_log_scroll(self, first: str, last: str) -> None:
        if not hasattr(self, "tx_log_scroll"):
            return
        self.tx_log_scroll.set(first, last)
        try:
            first_val = float(first)
            last_val = float(last)
        except (TypeError, ValueError):
            return
        needs_scroll = first_val > 0.0 or last_val < 1.0
        if needs_scroll:
            if not self.tx_log_scroll.winfo_ismapped():
                self.tx_log_scroll.grid(row=0, column=1, sticky="ns")
        else:
            if self.tx_log_scroll.winfo_ismapped():
                self.tx_log_scroll.grid_remove()

    def _update_tx_log_scrollbar(self) -> None:
        if not hasattr(self, "tx_log"):
            return
        first, last = self.tx_log.yview()
        self._on_tx_log_scroll(str(first), str(last))

    def _update_gen_scrollbar(self) -> None:
        if not hasattr(self, "gen_canvas"):
            return
        bbox = self.gen_canvas.bbox("all")
        if not bbox:
            if self.gen_scroll.winfo_ismapped():
                self.gen_scroll.grid_remove()
            self.gen_canvas.configure(yscrollcommand=None)
            self.gen_canvas.yview_moveto(0)
            self._gen_scroll_active = False
            return
        content_height = bbox[3] - bbox[1]
        canvas_height = self.gen_canvas.winfo_height()
        needs_scroll = content_height > (canvas_height + 1)
        if needs_scroll:
            if not self.gen_scroll.winfo_ismapped():
                self.gen_scroll.grid(row=0, column=1, sticky="ns")
            self.gen_canvas.configure(yscrollcommand=self.gen_scroll.set)
            self._gen_scroll_active = True
        else:
            if self.gen_scroll.winfo_ismapped():
                self.gen_scroll.grid_remove()
            self.gen_canvas.configure(yscrollcommand=None)
            self.gen_canvas.yview_moveto(0)
            self._gen_scroll_active = False

    def _update_rx_scrollbar(self, tab_name: str | None = None) -> None:
        try:
            target = self._get_rx_plot_target(tab_name)
        except KeyError:
            return
        name = target["name"]
        canvas = target["canvas"]
        vscroll = target["scrollbar"]
        bbox = canvas.bbox("all")
        if not bbox:
            if vscroll.winfo_ismapped():
                vscroll.grid_remove()
            canvas.configure(yscrollcommand=None)
            canvas.yview_moveto(0)
            self._rx_scroll_active[name] = False
            return
        content_height = bbox[3] - bbox[1]
        canvas_height = canvas.winfo_height()
        needs_scroll = content_height > (canvas_height + 1)
        if needs_scroll:
            if not vscroll.winfo_ismapped():
                vscroll.grid(row=0, column=1, sticky="ns")
            canvas.configure(yscrollcommand=vscroll.set)
            self._rx_scroll_active[name] = True
        else:
            if vscroll.winfo_ismapped():
                vscroll.grid_remove()
            canvas.configure(yscrollcommand=None)
            canvas.yview_moveto(0)
            self._rx_scroll_active[name] = False

    # ----- Mousewheel helpers -----
    def _bind_gen_mousewheel(self, _event) -> None:
        self.gen_canvas.bind_all("<MouseWheel>", self._on_gen_mousewheel)
        self.gen_canvas.bind_all("<Button-4>", self._on_gen_mousewheel)
        self.gen_canvas.bind_all("<Button-5>", self._on_gen_mousewheel)

    def _unbind_gen_mousewheel(self, _event) -> None:
        self.gen_canvas.unbind_all("<MouseWheel>")
        self.gen_canvas.unbind_all("<Button-4>")
        self.gen_canvas.unbind_all("<Button-5>")

    def _on_gen_mousewheel(self, event) -> None:
        if not self._gen_scroll_active:
            return
        delta = 0
        if hasattr(event, "delta") and event.delta:
            delta = -1 * int(event.delta / 120)
        elif event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        if delta:
            self.gen_canvas.yview_scroll(delta, "units")

    def _bind_rx_mousewheel(self, _event, tab_name: str) -> None:
        self._rx_active_scroll_tab = tab_name
        target = self._get_rx_plot_target(tab_name)
        canvas = target["canvas"]
        canvas.bind_all("<MouseWheel>", self._on_rx_mousewheel)
        canvas.bind_all("<Button-4>", self._on_rx_mousewheel)
        canvas.bind_all("<Button-5>", self._on_rx_mousewheel)

    def _unbind_rx_mousewheel(self, _event) -> None:
        self._rx_active_scroll_tab = None
        self.rx_canvas.unbind_all("<MouseWheel>")
        self.rx_canvas.unbind_all("<Button-4>")
        self.rx_canvas.unbind_all("<Button-5>")

    def _on_rx_mousewheel(self, event) -> None:
        tab_name = self._rx_active_scroll_tab or self._get_rx_active_tab()
        try:
            target = self._get_rx_plot_target(tab_name)
        except KeyError:
            return
        if not self._rx_scroll_active.get(tab_name, False):
            return
        canvas = target["canvas"]
        delta = 0
        if hasattr(event, "delta") and event.delta:
            delta = -1 * int(event.delta / 120)
        elif event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        if delta:
            canvas.yview_scroll(delta, "units")

    def _process_queue(self) -> None:
        chunks: list[str] = []
        processed = 0
        max_items = 200
        max_chars = 20000
        total_chars = 0
        while not self._out_queue.empty() and processed < max_items:
            line = self._out_queue.get_nowait()
            chunks.append(line)
            processed += 1
            total_chars += len(line)
            if total_chars >= max_chars:
                break
        if chunks:
            text = "".join(chunks)
            if self.console and self.console.winfo_exists():
                self.console.append(text)
            if hasattr(self, "tx_log") and self.tx_log.winfo_exists():
                self.tx_log.insert(tk.END, text)
                self.tx_log.see(tk.END)
        if self._cmd_running or not self._out_queue.empty():
            delay = 10 if self._out_queue.empty() else 1
            self.after(delay, self._process_queue)

    def _ui(self, callback) -> None:
        if self._closing:
            return
        try:
            self.after(0, callback)
        except tk.TclError:
            pass

    def _reset_tx_buttons(self) -> None:
        if hasattr(self, "tx_stop"):
            self.tx_stop.configure(state="disabled")
        if hasattr(self, "tx_button"):
            self.tx_button.configure(state="normal")
        if hasattr(self, "tx_retrans"):
            self.tx_retrans.configure(state="disabled")
        self._set_tx_indicator_state("idle")

    def _run_rx_thread(
        self, arg_list: list[str], channels: int, rate: float
    ) -> None:
        try:
            from .helpers import rx_to_file

            args = rx_to_file.parse_args(arg_list)
            rx_to_file.main(args=args)
        except Exception as exc:
            self._out_queue.put(f"Receive error: {exc}\n")
            args = None
        finally:
            self._cmd_running = False
            self._proc = None
            self._ui(self._reset_rx_buttons)

        if args is not None:
            try:
                path = Path(args.output_file)
                try:
                    data = rx_convert.load_iq_file(
                        path, channels=channels, layout="blocked"
                    )
                except ValueError:
                    data = rx_convert.load_iq_file(
                        path, channels=channels, layout="interleaved"
                    )
                self._ui(
                    lambda: self._display_rx_plots(data, rate, target_tab="Single")
                )
            except Exception as exc:
                self._out_queue.put(f"Receive plot error: {exc}\n")

    def _reset_rx_buttons(self) -> None:
        if hasattr(self, "rx_stop"):
            self.rx_stop.configure(state="disabled")
        if hasattr(self, "rx_button"):
            self.rx_button.configure(state="normal")

    def _set_manual_xcorr_lag(self, kind: str, lag_value: float) -> None:
        """Store manual lag selection and refresh LOS/echo stats."""
        if kind not in ("los", "echo"):
            return
        self.manual_xcorr_lags[kind] = int(round(lag_value))
        self._refresh_rx_stats()

    def _refresh_rx_stats(self) -> None:
        label_groups = []
        if hasattr(self, "rx_stats_labels"):
            label_groups = [
                label for label in self.rx_stats_labels if label is not None
            ]
        elif hasattr(self, "rx_stats_label"):
            label_groups = [self.rx_stats_label]
        if not label_groups:
            return
        if not hasattr(self, "latest_data") or self.latest_data is None:
            return
        ref_data, _ref_label = self._get_crosscorr_reference()
        path_cancel_info = None
        if self.rx_path_cancel_enable.get():
            info = self._last_path_cancel_info
            if info and info.get("applied"):
                path_cancel_info = info
        stats = _calc_stats(
            self.latest_data,
            self.latest_fs,
            ref_data,
            manual_lags=self.manual_xcorr_lags,
            xcorr_reduce=True,
            path_cancel_info=path_cancel_info,
            xcorr_normalized=self.rx_xcorr_normalized_enable.get(),
        )
        stats_rows = _format_rx_stats_rows(stats)
        text = "\n".join(f"{label}: {value}" for label, value in stats_rows)
        for label_group in label_groups:
            if isinstance(label_group, (list, tuple)):
                for idx, value_label in enumerate(label_group):
                    if idx < len(stats_rows):
                        value_label.configure(text=stats_rows[idx][1])
                    else:
                        value_label.configure(text="")
            else:
                label_group.configure(text=text)

    def _start_manual_xcorr_polling(self, output_path: str | None) -> None:
        if not output_path:
            return
        self._xcorr_manual_file = Path(output_path)
        if not self._xcorr_polling:
            self._xcorr_polling = True
            self._poll_manual_xcorr_updates()

    def _poll_manual_xcorr_updates(self) -> None:
        if self._closing:
            self._xcorr_polling = False
            return
        manual_file = self._xcorr_manual_file
        if manual_file is not None:
            updates = None
            try:
                with manual_file.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    updates = {
                        "los": data.get("los"),
                        "echo": data.get("echo"),
                    }
            except Exception:
                updates = None
            if updates is not None:
                changed = False
                for key in ("los", "echo"):
                    value = updates.get(key)
                    if value is not None:
                        value = int(value)
                    if self.manual_xcorr_lags.get(key) != value:
                        self.manual_xcorr_lags[key] = value
                        changed = True
                if changed:
                    self._refresh_rx_stats()
        self.after(200, self._poll_manual_xcorr_updates)

    def _show_fullscreen(
        self,
        data: np.ndarray,
        fs: float,
        mode: str,
        title: str,
        ref_data: np.ndarray | None = None,
        crosscorr_compare: np.ndarray | None = None,
    ) -> None:
        if data is None:
            return
        if mode == "Crosscorr" and ref_data is None:
            ref_data, _ref_label = self._get_crosscorr_reference()
        if (
            mode == "Crosscorr"
            and ref_data is not None
            and self.rx_magnitude_enable.get()
        ):
            ref_data = np.abs(ref_data)
        output_path = _spawn_plot_worker(
            data,
            fs,
            mode,
            title,
            ref_data=ref_data if ref_data is not None else getattr(self, "tx_data", None),
            manual_lags=self.manual_xcorr_lags,
            crosscorr_compare=crosscorr_compare,
            fullscreen=True,
            xcorr_normalized=self.rx_xcorr_normalized_enable.get(),
        )
        if mode == "Crosscorr":
            self._start_manual_xcorr_polling(output_path)

    def _show_toast(self, text: str, duration: int = 2000) -> None:
        """Show a temporary notification on top of the main window."""
        win = ctk.CTkToplevel(self)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        lbl = ctk.CTkLabel(win, text=text)
        lbl.pack(padx=12, pady=8)
        win.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - win.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")
        win.after(duration, win.destroy)

    def crosscorr_full(self) -> None:
        """Calculate cross-correlation without downsampling."""
        if not hasattr(self, "latest_data") or self.latest_data is None:
            messagebox.showerror("XCorr Full", "No RX data available")
            return
        ref_data, _ref_label = self._get_crosscorr_reference()
        if ref_data.size == 0:
            messagebox.showerror("XCorr Full", "No TX data available")
            return
        data = self.latest_data
        ref = np.abs(ref_data) if self.rx_magnitude_enable.get() else ref_data
        n = min(len(data), len(ref))
        cc = _xcorr_fft(data[:n], ref[:n])
        self.full_xcorr_lags = np.arange(-n + 1, n)
        if self.rx_xcorr_normalized_enable.get():
            denom = float(np.linalg.norm(data[:n]) * np.linalg.norm(ref[:n]))
            self.full_xcorr_mag = np.abs(cc) / denom if denom > 0.0 and np.isfinite(denom) else np.abs(cc)
        else:
            self.full_xcorr_mag = np.abs(cc)
        self.echo_aoa_results = []
        self._show_toast("Cross-correlation calculated")

    # ----- Preset handling --------------------------------------------------
    def _get_current_params(self) -> dict:
        return {
            "waveform": self.wave_var.get(),
            "fs": self.fs_entry.get(),
            "f": self.f_entry.get(),
            "f1": self.f1_entry.get(),
            "q": self.q_entry.get(),
            "samples": self.samples_entry.get(),
            "filter_enabled": self.fdz_enable.get(),
            "filter_mode": (self.filter_mode_var.get() or "frequency_domain_zeroing"),
            "filter_bandwidth_hz": self.filter_bandwidth_entry.get(),
            "repeats": self.repeat_entry.get(),
            "repeats_enabled": self.repeat_enable.get(),
            "zeros": self.zeros_var.get(),
            "zeros_enabled": self.zeros_enable.get(),
            "amplitude": self.amp_entry.get(),
            "ofdm_nfft": self.ofdm_nfft_entry.get(),
            "ofdm_cp": self.ofdm_cp_entry.get(),
            "ofdm_symbols": self.ofdm_symbols_entry.get(),
            "ofdm_short_repeats": self.ofdm_short_entry.get(),
            "file": self.file_entry.get(),
            "sync_rates": self.sync_var.get(),
            "tx_args": self.tx_args.get(),
            "tx_rate": self.tx_rate.get(),
            "tx_freq": self.tx_freq.get(),
            "tx_gain": self.tx_gain.get(),
            "tx_file": self.tx_file.get(),
            "rx_args": self.rx_args.get(),
            "rx_rate": self.rx_rate.get(),
            "rx_freq": self.rx_freq.get(),
            "rx_dur": self.rx_dur.get(),
            "rx_gain": self.rx_gain.get(),
            "rx_magnitude_enabled": self.rx_magnitude_enable.get(),
            "rx_xcorr_normalized_enabled": self.rx_xcorr_normalized_enable.get(),
            "rx_path_cancellation_enabled": self.rx_path_cancel_enable.get(),
            "rx_channel_2": self.rx_channel_2.get(),
            "rx_channel_view": self.rx_channel_view.get(),
            "rx_file": self.rx_file.get(),
            "rx_view": self.rx_view.get(),
            "rx_ant_spacing": self.rx_ant_spacing.get(),
            "rx_wavelength": self.rx_wavelength.get(),
            "trim": self.trim_var.get(),
            "trim_start": self.trim_start.get(),
            "trim_end": self.trim_end.get(),
            "pn_chip_rate": self.pn_chip_entry.get(),
            "pn_seed": self.pn_seed_entry.get(),
            "rx_cont_rate": self.rx_cont_rate.get(),
            "rx_cont_freq": self.rx_cont_freq.get(),
            "rx_cont_ring_seconds": self.rx_cont_ring_seconds.get(),
            "rx_cont_gain": self.rx_cont_gain.get(),
            "rx_cont_restart_margin": self.rx_cont_restart_margin.get(),
            "rx_cont_args": self.rx_cont_args.get(),
            "rx_cont_snippet_seconds": self.rx_cont_snippet_seconds.get(),
            "rx_cont_snippet_interval": self.rx_cont_snippet_interval.get(),
            "rx_cont_output_prefix": self.rx_cont_output_prefix.get(),
            "rx_active_tab": self._get_rx_active_tab(),
        }

    def _autosave_state(self) -> None:
        current = self._get_current_params()
        if current != getattr(self, "_last_saved_state", None):
            _save_state(current)
            self._last_saved_state = current
        self.after(AUTOSAVE_INTERVAL * 1000, self._autosave_state)

    def _apply_params(self, params: dict) -> None:
        self.wave_var.set(params.get("waveform", "sinus"))
        self.update_waveform_fields()
        self.fs_entry.delete(0, tk.END)
        self.fs_entry.insert(0, params.get("fs", ""))
        self.f_entry.delete(0, tk.END)
        self.f_entry.insert(0, params.get("f", ""))
        self.f1_entry.delete(0, tk.END)
        self.f1_entry.insert(0, params.get("f1", ""))
        self.q_entry.delete(0, tk.END)
        self.q_entry.insert(0, params.get("q", ""))
        self.samples_entry.delete(0, tk.END)
        self.samples_entry.insert(0, params.get("samples", ""))
        filter_bandwidth_value = params.get("filter_bandwidth_hz")
        if filter_bandwidth_value is None:
            filter_bandwidth_value = params.get("filter_bandwidth")
        if filter_bandwidth_value is None and "rrc_oversampling" in params:
            fs_for_filter = _try_parse_number_expr(str(params.get("fs", "")), default=0.0)
            oversampling = _try_parse_number_expr(
                str(params.get("rrc_oversampling", "")),
                default=0.0,
            )
            if fs_for_filter > 0 and oversampling > 0:
                filter_bandwidth_value = str(fs_for_filter / oversampling)
        if filter_bandwidth_value is None:
            filter_bandwidth_value = "1e6"
        self.filter_bandwidth_entry.delete(0, tk.END)
        self.filter_bandwidth_entry.insert(0, str(filter_bandwidth_value))
        self.repeat_entry.delete(0, tk.END)
        repeats_value = params.get("repeats", "1")
        self.repeat_entry.insert(0, repeats_value)
        if str(repeats_value).strip() not in ("", "0"):
            self._repeat_last_value = str(repeats_value)
        repeat_enabled = params.get("repeats_enabled")
        if repeat_enabled is None:
            repeat_enabled = str(repeats_value).strip() != "0"
        self.repeat_enable.set(bool(repeat_enabled))
        self._on_repeat_toggle()
        filter_enabled = params.get("filter_enabled")
        if filter_enabled is None:
            if "fdz_enabled" in params:
                filter_enabled = params.get("fdz_enabled")
            elif "rrc_enabled" in params:
                filter_enabled = params.get("rrc_enabled")
            else:
                filter_enabled = True
        self.fdz_enable.set(bool(filter_enabled))

        filter_mode = params.get("filter_mode")
        if filter_mode is None:
            if "fdz_enabled" in params:
                filter_mode = "frequency_domain_zeroing"
            elif "rrc_enabled" in params:
                filter_mode = "frequency_domain_zeroing"
            else:
                filter_mode = "frequency_domain_zeroing"
        self.filter_mode_var.set(str(filter_mode))

        state = "normal" if self.fdz_enable.get() else "disabled"
        self.filter_bandwidth_entry.entry.configure(state=state)
        zeros_value = params.get("zeros", "same")
        zeros_enabled = params.get("zeros_enabled")
        if zeros_enabled is None:
            zeros_enabled = str(zeros_value).strip() not in ("", "none")
        if zeros_value not in self.zeros_values:
            zeros_value = "same"
        self.zeros_var.set(zeros_value)
        if zeros_value:
            self._zeros_last_value = zeros_value
        self.zeros_enable.set(bool(zeros_enabled))
        self._on_zeros_toggle()
        self.amp_entry.delete(0, tk.END)
        self.amp_entry.insert(0, params.get("amplitude", ""))
        self.ofdm_nfft_entry.delete(0, tk.END)
        self.ofdm_nfft_entry.insert(0, params.get("ofdm_nfft", "64"))
        self.ofdm_cp_entry.delete(0, tk.END)
        self.ofdm_cp_entry.insert(0, params.get("ofdm_cp", "16"))
        self.ofdm_symbols_entry.delete(0, tk.END)
        self.ofdm_symbols_entry.insert(0, params.get("ofdm_symbols", "2"))
        self.ofdm_short_entry.delete(0, tk.END)
        self.ofdm_short_entry.insert(0, params.get("ofdm_short_repeats", "10"))
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, params.get("file", ""))
        self.tx_args.delete(0, tk.END)
        self.tx_args.insert(0, params.get("tx_args", ""))
        self.tx_rate.delete(0, tk.END)
        self.tx_rate.insert(0, params.get("tx_rate", ""))
        self.tx_freq.delete(0, tk.END)
        self.tx_freq.insert(0, params.get("tx_freq", ""))
        self.tx_gain.delete(0, tk.END)
        self.tx_gain.insert(0, params.get("tx_gain", ""))
        self.tx_file.delete(0, tk.END)
        self.tx_file.insert(0, params.get("tx_file", ""))
        self.rx_args.delete(0, tk.END)
        self.rx_args.insert(0, params.get("rx_args", ""))
        self.rx_rate.delete(0, tk.END)
        self.rx_rate.insert(0, params.get("rx_rate", ""))
        self.rx_freq.delete(0, tk.END)
        self.rx_freq.insert(0, params.get("rx_freq", ""))
        self.rx_dur.delete(0, tk.END)
        self.rx_dur.insert(0, params.get("rx_dur", ""))
        self.rx_gain.delete(0, tk.END)
        self.rx_gain.insert(0, params.get("rx_gain", ""))
        self.rx_magnitude_enable.set(params.get("rx_magnitude_enabled", False))
        self.rx_xcorr_normalized_enable.set(
            params.get("rx_xcorr_normalized_enabled", False)
        )
        self.rx_path_cancel_enable.set(
            params.get("rx_path_cancellation_enabled", False)
        )
        self._update_path_cancellation_status()
        self.rx_channel_2.set(params.get("rx_channel_2", False))
        self.rx_channel_view.set(params.get("rx_channel_view", "Kanal 1"))
        self.rx_file.delete(0, tk.END)
        self.rx_file.insert(0, params.get("rx_file", ""))
        self.rx_cont_rate.delete(0, tk.END)
        self.rx_cont_rate.insert(0, params.get("rx_cont_rate", "200e6"))
        self.rx_cont_freq.delete(0, tk.END)
        self.rx_cont_freq.insert(0, params.get("rx_cont_freq", "5.18e9"))
        self.rx_cont_ring_seconds.delete(0, tk.END)
        self.rx_cont_ring_seconds.insert(0, params.get("rx_cont_ring_seconds", "4.0"))
        self.rx_cont_gain.delete(0, tk.END)
        self.rx_cont_gain.insert(0, params.get("rx_cont_gain", "80"))
        self.rx_cont_restart_margin.delete(0, tk.END)
        self.rx_cont_restart_margin.insert(
            0, params.get("rx_cont_restart_margin", "1.5")
        )
        self.rx_cont_args.delete(0, tk.END)
        self.rx_cont_args.insert(
            0,
            params.get(
                "rx_cont_args", "addr=192.168.20.2,clock_source=external"
            ),
        )
        self.rx_cont_snippet_seconds.delete(0, tk.END)
        self.rx_cont_snippet_seconds.insert(
            0, params.get("rx_cont_snippet_seconds", "0.05")
        )
        self.rx_cont_snippet_interval.delete(0, tk.END)
        self.rx_cont_snippet_interval.insert(
            0, params.get("rx_cont_snippet_interval", "1.0")
        )
        self.rx_cont_output_prefix.delete(0, tk.END)
        self.rx_cont_output_prefix.insert(
            0, params.get("rx_cont_output_prefix", "signals/rx/snippet")
        )
        self.rx_view.set(params.get("rx_view", "Signal"))
        rx_active_tab = params.get("rx_active_tab", "Single")
        if hasattr(self, "rx_tabs"):
            try:
                self.rx_tabs.set(rx_active_tab)
            except Exception:
                self.rx_tabs.set("Single")
        self.rx_ant_spacing.delete(0, tk.END)
        self.rx_ant_spacing.insert(0, params.get("rx_ant_spacing", "0.03"))
        self.rx_wavelength.delete(0, tk.END)
        self.rx_wavelength.insert(0, params.get("rx_wavelength", "3e8/5.18e9"))
        self.trim_var.set(params.get("trim", False))
        self.trim_start.set(params.get("trim_start", 0.0))
        self.trim_end.set(params.get("trim_end", 100.0))
        self.update_trim()
        self.sync_var.set(params.get("sync_rates", True))
        self.toggle_rate_sync(self.sync_var.get())

    def open_load_preset_window(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("Load Preset")
        lb = tk.Listbox(win, exportselection=False)
        for name in sorted(_PRESETS.keys()):
            lb.insert(tk.END, name)
        lb.pack(fill="both", expand=True, padx=5, pady=5)

        def load_selected() -> None:
            sel = lb.curselection()
            if not sel:
                messagebox.showerror("Preset", "No preset selected")
                return
            name = lb.get(sel[0])
            self.load_preset(name)
            win.destroy()

        ctk.CTkButton(win, text="Load", command=load_selected).pack(pady=5)
        _apply_button_cursor(win)

    def open_save_preset_window(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("Save Preset")
        ctk.CTkLabel(win, text="Name:", anchor="e").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )
        name_var = tk.StringVar()
        ctk.CTkEntry(win, textvariable=name_var).grid(
            row=0, column=1, padx=5, pady=5
        )

        def save() -> None:
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Save Preset", "Name cannot be empty")
                return
            self.save_preset(name)
            win.destroy()

        ctk.CTkButton(win, text="Save", command=save).grid(
            row=1, column=0, columnspan=2, pady=5
        )
        _apply_input_margins(win)
        _apply_button_cursor(win)

    def load_preset(self, name: str) -> None:
        preset = _PRESETS.get(name)
        if not preset:
            messagebox.showerror("Preset", f"Preset '{name}' not found")
            return
        self._apply_params(preset)

    def save_preset(self, name: str) -> None:
        _PRESETS[name] = self._get_current_params()
        _save_presets(_PRESETS)

    def delete_preset(self) -> None:
        if not _PRESETS:
            messagebox.showinfo("Delete Preset", "No presets available")
            return
        win = ctk.CTkToplevel(self)
        win.title("Delete Preset")
        lb = tk.Listbox(win, exportselection=False)
        for p in sorted(_PRESETS.keys()):
            lb.insert(tk.END, p)
        lb.pack(fill="both", expand=True, padx=5, pady=5)

        def delete_selected():
            sel = lb.curselection()
            if not sel:
                messagebox.showerror("Delete Preset", "No preset selected")
                return
            name = lb.get(sel[0])
            if not messagebox.askyesno("Delete Preset", f"Delete preset '{name}'?"):
                return
            _PRESETS.pop(name, None)
            _save_presets(_PRESETS)
            win.destroy()

        ctk.CTkButton(win, text="Delete", command=delete_selected).pack(pady=5)
        _apply_button_cursor(win)

    # ----- Actions -----
    def generate(self):
        try:
            fs = _parse_number_expr_or_error(self.fs_entry.get())
            samples = int(self.samples_entry.get())
            repeats = self._get_repeat_count() if self.repeat_entry.get() else 1
            zeros_mode = self.zeros_var.get() if self.zeros_enable.get() else "none"
            amp = _parse_number_expr_or_error(self.amp_entry.get())
            waveform = self.wave_var.get()

            if waveform == "sinus":
                freq = _parse_number_expr_or_error(
                    self.f_entry.get(), allow_empty=True, empty_value=0.0
                )
                data = generate_waveform(waveform, fs, freq, samples)
            elif waveform == "doppelsinus":
                f1 = _parse_number_expr_or_error(
                    self.f_entry.get(), allow_empty=True, empty_value=0.0
                )
                f2 = _parse_number_expr_or_error(self.f1_entry.get())
                data = generate_waveform(waveform, fs, f1, samples, f1=f2)
            elif waveform == "zadoffchu":
                q = int(self.q_entry.get()) if self.q_entry.get() else 1
                data = generate_waveform(waveform, fs, 0.0, samples, q=q)
            elif waveform == "ofdm_preamble":
                nfft = int(_parse_number_expr_or_error(self.ofdm_nfft_entry.get()))
                cp_len = int(_parse_number_expr_or_error(self.ofdm_cp_entry.get()))
                num_symbols = int(_parse_number_expr_or_error(self.ofdm_symbols_entry.get()))
                short_repeats = int(_parse_number_expr_or_error(self.ofdm_short_entry.get()))
                if cp_len >= nfft:
                    raise ValueError("OFDM CP muss kleiner als NFFT sein.")
                if nfft <= 0:
                    raise ValueError("OFDM NFFT muss > 0 sein.")
                if num_symbols <= 0:
                    raise ValueError("OFDM Symbols muss > 0 sein.")
                if short_repeats < 0:
                    raise ValueError("OFDM Short Repeats muss >= 0 sein.")
                data = generate_waveform(
                    waveform,
                    fs,
                    0.0,
                    samples,
                    ofdm_nfft=nfft,
                    ofdm_cp_len=cp_len,
                    ofdm_num_symbols=num_symbols,
                    ofdm_short_repeats=short_repeats,
                )
            elif waveform == "pseudo_noise":
                pn_chip_rate = _parse_number_expr_or_error(self.pn_chip_entry.get())
                pn_seed = int(self.pn_seed_entry.get()) if self.pn_seed_entry.get() else 1
                data = generate_waveform(
                    waveform,
                    fs,
                    0.0,
                    samples,
                    pn_chip_rate=pn_chip_rate,
                    pn_seed=pn_seed,
                )
            else:  # chirp
                f0 = _parse_number_expr_or_error(
                    self.f_entry.get(), allow_empty=True, empty_value=0.0
                )
                f1_text = self.f1_entry.get()
                f1 = _parse_number_expr_or_error(f1_text) if f1_text else None
                data = generate_waveform(waveform, fs, f0, samples, f0=f0, f1=f1)

            filter_data = None
            if self._is_filter_active():
                bandwidth_hz = _parse_number_expr_or_error(
                    self.filter_bandwidth_entry.get(),
                    allow_empty=False,
                )
                if bandwidth_hz <= 0:
                    raise ValueError("Bandwidth muss > 0 Hz sein.")
                filter_data = apply_frequency_domain_zeroing(data, fs, bandwidth_hz)
                data = filter_data

            zeros = 0
            if zeros_mode == "same":
                zeros = 1
            elif zeros_mode == "half":
                zeros = 0.5
            elif zeros_mode == "quarter":
                zeros = 0.25
            elif zeros_mode == "double":
                zeros = 2
            elif zeros_mode == "quadruple":
                zeros = 4
            elif zeros_mode == "octuple":
                zeros = 8

            def _append_zeros(signal: np.ndarray | None) -> np.ndarray | None:
                if signal is None or zeros == 0:
                    return signal
                zeros_len = int(round(len(signal) * zeros))
                if zeros_len <= 0:
                    return signal
                return np.concatenate([signal, np.zeros(zeros_len, dtype=np.complex64)])

            repeated_data = np.tile(data, repeats) if repeats > 1 else None
            final_data = repeated_data if repeated_data is not None else data
            zeros_data = _append_zeros(final_data) if self.zeros_enable.get() and zeros > 0 else None

            save_interleaved(self.file_entry.get(), data, amplitude=amp)
            if filter_data is not None:
                filtered_filename = self._filtered_tx_file or _gen_filtered_tx_filename(self.file_entry.get())
                self._filtered_tx_file = filtered_filename
                save_interleaved(filtered_filename, filter_data, amplitude=amp)
            else:
                self._filtered_tx_file = None

            if repeats > 1 and repeated_data is not None:
                repeat_base = self._filtered_tx_file if filter_data is not None else self.file_entry.get()
                repeat_filename = _gen_repeat_tx_filename(repeat_base)
                self._repeat_tx_file = repeat_filename
                self.tx_file.delete(0, tk.END)
                self.tx_file.insert(0, repeat_filename)
                save_interleaved(repeat_filename, repeated_data, amplitude=amp)
                self._reset_manual_xcorr_lags("TX-Datei geändert")
            else:
                self._repeat_tx_file = None
                target_file = self._filtered_tx_file if filter_data is not None else self.file_entry.get()
                self.tx_file.delete(0, tk.END)
                self.tx_file.insert(0, target_file)
                if filter_data is not None:
                    self._reset_manual_xcorr_lags("TX-Datei geändert")

            if zeros_data is not None:
                zeros_base = self._repeat_tx_file if self._repeat_tx_file is not None else (self._filtered_tx_file if filter_data is not None else self.file_entry.get())
                zeros_filename = _gen_zeros_tx_filename(zeros_base)
                self._zeros_tx_file = zeros_filename
                self.tx_file.delete(0, tk.END)
                self.tx_file.insert(0, zeros_filename)
                save_interleaved(zeros_filename, zeros_data, amplitude=amp)
                self._reset_manual_xcorr_lags("TX-Datei geändert")
            else:
                self._zeros_tx_file = None
                if not self.tx_file.get():
                    self.tx_file.insert(0, self._tx_transmit_file())

            self._last_generated_tx_file = self._tx_transmit_file()

            def _scale_for_display(signal: np.ndarray) -> np.ndarray:
                max_abs = np.max(np.abs(signal)) if np.any(signal) else 1.0
                scale = amp / max_abs if max_abs > 1e-9 else 1.0
                return signal * scale

            scaled_data = _scale_for_display(data)
            scaled_repeated = _scale_for_display(repeated_data) if repeated_data is not None else None
            scaled_zeros = _scale_for_display(zeros_data) if zeros_data is not None else None
            symbol_rate = fs if waveform == "zadoffchu" else None
            repeated_symbol_rate = symbol_rate
            zeros_symbol_rate = symbol_rate if zeros_data is not None else None

            if scaled_repeated is not None or scaled_zeros is not None:
                self._display_gen_plots(
                    scaled_data,
                    fs,
                    repeated_data=scaled_repeated,
                    repeated_fs=fs,
                    zeros_data=scaled_zeros,
                    zeros_fs=fs,
                    symbol_rate=symbol_rate,
                    repeated_symbol_rate=repeated_symbol_rate,
                    zeros_symbol_rate=zeros_symbol_rate,
                )
            else:
                self._display_gen_plots(
                    scaled_data,
                    fs,
                    zeros_data=scaled_zeros,
                    zeros_fs=fs,
                    symbol_rate=symbol_rate,
                    zeros_symbol_rate=zeros_symbol_rate,
                )

        except ValueError as exc:
            messagebox.showerror("Generate", str(exc))
        except Exception as exc:
            messagebox.showerror("Generate error", str(exc))

    def transmit(self):
        now = time.monotonic()
        MIN_GAP = 0.3  # Sekunden (statt 10)

        if now - self._last_tx_end < MIN_GAP:
            self._set_tx_indicator_state("pending")
            wait = MIN_GAP - (now - self._last_tx_end)
            self.after(int(wait * 1000), self.transmit)
            return
        self._set_tx_indicator_state("pending")
        self._stop_requested = False
        tx_args = self.tx_args.get()
        try:
            rate = _parse_number_expr_or_error(self.tx_rate.get())
            freq = _parse_number_expr_or_error(self.tx_freq.get())
            gain = _parse_number_expr_or_error(self.tx_gain.get())
        except ValueError as exc:
            messagebox.showerror("Transmit", str(exc))
            self._set_tx_indicator_state("idle")
            return
        controller = TxController.for_args(tx_args)
        self._tx_controller = controller
        if hasattr(self, "tx_log"):
            self.tx_log.delete("1.0", tk.END)
        self._cmd_running = True
        self._tx_running = True
        if hasattr(self, "tx_button"):
            self.tx_button.configure(state="disabled")
        if hasattr(self, "tx_stop"):
            self.tx_stop.configure(state="normal")
        if hasattr(self, "tx_retrans"):
            self.tx_retrans.configure(state="normal")
        self._start_tx_output_capture()
        started = False
        tx_file = self._tx_transmit_file_for_start()
        self._active_tx_file = tx_file
        try:
            started = controller.start_tx(
                tx_file,
                repeat=True,
                rate=rate,
                freq=freq,
                gain=gain,
                chan=0,
            )
        finally:
            if not started:
                self._active_tx_file = None
                self._stop_tx_output_capture()
        if not started:
            self._out_queue.put("TX start failed; controller still running.\n")
            self._process_queue()
            self._set_tx_indicator_state("idle")
            return
        self._set_tx_indicator_state("active")
        self._process_queue()
        self._monitor_tx_state()

    def _monitor_tx_state(self) -> None:
        controller = self._tx_controller
        if not controller:
            return
        if controller.is_running and not self._closing:
            self.after(200, self._monitor_tx_state)
            return
        if controller.last_error:
            self._out_queue.put(f"TX error: {controller.last_error}\n")
        self._stop_tx_output_capture()
        self._cmd_running = False
        self._tx_running = False
        self._active_tx_file = None
        self._last_tx_end = controller.last_end_monotonic or time.monotonic()
        self._ui(self._reset_tx_buttons)

    def stop_transmit(self, reset_ui: bool = True) -> None:
        """Gracefully stop TX via the in-process UHD controller."""
        self._stop_requested = True
        controller = self._tx_controller
        if controller is None:
            self._tx_running = False
            self._cmd_running = False
            self._active_tx_file = None
            self._last_tx_end = time.monotonic()
            self._stop_tx_output_capture()
            if reset_ui:
                self._ui(self._reset_tx_buttons)
            return
        stopped = controller.stop_tx(timeout=5.0)
        if not stopped:
            self._out_queue.put("TX stop timed out; controller still running.\n")
        self._tx_running = controller.is_running
        self._cmd_running = controller.is_running
        if not controller.is_running:
            self._active_tx_file = None
        self._last_tx_end = controller.last_end_monotonic or time.monotonic()
        if not controller.is_running:
            self._stop_tx_output_capture()
            if reset_ui:
                self._ui(self._reset_tx_buttons)

    def retransmit(self) -> None:
        """Stop any ongoing transmission and start a new one."""
        self.stop_transmit(reset_ui=False)
        # Give the previous process a moment to terminate
        self.transmit()

    def stop_receive(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
                self._proc = None
            except Exception:
                pass
        if hasattr(self, "rx_stop"):
            self.rx_stop.configure(state="disabled")
        if hasattr(self, "rx_button"):
            self.rx_button.configure(state="normal")

    def receive(self):
        out_file = self.rx_file.get().strip()
        channels = 2 if self.rx_channel_2.get() else 1
        try:
            rate = _parse_number_expr_or_error(self.rx_rate.get())
        except ValueError as exc:
            messagebox.showerror("Receive", str(exc))
            return
        arg_list = [
            "-a",
            self.rx_args.get(),
            "-f",
            self.rx_freq.get(),
            "-r",
            self.rx_rate.get(),
            "-d",
            self.rx_dur.get(),
            "-g",
            self.rx_gain.get(),
            "--dram",
        ]
        if out_file:
            arg_list += ["--output-file", out_file]
        if self.rx_channel_2.get():
            arg_list += ["--channels", "0", "1"]
        self._cmd_running = True
        if hasattr(self, "rx_stop"):
            self.rx_stop.configure(state="normal")
        if hasattr(self, "rx_button"):
            self.rx_button.configure(state="disabled")
        threading.Thread(
            target=self._run_rx_thread,
            args=(arg_list, channels, rate),
            daemon=True,
        ).start()
        self._process_queue()

    def start_continuous(self) -> None:
        if getattr(self, "_cont_thread", None):
            return
        output_prefix = self.rx_cont_output_prefix.get().strip()
        if not output_prefix:
            messagebox.showerror("Continuous", "Output prefix ist erforderlich.")
            return
        ring_text = self.rx_cont_ring_seconds.get().strip()
        if not ring_text:
            messagebox.showerror("Continuous", "Ring Buffer (s) ist erforderlich.")
            return
        try:
            rate = _parse_number_expr_or_error(self.rx_cont_rate.get())
            freq = _parse_number_expr_or_error(self.rx_cont_freq.get())
            gain = _parse_number_expr_or_error(self.rx_cont_gain.get())
            ring_seconds = _parse_number_expr_or_error(ring_text)
            snippet_seconds = _parse_number_expr_or_error(
                self.rx_cont_snippet_seconds.get()
            )
            snippet_interval = _parse_number_expr_or_error(
                self.rx_cont_snippet_interval.get()
            )
            restart_margin = _parse_number_expr_or_error(
                self.rx_cont_restart_margin.get()
            )
        except ValueError as exc:
            messagebox.showerror("Continuous", str(exc))
            return
        tx_reference_path = _strip_zeros_tx_filename(self.tx_file.get())
        self._cont_runtime_config = {
            'trim_enabled': bool(self.trim_var.get()),
            'trim_start': float(self.trim_start.get()),
            'trim_end': float(self.trim_end.get()),
            'rx_channel_view': self.rx_channel_view.get(),
            'magnitude_enabled': bool(self.rx_magnitude_enable.get()),
            'xcorr_normalized_enabled': bool(self.rx_xcorr_normalized_enable.get()),
            'normalize_enabled': bool(self.rx_xcorr_normalized_enable.get()),
            'path_cancel_enabled': bool(self.rx_path_cancel_enable.get()),
            'tx_path': tx_reference_path,
        }

        cmd = [
            "-a",
            self.rx_cont_args.get(),
            "-f",
            str(freq),
            "-g",
            str(int(round(gain))),
            "-r",
            str(rate),
            "--ring-seconds",
            str(ring_seconds),
            "--snippet-seconds",
            str(snippet_seconds),
            "--snippet-interval",
            str(snippet_interval),
            "--restart-margin-seconds",
            str(restart_margin),
            "-o",
            output_prefix,
            "--memory-only",
            "--stdout-binary",
            "--numpy",
            "--cpu-format",
            "fc32",
        ]
        self._cont_stop_event = threading.Event()
        pg_state = getattr(self, "_rx_cont_pg_state", None)
        tabview = pg_state.get("tabview") if isinstance(pg_state, dict) else None
        if tabview is not None:
            try:
                tabview.set("Signal")
            except Exception:
                pass
        self._cmd_running = True
        if hasattr(self, "rx_cont_start"):
            self.rx_cont_start.configure(state="disabled")
        if hasattr(self, "rx_cont_stop"):
            self.rx_cont_stop.configure(state="normal")
        self._start_continuous_pipeline(rate=rate, snippet_seconds=snippet_seconds)
        self._cont_thread = threading.Thread(
            target=self._run_continuous_thread,
            args=(cmd, rate, self._cont_stop_event),
            daemon=True,
        )
        self._cont_thread.start()
        self._process_queue()

    def _start_continuous_pipeline(self, *, rate: float, snippet_seconds: float) -> None:
        self._cont_task_queue = multiprocessing.Queue(maxsize=2)
        self._cont_result_queue = multiprocessing.Queue(maxsize=2)
        self._prepare_continuous_input_slots(rate=rate, snippet_seconds=snippet_seconds)
        self._cont_task_queue_drops = 0
        self._cont_rendered_frames = 0
        self._cont_last_processing_ms = 0.0
        self._cont_last_end_to_end_ms = 0.0
        self._cont_worker_result_drops = 0
        self._cont_worker_process = Process(
            target=continuous_processing_worker,
            args=(
                self._cont_task_queue,
                self._cont_result_queue,
                [slot.name for slot in self._cont_input_slots],
                self._cont_input_slot_size,
            ),
            daemon=True,
        )
        self._cont_worker_process.start()
        self.after(25, self._poll_continuous_results)

    def _stop_continuous_pipeline(self) -> None:
        if self._cont_task_queue is not None:
            with contextlib.suppress(Exception):
                self._cont_task_queue.put_nowait(None)
        if self._cont_worker_process is not None and self._cont_worker_process.is_alive():
            self._cont_worker_process.join(timeout=2)
            if self._cont_worker_process.is_alive():
                self._cont_worker_process.terminate()
                self._cont_worker_process.join(timeout=1)
        for q in (self._cont_task_queue, self._cont_result_queue):
            if q is None:
                continue
            with contextlib.suppress(Exception):
                q.close()
            with contextlib.suppress(Exception):
                q.join_thread()
        self._cont_task_queue = None
        self._cont_result_queue = None
        self._cont_worker_process = None
        self._cleanup_continuous_input_slots()

    def _prepare_continuous_input_slots(self, *, rate: float, snippet_seconds: float) -> None:
        self._cleanup_continuous_input_slots()
        estimated_bytes = int(
            max(
                CONTINUOUS_INPUT_SLOT_MIN_BYTES,
                min(
                    CONTINUOUS_INPUT_SLOT_MAX_BYTES,
                    rate * max(snippet_seconds, 0.001) * 2 * np.dtype(np.complex64).itemsize * CONTINUOUS_INPUT_SLOT_HEADROOM,
                ),
            )
        )
        self._cont_input_slot_size = estimated_bytes
        self._cont_input_slots = [_create_shared_memory(estimated_bytes) for _ in range(CONTINUOUS_INPUT_SLOT_COUNT)]
        self._cont_input_free_slots = deque(range(CONTINUOUS_INPUT_SLOT_COUNT))

    def _cleanup_continuous_input_slots(self) -> None:
        while self._cont_input_slots:
            slot = self._cont_input_slots.pop()
            with contextlib.suppress(Exception):
                slot.close()
            with contextlib.suppress(Exception):
                slot.unlink()
        self._cont_input_free_slots.clear()
        self._cont_input_slot_size = 0

    def _release_continuous_input_slot(self, slot_id: object) -> None:
        if not isinstance(slot_id, int):
            return
        if slot_id < 0 or slot_id >= len(self._cont_input_slots):
            return
        if slot_id in self._cont_input_free_slots:
            return
        self._cont_input_free_slots.append(slot_id)

    def _enqueue_continuous_task(self, task: dict[str, object]) -> None:
        q = self._cont_task_queue
        if q is None:
            return

        if not self._cont_input_slots or self._cont_input_slot_size <= 0:
            return

        raw = np.ascontiguousarray(task.get('data', np.array([], dtype=np.complex64)))
        if raw.size == 0:
            return
        if raw.nbytes > self._cont_input_slot_size:
            self._cont_task_queue_drops += 1
            return

        if not self._cont_input_free_slots:
            try:
                dropped = q.get_nowait()
            except Exception:
                dropped = None
            if isinstance(dropped, dict):
                self._release_continuous_input_slot(dropped.get('slot_id'))
                self._cont_task_queue_drops += 1

        if not self._cont_input_free_slots:
            self._cont_task_queue_drops += 1
            return

        slot_id = self._cont_input_free_slots.popleft()
        slot = self._cont_input_slots[slot_id]
        slot.buf[: raw.nbytes] = raw.view(np.uint8).reshape(-1)

        payload = {
            'slot_id': slot_id,
            'nbytes': raw.nbytes,
            'shape': raw.shape,
            'dtype': raw.dtype.str,
            'fs': task.get('fs'),
            'frame_ts': task.get('frame_ts'),
        }
        payload.update({k: v for k, v in task.items() if k not in {'data', 'fs', 'frame_ts'}})

        try:
            q.put_nowait(payload)
            return
        except Exception:
            self._cont_task_queue_drops += 1

        try:
            dropped = q.get_nowait()
        except Exception:
            self._release_continuous_input_slot(slot_id)
            return

        if isinstance(dropped, dict):
            self._release_continuous_input_slot(dropped.get('slot_id'))

        try:
            q.put_nowait(payload)
        except Exception:
            self._release_continuous_input_slot(slot_id)

    def _poll_continuous_results(self) -> None:
        q = self._cont_result_queue
        if q is not None:
            latest = None
            worker_drops = 0
            while True:
                try:
                    item = q.get_nowait()
                except ThreadQueueEmpty:
                    break
                except Exception:
                    break
                self._release_continuous_input_slot(item.get('input_slot_id') if isinstance(item, dict) else None)
                if latest is not None:
                    worker_drops += 1
                latest = item
            self._cont_worker_result_drops += worker_drops
            if latest is not None:
                self._render_continuous_payload(latest)
        if getattr(self, '_cont_thread', None) is not None or q is not None:
            self.after(25, self._poll_continuous_results)

    def _render_continuous_payload(
        self,
        payload: dict[str, object],
    ) -> None:
        plot_data = np.asarray(payload.get('plot_data', np.array([], dtype=np.complex64)))
        fs = float(payload.get('fs', self.latest_fs))
        self.raw_rx_data = plot_data
        self.latest_fs_raw = fs
        self._cont_rendered_frames += 1
        self._cont_last_processing_ms = float(payload.get('processing_ms', 0.0))
        frame_ts = float(payload.get('frame_ts', time.monotonic()))
        self._cont_last_end_to_end_ms = (time.monotonic() - frame_ts) * 1000.0

        normalize_enabled = payload.get('normalize_enabled')
        if normalize_enabled is None:
            normalize_enabled = payload.get('xcorr_normalized_enabled')
        if normalize_enabled is not None:
            normalize_bool = bool(normalize_enabled)
            self.rx_xcorr_normalized_enable.set(normalize_bool)
            if self._cont_runtime_config:
                self._cont_runtime_config['normalize_enabled'] = normalize_bool
                self._cont_runtime_config['xcorr_normalized_enabled'] = normalize_bool

        self._display_rx_plots(plot_data, fs, reset_manual=False, target_tab='Continuous')

        if hasattr(self, 'rx_aoa_label'):
            self.rx_aoa_label.configure(text=str(payload.get('aoa_text', 'AoA (ESPRIT): deaktiviert')))
        if hasattr(self, 'rx_echo_aoa_label'):
            self.rx_echo_aoa_label.configure(
                text=str(payload.get('echo_aoa_text', 'Echo AoA: deaktiviert'))
            )

    def _wait_for_continuous_stop(
        self, cont_thread: threading.Thread, timeout: float = 10.0
    ) -> bool:
        dialog = tk.Toplevel(self)
        dialog.title("Continuous")
        dialog.transient(self)
        dialog.grab_set()
        dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        dialog.resizable(False, False)
        label = ttk.Label(
            dialog,
            text="Continuous-Modus wird beendet. Bitte warten…",
            padding=(20, 12),
        )
        label.pack()
        progress = ttk.Progressbar(dialog, mode="indeterminate", length=260)
        progress.pack(pady=(0, 16), padx=20)
        progress.start(10)
        end_time = time.monotonic() + timeout
        while cont_thread.is_alive() and time.monotonic() < end_time:
            cont_thread.join(timeout=0.2)
            dialog.update()
        progress.stop()
        dialog.grab_release()
        dialog.destroy()
        return not cont_thread.is_alive()

    def stop_continuous(self) -> None:
        stop_event = getattr(self, "_cont_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        cont_thread = getattr(self, "_cont_thread", None)
        if cont_thread and cont_thread.is_alive():
            cont_thread.join(timeout=5)
            if cont_thread.is_alive():
                stopped = self._wait_for_continuous_stop(cont_thread)
                if not stopped:
                    messagebox.showerror(
                        "Continuous",
                        "Continuous-Modus konnte nicht beendet werden. "
                        "Bitte erneut stoppen oder warten, bevor Single-Mode "
                        "gestartet wird.",
                    )
                    return
        self._cont_thread = None
        self._cont_runtime_config = {}
        self._stop_continuous_pipeline()
        if hasattr(self, "rx_cont_stop"):
            self.rx_cont_stop.configure(state="disabled")
        if hasattr(self, "rx_cont_start"):
            self.rx_cont_start.configure(state="normal")

    def _run_continuous_thread(
        self,
        arg_list: list[str],
        rate: float,
        stop_event: threading.Event,
    ) -> None:
        try:
            from .helpers import rx_continous

            args = rx_continous.parse_args(arg_list)

            def _callback(*, data: np.ndarray, **_info: object) -> None:
                if not data.size:
                    return
                payload = {
                    'data': data,
                    'fs': rate,
                    'frame_ts': time.monotonic(),
                }
                payload.update(self._cont_runtime_config)
                self._enqueue_continuous_task(payload)

            rx_continous.main(callback=_callback, args=args, stop_event=stop_event)
        except Exception as exc:
            self._out_queue.put(f"Continuous error: {exc}\n")
        finally:
            self._cmd_running = False
            self._cont_thread = None
            self._stop_continuous_pipeline()
            self._ui(self._reset_cont_buttons)

    def _reset_cont_buttons(self) -> None:
        if hasattr(self, "rx_cont_stop"):
            self.rx_cont_stop.configure(state="disabled")
        if hasattr(self, "rx_cont_start"):
            self.rx_cont_start.configure(state="normal")

    def on_close(self) -> None:
        self._closing = True
        self.stop_transmit()
        self.stop_receive()
        self.stop_continuous()
        if getattr(self, "_plot_worker_manager", None) is not None:
            self._plot_worker_manager.stop()
        if hasattr(self, "_tx_logger") and hasattr(self, "_tx_log_handler"):
            self._tx_logger.removeHandler(self._tx_log_handler)
        self._cmd_running = False
        _cleanup_shared_memory()
        _save_state(self._get_current_params())
        self.quit()
        self.destroy()


def main() -> None:
    _configure_multiprocessing()
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("dark-blue")
    app = TransceiverUI()
    app.mainloop()


if __name__ == "__main__":
    main()
