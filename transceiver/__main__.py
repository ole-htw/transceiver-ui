#!/usr/bin/env python3
"""Simple GUI to generate, transmit and receive signals."""
import subprocess
import logging
import threading
import queue
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

import numpy as np
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

import sys
from .helpers.tx_generator import generate_waveform
from .helpers.iq_utils import save_interleaved
from .helpers import rx_convert
from .helpers import doa_esprit
from .helpers.correlation_utils import (
    apply_manual_lags as _apply_manual_lags,
    autocorr_fft as _autocorr_fft,
    find_los_echo as _find_los_echo,
    lag_overlap as _lag_overlap,
    xcorr_fft as _xcorr_fft,
)
from .helpers.path_cancellation import apply_path_cancellation
from .helpers.number_parser import parse_number_expr
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
SHM_SIZE_THRESHOLD_BYTES = 25 * 1024 * 1024  # 25 MB


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
    header.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
    body = ctk.CTkFrame(frame, fg_color="transparent")
    body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(6, 10))
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
    header.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 0))
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
    body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 8))
    body.columnconfigure(1, weight=1)
    frame.rowconfigure(1, weight=1)
    return frame, body, toggle


def _resolve_theme_color(color: str | tuple[str, str]) -> str:
    if isinstance(color, (tuple, list)):
        if ctk.get_appearance_mode() == "Light":
            return color[0]
        return color[1]
    return color


def _resolve_ctk_frame_bg(widget: tk.Misc) -> str:
    parent = widget
    while isinstance(parent, ctk.CTkBaseClass):
        resolved = _resolve_theme_color(parent.cget("fg_color"))
        if resolved != "transparent":
            return resolved
        parent = parent.master
    return _resolve_theme_color(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])


def _apply_mpl_transparent(fig: Figure, ax) -> None:
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)


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
    header.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
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
    body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(6, 12))
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
    body.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
    body.columnconfigure(1, weight=1)
    body.columnconfigure(2, weight=1)
    if toggle_var is not None:
        header = ctk.CTkCheckBox(
            body, text=title, variable=toggle_var, command=toggle_command, width=24
        )
    else:
        header = ctk.CTkLabel(body, text=title, font=ctk.CTkFont(weight="bold"))
    header.grid(row=0, column=0, sticky="nw", padx=(6, 12), pady=2)
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
    toggle.grid(row=0, column=0, sticky="w", padx=(10, 6), pady=6)
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
        self.signal_color = "#e3e7ff"
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
        self.region = self.canvas.create_rectangle(
            0, 0, 0, height, fill="#4b5fbf", outline=""
        )
        self.handle_start = self.canvas.create_line(
            0, 0, 0, height, fill="red", width=2
        )
        self.handle_end = self.canvas.create_line(
            width, 0, width, height, fill="red", width=2
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
        self.canvas.coords(self.handle_start, x1, 0, x1, self.height)
        self.canvas.coords(self.handle_end, x2, 0, x2, self.height)
        self.canvas.coords(self.region, x1, 0, x2, self.height)

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
        trim_frame.columnconfigure(1, weight=1)

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
        self.range_slider.grid(row=0, column=1, sticky="ew", padx=2)

        self.apply_trim_btn = ctk.CTkButton(
            trim_frame,
            text="Apply",
            command=self.update_trim,
        )
        self.apply_trim_btn.grid(row=0, column=2, padx=2)
        self.apply_trim_btn.configure(state="disabled")

        self.trim_start_label = ctk.CTkLabel(trim_frame, width=50, text="")
        self.trim_start_label.grid(row=1, column=1, sticky="e")
        self.trim_end_label = ctk.CTkLabel(trim_frame, width=50, text="")
        self.trim_end_label.grid(row=1, column=2, sticky="e")

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
        trim_frame.columnconfigure(1, weight=1)

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
        self.range_slider.grid(row=0, column=1, sticky="ew", padx=2)

        self.apply_trim_btn = ctk.CTkButton(
            trim_frame,
            text="Apply",
            command=self.update_trim,
        )
        self.apply_trim_btn.grid(row=0, column=2, padx=2)
        self.apply_trim_btn.configure(state="disabled")

        self.trim_start_label = ctk.CTkLabel(trim_frame, width=50, text="")
        self.trim_start_label.grid(row=1, column=1, sticky="e")
        self.trim_end_label = ctk.CTkLabel(trim_frame, width=50, text="")
        self.trim_end_label.grid(row=1, column=2, sticky="e")

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
            rm.pack(side="right")
            frame.pack(side="left", padx=2, pady=1)
            frame.bind("<Button-1>", lambda _e, v=val: self._fill_entry(v))
            lbl.bind("<Button-1>", lambda _e, v=val: self._fill_entry(v))


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
    """Downsample *a* and *b* using the same step so both stay aligned."""
    step = 1
    max_nbytes = max(a.nbytes, b.nbytes)
    if max_nbytes > max_bytes:
        step = int(np.ceil(max_nbytes / max_bytes))
        a = a[::step]
        b = b[::step]
    return a, b, step


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
) -> tuple[DraggableLagMarker | None, DraggableLagMarker | None]:
    """Attach draggable LOS/echo markers to a plot."""
    view_box = plot.getViewBox()
    los_marker = None
    echo_marker = None
    if los_idx is not None:
        los_marker = DraggableLagMarker(
            view_box,
            lags,
            magnitudes,
            los_idx,
            "r",
            on_drag=on_los_drag,
            on_drag_end=on_los_drag_end,
        )
        plot.addItem(los_marker)
    if echo_idx is not None:
        echo_marker = DraggableLagMarker(
            view_box,
            lags,
            magnitudes,
            echo_idx,
            "g",
            on_drag=on_echo_drag,
            on_drag_end=on_echo_drag_end,
        )
        plot.addItem(echo_marker)
    return los_marker, echo_marker


def _echo_delay_samples(
    lags: np.ndarray, los_idx: int | None, echo_idx: int | None
) -> int | None:
    """Return the absolute LOS/echo lag distance in samples."""
    if los_idx is None or echo_idx is None:
        return None
    return int(abs(lags[echo_idx] - lags[los_idx]))


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
    los_idx, _echo_idx = _find_los_echo(cc)
    los_idx, _ = _apply_manual_lags(lags, los_idx, None, manual_lags)
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


def _find_peaks_simple(
    mag: np.ndarray, rel_thresh: float = 0.2, min_dist: int = 200
) -> list[int]:
    """Find local maxima in *mag* with a relative threshold and spacing."""
    if mag.size < 3:
        return []
    thr = rel_thresh * float(np.max(mag))
    candidates = []
    for i in range(1, len(mag) - 1):
        if mag[i] >= thr and mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]:
            candidates.append(i)

    candidates.sort(key=lambda i: mag[i], reverse=True)
    picked = []
    for i in candidates:
        if all(abs(i - j) >= min_dist for j in picked):
            picked.append(i)
    picked.sort()
    return picked


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


def _aoa_from_corr_peak(
    cc1: np.ndarray,
    cc2: np.ndarray,
    peak_index: int,
    antenna_spacing: float,
    wavelength: float,
    win: int = 50,
) -> tuple[float, float]:
    """Estimate AoA from correlation outputs around *peak_index*."""
    if antenna_spacing <= 0 or wavelength <= 0:
        return float("nan"), 0.0
    start = max(0, peak_index - win)
    end = min(len(cc1), peak_index + win + 1)
    w1 = cc1[start:end]
    w2 = cc2[start:end]
    mag = np.abs(w1) + np.abs(w2)
    if np.all(mag == 0):
        return float("nan"), 0.0

    weight = mag
    z = np.sum(weight * (w2 * np.conj(w1)))
    denom = np.sum(weight * (np.abs(w1) * np.abs(w2))) + 1e-12
    coherence = float(np.abs(z) / denom)
    phi = float(np.angle(z))
    sin_theta = phi * wavelength / (2.0 * np.pi * antenna_spacing)
    sin_theta = max(-1.0, min(1.0, sin_theta))
    theta = float(np.degrees(np.arcsin(sin_theta)))
    return theta, coherence


def _correlate_and_estimate_echo_aoa(
    rx_data: np.ndarray,
    tx_data: np.ndarray,
    antenna_spacing: float,
    wavelength: float,
    rel_thresh: float = 0.2,
    min_dist: int = 200,
    peak_win: int = 50,
) -> dict:
    """Cross-correlate per-channel RX with TX and estimate AoA per peak."""
    rx = np.asarray(rx_data)
    if rx.ndim != 2 or rx.shape[0] < 2:
        raise ValueError("Need two RX channels for echo AoA estimation.")
    ch1 = rx[0]
    ch2 = rx[1]
    n = min(len(ch1), len(ch2), len(tx_data))
    ch1 = ch1[:n]
    ch2 = ch2[:n]
    txr = tx_data[:n]

    cc1 = _xcorr_fft(ch1, txr)
    cc2 = _xcorr_fft(ch2, txr)
    lags = np.arange(-(len(txr) - 1), len(ch1))
    mag = np.abs(cc1) + np.abs(cc2)
    peaks = _find_peaks_simple(mag, rel_thresh=rel_thresh, min_dist=min_dist)
    results = []
    for p in peaks:
        theta, coh = _aoa_from_corr_peak(
            cc1,
            cc2,
            p,
            antenna_spacing=antenna_spacing,
            wavelength=wavelength,
            win=peak_win,
        )
        results.append(
            {
                "peak_index": int(p),
                "lag_samp": int(lags[p]),
                "strength": float(mag[p]),
                "theta_deg": float(theta),
                "coherence": float(coh),
            }
        )

    return {
        "lags": lags,
        "cc1": cc1,
        "cc2": cc2,
        "mag": mag,
        "peaks": peaks,
        "results": results,
    }


def _pretty(val: float) -> str:
    """Shorten numeric values for filenames."""
    abs_v = abs(val)
    if abs_v >= 1e6 and abs_v % 1e6 == 0:
        return f"{int(val/1e6)}M"
    if abs_v >= 1e3 and abs_v % 1e3 == 0:
        return f"{int(val/1e3)}k"
    return f"{int(val)}"


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
    try:
        oversampling = int(app.os_entry.get())
    except Exception:
        oversampling = 1
    if not getattr(app, "rrc_enable", tk.BooleanVar(value=False)).get():
        oversampling = 1

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
        if oversampling != 1:
            parts.append(f"os{oversampling}")
    elif w == "chirp":
        f0 = _try_parse_number_expr(app.f_entry.get(), default=0.0)
        f1 = _try_parse_number_expr(app.f1_entry.get(), default=f0)
        parts.append(f"{_pretty(f0)}_{_pretty(f1)}")

    parts.append(f"fs{_pretty(fs)}")
    if w == "zadoffchu":
        parts.append(f"Nsym{samples}")
        if oversampling != 1:
            parts.append(f"Nsamp{samples * oversampling}")
    else:
        parts.append(f"N{samples}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "_".join(parts) + f"_{stamp}.bin"
    return str(Path("signals/tx") / name)


def _gen_rrc_tx_filename(filename: str) -> str:
    """Return a filtered filename derived from *filename*."""
    path = Path(filename)
    stem = path.stem if path.suffix else path.name
    return str(path.with_name(f"{stem}_rrc{path.suffix}"))


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

    stats["amp"] = float(np.max(np.abs(data))) if np.any(data) else 0.0

    # Suppress DC components which would otherwise dominate the spectrum and
    # mask the actual signal.  This prevents the f_low/f_high detection from
    # always returning 0 Hz when a noticeable DC offset is present in the
    # received samples.
    data = data - np.mean(data)

    spec = np.fft.fftshift(np.fft.fft(data))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(data), d=1 / fs))
    mag = 20 * np.log10(np.abs(spec) / len(data) + 1e-12)
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

    if path_cancel_info is not None:
        if (
            path_cancel_info.get("k0") is not None
            and path_cancel_info.get("k1") is not None
        ):
            stats["echo_delay"] = path_cancel_info.get("delta_k")
        return stats
    if ref_data is not None and ref_data.size and data.size:
        xcorr_data = data
        xcorr_ref = ref_data
        xcorr_step = 1
        if xcorr_reduce:
            xcorr_data, xcorr_ref, xcorr_step = _reduce_pair(
                xcorr_data, xcorr_ref
            )
        n = min(len(xcorr_data), len(xcorr_ref))
        cc = _xcorr_fft(xcorr_data[:n], xcorr_ref[:n])
        lags = np.arange(-n + 1, n) * xcorr_step
        los_idx, echo_idx = _find_los_echo(cc)
        los_idx, echo_idx = _apply_manual_lags(
            lags, los_idx, echo_idx, manual_lags
        )
        stats["echo_delay"] = _echo_delay_samples(lags, los_idx, echo_idx)

    return stats


def _format_stats_rows(stats: dict, *, include_bw_extras: bool = True) -> list[tuple[str, str]]:
    """Return rows of labels/values for signal statistics."""
    rows = [
        ("fmin", _format_hz(stats["f_low"])),
        ("fmax", _format_hz(stats["f_high"])),
        ("max Amp", f"{stats['amp']:.1f}"),
        ("BW (3dB)", _format_hz(stats["bw"])),
    ]
    if include_bw_extras:
        if stats.get("bw_norm_nyq") is not None:
            rows.append(("BW (Nyq)", f"{stats['bw_norm_nyq']:.3f}"))
        if stats.get("bw_rs") is not None:
            rows.append(("BW (Rs)", f"{stats['bw_rs']:.3f}×Rs"))
    if stats.get("echo_delay") is not None:
        meters = stats["echo_delay"] * 1.5
        rows.append(("LOS-Echo", f"{stats['echo_delay']} samp ({meters:.1f} m)"))
    return rows


def _format_stats_text(stats: dict) -> str:
    """Return a formatted multi-line string for signal statistics."""
    rows = _format_stats_rows(stats, include_bw_extras=True)
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
        data_contiguous, ref_contiguous, reduction_step = _reduce_pair(
            data_contiguous, ref_contiguous
        )
        if crosscorr_compare is not None and np.size(crosscorr_compare) != 0:
            compare_contiguous = np.ascontiguousarray(crosscorr_compare)[
                ::reduction_step
            ]
    else:
        data_contiguous, reduction_step = _reduce_data(data_contiguous)
    fs = float(fs) / reduction_step
    if data_contiguous.nbytes >= SHM_SIZE_THRESHOLD_BYTES:
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
        if ref_contiguous.nbytes >= SHM_SIZE_THRESHOLD_BYTES:
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
        if compare_contiguous.nbytes >= SHM_SIZE_THRESHOLD_BYTES:
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
    on_los_drag=None,
    on_echo_drag=None,
    on_los_drag_end=None,
    on_echo_drag_end=None,
    *,
    reduce_data: bool = True,
    reduction_step: int = 1,
) -> None:
    """Helper to draw the selected visualization on a PyQtGraph PlotItem."""
    step = max(1, int(reduction_step))
    scene = plot.scene()
    if scene is not None and hasattr(plot, "_xcorr_click_handler"):
        try:
            scene.sigMouseClicked.disconnect(plot._xcorr_click_handler)
        except (TypeError, RuntimeError):
            pass
        delattr(plot, "_xcorr_click_handler")
    if reduce_data and mode != "Crosscorr":
        data, step = _reduce_data(data)
        fs /= step
    if mode == "Signal":
        plot.addLegend()
        plot.plot(np.real(data), pen=pg.mkPen("b"), name="Real")
        plot.plot(
            np.imag(data), pen=pg.mkPen("r", style=QtCore.Qt.DashLine), name="Imag"
        )
        plot.setTitle(title)
        plot.setLabel("bottom", "Sample Index")
        plot.setLabel("left", "Amplitude")
    elif mode in ("Freq", "Freq Analysis"):
        spec = np.fft.fftshift(np.fft.fft(data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data), d=1 / fs))
        plot.plot(freqs, 20 * np.log10(np.abs(spec) + 1e-9), pen="b")
        plot.setTitle(f"Spectrum: {title}")
        plot.setLabel("bottom", "Frequency [Hz]")
        plot.setLabel("left", "Magnitude [dB]")
    elif mode == "Autocorr":
        ac = _autocorr_fft(data)
        lags = np.arange(-len(data) + 1, len(data))
        plot.plot(lags, np.abs(ac), pen="b")
        plot.setTitle(f"Autocorrelation: {title}")
        plot.setLabel("bottom", "Lag")
        plot.setLabel("left", "Magnitude")
    elif mode == "Crosscorr":
        if ref_data is None or ref_data.size == 0:
            plot.setTitle("No TX data")
            plot.showGrid(x=True, y=True)
            return
        step_r = step
        if reduce_data:
            data, ref_data, step_r = _reduce_pair(data, ref_data)
            fs /= step_r
            if crosscorr_compare is not None and crosscorr_compare.size:
                crosscorr_compare = crosscorr_compare[::step_r]
        n = min(len(data), len(ref_data))
        cc = _xcorr_fft(data[:n], ref_data[:n])
        lags = np.arange(-n + 1, n) * step_r
        mag = np.abs(cc)
        legend = plot.addLegend()
        main_label = (
            "mit Pfad-Cancellation"
            if crosscorr_compare is not None and crosscorr_compare.size
            else "Kreuzkorrelation"
        )
        plot.plot(lags, mag, pen="b", name=main_label)
        if crosscorr_compare is not None and crosscorr_compare.size:
            n2 = min(len(crosscorr_compare), len(ref_data))
            cc2 = _xcorr_fft(crosscorr_compare[:n2], ref_data[:n2])
            lags2 = np.arange(-n2 + 1, n2) * step_r
            plot.plot(
                lags2,
                np.abs(cc2),
                pen=pg.mkPen("m", style=QtCore.Qt.DashLine),
                name="ohne Pfad-Cancellation",
            )
        if legend is None:
            legend = plot.addLegend()
        base_los_idx, base_echo_idx = _find_los_echo(cc)
        los_idx, echo_idx = _apply_manual_lags(
            lags, base_los_idx, base_echo_idx, manual_lags
        )

        echo_text = pg.TextItem(color="k", anchor=(0, 1))

        def _position_echo_text() -> None:
            view_box = plot.getViewBox()
            x_range, y_range = view_box.viewRange()
            echo_text.setPos(x_range[0], y_range[0])

        def _update_echo_text() -> None:
            adj_los_idx, adj_echo_idx = _apply_manual_lags(
                lags, base_los_idx, base_echo_idx, manual_lags
            )
            delay = _echo_delay_samples(lags, adj_los_idx, adj_echo_idx)
            if delay is None:
                echo_text.setText("LOS-Echo: --")
            else:
                meters = delay * 1.5
                echo_text.setText(f"LOS-Echo: {delay} samp ({meters:.1f} m)")
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
        los_color = pg.mkColor("r")
        echo_color = pg.mkColor("g")
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
        los_legend.setData([0], [0])
        echo_legend.setData([0], [0])
        legend.addItem(los_legend, "LOS")
        legend.addItem(echo_legend, "Echo")

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
                idx = int(np.abs(lags - pos.x()).argmin())
                lag_value = float(lags[idx])
                if modifiers & QtCore.Qt.ShiftModifier:
                    manual_lags["los"] = int(round(lag_value))
                    if los_marker is not None:
                        los_marker.set_index(idx)
                    callback = los_drag_callback or los_end_callback
                    if callback is not None:
                        callback(idx, lag_value)
                if modifiers & QtCore.Qt.AltModifier:
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
        plot.setLabel("left", "Magnitude")
    plot.showGrid(x=True, y=True)


def _plot_on_mpl(
    ax,
    data: np.ndarray,
    fs: float,
    mode: str,
    title: str,
    ref_data: np.ndarray | None = None,
    crosscorr_compare: np.ndarray | None = None,
    manual_lags: dict[str, int | None] | None = None,
) -> None:
    """Helper to draw a small matplotlib preview plot."""
    if data.ndim != 1:
        data = np.asarray(data)
        if data.ndim >= 2:
            data = data[0]
    data, step = _reduce_data(data)
    fs /= step
    if mode == "Signal":
        ax.plot(np.real(data), "b", label="Real")
        ax.plot(np.imag(data), "r--", label="Imag")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.legend()
    elif mode in ("Freq", "Freq Analysis"):
        spec = np.fft.fftshift(np.fft.fft(data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data), d=1 / fs))
        ax.plot(freqs, 20 * np.log10(np.abs(spec) + 1e-9), "b")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
    elif mode == "Autocorr":
        ac = _autocorr_fft(data)
        lags = np.arange(-len(data) + 1, len(data))
        ax.plot(lags, np.abs(ac), "b")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Magnitude")
    elif mode == "Crosscorr":
        if ref_data is None or ref_data.size == 0:
            ax.set_title("No TX data")
            ax.grid(True)
            return
        data, ref_data, step_r = _reduce_pair(data, ref_data)
        if crosscorr_compare is not None and crosscorr_compare.size:
            crosscorr_compare = crosscorr_compare[::step_r]
        fs /= step_r
        n = min(len(data), len(ref_data))
        cc = _xcorr_fft(data[:n], ref_data[:n])
        lags = np.arange(-n + 1, n) * step_r
        mag = np.abs(cc)
        ax.plot(lags, mag, "b")
        compare_handles: list[Line2D] = []
        if crosscorr_compare is not None and crosscorr_compare.size:
            n2 = min(len(crosscorr_compare), len(ref_data))
            cc2 = _xcorr_fft(crosscorr_compare[:n2], ref_data[:n2])
            lags2 = np.arange(-n2 + 1, n2) * step_r
            mag2 = np.abs(cc2)
            ax.plot(lags2, mag2, "m--", alpha=0.8)
            compare_handles = [
                Line2D([0], [0], color="b", label="mit Pfad-Cancellation"),
                Line2D(
                    [0],
                    [0],
                    color="m",
                    linestyle="--",
                    label="ohne Pfad-Cancellation",
                ),
            ]
        los_idx, echo_idx = _find_los_echo(cc)
        los_idx, echo_idx = _apply_manual_lags(
            lags, los_idx, echo_idx, manual_lags
        )
        if los_idx is not None:
            ax.plot(lags[los_idx], mag[los_idx], "ro")
        if echo_idx is not None:
            ax.plot(lags[echo_idx], mag[echo_idx], "go")
        ax.legend(
            handles=[
                *compare_handles,
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor="r",
                    markeredgecolor="r",
                    label="LOS",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor="g",
                    markeredgecolor="g",
                    label="Echo",
                ),
            ],
            loc="upper right",
        )
        ax.set_xlabel("Lag")
        ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(True)


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
        self._tx_controller = None
        self._tx_output_capture: _FDCapture | None = None
        self._closing = False
        self._plot_worker_manager = _get_plot_worker_manager()
        self._plot_worker_manager.start()
        self._xcorr_manual_file: Path | None = None
        self._xcorr_polling = False
        self._last_path_cancel_log: tuple[object, ...] | None = None
        self._last_path_cancel_info: dict[str, object] | None = None
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

        ctk.CTkLabel(waveform_left, text="Waveform").grid(
            row=0, column=0, sticky="w", padx=label_padx
        )
        self.wave_var = tk.StringVar(value="sinus")
        wave_box = ctk.CTkComboBox(
            waveform_left,
            variable=self.wave_var,
            values=["sinus", "doppelsinus", "zadoffchu", "chirp"],
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

        ctk.CTkLabel(waveform_right, text="fs").grid(
            row=0, column=0, sticky="w", padx=label_padx
        )
        self.fs_entry = SuggestEntry(
            waveform_right, "fs_entry", textvariable=self.fs_var
        )
        self.fs_entry.grid(row=0, column=1, sticky="ew")
        self.fs_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        self.f_label = ctk.CTkLabel(waveform_left, text="f")
        self.f_label.grid(row=1, column=0, sticky="w", padx=label_padx)
        self.f_entry = SuggestEntry(waveform_left, "f_entry")
        self.f_entry.insert(0, "25e3")
        self.f_entry.grid(row=1, column=1, sticky="ew")
        self.f_entry.entry.bind("<FocusOut>", lambda _e: self.auto_update_tx_filename())

        self.f1_label = ctk.CTkLabel(waveform_left, text="f1")
        self.f1_entry = SuggestEntry(waveform_left, "f1_entry")
        self.f1_label.grid(row=2, column=0, sticky="w", padx=label_padx)
        self.f1_entry.grid(row=2, column=1, sticky="ew")
        self.f1_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        self.q_label = ctk.CTkLabel(waveform_left, text="q")
        self.q_entry = SuggestEntry(waveform_left, "q_entry")
        self.q_entry.insert(0, "1")
        # row placement will be adjusted in update_waveform_fields
        self.q_label.grid(row=1, column=0, sticky="w", padx=label_padx)
        self.q_entry.grid(row=1, column=1, sticky="ew")
        self.q_entry.entry.bind("<FocusOut>", lambda _e: self.auto_update_tx_filename())

        ctk.CTkLabel(waveform_right, text="Samples").grid(
            row=1, column=0, sticky="w", padx=label_padx
        )
        self.samples_entry = SuggestEntry(waveform_right, "samples_entry")
        self.samples_entry.insert(0, "40000")
        self.samples_entry.grid(row=1, column=1, sticky="ew")
        self.samples_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        ctk.CTkLabel(waveform_right, text="Amplitude").grid(
            row=2, column=0, sticky="w", padx=label_padx
        )
        self.amp_entry = SuggestEntry(waveform_right, "amp_entry")
        self.amp_entry.insert(0, "10000")
        self.amp_entry.grid(row=2, column=1, sticky="ew")

        self.rrc_enable = tk.BooleanVar(value=True)
        filter_frame, filter_body, _ = _make_side_bordered_group(
            gen_body,
            "Filter",
            toggle_var=self.rrc_enable,
            toggle_command=self._on_rrc_toggle,
        )
        filter_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        filter_left = ctk.CTkFrame(filter_body, fg_color="transparent")
        filter_left.grid(row=0, column=1, sticky="nsew")
        filter_left.columnconfigure(1, weight=1)
        filter_right = ctk.CTkFrame(filter_body, fg_color="transparent")
        filter_right.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        filter_right.columnconfigure(1, weight=1)

        self.rrc_beta_label = ctk.CTkLabel(filter_left, text="RRC β")
        self.rrc_beta_label.grid(row=0, column=0, sticky="w", padx=label_padx)
        self.rrc_beta_entry = SuggestEntry(filter_left, "rrc_beta_entry")
        self.rrc_beta_entry.insert(0, "0.25")
        self.rrc_beta_entry.grid(row=0, column=1, sticky="ew")
        self.rrc_beta_entry.entry.bind(
            "<FocusOut>",
            lambda _e: self.auto_update_tx_filename(),
        )

        self.rrc_span_label = ctk.CTkLabel(filter_right, text="RRC Span")
        self.rrc_span_label.grid(row=0, column=0, sticky="w", padx=label_padx)
        self.rrc_span_entry = SuggestEntry(filter_right, "rrc_span_entry")
        self.rrc_span_entry.insert(0, "6")
        self.rrc_span_entry.grid(row=0, column=1, sticky="ew")
        self.rrc_span_entry.entry.bind(
            "<FocusOut>",
            lambda _e: self.auto_update_tx_filename(),
        )

        ctk.CTkLabel(filter_left, text="Oversampling").grid(
            row=1, column=0, sticky="w", padx=label_padx
        )
        self.os_entry = SuggestEntry(filter_left, "os_entry")
        self.os_entry.insert(0, "1")
        self.os_entry.grid(row=1, column=1, sticky="ew")
        self.os_entry.entry.bind(
            "<FocusOut>",
            lambda _e: (
                self.auto_update_tx_filename(),
                self._reset_manual_xcorr_lags("Oversampling geändert"),
            ),
        )

        if not self.rrc_enable.get():
            self.rrc_beta_entry.entry.configure(state="disabled")
            self.rrc_span_entry.entry.configure(state="disabled")
            self.os_entry.entry.configure(state="disabled")
        else:
            self.os_entry.entry.configure(state="normal")

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
        self.gen_canvas.configure(yscrollcommand=self.gen_scroll.set)

        # enable mouse wheel scrolling
        self.gen_canvas.bind("<Enter>", self._bind_gen_mousewheel)
        self.gen_canvas.bind("<Leave>", self._unbind_gen_mousewheel)

        self.gen_plots_frame = ctk.CTkFrame(
            self.gen_canvas,
            fg_color=terminal_container_fg,
            corner_radius=terminal_container_corner,
        )
        self.gen_plots_frame.columnconfigure(0, weight=1)
        self.gen_plots_window = self.gen_canvas.create_window(
            (0, 0), window=self.gen_plots_frame, anchor="n"
        )
        self.gen_plots_frame.bind(
            "<Configure>",
            lambda _e: self.gen_canvas.configure(
                scrollregion=self.gen_canvas.bbox("all")
            ),
        )
        self.gen_canvas.bind(
            "<Configure>",
            lambda _e: self._center_canvas_window(
                self.gen_canvas, self.gen_plots_window
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

        tx_fields = ctk.CTkFrame(tx_body, fg_color="transparent")
        tx_fields.grid(row=0, column=0, sticky="ew")
        tx_fields.columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(tx_fields, text="Args").grid(row=0, column=0, sticky="w")
        self.tx_args = SuggestEntry(tx_fields, "tx_args")
        self.tx_args.insert(0, "addr=192.168.10.2")
        self.tx_args.grid(row=0, column=1, sticky="ew", padx=(0, 12))

        ctk.CTkLabel(tx_fields, text="Rate").grid(row=0, column=2, sticky="w")
        self.tx_rate = SuggestEntry(tx_fields, "tx_rate", textvariable=self.tx_rate_var)
        self.tx_rate.grid(row=0, column=3, sticky="ew")

        ctk.CTkLabel(tx_fields, text="Freq").grid(row=1, column=0, sticky="w")
        self.tx_freq = SuggestEntry(tx_fields, "tx_freq")
        self.tx_freq.insert(0, "5.18e9")
        self.tx_freq.grid(row=1, column=1, sticky="ew", padx=(0, 12))

        ctk.CTkLabel(tx_fields, text="Gain").grid(row=1, column=2, sticky="w")
        self.tx_gain = SuggestEntry(tx_fields, "tx_gain")
        self.tx_gain.insert(0, "30")
        self.tx_gain.grid(row=1, column=3, sticky="ew")

        ctk.CTkLabel(tx_fields, text="File").grid(row=2, column=0, sticky="w")
        self.tx_file = SuggestEntry(tx_fields, "tx_file")
        self.tx_file.insert(0, "tx_signal.bin")
        self.tx_file.grid(row=2, column=1, columnspan=3, sticky="ew")
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
        btn_frame.grid(row=1, column=0, pady=5)
        btn_frame.columnconfigure((0, 1, 2), weight=1)

        self.tx_button = ctk.CTkButton(btn_frame, text="Transmit", command=self.transmit)
        self.tx_button.grid(row=0, column=0, padx=2)

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
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.tx_log = ctk.CTkTextbox(log_frame, height=150, wrap="word")
        self.tx_log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ctk.CTkScrollbar(
            log_frame, orientation="vertical", command=self.tx_log.yview
        )
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.tx_log.configure(yscrollcommand=log_scroll.set)
        tx_body.rowconfigure(2, weight=1)

        # ----- Column 3: Receive -----
        rx_frame, rx_body = _make_section(self, "Receive")
        rx_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        rx_body.columnconfigure(1, weight=1)

        ctk.CTkLabel(rx_body, text="Args").grid(row=0, column=0, sticky="w")
        self.rx_args = SuggestEntry(rx_body, "rx_args")
        self.rx_args.insert(0, "addr=192.168.20.2,clock_source=external")
        self.rx_args.grid(row=0, column=1, sticky="ew")

        ctk.CTkLabel(rx_body, text="Rate").grid(row=1, column=0, sticky="w")
        self.rx_rate = SuggestEntry(rx_body, "rx_rate", textvariable=self.rx_rate_var)
        self.rx_rate.grid(row=1, column=1, sticky="ew")
        self.rx_rate.entry.bind("<FocusOut>", lambda _e: self.auto_update_rx_filename())

        ctk.CTkLabel(rx_body, text="Freq").grid(row=2, column=0, sticky="w")
        self.rx_freq = SuggestEntry(rx_body, "rx_freq")
        self.rx_freq.insert(0, "5.18e9")
        self.rx_freq.grid(row=2, column=1, sticky="ew")
        self.rx_freq.entry.bind("<FocusOut>", lambda _e: self.auto_update_rx_filename())

        ctk.CTkLabel(rx_body, text="Duration").grid(row=3, column=0, sticky="w")
        self.rx_dur = SuggestEntry(rx_body, "rx_dur")
        self.rx_dur.insert(0, "0.01")
        self.rx_dur.grid(row=3, column=1, sticky="ew")
        self.rx_dur.entry.bind("<FocusOut>", lambda _e: self.auto_update_rx_filename())

        ctk.CTkLabel(rx_body, text="Gain").grid(row=4, column=0, sticky="w")
        self.rx_gain = SuggestEntry(rx_body, "rx_gain")
        self.rx_gain.insert(0, "80")
        self.rx_gain.grid(row=4, column=1, sticky="ew")
        self.rx_gain.entry.bind("<FocusOut>", lambda _e: self.auto_update_rx_filename())

        self.rx_channel_2 = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            rx_body,
            text="RX Antenne 2 aktivieren",
            variable=self.rx_channel_2,
        ).grid(row=5, column=0, columnspan=2, sticky="w")

        self.rx_channel_view_label = ctk.CTkLabel(rx_body, text="RX Ansicht")
        self.rx_channel_view_label.grid(row=6, column=0, sticky="w")
        self.rx_channel_view_box = ctk.CTkComboBox(
            rx_body,
            variable=self.rx_channel_view,
            values=["Kanal 1", "Kanal 2", "Differenz"],
            width=140,
            command=lambda _value: self.update_trim(),
        )
        self.rx_channel_view_box.grid(row=6, column=1, sticky="w")
        self.rx_channel_view_box.configure(state="disabled")

        self.rx_path_cancel_enable = tk.BooleanVar(value=False)
        self.rx_path_cancel_check = ctk.CTkCheckBox(
            rx_body,
            text="Pfad-Cancellation (LOS entfernen)",
            variable=self.rx_path_cancel_enable,
            command=self._on_rx_path_cancel_toggle,
        )
        self.rx_path_cancel_check.grid(row=7, column=0, columnspan=2, sticky="w")

        ctk.CTkLabel(rx_body, text="Output").grid(row=8, column=0, sticky="w")
        self.rx_file = SuggestEntry(rx_body, "rx_file")
        self.rx_file.insert(0, "rx_signal.bin")
        self.rx_file.grid(row=8, column=1, sticky="ew")

        ctk.CTkLabel(rx_body, text="View").grid(row=9, column=0, sticky="w")
        ctk.CTkComboBox(
            rx_body,
            variable=self.rx_view,
            values=["Signal", "Freq", "Crosscorr", "AoA (ESPRIT)"],
            width=140,
        ).grid(row=9, column=1, sticky="w")

        ctk.CTkLabel(rx_body, text="Antennenabstand [m]").grid(
            row=10, column=0, sticky="w"
        )
        self.rx_ant_spacing = SuggestEntry(rx_body, "rx_ant_spacing")
        self.rx_ant_spacing.insert(0, "0.03")
        self.rx_ant_spacing.grid(row=10, column=1, sticky="ew")

        ctk.CTkLabel(rx_body, text="Wellenlänge [m]").grid(
            row=11, column=0, sticky="w"
        )
        self.rx_wavelength = SuggestEntry(rx_body, "rx_wavelength")
        self.rx_wavelength.insert(0, "3e8/5.18e9")
        self.rx_wavelength.grid(row=11, column=1, sticky="ew")

        self.rx_aoa_label = ctk.CTkLabel(rx_body, text="AoA (ESPRIT): --")
        self.rx_aoa_label.grid(row=12, column=0, columnspan=2, sticky="w")
        self.rx_echo_aoa_label = ctk.CTkLabel(rx_body, text="Echo AoA: --")
        self.rx_echo_aoa_label.grid(row=13, column=0, columnspan=2, sticky="w")

        # --- Trim controls -------------------------------------------------
        self.trim_var = tk.BooleanVar(value=False)
        self.trim_start = tk.DoubleVar(value=0.0)
        self.trim_end = tk.DoubleVar(value=100.0)
        self.trim_dirty = False

        trim_frame = ctk.CTkFrame(rx_body)
        trim_frame.grid(row=14, column=0, columnspan=2, sticky="ew")
        trim_frame.columnconfigure(1, weight=1)

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
        self.range_slider.grid(row=0, column=1, sticky="ew", padx=2)

        self.apply_trim_btn = ctk.CTkButton(
            trim_frame,
            text="Apply",
            command=self.update_trim,
        )
        self.apply_trim_btn.grid(row=0, column=2, padx=2)
        self.apply_trim_btn.configure(state="disabled")

        self.trim_start_label = ctk.CTkLabel(trim_frame, width=50, text="")
        self.trim_start_label.grid(row=1, column=1, sticky="e")
        self.trim_end_label = ctk.CTkLabel(trim_frame, width=50, text="")
        self.trim_end_label.grid(row=1, column=2, sticky="e")

        rx_btn_frame = ctk.CTkFrame(rx_body)
        rx_btn_frame.grid(row=15, column=0, columnspan=2, pady=5)
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
            rx_body,
            fg_color=terminal_container_fg,
            corner_radius=terminal_container_corner,
        )
        rx_scroll_container.grid(row=16, column=0, columnspan=2, sticky="nsew")
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
        self.rx_canvas.configure(yscrollcommand=self.rx_vscroll.set)
        self.rx_canvas.bind("<Enter>", self._bind_rx_mousewheel)
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
            lambda _e: self.rx_canvas.configure(
                scrollregion=self.rx_canvas.bbox("all")
            ),
        )
        self.rx_canvas.bind(
            "<Configure>",
            lambda _e: self._center_canvas_window(
                self.rx_canvas, self.rx_plots_window
            ),
        )
        rx_body.rowconfigure(16, weight=1)
        self.rx_canvases = []
        self.update_waveform_fields()
        self.auto_update_tx_filename()
        self.auto_update_rx_filename()
        self.toggle_rate_sync(self.sync_var.get())
        self.update_trim()

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
        self.rrc_beta_label.grid_remove()
        self.rrc_beta_entry.grid_remove()
        self.rrc_span_label.grid_remove()
        self.rrc_span_entry.grid_remove()
        self.rrc_beta_entry.entry.configure(state="disabled")
        self.rrc_span_entry.entry.configure(state="disabled")

        if w == "sinus":
            self.f_label.configure(text="f")
            self.f_label.grid(row=1, column=0, sticky="w", padx=self._label_padx)
            self.f_entry.grid(row=1, column=1, sticky="ew")
        elif w == "doppelsinus":
            self.f_label.configure(text="f1")
            self.f_label.grid(row=1, column=0, sticky="w", padx=self._label_padx)
            self.f_entry.grid(row=1, column=1, sticky="ew")
            self.f1_label.configure(text="f2")
            self.f1_label.grid(row=2, column=0, sticky="w", padx=self._label_padx)
            self.f1_entry.grid(row=2, column=1, sticky="ew")
        elif w == "zadoffchu":
            self.q_label.grid(row=1, column=0, sticky="w", padx=self._label_padx)
            self.q_entry.grid(row=1, column=1, sticky="ew")
            self.rrc_beta_label.grid(row=0, column=0, sticky="w", padx=self._label_padx)
            self.rrc_beta_entry.grid(row=0, column=1, sticky="ew")
            self.rrc_span_label.grid(row=0, column=0, sticky="w", padx=self._label_padx)
            self.rrc_span_entry.grid(row=0, column=1, sticky="ew")
            state = "normal" if self.rrc_enable.get() else "disabled"
            self.rrc_beta_entry.entry.configure(state=state)
            self.rrc_span_entry.entry.configure(state=state)
        elif w == "chirp":
            self.f_label.configure(text="f0")
            self.f_label.grid(row=1, column=0, sticky="w", padx=self._label_padx)
            self.f_entry.grid(row=1, column=1, sticky="ew")
            self.f1_label.grid(row=2, column=0, sticky="w", padx=self._label_padx)
            self.f1_entry.grid(row=2, column=1, sticky="ew")

        self.auto_update_tx_filename()

    def _rrc_active(self) -> bool:
        return self.rrc_enable.get() and self.wave_var.get().lower() == "zadoffchu"

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
        if self._rrc_active():
            return self._filtered_tx_file or self.tx_file.get()
        return self.tx_file.get()

    def _on_rrc_toggle(self) -> None:
        state = "normal" if self.rrc_enable.get() else "disabled"
        self.rrc_beta_entry.entry.configure(state=state)
        self.rrc_span_entry.entry.configure(state=state)
        self.os_entry.entry.configure(state=state)
        self.auto_update_tx_filename()
        self._reset_manual_xcorr_lags("RRC/Oversampling geändert")

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

    def _on_rx_path_cancel_toggle(self) -> None:
        self._reset_manual_xcorr_lags("Pfad-Cancellation geändert")
        self.update_trim()

    def _apply_path_cancellation(
        self, data: np.ndarray, ref_data: np.ndarray
    ) -> tuple[np.ndarray, dict[str, object]]:
        return apply_path_cancellation(
            data, ref_data, manual_lags=self.manual_xcorr_lags
        )

    def _path_cancellation_note(
        self, info: dict[str, object], ref_data: np.ndarray
    ) -> str | None:
        if not self.rx_path_cancel_enable.get():
            return None
        if ref_data.size == 0:
            return "LOS entfernt: nein (keine TX-Daten)"
        if info.get("applied"):
            return "LOS entfernt: ja"
        return "LOS entfernt: nein"

    def _update_path_cancellation_status(self) -> None:
        """Placeholder to keep path cancellation controls in sync."""
        if hasattr(self, "rx_path_cancel_check"):
            self.rx_path_cancel_check.configure(state="normal")

    def _reset_manual_xcorr_lags(self, reason: str | None = None) -> None:
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
        if self._rrc_active():
            filtered_name = _gen_rrc_tx_filename(name)
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
        stats_frame.columnconfigure(1, weight=1)
        for idx, (label, value) in enumerate(stats_rows):
            ctk.CTkLabel(stats_frame, justify="left", anchor="w", text=f"{label}:").grid(
                row=idx, column=0, sticky="w", padx=(0, 6)
            )
            ctk.CTkLabel(stats_frame, justify="left", anchor="w", text=value).grid(
                row=idx, column=1, sticky="w"
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

        if filtered_data is None and repeated_data is None and zeros_data is None:
            tab_frame = ctk.CTkFrame(self.gen_plots_frame)
            tab_frame.grid(row=0, column=0, sticky="nsew")
            tab_frame.columnconfigure(0, weight=1)
            self._render_gen_tab(tab_frame, data, fs, symbol_rate=symbol_rate)
            return

        notebook = ctk.CTkTabview(self.gen_plots_frame)
        notebook.grid(row=0, column=0, sticky="nsew")
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
        self, data: np.ndarray, fs: float, reset_manual: bool = True
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

        data_uncanceled = data

        def _load_tx_samples(path: str) -> np.ndarray:
            raw = np.fromfile(path, dtype=np.int16)
            if raw.size % 2:
                raw = raw[:-1]
            raw = raw.reshape(-1, 2).astype(np.float32)
            return raw[:, 0] + 1j * raw[:, 1]

        tx_reference_path = _strip_zeros_tx_filename(self.tx_file.get())
        try:
            self.tx_data = _load_tx_samples(tx_reference_path)
        except Exception:
            self.tx_data = np.array([], dtype=np.complex64)
        ref_data, ref_label = self._get_crosscorr_reference()
        aoa_text = "AoA (ESPRIT): --"
        echo_aoa_text = "Echo AoA: --"
        self.echo_aoa_results = []
        aoa_raw_data = None
        aoa_time = None
        aoa_series = None
        if self.raw_rx_data.ndim == 2 and self.raw_rx_data.shape[0] >= 2:
            aoa_raw_data = self.raw_rx_data[:2]
            if self.trim_var.get():
                aoa_raw_data = self._trim_data_multichannel(aoa_raw_data)
            aoa_data = aoa_raw_data
            try:
                antenna_spacing = _parse_number_expr_or_error(
                    self.rx_ant_spacing.get()
                )
                wavelength = _parse_number_expr_or_error(self.rx_wavelength.get())
                aoa_angle = doa_esprit.estimate_aoa_esprit(
                    aoa_data, antenna_spacing, wavelength
                )
                if not np.isnan(aoa_angle):
                    aoa_text = f"AoA (ESPRIT): {aoa_angle:.1f}°"
                    if self.rx_view.get() == "AoA (ESPRIT)":
                        aoa_time, aoa_series = doa_esprit.estimate_aoa_esprit_series(
                            aoa_data, antenna_spacing, wavelength
                        )
                if ref_data.size > 0:
                    echo_data = aoa_data
                    echo_out = _correlate_and_estimate_echo_aoa(
                        echo_data,
                        ref_data,
                        antenna_spacing=antenna_spacing,
                        wavelength=wavelength,
                    )
                    self.echo_aoa_results = echo_out["results"]
                    if self.echo_aoa_results:
                        items = []
                        for result in self.echo_aoa_results[:3]:
                            theta = result["theta_deg"]
                            theta_text = (
                                "nan" if np.isnan(theta) else f"{theta:.1f}°"
                            )
                            items.append(
                                "Lag {lag}: {theta} (ρ {coh:.2f})".format(
                                    lag=result["lag_samp"],
                                    theta=theta_text,
                                    coh=result["coherence"],
                                )
                            )
                        if len(self.echo_aoa_results) > 3:
                            items.append(
                                f"+{len(self.echo_aoa_results) - 3} weitere"
                            )
                        echo_aoa_text = "Echo AoA:\n" + "\n".join(items)
                    else:
                        echo_aoa_text = "Echo AoA: keine Peaks"
            except ValueError as exc:
                messagebox.showerror("AoA (ESPRIT)", str(exc))
                aoa_text = "AoA (ESPRIT): Parameter ungültig"
                echo_aoa_text = "Echo AoA: Parameter ungültig"
        else:
            aoa_text = "AoA (ESPRIT): 2 Kanäle erforderlich"
            echo_aoa_text = "Echo AoA: 2 Kanäle erforderlich"

        if (
            self.raw_rx_data.ndim == 2
            and self.raw_rx_data.shape[0] >= 2
            and ref_data.size == 0
        ):
            echo_aoa_text = "Echo AoA: TX-Daten erforderlich"

        cancel_note = None
        cancel_info: dict[str, object] | None = None
        if self.rx_path_cancel_enable.get():
            data, cancel_info = self._apply_path_cancellation(data, ref_data)
            cancel_note = self._path_cancellation_note(cancel_info, ref_data)
            if cancel_info is not None:
                self._log_path_cancellation(cancel_info, "RX")
            self._last_path_cancel_info = cancel_info
        else:
            self._last_path_cancel_info = None

        self.latest_fs = fs
        self.latest_data = data

        for c in self.rx_canvases:
            if hasattr(c, "get_tk_widget"):
                c.get_tk_widget().destroy()
            else:
                c.destroy()
        self.rx_canvases.clear()

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
            path_note: str | None,
            path_cancel_info: dict[str, object] | None,
            crosscorr_compare: np.ndarray | None,
        ) -> None:
            target_frame.columnconfigure(0, weight=1)
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
                self.rx_canvases.append(canvas)

            stats = _calc_stats(
                plot_data,
                plot_fs,
                plot_ref_data,
                manual_lags=self.manual_xcorr_lags,
                xcorr_reduce=True,
                path_cancel_info=path_cancel_info,
            )
            text = _format_stats_text(stats)
            if path_note:
                text = f"{text}\n{path_note}"
            stats_label = ctk.CTkLabel(target_frame, justify="left", anchor="w", text="")
            stats_label.grid(row=len(modes), column=0, sticky="ew", pady=2)
            stats_label.configure(text=text)
            self.rx_stats_labels.append(stats_label)
            if self.rx_view.get() == "AoA (ESPRIT)":
                fig = Figure(figsize=(5, 2), dpi=100)
                ax = fig.add_subplot(111)
                _apply_mpl_transparent(fig, ax)
                if (
                    aoa_plot_time is None
                    or aoa_plot_series is None
                    or aoa_plot_series.size == 0
                ):
                    ax.set_title("AoA (ESPRIT)")
                    ax.text(
                        0.5,
                        0.5,
                        "Keine AoA-Daten",
                        ha="center",
                        va="center",
                    )
                    ax.set_axis_off()
                else:
                    t = aoa_plot_time / plot_fs
                    ax.plot(t, aoa_plot_series, "b")
                    ax.set_title("AoA (ESPRIT)")
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel("Angle [deg]")
                    ax.grid(True)
                canvas = FigureCanvasTkAgg(fig, master=target_frame)
                canvas.draw()
                widget = canvas.get_tk_widget()
                widget.configure(bg=_resolve_ctk_frame_bg(target_frame))
                widget.grid(row=len(modes) + 1, column=0, sticky="n", pady=2)
                self.rx_canvases.append(canvas)

        _render_rx_preview(
            self.rx_plots_frame,
            data,
            fs,
            ref_data,
            ref_label,
            aoa_time,
            aoa_series,
            cancel_note,
            cancel_info if cancel_info and cancel_info.get("applied") else None,
            data_uncanceled if self.rx_path_cancel_enable.get() else None,
        )

        if hasattr(self, "rx_aoa_label"):
            self.rx_aoa_label.configure(text=aoa_text)
        if hasattr(self, "rx_echo_aoa_label"):
            self.rx_echo_aoa_label.configure(text=echo_aoa_text)

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
        delta = 0
        if hasattr(event, "delta") and event.delta:
            delta = -1 * int(event.delta / 120)
        elif event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        if delta:
            self.gen_canvas.yview_scroll(delta, "units")

    def _bind_rx_mousewheel(self, _event) -> None:
        self.rx_canvas.bind_all("<MouseWheel>", self._on_rx_mousewheel)
        self.rx_canvas.bind_all("<Button-4>", self._on_rx_mousewheel)
        self.rx_canvas.bind_all("<Button-5>", self._on_rx_mousewheel)

    def _unbind_rx_mousewheel(self, _event) -> None:
        self.rx_canvas.unbind_all("<MouseWheel>")
        self.rx_canvas.unbind_all("<Button-4>")
        self.rx_canvas.unbind_all("<Button-5>")

    def _on_rx_mousewheel(self, event) -> None:
        delta = 0
        if hasattr(event, "delta") and event.delta:
            delta = -1 * int(event.delta / 120)
        elif event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        if delta:
            self.rx_canvas.yview_scroll(delta, "units")

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

    def _run_rx_cmd(
        self, cmd: list[str], out_file: str, channels: int, rate: float
    ) -> None:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._proc = proc
            for line in proc.stdout:
                self._out_queue.put(line)
            proc.wait()
            self._out_queue.put(f"[Exited with code {proc.returncode}]\n")
        except Exception as exc:
            self._out_queue.put(f"Error: {exc}\n")
            proc = None
        finally:
            self._cmd_running = False
            self._proc = None
            self._ui(self._reset_rx_buttons)

        if proc is not None and proc.returncode == 0:
            try:
                path = Path(out_file)
                try:
                    data = rx_convert.load_iq_file(
                        path, channels=channels, layout="blocked"
                    )
                except ValueError:
                    data = rx_convert.load_iq_file(
                        path, channels=channels, layout="interleaved"
                    )
                self._ui(lambda: self._display_rx_plots(data, rate))
            except Exception as exc:
                self._out_queue.put(f"Error: {exc}\n")

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
        labels = []
        if hasattr(self, "rx_stats_labels"):
            labels = [label for label in self.rx_stats_labels if label is not None]
        elif hasattr(self, "rx_stats_label"):
            labels = [self.rx_stats_label]
        if not labels:
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
        )
        text = _format_stats_text(stats)
        for label in labels:
            label.configure(text=text)

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
        output_path = _spawn_plot_worker(
            data,
            fs,
            mode,
            title,
            ref_data=ref_data if ref_data is not None else getattr(self, "tx_data", None),
            manual_lags=self.manual_xcorr_lags,
            crosscorr_compare=crosscorr_compare,
            fullscreen=True,
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
        if (
            hasattr(self, "raw_rx_data")
            and self.raw_rx_data is not None
            and self.raw_rx_data.ndim == 2
            and self.raw_rx_data.shape[0] >= 2
        ):
            echo_data = self.raw_rx_data[:2]
            if self.trim_var.get():
                echo_data = self._trim_data_multichannel(echo_data)
            try:
                antenna_spacing = _parse_number_expr_or_error(
                    self.rx_ant_spacing.get()
                )
                wavelength = _parse_number_expr_or_error(self.rx_wavelength.get())
            except ValueError as exc:
                messagebox.showerror("XCorr Full", str(exc))
                return
            echo_out = _correlate_and_estimate_echo_aoa(
                echo_data,
                ref_data,
                antenna_spacing=antenna_spacing,
                wavelength=wavelength,
            )
            self.full_xcorr_lags = echo_out["lags"]
            self.full_xcorr_mag = echo_out["mag"]
            self.echo_aoa_results = echo_out["results"]
        else:
            data = self.latest_data
            ref = ref_data
            n = min(len(data), len(ref))
            cc = _xcorr_fft(data[:n], ref[:n])
            self.full_xcorr_lags = np.arange(-n + 1, n)
            self.full_xcorr_mag = np.abs(cc)
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
            "rrc_oversampling": self.os_entry.get(),
            "repeats": self.repeat_entry.get(),
            "repeats_enabled": self.repeat_enable.get(),
            "rrc_beta": self.rrc_beta_entry.get(),
            "rrc_span": self.rrc_span_entry.get(),
            "rrc_enabled": self.rrc_enable.get(),
            "zeros": self.zeros_var.get(),
            "zeros_enabled": self.zeros_enable.get(),
            "amplitude": self.amp_entry.get(),
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
        self.os_entry.delete(0, tk.END)
        self.os_entry.insert(
            0,
            params.get("rrc_oversampling", params.get("oversampling", "1")),
        )
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
        self.rrc_beta_entry.delete(0, tk.END)
        self.rrc_beta_entry.insert(0, params.get("rrc_beta", "0.25"))
        self.rrc_span_entry.delete(0, tk.END)
        self.rrc_span_entry.insert(0, params.get("rrc_span", "6"))
        self.rrc_enable.set(params.get("rrc_enabled", True))
        state = "normal" if self.rrc_enable.get() else "disabled"
        self.rrc_beta_entry.entry.configure(state=state)
        self.rrc_span_entry.entry.configure(state=state)
        self.os_entry.entry.configure(state=state)
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
        self.rx_path_cancel_enable.set(
            params.get("rx_path_cancellation_enabled", False)
        )
        self._update_path_cancellation_status()
        self.rx_channel_2.set(params.get("rx_channel_2", False))
        self.rx_channel_view.set(params.get("rx_channel_view", "Kanal 1"))
        self.rx_file.delete(0, tk.END)
        self.rx_file.insert(0, params.get("rx_file", ""))
        self.rx_view.set(params.get("rx_view", "Signal"))
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

    def open_save_preset_window(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("Save Preset")
        ctk.CTkLabel(win, text="Name:").grid(row=0, column=0, padx=5, pady=5)
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

    # ----- Actions -----
    def generate(self):
        try:
            fs = _parse_number_expr_or_error(self.fs_entry.get())
            samples = int(self.samples_entry.get())
            oversampling = int(self.os_entry.get()) if self.os_entry.get() else 1
            if not self.rrc_enable.get():
                oversampling = 1
            repeats = self._get_repeat_count() if self.repeat_entry.get() else 1
            zeros_mode = self.zeros_var.get() if self.zeros_enable.get() else "none"
            amp = _parse_number_expr_or_error(self.amp_entry.get())
            waveform = self.wave_var.get()
            rrc_active = self._rrc_active()
            self._last_tx_os = 1
            if waveform == "zadoffchu" and oversampling > 1 and rrc_active:
                self._last_tx_os = oversampling

            unfiltered_data = None
            filtered_data = None

            if waveform == "sinus":
                freq = _parse_number_expr_or_error(
                    self.f_entry.get(), allow_empty=True, empty_value=0.0
                )
                data = generate_waveform(
                    waveform, fs, freq, samples, oversampling=1
                )
            elif waveform == "doppelsinus":
                f1 = _parse_number_expr_or_error(
                    self.f_entry.get(), allow_empty=True, empty_value=0.0
                )
                f2 = _parse_number_expr_or_error(self.f1_entry.get())
                data = generate_waveform(
                    waveform,
                    fs,
                    f1,
                    samples,
                    f1=f2,
                    oversampling=1,
                )
            elif waveform == "zadoffchu":
                q = int(self.q_entry.get()) if self.q_entry.get() else 1
                beta = (
                    float(self.rrc_beta_entry.get())
                    if self.rrc_beta_entry.get()
                    else 0.25
                )
                span = (
                    int(self.rrc_span_entry.get()) if self.rrc_span_entry.get() else 6
                )
                if not self.rrc_enable.get():
                    span = 0

                if rrc_active:
                    unfiltered_data = generate_waveform(
                        waveform,
                        fs,
                        0.0,
                        samples,
                        q=q,
                        rrc_beta=beta,
                        rrc_span=0,
                        oversampling=1,
                    )
                    filtered_data = generate_waveform(
                        waveform,
                        fs,
                        0.0,
                        samples,
                        q=q,
                        rrc_beta=beta,
                        rrc_span=span,
                        oversampling=oversampling,
                    )
                    data = filtered_data
                else:
                    data = generate_waveform(
                        waveform,
                        fs,
                        0.0,
                        samples,
                        q=q,
                        rrc_beta=beta,
                        rrc_span=span,
                        oversampling=oversampling,
                    )
            else:  # chirp
                f0 = _parse_number_expr_or_error(
                    self.f_entry.get(), allow_empty=True, empty_value=0.0
                )
                f1_text = self.f1_entry.get()
                if f1_text:
                    f1 = _parse_number_expr_or_error(f1_text)
                else:
                    f1 = None
                data = generate_waveform(
                    waveform,
                    fs,
                    f0,
                    samples,
                    f0=f0,
                    f1=f1,
                    oversampling=1,
                )

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
                return np.concatenate(
                    [signal, np.zeros(zeros_len, dtype=np.complex64)]
                )

            repeated_data = None
            if repeats > 1:
                repeat_source = filtered_data if filtered_data is not None else data
                repeated_data = np.tile(repeat_source, repeats)

            base_data = filtered_data if filtered_data is not None else data
            final_data = repeated_data if repeated_data is not None else base_data
            zeros_data = None
            if self.zeros_enable.get() and zeros > 0:
                zeros_data = _append_zeros(final_data)
                final_data = zeros_data

            save_interleaved(
                self.file_entry.get(),
                unfiltered_data if unfiltered_data is not None else data,
                amplitude=amp,
            )
            if filtered_data is not None:
                filtered_filename = self._filtered_tx_file or _gen_rrc_tx_filename(
                    self.file_entry.get()
                )
                self._filtered_tx_file = filtered_filename
                save_interleaved(filtered_filename, filtered_data, amplitude=amp)

            if repeats > 1 and repeated_data is not None:
                repeat_base = (
                    self._filtered_tx_file
                    if filtered_data is not None
                    else self.file_entry.get()
                )
                repeat_filename = _gen_repeat_tx_filename(repeat_base)
                self._repeat_tx_file = repeat_filename
                self.tx_file.delete(0, tk.END)
                self.tx_file.insert(0, repeat_filename)
                save_interleaved(repeat_filename, repeated_data, amplitude=amp)
                self._reset_manual_xcorr_lags("TX-Datei geändert")
            else:
                self._repeat_tx_file = None
                target_file = (
                    self._filtered_tx_file
                    if filtered_data is not None
                    else self.file_entry.get()
                )
                self.tx_file.delete(0, tk.END)
                self.tx_file.insert(0, target_file)
                if filtered_data is not None:
                    self._reset_manual_xcorr_lags("TX-Datei geändert")

            if zeros_data is not None:
                zeros_base = (
                    self._repeat_tx_file
                    if self._repeat_tx_file is not None
                    else (
                        self._filtered_tx_file
                        if filtered_data is not None
                        else self.file_entry.get()
                    )
                )
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

            def _scale_for_display(signal: np.ndarray) -> np.ndarray:
                max_abs = np.max(np.abs(signal)) if np.any(signal) else 1.0
                scale = amp / max_abs if max_abs > 1e-9 else 1.0
                return signal * scale

            scaled_data = _scale_for_display(data)
            scaled_unfiltered = (
                _scale_for_display(unfiltered_data)
                if unfiltered_data is not None
                else None
            )
            scaled_filtered = (
                _scale_for_display(filtered_data)
                if filtered_data is not None
                else None
            )
            scaled_repeated = (
                _scale_for_display(repeated_data)
                if repeated_data is not None
                else None
            )
            scaled_zeros = (
                _scale_for_display(zeros_data)
                if zeros_data is not None
                else None
            )
            symbol_rate = None
            filtered_symbol_rate = None
            repeated_symbol_rate = None
            zeros_symbol_rate = None
            if waveform == "zadoffchu":
                symbol_rate = fs
                if oversampling > 1 and self.rrc_enable.get():
                    # Oversampling adds samples but does not change the DAC
                    # playback rate; keep the spectrum in Hz referenced to fs.
                    filtered_symbol_rate = fs / oversampling
                    repeated_symbol_rate = filtered_symbol_rate
                else:
                    repeated_symbol_rate = symbol_rate
            if zeros_data is not None:
                zeros_symbol_rate = repeated_symbol_rate or filtered_symbol_rate or symbol_rate
            if scaled_unfiltered is not None and scaled_filtered is not None:
                self._display_gen_plots(
                    scaled_unfiltered,
                    fs,
                    scaled_filtered,
                    fs,
                    repeated_data=scaled_repeated,
                    repeated_fs=fs,
                    zeros_data=scaled_zeros,
                    zeros_fs=fs,
                    symbol_rate=symbol_rate,
                    filtered_symbol_rate=filtered_symbol_rate,
                    repeated_symbol_rate=repeated_symbol_rate,
                    zeros_symbol_rate=zeros_symbol_rate,
                )
            elif scaled_repeated is not None or scaled_zeros is not None:
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
            wait = MIN_GAP - (now - self._last_tx_end)
            self.after(int(wait * 1000), self.transmit)
            return
        self._stop_requested = False
        tx_args = self.tx_args.get()
        try:
            rate = _parse_number_expr_or_error(self.tx_rate.get())
            freq = _parse_number_expr_or_error(self.tx_freq.get())
            gain = _parse_number_expr_or_error(self.tx_gain.get())
        except ValueError as exc:
            messagebox.showerror("Transmit", str(exc))
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
        try:
            started = controller.start_tx(
                self._tx_transmit_file(),
                repeat=True,
                rate=rate,
                freq=freq,
                gain=gain,
                chan=0,
            )
        finally:
            if not started:
                self._stop_tx_output_capture()
        if not started:
            self._out_queue.put("TX start failed; controller still running.\n")
            self._process_queue()
            return
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
        self._last_tx_end = controller.last_end_monotonic or time.monotonic()
        self._ui(self._reset_tx_buttons)

    def stop_transmit(self) -> None:
        """Gracefully stop TX via the in-process UHD controller."""
        self._stop_requested = True
        controller = self._tx_controller
        if controller is None:
            self._tx_running = False
            self._cmd_running = False
            self._last_tx_end = time.monotonic()
            self._stop_tx_output_capture()
            self._ui(self._reset_tx_buttons)
            return
        stopped = controller.stop_tx(timeout=5.0)
        if not stopped:
            self._out_queue.put("TX stop timed out; controller still running.\n")
        self._tx_running = controller.is_running
        self._cmd_running = controller.is_running
        self._last_tx_end = controller.last_end_monotonic or time.monotonic()
        if not controller.is_running:
            self._stop_tx_output_capture()
            self._ui(self._reset_tx_buttons)

    def retransmit(self) -> None:
        """Stop any ongoing transmission and start a new one."""
        self.stop_transmit()
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
        out_file = self.rx_file.get()
        channels = 2 if self.rx_channel_2.get() else 1
        try:
            rate = _parse_number_expr_or_error(self.rx_rate.get())
        except ValueError as exc:
            messagebox.showerror("Receive", str(exc))
            return
        cmd = [
            sys.executable,
            "-m",
            "transceiver.helpers.rx_to_file",
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
            "--output-file",
            out_file,
        ]
        if self.rx_channel_2.get():
            cmd += ["--channels", "0", "1"]
        self._cmd_running = True
        if hasattr(self, "rx_stop"):
            self.rx_stop.configure(state="normal")
        if hasattr(self, "rx_button"):
            self.rx_button.configure(state="disabled")
        threading.Thread(
            target=self._run_rx_cmd,
            args=(cmd, out_file, channels, rate),
            daemon=True,
        ).start()
        self._process_queue()

    def on_close(self) -> None:
        self._closing = True
        self.stop_transmit()
        self.stop_receive()
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
