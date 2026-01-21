#!/usr/bin/env python3
"""Simple GUI to generate, transmit and receive signals."""
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
import math
import signal
import contextlib
import tempfile
from multiprocessing import shared_memory, Pipe, Process
from pathlib import Path
from datetime import datetime

import numpy as np
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

import sys
from .helpers.tx_generator import generate_waveform, rrc_coeffs
from .helpers.iq_utils import save_interleaved
from .helpers import rx_convert
from .helpers import doa_esprit
from .helpers.number_parser import parse_number_expr

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

# Paths to external helpers
ROOT_DIR = Path(__file__).resolve().parents[1]
BIN_DIR = ROOT_DIR / "bin"
REPLAY_BIN = str(BIN_DIR / "rfnoc_replay_samples_from_file")


class RangeSlider(ttk.Frame):
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
        self.canvas = tk.Canvas(self, width=width, height=height, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", self._on_resize)
        self.data = np.array([], dtype=np.float32)
        self.region = self.canvas.create_rectangle(
            0, 0, 0, height, fill="#ccf", outline=""
        )
        self.handle_start = self.canvas.create_line(
            0, 0, 0, height, fill="red", width=2
        )
        self.handle_end = self.canvas.create_line(
            width, 0, width, height, fill="red", width=2
        )
        self.active = None
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
    def _draw_signal(self) -> None:
        self.canvas.delete("signal")
        if self.data.size:
            y = np.abs(self.data)
            step = max(1, len(y) // self.width)
            y = y[::step]
            if np.max(y) > 0:
                y = y / np.max(y)
            prev_x = 0
            prev_y = self.height / 2
            for i, val in enumerate(y):
                x = int(i * self.width / (len(y) - 1)) if len(y) > 1 else 0
                yv = self.height / 2 - val * (self.height / 2 - 2)
                self.canvas.create_line(
                    prev_x, prev_y, x, yv, fill="gray", tags="signal"
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
        self.active = "start" if abs(x - x1) <= abs(x - x2) else "end"
        self._move(x)

    def _on_drag(self, event) -> None:
        if not self.enabled or self.active is None:
            return
        self._move(event.x)

    def _move(self, x: float) -> None:
        x = max(0, min(self.width, x))
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


class ConsoleWindow(tk.Toplevel):
    """Simple window to display text output."""

    def __init__(self, parent, title: str = "Console") -> None:
        super().__init__(parent)
        self.title(title)
        self.text = tk.Text(self, wrap="none")
        self.text.pack(fill="both", expand=True)

    def append(self, text: str) -> None:
        self.text.insert(tk.END, text)
        self.text.see(tk.END)


class SignalViewer(tk.Toplevel):
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

        trim_frame = ttk.Frame(self)
        trim_frame.grid(row=0, column=0, sticky="ew")
        trim_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
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

        self.apply_trim_btn = ttk.Button(
            trim_frame,
            text="Apply",
            command=self.update_trim,
            state="disabled",
        )
        self.apply_trim_btn.grid(row=0, column=2, padx=2)

        self.trim_start_label = ttk.Label(trim_frame, width=5)
        self.trim_start_label.grid(row=1, column=1, sticky="e")
        self.trim_end_label = ttk.Label(trim_frame, width=5)
        self.trim_end_label.grid(row=1, column=2, sticky="e")

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=1, column=0, pady=5)
        btn_frame.columnconfigure(0, weight=1)

        ttk.Button(btn_frame, text="Save Trim", command=self.save_trimmed).grid(
            row=0, column=0, padx=2
        )

        scroll = ttk.Frame(self)
        scroll.grid(row=2, column=0, sticky="nsew")
        scroll.columnconfigure(0, weight=1)
        scroll.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(scroll)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(scroll, orient="vertical", command=self.canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=vscroll.set)

        self.plots_frame = ttk.Frame(self.canvas)
        self.plots_frame.columnconfigure(0, weight=1)
        self.canvas.create_window((0, 0), window=self.plots_frame, anchor="nw")
        self.plots_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvases: list[FigureCanvasTkAgg] = []

        self.stats_label = ttk.Label(self.plots_frame, justify="left", anchor="w")

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

        modes = ["Signal", "Freq", "InstantFreq", "Crosscorr"]
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
        )
        text = _format_stats_text(stats)
        self.stats_label.grid(row=len(modes), column=0, sticky="ew", pady=2)
        self.stats_label.configure(text=text)


class OpenSignalDialog(tk.Toplevel):
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

        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Button(btn_frame, text="Open", command=self._on_open).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="left", padx=5)

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


class SignalColumn(ttk.Frame):
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

        ttk.Button(self, text="Open Signal", command=self.open_signal).grid(
            row=0, column=0, pady=2
        )

        trim_frame = ttk.Frame(self)
        trim_frame.grid(row=1, column=0, sticky="ew")
        trim_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
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

        self.apply_trim_btn = ttk.Button(
            trim_frame,
            text="Apply",
            command=self.update_trim,
            state="disabled",
        )
        self.apply_trim_btn.grid(row=0, column=2, padx=2)

        self.trim_start_label = ttk.Label(trim_frame, width=5)
        self.trim_start_label.grid(row=1, column=1, sticky="e")
        self.trim_end_label = ttk.Label(trim_frame, width=5)
        self.trim_end_label.grid(row=1, column=2, sticky="e")

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, column=0, pady=5)
        btn_frame.columnconfigure(0, weight=1)

        ttk.Button(btn_frame, text="Save Trim", command=self.save_trimmed).grid(
            row=0, column=0, padx=2
        )

        scroll = ttk.Frame(self)
        scroll.grid(row=3, column=0, sticky="nsew")
        scroll.columnconfigure(0, weight=1)
        scroll.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(scroll)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(scroll, orient="vertical", command=self.canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=vscroll.set)

        self.plots_frame = ttk.Frame(self.canvas)
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
        self.stats_label = ttk.Label(self.plots_frame, justify="left", anchor="w")

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

        modes = ["Signal", "Freq", "InstantFreq", "Crosscorr"]
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

        stats = _calc_stats(data, fs, tx_data)
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


class CompareWindow(tk.Toplevel):
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


class SuggestEntry(tk.Frame):
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
        self.entry = ttk.Entry(self, width=width, textvariable=textvariable)
        self.var = textvariable
        self.entry.grid(row=0, column=0, sticky="ew")
        self.sugg_frame = tk.Frame(self)
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
        for val in self.suggestions:
            frame = tk.Frame(self.sugg_frame, bd=1, relief="ridge", padx=2)
            lbl = tk.Label(frame, text=val)
            lbl.pack(side="left")
            rm = tk.Button(
                frame,
                text="x",
                command=lambda v=val: self._remove_suggestion(v),
                width=2,
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
        on_drag_end=None,
    ) -> None:
        self._view_box = view_box
        self._lags = np.asarray(lags)
        self._magnitudes = np.asarray(magnitudes)
        self._index = int(index)
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
    on_los_drag_end=None,
    on_echo_drag_end=None,
) -> None:
    """Attach draggable LOS/echo markers to a plot."""
    view_box = plot.getViewBox()
    if los_idx is not None:
        plot.addItem(
            DraggableLagMarker(
                view_box,
                lags,
                magnitudes,
                los_idx,
                "r",
                on_drag_end=on_los_drag_end,
            )
        )
    if echo_idx is not None:
        plot.addItem(
            DraggableLagMarker(
                view_box,
                lags,
                magnitudes,
                echo_idx,
                "g",
                on_drag_end=on_echo_drag_end,
            )
        )


def _apply_manual_lags(
    lags: np.ndarray,
    los_idx: int | None,
    echo_idx: int | None,
    manual_lags: dict[str, int | None] | None,
) -> tuple[int | None, int | None]:
    """Return marker indices adjusted by manual lag selections."""
    if manual_lags is None or lags.size == 0:
        return los_idx, echo_idx
    manual_los = manual_lags.get("los")
    manual_echo = manual_lags.get("echo")
    min_lag = float(lags.min())
    max_lag = float(lags.max())
    if manual_los is not None and min_lag <= manual_los <= max_lag:
        los_idx = int(np.abs(lags - manual_los).argmin())
    if manual_echo is not None and min_lag <= manual_echo <= max_lag:
        echo_idx = int(np.abs(lags - manual_echo).argmin())
    return los_idx, echo_idx


def _echo_delay_samples(
    lags: np.ndarray, los_idx: int | None, echo_idx: int | None
) -> int | None:
    """Return the absolute LOS/echo lag distance in samples."""
    if los_idx is None or echo_idx is None:
        return None
    return int(abs(lags[echo_idx] - lags[los_idx]))


def _xcorr_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the full cross-correlation of *a* and *b* using FFT."""
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    A = np.fft.fft(a, nfft)
    B = np.fft.fft(b, nfft)
    cc = np.fft.ifft(A * np.conj(B))
    return np.concatenate((cc[-(len(b) - 1) :], cc[: len(a)]))


def _autocorr_fft(x: np.ndarray) -> np.ndarray:
    """Return the full autocorrelation of *x* using FFT."""
    return _xcorr_fft(x, x)


def _find_los_echo(cc: np.ndarray) -> tuple[int | None, int | None]:
    """Return indices of the LOS peak and the first echo in ``cc``.

    Parameters
    ----------
    cc : np.ndarray
        Complex cross-correlation data.

    Returns
    -------
    tuple[int | None, int | None]
        Index of the LOS peak and the first echo. ``None`` if not found.
    """

    mag = np.abs(cc)
    if mag.size == 0:
        return None, None

    los = int(np.argmax(mag))
    echo = None
    for i in range(los + 1, len(mag) - 1):
        if mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]:
            echo = int(i)
            break

    if echo is None and los + 1 < len(mag):
        echo = int(np.argmax(mag[los + 1 :]) + los + 1)

    return los, echo


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

    if ref_data is not None and ref_data.size and data.size:
        n = min(len(data), len(ref_data))
        cc = _xcorr_fft(data[:n], ref_data[:n])
        lags = np.arange(-n + 1, n)
        los_idx, echo_idx = _find_los_echo(cc)
        los_idx, echo_idx = _apply_manual_lags(
            lags, los_idx, echo_idx, manual_lags
        )
        stats["echo_delay"] = _echo_delay_samples(lags, los_idx, echo_idx)

    return stats


def _format_stats_text(stats: dict) -> str:
    """Return a formatted multi-line string for signal statistics."""
    lines = [
        f"fmin: {_format_hz(stats['f_low'])}",
        f"fmax: {_format_hz(stats['f_high'])}",
        f"max Amp: {stats['amp']:.1f}",
        f"BW (3dB): {_format_hz(stats['bw'])}",
    ]
    if stats.get("bw_norm_nyq") is not None:
        lines.append(f"BW (Nyq): {stats['bw_norm_nyq']:.3f}")
    if stats.get("bw_rs") is not None:
        lines.append(f"BW (Rs): {stats['bw_rs']:.3f}×Rs")
    if stats.get("echo_delay") is not None:
        meters = stats["echo_delay"] * 1.5
        lines.append(f"LOS-Echo: {stats['echo_delay']} samp ({meters:.1f} m)")
    return "\n".join(lines)


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
    fullscreen: bool = False,
) -> None:
    """Launch the PyQtGraph plot worker in a separate process."""
    temp_dir = Path(tempfile.mkdtemp(prefix="transceiver_plot_"))
    data_path = None
    shm_name = None
    shm_shape = None
    shm_dtype = None
    data_contiguous = np.ascontiguousarray(data)
    reduction_step = 1
    ref_contiguous = None
    if ref_data is not None and np.size(ref_data) != 0:
        ref_contiguous = np.ascontiguousarray(ref_data)
        data_contiguous, ref_contiguous, reduction_step = _reduce_pair(
            data_contiguous, ref_contiguous
        )
    else:
        data_contiguous, reduction_step = _reduce_data(data_contiguous)
    fs = float(fs) / reduction_step
    if data_contiguous.nbytes >= SHM_SIZE_THRESHOLD_BYTES:
        data_path = temp_dir / "data.npy"
        np.save(data_path, data_contiguous)
    else:
        try:
            shm = shared_memory.SharedMemory(
                create=True, size=data_contiguous.nbytes
            )
            shm_view = np.ndarray(
                data_contiguous.shape, dtype=data_contiguous.dtype, buffer=shm.buf
            )
            shm_view[...] = data_contiguous
            shm_name = shm.name
            shm_shape = list(data_contiguous.shape)
            shm_dtype = data_contiguous.dtype.str
            shm.close()

            def _unlink_shared_memory(name: str) -> None:
                try:
                    shm_cleanup = shared_memory.SharedMemory(name=name)
                except (FileNotFoundError, OSError):
                    return
                with contextlib.suppress(FileNotFoundError):
                    shm_cleanup.unlink()
                shm_cleanup.close()

            cleanup_timer = threading.Timer(
                30.0, _unlink_shared_memory, args=(shm_name,)
            )
            cleanup_timer.daemon = True
            cleanup_timer.start()
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
                ref_shm = shared_memory.SharedMemory(
                    create=True, size=ref_contiguous.nbytes
                )
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

                def _unlink_ref_shared_memory(name: str) -> None:
                    try:
                        ref_cleanup = shared_memory.SharedMemory(name=name)
                    except (FileNotFoundError, OSError):
                        return
                    with contextlib.suppress(FileNotFoundError):
                        ref_cleanup.unlink()
                    ref_cleanup.close()

                ref_timer = threading.Timer(
                    30.0, _unlink_ref_shared_memory, args=(ref_shm_name,)
                )
                ref_timer.daemon = True
                ref_timer.start()
            except (BufferError, FileNotFoundError, OSError, ValueError):
                ref_path = temp_dir / "ref.npy"
                np.save(ref_path, ref_contiguous)
        if ref_shm_name:
            payload["ref_shm_name"] = ref_shm_name
            payload["ref_shape"] = ref_shm_shape
            payload["ref_dtype"] = ref_shm_dtype
        if ref_path is not None:
            payload["ref_file"] = str(ref_path)
    if manual_lags is not None:
        payload["manual_lags"] = {
            key: (int(val) if val is not None else None)
            for key, val in manual_lags.items()
        }
        payload["output_path"] = str(temp_dir / "manual_lags.json")
    _get_plot_worker_manager().send_payload(payload)


def _plot_on_pg(
    plot: pg.PlotItem,
    data: np.ndarray,
    fs: float,
    mode: str,
    title: str,
    ref_data: np.ndarray | None = None,
    manual_lags: dict[str, int | None] | None = None,
    on_los_drag_end=None,
    on_echo_drag_end=None,
    *,
    reduce_data: bool = True,
    reduction_step: int = 1,
) -> None:
    """Helper to draw the selected visualization on a PyQtGraph PlotItem."""
    step = max(1, int(reduction_step))
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
    elif mode == "InstantFreq":
        phase = np.unwrap(np.angle(data))
        inst = np.diff(phase)
        fi = fs * inst / (2 * np.pi)
        t = np.arange(len(fi)) / fs
        plot.plot(t, fi, pen="b")
        plot.setTitle(f"Instantaneous Frequency: {title}")
        plot.setLabel("bottom", "Time [s]")
        plot.setLabel("left", "Frequency [Hz]")
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
        n = min(len(data), len(ref_data))
        cc = _xcorr_fft(data[:n], ref_data[:n])
        lags = np.arange(-n + 1, n) * step_r
        mag = np.abs(cc)
        plot.plot(lags, mag, pen="b")
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

        _add_draggable_markers(
            plot,
            lags,
            mag,
            los_idx,
            echo_idx,
            on_los_drag_end=_wrap_drag(on_los_drag_end),
            on_echo_drag_end=_wrap_drag(on_echo_drag_end),
        )
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
    elif mode == "InstantFreq":
        phase = np.unwrap(np.angle(data))
        inst = np.diff(phase)
        fi = fs * inst / (2 * np.pi)
        t = np.arange(len(fi)) / fs
        ax.plot(t, fi, "b")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
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
        fs /= step_r
        n = min(len(data), len(ref_data))
        cc = _xcorr_fft(data[:n], ref_data[:n])
        lags = np.arange(-n + 1, n) * step_r
        mag = np.abs(cc)
        ax.plot(lags, mag, "b")
        los_idx, echo_idx = _find_los_echo(cc)
        los_idx, echo_idx = _apply_manual_lags(
            lags, los_idx, echo_idx, manual_lags
        )
        if los_idx is not None:
            ax.plot(lags[los_idx], mag[los_idx], "ro")
        if echo_idx is not None:
            ax.plot(lags[echo_idx], mag[echo_idx], "go")
        delay = _echo_delay_samples(lags, los_idx, echo_idx)
        if delay is None:
            delay_text = "LOS-Echo: --"
        else:
            meters = delay * 1.5
            delay_text = f"LOS-Echo: {delay} samp ({meters:.1f} m)"
        ax.text(
            0.01,
            0.01,
            delay_text,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            color="black",
        )
        ax.set_xlabel("Lag")
        ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(True)


class TransceiverUI(tk.Tk):
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
        self._cmd_running = False
        self._proc = None
        self._stop_requested = False
        self._plot_win = None
        self.manual_xcorr_lags = {"los": None, "echo": None}
        self._tx_running = False
        self._last_tx_end = 0.0
        self._filtered_tx_file = None
        self._closing = False
        self._plot_worker_manager = _get_plot_worker_manager()
        self._plot_worker_manager.start()
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

    def create_widgets(self):
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1, uniform="cols")
        self.columnconfigure(1, weight=1, uniform="cols")
        self.columnconfigure(2, weight=1, uniform="cols")

        # ----- Column 1: Generation -----
        gen_frame = ttk.LabelFrame(self, text="Signal Generation")
        gen_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        gen_frame.columnconfigure(1, weight=1)
        gen_frame.columnconfigure(2, weight=0)

        ttk.Label(gen_frame, text="Waveform").grid(row=0, column=0, sticky="w")
        self.wave_var = tk.StringVar(value="sinus")
        wave_box = ttk.Combobox(
            gen_frame,
            textvariable=self.wave_var,
            values=["sinus", "zadoffchu", "chirp"],
            width=10,
            state="readonly",
        )
        wave_box.grid(row=0, column=1)
        wave_box.bind(
            "<<ComboboxSelected>>",
            lambda _e: (self.update_waveform_fields(), self.auto_update_tx_filename()),
        )

        ttk.Label(gen_frame, text="fs").grid(row=1, column=0, sticky="w")
        self.fs_entry = SuggestEntry(gen_frame, "fs_entry", textvariable=self.fs_var)
        self.fs_entry.grid(row=1, column=1, sticky="ew")
        self.fs_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        self.f_label = ttk.Label(gen_frame, text="f")
        self.f_label.grid(row=2, column=0, sticky="w")
        self.f_entry = SuggestEntry(gen_frame, "f_entry")
        self.f_entry.insert(0, "25e3")
        self.f_entry.grid(row=2, column=1, sticky="ew")
        self.f_entry.entry.bind("<FocusOut>", lambda _e: self.auto_update_tx_filename())

        self.f1_label = ttk.Label(gen_frame, text="f1")
        self.f1_entry = SuggestEntry(gen_frame, "f1_entry")
        self.f1_label.grid(row=3, column=0, sticky="w")
        self.f1_entry.grid(row=3, column=1, sticky="ew")
        self.f1_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        self.q_label = ttk.Label(gen_frame, text="q")
        self.q_entry = SuggestEntry(gen_frame, "q_entry")
        self.q_entry.insert(0, "1")
        # row placement will be adjusted in update_waveform_fields
        self.q_label.grid(row=2, column=0, sticky="w")
        self.q_entry.grid(row=2, column=1, sticky="ew")
        self.q_entry.entry.bind("<FocusOut>", lambda _e: self.auto_update_tx_filename())

        ttk.Label(gen_frame, text="Samples").grid(row=4, column=0, sticky="w")
        self.samples_entry = SuggestEntry(gen_frame, "samples_entry")
        self.samples_entry.insert(0, "40000")
        self.samples_entry.grid(row=4, column=1, sticky="ew")
        self.samples_entry.entry.bind(
            "<FocusOut>", lambda _e: self.auto_update_tx_filename()
        )

        ttk.Label(gen_frame, text="Repeats").grid(row=5, column=0, sticky="w")
        self.repeat_entry = SuggestEntry(gen_frame, "repeat_entry")
        self.repeat_entry.insert(0, "1")
        self.repeat_entry.grid(row=5, column=1, sticky="ew")

        self.rrc_enable = tk.BooleanVar(value=True)
        filter_label_frame = ttk.Frame(gen_frame)
        ttk.Label(filter_label_frame, text="Filter").grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            filter_label_frame,
            variable=self.rrc_enable,
            command=self._on_rrc_toggle,
        ).grid(row=0, column=1, sticky="w", padx=(4, 0))

        filter_frame = ttk.Labelframe(gen_frame, labelwidget=filter_label_frame)
        filter_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        filter_frame.columnconfigure(1, weight=1)

        self.rrc_beta_label = ttk.Label(filter_frame, text="RRC β")
        self.rrc_beta_label.grid(row=0, column=0, sticky="w")
        self.rrc_beta_entry = SuggestEntry(filter_frame, "rrc_beta_entry")
        self.rrc_beta_entry.insert(0, "0.25")
        self.rrc_beta_entry.grid(row=0, column=1, sticky="ew")
        self.rrc_beta_entry.entry.bind(
            "<FocusOut>",
            lambda _e: (self._sync_rx_inv_rrc_params(), self.auto_update_tx_filename()),
        )

        self.rrc_span_label = ttk.Label(filter_frame, text="RRC Span")
        self.rrc_span_label.grid(row=1, column=0, sticky="w")
        self.rrc_span_entry = SuggestEntry(filter_frame, "rrc_span_entry")
        self.rrc_span_entry.insert(0, "6")
        self.rrc_span_entry.grid(row=1, column=1, sticky="ew")
        self.rrc_span_entry.entry.bind(
            "<FocusOut>",
            lambda _e: (self._sync_rx_inv_rrc_params(), self.auto_update_tx_filename()),
        )

        ttk.Label(filter_frame, text="Oversampling").grid(row=2, column=0, sticky="w")
        self.os_entry = SuggestEntry(filter_frame, "os_entry")
        self.os_entry.insert(0, "1")
        self.os_entry.grid(row=2, column=1, sticky="ew")
        self.os_entry.entry.bind(
            "<FocusOut>",
            lambda _e: (
                self.auto_update_tx_filename(),
                self._sync_rx_inv_rrc_params(),
                self._reset_manual_xcorr_lags("Oversampling geändert"),
            ),
        )

        if not self.rrc_enable.get():
            self.rrc_beta_entry.entry.configure(state="disabled")
            self.rrc_span_entry.entry.configure(state="disabled")
            self.os_entry.entry.configure(state="disabled")
        else:
            self.os_entry.entry.configure(state="normal")

        ttk.Label(gen_frame, text="Zeros").grid(row=7, column=0, sticky="w")
        self.zeros_var = tk.StringVar(value="none")
        ttk.Combobox(
            gen_frame,
            textvariable=self.zeros_var,
            values=[
                "none",
                "same",
                "half",
                "quarter",
                "double",
                "quadruple",
                "octuple",
            ],
            state="readonly",
            width=10,
        ).grid(row=7, column=1, sticky="ew")

        ttk.Label(gen_frame, text="Amplitude").grid(row=8, column=0, sticky="w")
        self.amp_entry = SuggestEntry(gen_frame, "amp_entry")
        self.amp_entry.insert(0, "10000")
        self.amp_entry.grid(row=8, column=1, sticky="ew")

        ttk.Label(gen_frame, text="File").grid(row=9, column=0, sticky="w")
        self.file_entry = SuggestEntry(gen_frame, "file_entry")
        self.file_entry.insert(0, "tx_signal.bin")
        self.file_entry.grid(row=9, column=1, sticky="ew")

        ttk.Button(gen_frame, text="Generate", command=self.generate).grid(
            row=10, column=0, columnspan=2, pady=5
        )

        scroll_container = ttk.Frame(gen_frame)
        scroll_container.grid(row=11, column=0, columnspan=2, sticky="nsew")
        scroll_container.columnconfigure(0, weight=1)
        scroll_container.rowconfigure(0, weight=1)

        self.gen_canvas = tk.Canvas(scroll_container)
        self.gen_canvas.grid(row=0, column=0, sticky="nsew")
        self.gen_scroll = ttk.Scrollbar(
            scroll_container, orient="vertical", command=self.gen_canvas.yview
        )
        self.gen_scroll.grid(row=0, column=1, sticky="ns")
        self.gen_canvas.configure(yscrollcommand=self.gen_scroll.set)

        # enable mouse wheel scrolling
        self.gen_canvas.bind("<Enter>", self._bind_gen_mousewheel)
        self.gen_canvas.bind("<Leave>", self._unbind_gen_mousewheel)

        self.gen_plots_frame = ttk.Frame(self.gen_canvas)
        self.gen_plots_frame.columnconfigure(0, weight=1)
        self.gen_canvas.create_window((0, 0), window=self.gen_plots_frame, anchor="nw")
        self.gen_plots_frame.bind(
            "<Configure>",
            lambda _e: self.gen_canvas.configure(
                scrollregion=self.gen_canvas.bbox("all")
            ),
        )
        gen_frame.rowconfigure(13, weight=1)
        self.gen_canvases = []
        self.latest_data = None
        self.latest_fs = 0.0

        # ----- Presets -----
        preset_frame = ttk.LabelFrame(self, text="Presets")
        preset_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        ttk.Button(
            preset_frame, text="Load Preset", command=self.open_load_preset_window
        ).grid(row=0, column=0, padx=5)
        ttk.Button(
            preset_frame, text="Save Preset", command=self.open_save_preset_window
        ).grid(row=0, column=1, padx=5)
        ttk.Checkbutton(
            preset_frame,
            text="Sync Sample Rates",
            variable=self.sync_var,
            command=lambda: self.toggle_rate_sync(self.sync_var.get()),
        ).grid(row=0, column=2, padx=5)

        # ----- Column 2: Transmit -----
        tx_frame = ttk.LabelFrame(self, text="Transmit")
        tx_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        tx_frame.columnconfigure(1, weight=1)

        ttk.Label(tx_frame, text="Args").grid(row=0, column=0, sticky="w")
        self.tx_args = SuggestEntry(tx_frame, "tx_args")
        self.tx_args.insert(0, "addr=192.168.10.2")
        self.tx_args.grid(row=0, column=1, sticky="ew")

        ttk.Label(tx_frame, text="Rate").grid(row=1, column=0, sticky="w")
        self.tx_rate = SuggestEntry(tx_frame, "tx_rate", textvariable=self.tx_rate_var)
        self.tx_rate.grid(row=1, column=1, sticky="ew")

        ttk.Label(tx_frame, text="Freq").grid(row=2, column=0, sticky="w")
        self.tx_freq = SuggestEntry(tx_frame, "tx_freq")
        self.tx_freq.insert(0, "5.18e9")
        self.tx_freq.grid(row=2, column=1, sticky="ew")

        ttk.Label(tx_frame, text="Gain").grid(row=3, column=0, sticky="w")
        self.tx_gain = SuggestEntry(tx_frame, "tx_gain")
        self.tx_gain.insert(0, "30")
        self.tx_gain.grid(row=3, column=1, sticky="ew")

        ttk.Label(tx_frame, text="File").grid(row=4, column=0, sticky="w")
        self.tx_file = SuggestEntry(tx_frame, "tx_file")
        self.tx_file.insert(0, "tx_signal.bin")
        self.tx_file.grid(row=4, column=1, sticky="ew")
        self.tx_file.entry.bind(
            "<FocusOut>",
            lambda _e: self._reset_manual_xcorr_lags("TX-Datei geändert"),
        )

        btn_frame = ttk.Frame(tx_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=5)
        btn_frame.columnconfigure((0, 1, 2), weight=1)

        self.tx_button = ttk.Button(btn_frame, text="Transmit", command=self.transmit)
        self.tx_button.grid(row=0, column=0, padx=2)

        self.tx_retrans = ttk.Button(
            btn_frame, text="Retransmit", command=self.retransmit, state="disabled"
        )
        self.tx_retrans.grid(row=0, column=1, padx=2)

        self.tx_stop = ttk.Button(
            btn_frame, text="Stop", command=self.stop_transmit, state="disabled"
        )
        self.tx_stop.grid(row=0, column=2, padx=2)

        log_frame = ttk.Frame(tx_frame)
        log_frame.grid(row=6, column=0, columnspan=2, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.tx_log = tk.Text(log_frame, height=10, wrap="none")
        self.tx_log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.tx_log.yview
        )
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.tx_log.configure(yscrollcommand=log_scroll.set)
        tx_frame.rowconfigure(6, weight=1)

        # ----- Column 3: Receive -----
        rx_frame = ttk.LabelFrame(self, text="Receive")
        rx_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        rx_frame.columnconfigure(1, weight=1)

        ttk.Label(rx_frame, text="Args").grid(row=0, column=0, sticky="w")
        self.rx_args = SuggestEntry(rx_frame, "rx_args")
        self.rx_args.insert(0, "addr=192.168.20.2,clock_source=external")
        self.rx_args.grid(row=0, column=1, sticky="ew")

        ttk.Label(rx_frame, text="Rate").grid(row=1, column=0, sticky="w")
        self.rx_rate = SuggestEntry(rx_frame, "rx_rate", textvariable=self.rx_rate_var)
        self.rx_rate.grid(row=1, column=1, sticky="ew")
        self.rx_rate.entry.bind("<FocusOut>", lambda _e: self.auto_update_rx_filename())

        ttk.Label(rx_frame, text="Freq").grid(row=2, column=0, sticky="w")
        self.rx_freq = SuggestEntry(rx_frame, "rx_freq")
        self.rx_freq.insert(0, "5.18e9")
        self.rx_freq.grid(row=2, column=1, sticky="ew")
        self.rx_freq.entry.bind("<FocusOut>", lambda _e: self.auto_update_rx_filename())

        ttk.Label(rx_frame, text="Duration").grid(row=3, column=0, sticky="w")
        self.rx_dur = SuggestEntry(rx_frame, "rx_dur")
        self.rx_dur.insert(0, "0.01")
        self.rx_dur.grid(row=3, column=1, sticky="ew")
        self.rx_dur.entry.bind("<FocusOut>", lambda _e: self.auto_update_rx_filename())

        ttk.Label(rx_frame, text="Gain").grid(row=4, column=0, sticky="w")
        self.rx_gain = SuggestEntry(rx_frame, "rx_gain")
        self.rx_gain.insert(0, "80")
        self.rx_gain.grid(row=4, column=1, sticky="ew")
        self.rx_gain.entry.bind("<FocusOut>", lambda _e: self.auto_update_rx_filename())

        self.rx_channel_2 = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            rx_frame,
            text="RX Antenne 2 aktivieren",
            variable=self.rx_channel_2,
        ).grid(row=5, column=0, columnspan=2, sticky="w")

        self.rx_channel_view_label = ttk.Label(rx_frame, text="RX Ansicht")
        self.rx_channel_view_label.grid(row=6, column=0, sticky="w")
        self.rx_channel_view_box = ttk.Combobox(
            rx_frame,
            textvariable=self.rx_channel_view,
            values=["Kanal 1", "Kanal 2", "Differenz"],
            width=12,
            state="readonly",
        )
        self.rx_channel_view_box.grid(row=6, column=1, sticky="w")
        self.rx_channel_view_box.configure(state="disabled")
        self.rx_channel_view_box.bind(
            "<<ComboboxSelected>>",
            lambda _e: self.update_trim(),
        )

        self.rx_inv_rrc_enable = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            rx_frame,
            text="Inv. RRC",
            variable=self.rx_inv_rrc_enable,
            command=self._on_rx_inv_rrc_toggle,
        ).grid(row=7, column=0, columnspan=2, sticky="w")
        self.rx_inv_rrc_beta_label = ttk.Label(rx_frame, text="Inv. RRC β")
        self.rx_inv_rrc_beta_entry = SuggestEntry(rx_frame, "rx_inv_rrc_beta")
        self.rx_inv_rrc_beta_entry.insert(0, "0.25")
        self.rx_inv_rrc_beta_entry.entry.configure(state="disabled")

        self.rx_inv_rrc_span_label = ttk.Label(rx_frame, text="Inv. RRC Span")
        self.rx_inv_rrc_span_entry = SuggestEntry(rx_frame, "rx_inv_rrc_span")
        self.rx_inv_rrc_span_entry.insert(0, "6")
        self.rx_inv_rrc_span_entry.entry.configure(state="disabled")

        self.rx_inv_os_entry = SuggestEntry(rx_frame, "rx_inv_os_entry")
        self.rx_inv_os_entry.insert(0, "1")
        self.rx_inv_os_entry.entry.configure(state="disabled")
        ttk.Label(rx_frame, text="Output").grid(row=8, column=0, sticky="w")
        self.rx_file = SuggestEntry(rx_frame, "rx_file")
        self.rx_file.insert(0, "rx_signal.bin")
        self.rx_file.grid(row=8, column=1, sticky="ew")

        ttk.Label(rx_frame, text="View").grid(row=9, column=0, sticky="w")
        ttk.Combobox(
            rx_frame,
            textvariable=self.rx_view,
            values=["Signal", "Freq", "InstantFreq", "Crosscorr", "AoA (ESPRIT)"],
            width=12,
        ).grid(row=9, column=1)

        ttk.Label(rx_frame, text="Antennenabstand [m]").grid(
            row=10, column=0, sticky="w"
        )
        self.rx_ant_spacing = SuggestEntry(rx_frame, "rx_ant_spacing")
        self.rx_ant_spacing.insert(0, "0.03")
        self.rx_ant_spacing.grid(row=10, column=1, sticky="ew")

        ttk.Label(rx_frame, text="Wellenlänge [m]").grid(
            row=11, column=0, sticky="w"
        )
        self.rx_wavelength = SuggestEntry(rx_frame, "rx_wavelength")
        self.rx_wavelength.insert(0, "3e8/5.18e9")
        self.rx_wavelength.grid(row=11, column=1, sticky="ew")

        self.rx_aoa_label = ttk.Label(rx_frame, text="AoA (ESPRIT): --")
        self.rx_aoa_label.grid(row=12, column=0, columnspan=2, sticky="w")
        self.rx_echo_aoa_label = ttk.Label(rx_frame, text="Echo AoA: --")
        self.rx_echo_aoa_label.grid(row=13, column=0, columnspan=2, sticky="w")

        # --- Trim controls -------------------------------------------------
        self.trim_var = tk.BooleanVar(value=False)
        self.trim_start = tk.DoubleVar(value=0.0)
        self.trim_end = tk.DoubleVar(value=100.0)
        self.trim_dirty = False

        trim_frame = ttk.Frame(rx_frame)
        trim_frame.grid(row=14, column=0, columnspan=2, sticky="ew")
        trim_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
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

        self.apply_trim_btn = ttk.Button(
            trim_frame,
            text="Apply",
            command=self.update_trim,
            state="disabled",
        )
        self.apply_trim_btn.grid(row=0, column=2, padx=2)

        self.trim_start_label = ttk.Label(trim_frame, width=5)
        self.trim_start_label.grid(row=1, column=1, sticky="e")
        self.trim_end_label = ttk.Label(trim_frame, width=5)
        self.trim_end_label.grid(row=1, column=2, sticky="e")

        rx_btn_frame = ttk.Frame(rx_frame)
        rx_btn_frame.grid(row=15, column=0, columnspan=2, pady=5)
        rx_btn_frame.columnconfigure((0, 1, 2, 3), weight=1)

        self.rx_button = ttk.Button(rx_btn_frame, text="Receive", command=self.receive)
        self.rx_button.grid(row=0, column=0, padx=2)
        self.rx_stop = ttk.Button(
            rx_btn_frame, text="Stop", command=self.stop_receive, state="disabled"
        )
        self.rx_stop.grid(row=0, column=1, padx=2)
        self.rx_save_trim = ttk.Button(
            rx_btn_frame, text="Save Trim", command=self.save_trimmed
        )
        self.rx_save_trim.grid(row=0, column=2, padx=2)
        ttk.Button(rx_btn_frame, text="Compare", command=self.open_signal).grid(
            row=0, column=3, padx=2
        )

        rx_scroll_container = ttk.Frame(rx_frame)
        rx_scroll_container.grid(row=16, column=0, columnspan=2, sticky="nsew")
        rx_scroll_container.columnconfigure(0, weight=1)
        rx_scroll_container.rowconfigure(0, weight=1)

        self.rx_canvas = tk.Canvas(rx_scroll_container)
        self.rx_canvas.grid(row=0, column=0, sticky="nsew")
        self.rx_vscroll = ttk.Scrollbar(
            rx_scroll_container, orient="vertical", command=self.rx_canvas.yview
        )
        self.rx_vscroll.grid(row=0, column=1, sticky="ns")
        self.rx_canvas.configure(yscrollcommand=self.rx_vscroll.set)
        self.rx_canvas.bind("<Enter>", self._bind_rx_mousewheel)
        self.rx_canvas.bind("<Leave>", self._unbind_rx_mousewheel)

        self.rx_plots_frame = ttk.Frame(self.rx_canvas)
        self.rx_plots_frame.columnconfigure(0, weight=1)
        self.rx_canvas.create_window((0, 0), window=self.rx_plots_frame, anchor="nw")
        self.rx_plots_frame.bind(
            "<Configure>",
            lambda _e: self.rx_canvas.configure(
                scrollregion=self.rx_canvas.bbox("all")
            ),
        )
        rx_frame.rowconfigure(16, weight=1)
        self.rx_canvases = []
        self.update_waveform_fields()
        self._sync_rx_inv_rrc_params()
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
            self.f_label.grid(row=2, column=0, sticky="w")
            self.f_entry.grid(row=2, column=1, sticky="ew")
        elif w == "zadoffchu":
            self.q_label.grid(row=2, column=0, sticky="w")
            self.q_entry.grid(row=2, column=1, sticky="ew")
            self.rrc_beta_label.grid(row=6, column=0, sticky="w")
            self.rrc_beta_entry.grid(row=6, column=1, sticky="ew")
            self.rrc_span_label.grid(row=7, column=0, sticky="w")
            self.rrc_span_entry.grid(row=7, column=1, sticky="ew")
            state = "normal" if self.rrc_enable.get() else "disabled"
            self.rrc_beta_entry.entry.configure(state=state)
            self.rrc_span_entry.entry.configure(state=state)
        elif w == "chirp":
            self.f_label.configure(text="f0")
            self.f_label.grid(row=2, column=0, sticky="w")
            self.f_entry.grid(row=2, column=1, sticky="ew")
            self.f1_label.grid(row=3, column=0, sticky="w")
            self.f1_entry.grid(row=3, column=1, sticky="ew")

        self._sync_rx_inv_rrc_params()
        self.auto_update_tx_filename()

    def _rrc_active(self) -> bool:
        return self.rrc_enable.get() and self.wave_var.get().lower() == "zadoffchu"

    def _tx_transmit_file(self) -> str:
        if self._rrc_active():
            return self._filtered_tx_file or self.tx_file.get()
        return self.tx_file.get()

    def _on_rrc_toggle(self) -> None:
        state = "normal" if self.rrc_enable.get() else "disabled"
        self.rrc_beta_entry.entry.configure(state=state)
        self.rrc_span_entry.entry.configure(state=state)
        self.os_entry.entry.configure(state=state)
        self._sync_rx_inv_rrc_params()
        self.auto_update_tx_filename()
        self._reset_manual_xcorr_lags("RRC/Oversampling geändert")

    def _on_rx_inv_rrc_toggle(self) -> None:
        self._sync_rx_inv_rrc_params()
        self._reset_manual_xcorr_lags("Oversampling geändert")
        self.update_trim()

    def _sync_rx_inv_rrc_params(self) -> None:
        beta = self.rrc_beta_entry.get() or "0.25"
        span = self.rrc_span_entry.get() or "6"
        oversampling = self.os_entry.get() or "1"
        self.rx_inv_rrc_beta_entry.delete(0, tk.END)
        self.rx_inv_rrc_beta_entry.insert(0, beta)
        self.rx_inv_rrc_span_entry.delete(0, tk.END)
        self.rx_inv_rrc_span_entry.insert(0, span)
        self.rx_inv_os_entry.delete(0, tk.END)
        self.rx_inv_os_entry.insert(0, oversampling)
        self.rx_inv_rrc_beta_entry.entry.configure(state="disabled")
        self.rx_inv_rrc_span_entry.entry.configure(state="disabled")
        self.rx_inv_os_entry.entry.configure(state="disabled")

    def _reset_manual_xcorr_lags(self, reason: str | None = None) -> None:
        if self.manual_xcorr_lags.get("los") is None and self.manual_xcorr_lags.get(
            "echo"
        ) is None:
            return
        self.manual_xcorr_lags = {"los": None, "echo": None}
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
            self.tx_file.insert(0, filtered_name)
            self._filtered_tx_file = filtered_name
        else:
            self.tx_file.insert(0, name)
            self._filtered_tx_file = None
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
        frame: ttk.Frame,
        data: np.ndarray,
        fs: float,
        symbol_rate: float | None = None,
    ) -> None:
        modes = ["Signal", "Freq", "InstantFreq", "Autocorr"]
        for idx, mode in enumerate(modes):
            fig = Figure(figsize=(5, 2), dpi=100)
            ax = fig.add_subplot(111)
            _plot_on_mpl(ax, data, fs, mode, f"TX {mode}")
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.grid(row=idx, column=0, sticky="nsew", pady=2)
            widget.bind(
                "<Button-1>",
                lambda _e, m=mode, d=data, s=fs: self._show_fullscreen(
                    d, s, m, f"TX {m}"
                ),
            )
            self.gen_canvases.append(canvas)

        stats = _calc_stats(data, fs, symbol_rate=symbol_rate)
        text = _format_stats_text(stats)
        stats_label = ttk.Label(frame, justify="left", anchor="w")
        stats_label.grid(row=len(modes), column=0, sticky="ew", pady=2)
        stats_label.configure(text=text)

    def _display_gen_plots(
        self,
        data: np.ndarray,
        fs: float,
        filtered_data: np.ndarray | None = None,
        filtered_fs: float | None = None,
        symbol_rate: float | None = None,
        filtered_symbol_rate: float | None = None,
    ) -> None:
        """Render preview plots below the generation parameters."""
        if filtered_data is not None:
            self.latest_data = filtered_data
            self.latest_fs = filtered_fs if filtered_fs is not None else fs
        else:
            self.latest_data = data
            self.latest_fs = fs

        for child in self.gen_plots_frame.winfo_children():
            child.destroy()
        self.gen_canvases.clear()

        if filtered_data is None:
            tab_frame = ttk.Frame(self.gen_plots_frame)
            tab_frame.grid(row=0, column=0, sticky="nsew")
            tab_frame.columnconfigure(0, weight=1)
            self._render_gen_tab(tab_frame, data, fs, symbol_rate=symbol_rate)
            return

        notebook = ttk.Notebook(self.gen_plots_frame)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.gen_plots_frame.columnconfigure(0, weight=1)

        unfiltered_tab = ttk.Frame(notebook)
        unfiltered_tab.columnconfigure(0, weight=1)
        filtered_tab = ttk.Frame(notebook)
        filtered_tab.columnconfigure(0, weight=1)

        notebook.add(unfiltered_tab, text="Ungefiltert")
        notebook.add(filtered_tab, text="Gefiltert")
        notebook.select(filtered_tab)

        self._render_gen_tab(unfiltered_tab, data, fs, symbol_rate=symbol_rate)
        self._render_gen_tab(
            filtered_tab,
            filtered_data,
            filtered_fs if filtered_fs is not None else fs,
            symbol_rate=filtered_symbol_rate,
        )

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
        ref = getattr(self, "tx_data", np.array([], dtype=np.complex64))
        label = "TX"
        if self.rx_inv_rrc_enable.get():
            unfiltered = getattr(
                self, "tx_data_unfiltered", np.array([], dtype=np.complex64)
            )
            if unfiltered.size:
                ref = _strip_trailing_zeros(unfiltered)
                label = "TX ungefiltert"
            elif ref.size:
                label = "TX (gefiltert)"
        return ref, label

    def _display_rx_plots(
        self, data: np.ndarray, fs: float, reset_manual: bool = True
    ) -> None:
        """Render preview plots below the receive parameters."""
        if reset_manual:
            self._reset_manual_xcorr_lags("Neue RX-Daten")
        self.raw_rx_data = data
        self.latest_fs_raw = fs
        if data.ndim == 2 and data.shape[0] >= 2:
            self.rx_channel_view_box.configure(state="readonly")
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

        data_unfiltered = data
        fs_unfiltered = fs
        data, fac = self._apply_inverse_rrc(data)
        fs *= fac
        self.latest_fs = fs
        self.latest_data = data

        def _load_tx_samples(path: str) -> np.ndarray:
            raw = np.fromfile(path, dtype=np.int16)
            if raw.size % 2:
                raw = raw[:-1]
            raw = raw.reshape(-1, 2).astype(np.float32)
            return raw[:, 0] + 1j * raw[:, 1]

        try:
            self.tx_data = _load_tx_samples(self.tx_file.get())
        except Exception:
            self.tx_data = np.array([], dtype=np.complex64)
        self.tx_data_unfiltered = np.array([], dtype=np.complex64)
        if self.rx_inv_rrc_enable.get():
            unfiltered_path = self.file_entry.get() or self.tx_file.get()
            if unfiltered_path == self.tx_file.get():
                self.tx_data_unfiltered = self.tx_data
            else:
                try:
                    self.tx_data_unfiltered = _load_tx_samples(unfiltered_path)
                except Exception:
                    self.tx_data_unfiltered = np.array([], dtype=np.complex64)
        ref_data, ref_label = self._get_crosscorr_reference()
        unfiltered_ref_data = getattr(self, "tx_data", np.array([], dtype=np.complex64))
        unfiltered_ref_label = "TX"
        if self.rx_inv_rrc_enable.get() and unfiltered_ref_data.size:
            unfiltered_ref_label = "TX (gefiltert)"
        aoa_text = "AoA (ESPRIT): --"
        echo_aoa_text = "Echo AoA: --"
        self.echo_aoa_results = []
        aoa_data = None
        aoa_time = None
        aoa_series = None
        if self.raw_rx_data.ndim == 2 and self.raw_rx_data.shape[0] >= 2:
            aoa_data = self.raw_rx_data[:2]
            if self.trim_var.get():
                aoa_data = self._trim_data_multichannel(aoa_data)
            if self.rx_inv_rrc_enable.get():
                aoa_data = np.vstack(
                    [self._apply_inverse_rrc(chan)[0] for chan in aoa_data]
                )
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

        for c in self.rx_canvases:
            if hasattr(c, "get_tk_widget"):
                c.get_tk_widget().destroy()
            else:
                c.destroy()
        self.rx_canvases.clear()

        modes = ["Signal", "Freq", "InstantFreq", "Crosscorr"]
        title_suffix = f" ({channel_label})" if channel_label else ""
        for idx, mode in enumerate(modes):
            if mode == "Crosscorr" and self.rx_inv_rrc_enable.get():
                notebook = ttk.Notebook(self.rx_plots_frame)
                notebook.grid(row=idx, column=0, sticky="nsew", pady=2)
                self.rx_plots_frame.columnconfigure(0, weight=1)

                filtered_tab = ttk.Frame(notebook)
                filtered_tab.columnconfigure(0, weight=1)
                unfiltered_tab = ttk.Frame(notebook)
                unfiltered_tab.columnconfigure(0, weight=1)

                notebook.add(filtered_tab, text="Gefiltert")
                notebook.add(unfiltered_tab, text="Ungefiltert")
                notebook.select(filtered_tab)

                crosscorr_title = (
                    f"RX {mode}{title_suffix} ({ref_label})"
                    if ref_label
                    else f"RX {mode}{title_suffix}"
                )
                fig = Figure(figsize=(5, 2), dpi=100)
                ax = fig.add_subplot(111)
                _plot_on_mpl(
                    ax,
                    data,
                    fs,
                    mode,
                    crosscorr_title,
                    ref_data,
                    manual_lags=self.manual_xcorr_lags,
                )
                canvas = FigureCanvasTkAgg(fig, master=filtered_tab)
                canvas.draw()
                widget = canvas.get_tk_widget()
                widget.grid(row=0, column=0, sticky="nsew", pady=2)
                widget.bind(
                    "<Button-1>",
                    lambda _e, m=mode, d=data, s=fs, r=ref_data, t=crosscorr_title: (
                        self._show_fullscreen(d, s, m, t, ref_data=r)
                    ),
                )
                self.rx_canvases.append(canvas)

                unfiltered_title = (
                    f"RX {mode}{title_suffix} ({unfiltered_ref_label})"
                    if unfiltered_ref_label
                    else f"RX {mode}{title_suffix}"
                )
                fig = Figure(figsize=(5, 2), dpi=100)
                ax = fig.add_subplot(111)
                _plot_on_mpl(
                    ax,
                    data_unfiltered,
                    fs_unfiltered,
                    mode,
                    unfiltered_title,
                    unfiltered_ref_data,
                    manual_lags=self.manual_xcorr_lags,
                )
                canvas = FigureCanvasTkAgg(fig, master=unfiltered_tab)
                canvas.draw()
                widget = canvas.get_tk_widget()
                widget.grid(row=0, column=0, sticky="nsew", pady=2)
                widget.bind(
                    "<Button-1>",
                    lambda _e,
                    m=mode,
                    d=data_unfiltered,
                    s=fs_unfiltered,
                    r=unfiltered_ref_data,
                    t=unfiltered_title: (self._show_fullscreen(d, s, m, t, ref_data=r)),
                )
                self.rx_canvases.append(canvas)
                self.rx_canvases.append(notebook)
                continue

            fig = Figure(figsize=(5, 2), dpi=100)
            ax = fig.add_subplot(111)
            ref = ref_data if mode == "Crosscorr" else None
            crosscorr_title = (
                f"RX {mode}{title_suffix} ({ref_label})"
                if mode == "Crosscorr" and ref_label
                else f"RX {mode}{title_suffix}"
            )
            _plot_on_mpl(
                ax,
                data,
                fs,
                mode,
                crosscorr_title,
                ref,
                manual_lags=self.manual_xcorr_lags,
            )
            canvas = FigureCanvasTkAgg(fig, master=self.rx_plots_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.grid(row=idx, column=0, sticky="nsew", pady=2)
            if mode == "Crosscorr":
                handler = lambda _e, m=mode, d=data, s=fs, r=ref, t=crosscorr_title: (
                    self._show_fullscreen(d, s, m, t, ref_data=r)
                )
            else:
                handler = lambda _e, m=mode, d=data, s=fs: self._show_fullscreen(
                    d, s, m, f"RX {m}{title_suffix}"
                )
            widget.bind("<Button-1>", handler)
            self.rx_canvases.append(canvas)

        stats = _calc_stats(
            data,
            fs,
            ref_data,
            manual_lags=self.manual_xcorr_lags,
        )
        text = _format_stats_text(stats)
        if not hasattr(self, "rx_stats_label"):
            self.rx_stats_label = ttk.Label(
                self.rx_plots_frame, justify="left", anchor="w"
            )
        self.rx_stats_label.grid(row=len(modes), column=0, sticky="ew", pady=2)
        self.rx_stats_label.configure(text=text)
        if hasattr(self, "rx_aoa_label"):
            self.rx_aoa_label.configure(text=aoa_text)
        if hasattr(self, "rx_echo_aoa_label"):
            self.rx_echo_aoa_label.configure(text=echo_aoa_text)
        if self.rx_view.get() == "AoA (ESPRIT)":
            fig = Figure(figsize=(5, 2), dpi=100)
            ax = fig.add_subplot(111)
            if aoa_time is None or aoa_series is None or aoa_series.size == 0:
                ax.set_title("AoA (ESPRIT)")
                ax.text(0.5, 0.5, "Keine AoA-Daten", ha="center", va="center")
                ax.set_axis_off()
            else:
                t = aoa_time / fs
                ax.plot(t, aoa_series, "b")
                ax.set_title("AoA (ESPRIT)")
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Angle [deg]")
                ax.grid(True)
            canvas = FigureCanvasTkAgg(fig, master=self.rx_plots_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.grid(row=len(modes) + 1, column=0, sticky="nsew", pady=2)
            self.rx_canvases.append(canvas)

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

    def _apply_inverse_rrc(self, data: np.ndarray) -> tuple[np.ndarray, float]:
        """Return *data* after inverse RRC filtering and downsampling."""
        if not self.rx_inv_rrc_enable.get() or data.size == 0:
            return data, 1.0
        try:
            factor = int(self.rx_inv_os_entry.get())
        except Exception:
            factor = 1
        if factor < 1:
            factor = 1
        try:
            beta = float(self.rrc_beta_entry.get())
        except Exception:
            beta = 0.25
        try:
            span = int(self.rrc_span_entry.get())
        except Exception:
            span = 6
        filtered = data
        if span > 0:
            h = rrc_coeffs(beta, span, sps=factor).astype(np.complex64)
            n = len(filtered)
            H = np.fft.fft(h, n)
            eps = 1e-6
            filtered = np.fft.ifft(np.fft.fft(filtered, n) / (H + eps))
        if factor > 1:
            filtered = filtered[::factor]
            return filtered, 1.0 / float(factor)
        return filtered, 1.0

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
        if hasattr(self, "raw_rx_data") and self.raw_rx_data is not None:
            fs = getattr(self, "latest_fs_raw", self.latest_fs)
            self._display_rx_plots(self.raw_rx_data, fs, reset_manual=False)
        if (
            hasattr(self, "latest_data")
            and self.latest_data is not None
            and (
                self.trim_var.get()
                or (
                    self.rx_inv_rrc_enable.get()
                    and int(self.rx_inv_os_entry.get() or 1) > 1
                )
            )
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

    def _open_console(self, title: str) -> None:
        if self.console is None or not self.console.winfo_exists():
            self.console = ConsoleWindow(self, title)
        else:
            self.console.title(title)
            self.console.text.delete("1.0", tk.END)
        while not self._out_queue.empty():
            self._out_queue.get_nowait()

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
        while not self._out_queue.empty():
            line = self._out_queue.get_nowait()
            if self.console and self.console.winfo_exists():
                self.console.append(line)
            if hasattr(self, "tx_log") and self.tx_log.winfo_exists():
                self.tx_log.insert(tk.END, line)
                self.tx_log.see(tk.END)
        if self._cmd_running:
            self.after(100, self._process_queue)

    def _ui(self, callback) -> None:
        if self._closing:
            return
        try:
            self.after(0, callback)
        except tk.TclError:
            pass

    def _kill_stale_tx(self) -> None:
        """Terminate orphaned transmit processes from previous runs."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "rfnoc_replay_samples_from_file"],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                subprocess.run(
                    ["pkill", "-f", "rfnoc_replay_samples_from_file"],
                    capture_output=True,
                )
                # Force kill any remaining processes
                subprocess.run(
                    ["pkill", "-9", "-f", "rfnoc_replay_samples_from_file"],
                    capture_output=True,
                )
        except Exception:
            pass

    def _ping_device(self, arg_str: str) -> None:
        """Send a single ping to the configured device address."""
        addr = None
        for part in arg_str.split(","):
            if part.strip().startswith("addr="):
                addr = part.split("=", 1)[1].strip()
                break
        if not addr:
            return
        try:
            subprocess.run(
                ["ping", "-c", "1", addr],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except Exception:
            pass

    def _initial_stop(self) -> None:
        """Send a single stop signal to any running transmit helper."""
        if self._proc:
            try:
                self._proc.send_signal(signal.SIGINT)
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(Exception):
                    self._proc.terminate()
                    self._proc.wait(timeout=3)
            finally:
                if self._proc and self._proc.poll() is None:
                    with contextlib.suppress(Exception):
                        self._proc.kill()
                        self._proc.wait(timeout=2)
                self._proc = None

    def _run_cmd(
        self,
        cmd: list[str],
        max_attempts: int = 1,
        delay: float = 5.0,
        tx_args: str | None = None,
    ) -> None:
        attempt = 1
        # Ensure the replay block is stopped before retrying
        self._initial_stop()
        try:
            while attempt <= max_attempts and not self._stop_requested:
                self._kill_stale_tx()
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    self._proc = proc
                    output_lines = []
                    for line in proc.stdout:
                        output_lines.append(line)
                        self._out_queue.put(line)
                    proc.wait()
                    self._out_queue.put(f"[Exited with code {proc.returncode}]\n")
                except Exception as exc:
                    self._out_queue.put(f"Error: {exc}\n")
                    proc = None
                    output_lines = []
                finally:
                    self._proc = None

                # If the device was not found, try pinging the target once
                if (
                    output_lines
                    and any("No devices found" in l for l in output_lines)
                    and tx_args
                ):
                    self._ping_device(tx_args)

                if self._stop_requested or (proc is not None and proc.returncode == 0):
                    break

                if attempt < max_attempts:
                    self._out_queue.put(
                        f"Retry {attempt}/{max_attempts} failed, retrying...\n"
                    )
                    time.sleep(delay)
                attempt += 1
        finally:
            self._cmd_running = False
            self._proc = None
            self._tx_running = False
            self._last_tx_end = time.monotonic()
            self._ui(self._reset_tx_buttons)

    def _reset_tx_buttons(self) -> None:
        if hasattr(self, "tx_stop"):
            self.tx_stop.config(state="disabled")
        if hasattr(self, "tx_button"):
            self.tx_button.config(state="normal")
        if hasattr(self, "tx_retrans"):
            self.tx_retrans.config(state="disabled")

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
            self.rx_stop.config(state="disabled")
        if hasattr(self, "rx_button"):
            self.rx_button.config(state="normal")

    def _set_manual_xcorr_lag(self, kind: str, lag_value: float) -> None:
        """Store manual lag selection and refresh LOS/echo stats."""
        if kind not in ("los", "echo"):
            return
        self.manual_xcorr_lags[kind] = int(round(lag_value))
        self._refresh_rx_stats()

    def _refresh_rx_stats(self) -> None:
        if not hasattr(self, "rx_stats_label"):
            return
        if not hasattr(self, "latest_data") or self.latest_data is None:
            return
        ref_data, _ref_label = self._get_crosscorr_reference()
        stats = _calc_stats(
            self.latest_data,
            self.latest_fs,
            ref_data,
            manual_lags=self.manual_xcorr_lags,
        )
        self.rx_stats_label.configure(text=_format_stats_text(stats))

    def _show_fullscreen(
        self,
        data: np.ndarray,
        fs: float,
        mode: str,
        title: str,
        ref_data: np.ndarray | None = None,
    ) -> None:
        if data is None:
            return
        if mode == "Crosscorr" and ref_data is None:
            ref_data, _ref_label = self._get_crosscorr_reference()
        _spawn_plot_worker(
            data,
            fs,
            mode,
            title,
            ref_data=ref_data if ref_data is not None else getattr(self, "tx_data", None),
            manual_lags=self.manual_xcorr_lags,
            fullscreen=True,
        )

    def _show_toast(self, text: str, duration: int = 2000) -> None:
        """Show a temporary notification on top of the main window."""
        win = tk.Toplevel(self)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        lbl = ttk.Label(win, text=text, padding=10, relief="solid")
        lbl.pack()
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
            if self.rx_inv_rrc_enable.get():
                echo_data = np.vstack(
                    [self._apply_inverse_rrc(chan)[0] for chan in echo_data]
                )
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
            "rrc_beta": self.rrc_beta_entry.get(),
            "rrc_span": self.rrc_span_entry.get(),
            "rrc_enabled": self.rrc_enable.get(),
            "zeros": self.zeros_var.get(),
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
            "rx_inv_rrc_oversampling": self.os_entry.get(),
            "rx_inv_rrc_beta": self.rrc_beta_entry.get(),
            "rx_inv_rrc_span": self.rrc_span_entry.get(),
            "rx_inv_rrc_enabled": self.rx_inv_rrc_enable.get(),
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
            params.get(
                "rrc_oversampling",
                params.get(
                    "rx_inv_rrc_oversampling", params.get("oversampling", "1")
                ),
            ),
        )
        self.repeat_entry.delete(0, tk.END)
        self.repeat_entry.insert(0, params.get("repeats", "1"))
        self.rrc_beta_entry.delete(0, tk.END)
        self.rrc_beta_entry.insert(
            0, params.get("rrc_beta", params.get("rx_inv_rrc_beta", "0.25"))
        )
        self.rrc_span_entry.delete(0, tk.END)
        self.rrc_span_entry.insert(
            0, params.get("rrc_span", params.get("rx_inv_rrc_span", "6"))
        )
        self.rrc_enable.set(params.get("rrc_enabled", True))
        state = "normal" if self.rrc_enable.get() else "disabled"
        self.rrc_beta_entry.entry.configure(state=state)
        self.rrc_span_entry.entry.configure(state=state)
        self.os_entry.entry.configure(state=state)
        self.zeros_var.set(params.get("zeros", "none"))
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
        self.rx_inv_rrc_enable.set(
            params.get("rx_inv_rrc_enabled", params.get("rx_rrc_enabled", False))
        )
        self._sync_rx_inv_rrc_params()
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
        win = tk.Toplevel(self)
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

        ttk.Button(win, text="Load", command=load_selected).pack(pady=5)

    def open_save_preset_window(self) -> None:
        win = tk.Toplevel(self)
        win.title("Save Preset")
        tk.Label(win, text="Name:").grid(row=0, column=0, padx=5, pady=5)
        name_var = tk.StringVar()
        ttk.Entry(win, textvariable=name_var).grid(row=0, column=1, padx=5, pady=5)

        def save() -> None:
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Save Preset", "Name cannot be empty")
                return
            self.save_preset(name)
            win.destroy()

        ttk.Button(win, text="Save", command=save).grid(
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
        win = tk.Toplevel(self)
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

        ttk.Button(win, text="Delete", command=delete_selected).pack(pady=5)

    # ----- Actions -----
    def generate(self):
        try:
            fs = _parse_number_expr_or_error(self.fs_entry.get())
            samples = int(self.samples_entry.get())
            oversampling = int(self.os_entry.get()) if self.os_entry.get() else 1
            if not self.rrc_enable.get():
                oversampling = 1
            repeats = int(self.repeat_entry.get()) if self.repeat_entry.get() else 1
            zeros_mode = self.zeros_var.get()
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

            if repeats > 1:
                data = np.tile(data, repeats)
                if unfiltered_data is not None:
                    unfiltered_data = np.tile(unfiltered_data, repeats)
                if filtered_data is not None:
                    filtered_data = np.tile(filtered_data, repeats)

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

            unfiltered_data = _append_zeros(unfiltered_data)
            filtered_data = _append_zeros(filtered_data)
            if filtered_data is not None:
                data = filtered_data
            else:
                data = _append_zeros(data)

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
                self.tx_file.delete(0, tk.END)
                self.tx_file.insert(0, filtered_filename)
                save_interleaved(filtered_filename, filtered_data, amplitude=amp)
                self._reset_manual_xcorr_lags("TX-Datei geändert")

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
            symbol_rate = None
            filtered_symbol_rate = None
            if waveform == "zadoffchu":
                symbol_rate = fs
                if oversampling > 1 and self.rrc_enable.get():
                    # Oversampling adds samples but does not change the DAC
                    # playback rate; keep the spectrum in Hz referenced to fs.
                    filtered_symbol_rate = fs / oversampling
            if scaled_unfiltered is not None and scaled_filtered is not None:
                self._display_gen_plots(
                    scaled_unfiltered,
                    fs,
                    scaled_filtered,
                    fs,
                    symbol_rate=symbol_rate,
                    filtered_symbol_rate=filtered_symbol_rate,
                )
            else:
                self._display_gen_plots(scaled_data, fs, symbol_rate=symbol_rate)

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
        self._kill_stale_tx()
        self._stop_requested = False
        tx_args = self.tx_args.get()
        cmd = [
            REPLAY_BIN,
            "--args",
            tx_args,
            "--rate",
            self.tx_rate.get(),
            "--freq",
            self.tx_freq.get(),
            "--gain",
            self.tx_gain.get(),
            "--nsamps",
            "0",
            "--file",
            self._tx_transmit_file(),
        ]
        if hasattr(self, "tx_log"):
            self.tx_log.delete("1.0", tk.END)
        self._cmd_running = True
        self._tx_running = True
        if hasattr(self, "tx_button"):
            self.tx_button.config(state="disabled")
        if hasattr(self, "tx_stop"):
            self.tx_stop.config(state="normal")
        if hasattr(self, "tx_retrans"):
            self.tx_retrans.config(state="normal")
        threading.Thread(
            target=self._run_cmd,
            args=(cmd,),
            kwargs={"max_attempts": 10, "delay": 2.0, "tx_args": tx_args},
            daemon=True,
        ).start()
        self._process_queue()

    def stop_transmit(self) -> None:
        """Gracefully stop rfnoc_replay_samples_from_file in --nsamps 0 mode."""
        if self._proc:
            # 1) Freundlich: SIGINT (entspricht Ctrl‑C)
            try:
                self._proc.send_signal(signal.SIGINT)
                self._proc.wait(timeout=3)  # <‑ Helfer macht replay->stop()
            except subprocess.TimeoutExpired:
                # 2) Immer noch aktiv? Leicht härter: SIGTERM
                with contextlib.suppress(Exception):
                    self._proc.terminate()
                    self._proc.wait(timeout=3)
            finally:
                # 3) Wenn alles schiefgeht, letzter Ausweg SIGKILL
                if self._proc and self._proc.poll() is None:
                    with contextlib.suppress(Exception):
                        self._proc.kill()
                        self._proc.wait(timeout=2)
                self._proc = None

        # FPGA‑Block ist jetzt freigegeben → UI zurücksetzen
        self._stop_requested = True
        self._tx_running = False
        self._last_tx_end = time.monotonic()

        self.tx_stop.config(state="disabled")
        self.tx_button.config(state="normal")
        self.tx_retrans.config(state="disabled")

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
            self.rx_stop.config(state="disabled")
        if hasattr(self, "rx_button"):
            self.rx_button.config(state="normal")

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
            self.rx_stop.config(state="normal")
        if hasattr(self, "rx_button"):
            self.rx_button.config(state="disabled")
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
        _save_state(self._get_current_params())
        self.destroy()


def main() -> None:
    app = TransceiverUI()
    app.mainloop()


if __name__ == "__main__":
    main()
