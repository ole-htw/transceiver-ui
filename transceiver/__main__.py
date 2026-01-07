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
from pathlib import Path
from datetime import datetime

import numpy as np
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

import sys
from .helpers.tx_generator import generate_waveform
from .helpers import rx_convert

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
            _plot_on_mpl(ax, data, fs, mode, f"RX {mode}", self.tx_data)
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

        stats = _calc_stats(data, fs, self.tx_data)
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


def save_interleaved(
    filename: str, data: np.ndarray, amplitude: float = 10000.0
) -> None:
    """Save complex64 data as interleaved int16."""
    if data.ndim != 1:
        raise ValueError("Mehrkanal-Daten müssen vor dem Speichern gemischt werden.")
    max_abs = np.max(np.abs(data)) if np.any(data) else 1.0
    scale = amplitude / max_abs if max_abs > 1e-9 else 1.0
    scaled = data * scale
    real_i16 = np.int16(np.round(np.real(scaled)))
    imag_i16 = np.int16(np.round(np.imag(scaled)))
    interleaved = np.empty(real_i16.size + imag_i16.size, dtype=np.int16)
    interleaved[0::2] = np.clip(real_i16, -32768, 32767)
    interleaved[1::2] = np.clip(imag_i16, -32768, 32767)
    interleaved.tofile(filename)


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


def _gen_tx_filename(app) -> str:
    """Generate TX filename based on current UI settings."""
    w = app.wave_var.get().lower()
    parts = [w]
    try:
        fs = float(eval(app.fs_entry.get()))
    except Exception:
        fs = 0.0
    try:
        samples = int(app.samples_entry.get())
    except Exception:
        samples = 0
    try:
        oversampling = int(app.os_entry.get())
    except Exception:
        oversampling = 1
    if not getattr(app, "os_enable", tk.BooleanVar(value=False)).get():
        oversampling = 1

    if w == "sinus":
        try:
            f = float(eval(app.f_entry.get()))
        except Exception:
            f = 0.0
        parts.append(f"f{_pretty(f)}")
    elif w == "zadoffchu":
        q = app.q_entry.get() or "1"
        parts.append(f"q{q}")
        if oversampling != 1:
            parts.append(f"os{oversampling}")
    elif w == "chirp":
        try:
            f0 = float(eval(app.f_entry.get()))
        except Exception:
            f0 = 0.0
        try:
            f1 = float(eval(app.f1_entry.get()))
        except Exception:
            f1 = f0
        parts.append(f"{_pretty(f0)}_{_pretty(f1)}")

    parts.append(f"fs{_pretty(fs)}")
    # ``samples`` already represents the number of output samples. Using
    # ``samples * oversampling`` would misrepresent the actual length when
    # oversampling is enabled.
    parts.append(f"N{samples}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "_".join(parts) + f"_{stamp}.bin"
    return str(Path("signals/tx") / name)


def _gen_rx_filename(app) -> str:
    """Generate RX filename based on current UI settings."""
    try:
        freq = float(eval(app.rx_freq.get()))
    except Exception:
        freq = 0.0
    try:
        rate = float(eval(app.rx_rate.get()))
    except Exception:
        rate = 0.0
    dur = app.rx_dur.get() or "0"
    gain = app.rx_gain.get() or "0"
    parts = [f"f{_pretty(freq)}", f"r{_pretty(rate)}", f"d{dur}s", f"g{gain}"]
    try:
        os_factor = int(app.rx_os_entry.get())
    except Exception:
        os_factor = 1
    if (
        getattr(app, "rx_os_enable", tk.BooleanVar(value=False)).get()
        and os_factor != 1
    ):
        parts.append(f"os{os_factor}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "_".join(parts) + f"_{stamp}.bin"
    return str(Path("signals/rx") / name)


def _calc_stats(
    data: np.ndarray, fs: float, ref_data: np.ndarray | None = None
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

    if ref_data is not None and ref_data.size and data.size:
        n = min(len(data), len(ref_data))
        cc = _xcorr_fft(data[:n], ref_data[:n])
        lags = np.arange(-n + 1, n)
        los_idx, echo_idx = _find_los_echo(cc)
        if los_idx is not None and echo_idx is not None:
            stats["echo_delay"] = int(lags[echo_idx] - lags[los_idx])

    return stats


def _format_stats_text(stats: dict) -> str:
    """Return a formatted multi-line string for signal statistics."""
    lines = [
        f"fmin: {_format_hz(stats['f_low'])}",
        f"fmax: {_format_hz(stats['f_high'])}",
        f"max Amp: {stats['amp']:.1f}",
        f"BW (3dB): {_format_hz(stats['bw'])}",
    ]
    if stats.get("echo_delay") is not None:
        lines.append(f"LOS-Echo: {stats['echo_delay']} samp")
    return "\n".join(lines)


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

    data, step = _reduce_data(data)
    fs /= step

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = pg.mkQApp()

    if mode == "Signal":
        win = pg.plot(title=title)
        win.addLegend()
        win.plot(np.real(data), pen=pg.mkPen("b"), name="Real")
        win.plot(
            np.imag(data), pen=pg.mkPen("r", style=QtCore.Qt.DashLine), name="Imag"
        )
        win.setLabel("bottom", "Sample Index")
        win.setLabel("left", "Amplitude")
        win.showGrid(x=True, y=True)
    elif mode in ("Freq", "Freq Analysis"):
        spec = np.fft.fftshift(np.fft.fft(data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data), d=1 / fs))
        win = pg.plot(
            freqs,
            20 * np.log10(np.abs(spec) + 1e-9),
            pen="b",
            title=f"Spectrum: {title}",
        )
        win.setLabel("bottom", "Frequency [Hz]")
        win.setLabel("left", "Magnitude [dB]")
        win.showGrid(x=True, y=True)
    elif mode == "InstantFreq":
        phase = np.unwrap(np.angle(data))
        inst = np.diff(phase)
        fi = fs * inst / (2 * np.pi)
        t = np.arange(len(fi)) / fs
        win = pg.plot(t, fi, pen="b", title=f"Instantaneous Frequency: {title}")
        win.setLabel("bottom", "Time [s]")
        win.setLabel("left", "Frequency [Hz]")
        win.showGrid(x=True, y=True)
    elif mode == "Autocorr":
        ac = _autocorr_fft(data)
        lags = np.arange(-len(data) + 1, len(data))
        win = pg.plot(lags, np.abs(ac), pen="b", title=f"Autocorrelation: {title}")
        win.setLabel("bottom", "Lag")
        win.setLabel("left", "Magnitude")
        win.showGrid(x=True, y=True)
    elif mode == "Crosscorr":
        if ref_data is None or ref_data.size == 0:
            messagebox.showinfo("Info", "Crosscorrelation requires TX data.")
            return
        data, ref_data, step_r = _reduce_pair(data, ref_data)
        fs /= step_r
        n = min(len(data), len(ref_data))
        cc = _xcorr_fft(data[:n], ref_data[:n])
        lags = np.arange(-n + 1, n)
        mag = np.abs(cc)
        win = pg.plot(lags, mag, pen="b", title=f"Crosscorr. with TX: {title}")
        win.setLabel("bottom", "Lag")
        win.setLabel("left", "Magnitude")
        los_idx, echo_idx = _find_los_echo(cc)
        if los_idx is not None:
            win.plot(
                [lags[los_idx]], [mag[los_idx]], pen=None, symbol="o", symbolBrush="r"
            )
        if echo_idx is not None:
            win.plot(
                [lags[echo_idx]], [mag[echo_idx]], pen=None, symbol="o", symbolBrush="g"
            )
        win.showGrid(x=True, y=True)
    else:
        messagebox.showerror("Error", f"Unknown mode {mode}")
        return

    pg.exec()


def _plot_on_pg(
    plot: pg.PlotItem,
    data: np.ndarray,
    fs: float,
    mode: str,
    title: str,
    ref_data: np.ndarray | None = None,
) -> None:
    """Helper to draw the selected visualization on a PyQtGraph PlotItem."""
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
        data, ref_data, step_r = _reduce_pair(data, ref_data)
        fs /= step_r
        n = min(len(data), len(ref_data))
        cc = _xcorr_fft(data[:n], ref_data[:n])
        lags = np.arange(-n + 1, n)
        mag = np.abs(cc)
        plot.plot(lags, mag, pen="b")
        los_idx, echo_idx = _find_los_echo(cc)
        if los_idx is not None:
            plot.plot(
                [lags[los_idx]], [mag[los_idx]], pen=None, symbol="o", symbolBrush="r"
            )
        if echo_idx is not None:
            plot.plot(
                [lags[echo_idx]], [mag[echo_idx]], pen=None, symbol="o", symbolBrush="g"
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
        lags = np.arange(-n + 1, n)
        mag = np.abs(cc)
        ax.plot(lags, mag, "b")
        los_idx, echo_idx = _find_los_echo(cc)
        if los_idx is not None:
            ax.plot(lags[los_idx], mag[los_idx], "ro")
        if echo_idx is not None:
            ax.plot(lags[echo_idx], mag[echo_idx], "go")
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
        self._tx_running = False
        self._last_tx_end = 0.0
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

        ttk.Label(gen_frame, text="Oversampling").grid(row=5, column=0, sticky="w")
        self.os_entry = SuggestEntry(gen_frame, "os_entry")
        self.os_entry.insert(0, "1")
        self.os_entry.grid(row=5, column=1, sticky="ew")
        self.os_enable = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            gen_frame,
            variable=self.os_enable,
            command=lambda: self.os_entry.entry.configure(
                state="normal" if self.os_enable.get() else "disabled"
            ),
        ).grid(row=5, column=2, sticky="w")
        self.os_entry.entry.configure(state="disabled")

        ttk.Label(gen_frame, text="Repeats").grid(row=6, column=0, sticky="w")
        self.repeat_entry = SuggestEntry(gen_frame, "repeat_entry")
        self.repeat_entry.insert(0, "1")
        self.repeat_entry.grid(row=6, column=1, sticky="ew")

        self.rrc_beta_label = ttk.Label(gen_frame, text="RRC β")
        self.rrc_beta_label.grid(row=7, column=0, sticky="w")
        self.rrc_beta_entry = SuggestEntry(gen_frame, "rrc_beta_entry")
        self.rrc_beta_entry.insert(0, "0.25")
        self.rrc_beta_entry.grid(row=7, column=1, sticky="ew")

        self.rrc_span_label = ttk.Label(gen_frame, text="RRC Span")
        self.rrc_span_label.grid(row=8, column=0, sticky="w")
        self.rrc_span_entry = SuggestEntry(gen_frame, "rrc_span_entry")
        self.rrc_span_entry.insert(0, "6")
        self.rrc_span_entry.grid(row=8, column=1, sticky="ew")
        self.rrc_enable = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            gen_frame,
            variable=self.rrc_enable,
            command=lambda: [
                self.rrc_beta_entry.entry.configure(
                    state="normal" if self.rrc_enable.get() else "disabled"
                ),
                self.rrc_span_entry.entry.configure(
                    state="normal" if self.rrc_enable.get() else "disabled"
                ),
            ],
        ).grid(row=7, column=2, sticky="w", rowspan=2)
        if not self.rrc_enable.get():
            self.rrc_beta_entry.entry.configure(state="disabled")
            self.rrc_span_entry.entry.configure(state="disabled")

        ttk.Label(gen_frame, text="Zeros").grid(row=9, column=0, sticky="w")
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
        ).grid(row=9, column=1, sticky="ew")

        ttk.Label(gen_frame, text="Amplitude").grid(row=10, column=0, sticky="w")
        self.amp_entry = SuggestEntry(gen_frame, "amp_entry")
        self.amp_entry.insert(0, "10000")
        self.amp_entry.grid(row=10, column=1, sticky="ew")

        ttk.Label(gen_frame, text="File").grid(row=11, column=0, sticky="w")
        self.file_entry = SuggestEntry(gen_frame, "file_entry")
        self.file_entry.insert(0, "tx_signal.bin")
        self.file_entry.grid(row=11, column=1, sticky="ew")

        ttk.Button(gen_frame, text="Generate", command=self.generate).grid(
            row=12, column=0, columnspan=2, pady=5
        )

        scroll_container = ttk.Frame(gen_frame)
        scroll_container.grid(row=13, column=0, columnspan=2, sticky="nsew")
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

        self.rx_rrc_beta_label = ttk.Label(rx_frame, text="RRC β")
        self.rx_rrc_beta_label.grid(row=7, column=0, sticky="w")
        self.rx_rrc_beta_entry = SuggestEntry(rx_frame, "rx_rrc_beta")
        self.rx_rrc_beta_entry.insert(0, "0.25")
        self.rx_rrc_beta_entry.grid(row=7, column=1, sticky="ew")

        self.rx_rrc_span_label = ttk.Label(rx_frame, text="RRC Span")
        self.rx_rrc_span_label.grid(row=8, column=0, sticky="w")
        self.rx_rrc_span_entry = SuggestEntry(rx_frame, "rx_rrc_span")
        self.rx_rrc_span_entry.insert(0, "6")
        self.rx_rrc_span_entry.grid(row=8, column=1, sticky="ew")
        self.rx_rrc_enable = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            rx_frame,
            variable=self.rx_rrc_enable,
            command=lambda: [
                self.rx_rrc_beta_entry.entry.configure(
                    state="normal" if self.rx_rrc_enable.get() else "disabled"
                ),
                self.rx_rrc_span_entry.entry.configure(
                    state="normal" if self.rx_rrc_enable.get() else "disabled"
                ),
            ],
        ).grid(row=7, column=2, sticky="w", rowspan=2)
        if not self.rx_rrc_enable.get():
            self.rx_rrc_beta_entry.entry.configure(state="disabled")
            self.rx_rrc_span_entry.entry.configure(state="disabled")

        ttk.Label(rx_frame, text="Oversampling").grid(row=9, column=0, sticky="w")
        self.rx_os_entry = SuggestEntry(rx_frame, "rx_os_entry")
        self.rx_os_entry.insert(0, "1")
        self.rx_os_entry.grid(row=9, column=1, sticky="ew")
        self.rx_os_enable = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            rx_frame,
            variable=self.rx_os_enable,
            command=self._on_rx_os_change,
        ).grid(row=9, column=2, sticky="w")
        self.rx_os_entry.entry.configure(state="disabled")
        self.rx_os_entry.entry.bind(
            "<FocusOut>",
            lambda _e: (self.auto_update_rx_filename(), self.update_trim()),
        )
        ttk.Label(rx_frame, text="Output").grid(row=10, column=0, sticky="w")
        self.rx_file = SuggestEntry(rx_frame, "rx_file")
        self.rx_file.insert(0, "rx_signal.bin")
        self.rx_file.grid(row=10, column=1, sticky="ew")

        ttk.Label(rx_frame, text="View").grid(row=11, column=0, sticky="w")
        ttk.Combobox(
            rx_frame,
            textvariable=self.rx_view,
            values=["Signal", "Freq", "InstantFreq", "Crosscorr"],
            width=12,
        ).grid(row=11, column=1)

        # --- Trim controls -------------------------------------------------
        self.trim_var = tk.BooleanVar(value=False)
        self.trim_start = tk.DoubleVar(value=0.0)
        self.trim_end = tk.DoubleVar(value=100.0)
        self.trim_dirty = False

        trim_frame = ttk.Frame(rx_frame)
        trim_frame.grid(row=12, column=0, columnspan=2, sticky="ew")
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
        rx_btn_frame.grid(row=13, column=0, columnspan=2, pady=5)
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
        rx_scroll_container.grid(row=14, column=0, columnspan=2, sticky="nsew")
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
        rx_frame.rowconfigure(13, weight=1)
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
            self.f_label.grid(row=2, column=0, sticky="w")
            self.f_entry.grid(row=2, column=1, sticky="ew")
        elif w == "zadoffchu":
            self.q_label.grid(row=2, column=0, sticky="w")
            self.q_entry.grid(row=2, column=1, sticky="ew")
            self.rrc_beta_label.grid(row=7, column=0, sticky="w")
            self.rrc_beta_entry.grid(row=7, column=1, sticky="ew")
            self.rrc_span_label.grid(row=8, column=0, sticky="w")
            self.rrc_span_entry.grid(row=8, column=1, sticky="ew")
            state = "normal" if self.rrc_enable.get() else "disabled"
            self.rrc_beta_entry.entry.configure(state=state)
            self.rrc_span_entry.entry.configure(state=state)
        elif w == "chirp":
            self.f_label.configure(text="f0")
            self.f_label.grid(row=2, column=0, sticky="w")
            self.f_entry.grid(row=2, column=1, sticky="ew")
            self.f1_label.grid(row=3, column=0, sticky="w")
            self.f1_entry.grid(row=3, column=1, sticky="ew")

        self.auto_update_tx_filename()

    def auto_update_tx_filename(self) -> None:
        """Update TX filename entry based on current parameters."""
        name = _gen_tx_filename(self)
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, name)
        self.tx_file.delete(0, tk.END)
        self.tx_file.insert(0, name)

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

    def _display_gen_plots(self, data: np.ndarray, fs: float) -> None:
        """Render preview plots below the generation parameters."""
        self.latest_data = data
        self.latest_fs = fs

        for c in self.gen_canvases:
            c.get_tk_widget().destroy()
        self.gen_canvases.clear()

        modes = ["Signal", "Freq", "InstantFreq", "Autocorr"]
        for idx, mode in enumerate(modes):
            fig = Figure(figsize=(5, 2), dpi=100)
            ax = fig.add_subplot(111)
            _plot_on_mpl(ax, data, fs, mode, f"TX {mode}")
            canvas = FigureCanvasTkAgg(fig, master=self.gen_plots_frame)
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

        stats = _calc_stats(data, fs)
        text = _format_stats_text(stats)
        if not hasattr(self, "gen_stats_label"):
            self.gen_stats_label = ttk.Label(
                self.gen_plots_frame, justify="left", anchor="w"
            )
        self.gen_stats_label.grid(row=len(modes), column=0, sticky="ew", pady=2)
        self.gen_stats_label.configure(text=text)

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

    def _display_rx_plots(self, data: np.ndarray, fs: float) -> None:
        """Render preview plots below the receive parameters."""
        self.raw_rx_data = data
        self.latest_fs = fs
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
        data, fac = self._oversample_data(data)
        fs *= fac
        self.latest_data = data

        try:
            raw = np.fromfile(self.tx_file.get(), dtype=np.int16)
            if raw.size % 2:
                raw = raw[:-1]
            raw = raw.reshape(-1, 2).astype(np.float32)
            self.tx_data = raw[:, 0] + 1j * raw[:, 1]
        except Exception:
            self.tx_data = np.array([], dtype=np.complex64)

        for c in self.rx_canvases:
            c.get_tk_widget().destroy()
        self.rx_canvases.clear()

        modes = ["Signal", "Freq", "InstantFreq", "Crosscorr"]
        title_suffix = f" ({channel_label})" if channel_label else ""
        for idx, mode in enumerate(modes):
            fig = Figure(figsize=(5, 2), dpi=100)
            ax = fig.add_subplot(111)
            _plot_on_mpl(ax, data, fs, mode, f"RX {mode}{title_suffix}", self.tx_data)
            canvas = FigureCanvasTkAgg(fig, master=self.rx_plots_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.grid(row=idx, column=0, sticky="nsew", pady=2)
            widget.bind(
                "<Button-1>",
                lambda _e, m=mode, d=data, s=fs: self._show_fullscreen(
                    d, s, m, f"RX {m}{title_suffix}"
                ),
            )
            self.rx_canvases.append(canvas)

        stats = _calc_stats(data, fs, self.tx_data)
        text = _format_stats_text(stats)
        if not hasattr(self, "rx_stats_label"):
            self.rx_stats_label = ttk.Label(
                self.rx_plots_frame, justify="left", anchor="w"
            )
        self.rx_stats_label.grid(row=len(modes), column=0, sticky="ew", pady=2)
        self.rx_stats_label.configure(text=text)

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

    def _oversample_data(self, data: np.ndarray) -> tuple[np.ndarray, float]:
        """Return oversampled *data* and new sample rate factor."""
        if not self.rx_os_enable.get() or data.size == 0:
            return data, 1.0
        try:
            factor = int(self.rx_os_entry.get())
        except Exception:
            factor = 1
        if factor <= 1:
            return data, 1.0
        x_old = np.arange(len(data))
        x_new = np.linspace(0, len(data) - 1, len(data) * factor)
        real = np.interp(x_new, x_old, data.real)
        imag = np.interp(x_new, x_old, data.imag)
        return real + 1j * imag, float(factor)

    def _on_trim_change(self, *_args) -> None:
        state = "normal" if self.trim_var.get() else "disabled"
        self.range_slider.configure_state(state)
        try:
            self.trim_start_label.configure(text=f"{self.trim_start.get():.0f}%")
            self.trim_end_label.configure(text=f"{self.trim_end.get():.0f}%")
        except Exception:
            pass
        self.trim_dirty = True
        self.apply_trim_btn.configure(state="normal")

    def _on_rx_os_change(self, *_args) -> None:
        self.rx_os_entry.entry.configure(
            state="normal" if self.rx_os_enable.get() else "disabled"
        )
        self.update_trim()

    def update_trim(self, *_args) -> None:
        """Re-apply trimming and refresh RX plots."""
        self._on_trim_change()
        self.apply_trim_btn.configure(state="disabled")
        self.trim_dirty = False
        if hasattr(self, "raw_rx_data") and self.raw_rx_data is not None:
            self._display_rx_plots(self.raw_rx_data, self.latest_fs)
        if (
            hasattr(self, "latest_data")
            and self.latest_data is not None
            and (
                self.trim_var.get()
                or (
                    self.rx_os_enable.get()
                    and int(self.rx_os_entry.get() or 1) > 1
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
                if output_lines and any("No devices found" in l for l in output_lines):
                    self._ping_device(self.tx_args.get())

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
            if hasattr(self, "tx_stop"):
                self.tx_stop.config(state="disabled")
            if hasattr(self, "tx_button"):
                self.tx_button.config(state="normal")
            if hasattr(self, "tx_retrans"):
                self.tx_retrans.config(state="disabled")

    def _run_rx_cmd(self, cmd: list[str], out_file: str) -> None:
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
            if hasattr(self, "rx_stop"):
                self.rx_stop.config(state="disabled")
            if hasattr(self, "rx_button"):
                self.rx_button.config(state="normal")

        if proc is not None and proc.returncode == 0:
            try:
                channels = 2 if self.rx_channel_2.get() else 1
                path = Path(out_file)
                try:
                    data = rx_convert.load_iq_file(
                        path, channels=channels, layout="blocked"
                    )
                except ValueError:
                    data = rx_convert.load_iq_file(
                        path, channels=channels, layout="interleaved"
                    )
                self._display_rx_plots(data, float(eval(self.rx_rate.get())))
            except Exception as exc:
                self._out_queue.put(f"Error: {exc}\n")

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
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        if self._plot_win is not None and self._plot_win.isVisible():
            try:
                self._plot_win.close()
            except Exception:
                pass
            self._plot_win = None

        app = pg.mkQApp()
        win = pg.plot()
        self._plot_win = win
        _plot_on_pg(
            win.getPlotItem(),
            data,
            fs,
            mode,
            title,
            ref_data if ref_data is not None else getattr(self, "tx_data", None),
        )
        try:
            win.showMaximized()
        except Exception:
            pass
        try:
            win.raise_()
            win.activateWindow()
        except Exception:
            pass
        pg.exec()
        self._plot_win = None

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
        if not hasattr(self, "tx_data") or self.tx_data.size == 0:
            messagebox.showerror("XCorr Full", "No TX data available")
            return
        data = self.latest_data
        ref = self.tx_data
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
            "oversampling": self.os_entry.get(),
            "os_enabled": self.os_enable.get(),
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
            "rx_oversampling": self.rx_os_entry.get(),
            "rx_os_enabled": self.rx_os_enable.get(),
            "rx_rrc_beta": self.rx_rrc_beta_entry.get(),
            "rx_rrc_span": self.rx_rrc_span_entry.get(),
            "rx_rrc_enabled": self.rx_rrc_enable.get(),
            "rx_channel_2": self.rx_channel_2.get(),
            "rx_channel_view": self.rx_channel_view.get(),
            "rx_file": self.rx_file.get(),
            "rx_view": self.rx_view.get(),
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
        self.os_entry.insert(0, params.get("oversampling", "1"))
        self.os_enable.set(params.get("os_enabled", False))
        self.os_entry.entry.configure(
            state="normal" if self.os_enable.get() else "disabled"
        )
        self.repeat_entry.delete(0, tk.END)
        self.repeat_entry.insert(0, params.get("repeats", "1"))
        self.rrc_beta_entry.delete(0, tk.END)
        self.rrc_beta_entry.insert(0, params.get("rrc_beta", "0.25"))
        self.rrc_span_entry.delete(0, tk.END)
        self.rrc_span_entry.insert(0, params.get("rrc_span", "6"))
        self.rrc_enable.set(params.get("rrc_enabled", True))
        state = "normal" if self.rrc_enable.get() else "disabled"
        self.rrc_beta_entry.entry.configure(state=state)
        self.rrc_span_entry.entry.configure(state=state)
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
        self.rx_os_entry.delete(0, tk.END)
        self.rx_os_entry.insert(0, params.get("rx_oversampling", "1"))
        self.rx_os_enable.set(params.get("rx_os_enabled", False))
        self.rx_os_entry.entry.configure(
            state="normal" if self.rx_os_enable.get() else "disabled"
        )
        self.rx_rrc_beta_entry.delete(0, tk.END)
        self.rx_rrc_beta_entry.insert(0, params.get("rx_rrc_beta", "0.25"))
        self.rx_rrc_span_entry.delete(0, tk.END)
        self.rx_rrc_span_entry.insert(0, params.get("rx_rrc_span", "6"))
        self.rx_rrc_enable.set(params.get("rx_rrc_enabled", True))
        state = "normal" if self.rx_rrc_enable.get() else "disabled"
        self.rx_rrc_beta_entry.entry.configure(state=state)
        self.rx_rrc_span_entry.entry.configure(state=state)
        self.rx_channel_2.set(params.get("rx_channel_2", False))
        self.rx_channel_view.set(params.get("rx_channel_view", "Kanal 1"))
        self.rx_file.delete(0, tk.END)
        self.rx_file.insert(0, params.get("rx_file", ""))
        self.rx_view.set(params.get("rx_view", "Signal"))
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
            fs = float(eval(self.fs_entry.get()))
            samples = int(self.samples_entry.get())
            oversampling = int(self.os_entry.get()) if self.os_entry.get() else 1
            if not self.os_enable.get():
                oversampling = 1
            repeats = int(self.repeat_entry.get()) if self.repeat_entry.get() else 1
            zeros_mode = self.zeros_var.get()
            amp = float(self.amp_entry.get())
            waveform = self.wave_var.get()

            if waveform == "sinus":
                freq = float(eval(self.f_entry.get())) if self.f_entry.get() else 0.0
                data = generate_waveform(
                    waveform, fs, freq, samples, oversampling=oversampling
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

                if oversampling > 1:
                    if samples % oversampling != 0:
                        messagebox.showerror(
                            "Generate error",
                            "Samples muss durch Oversampling teilbar sein",
                        )
                        return
                    base_samples = samples // oversampling
                else:
                    base_samples = samples

                data = generate_waveform(
                    waveform,
                    fs,
                    0.0,
                    base_samples,
                    q=q,
                    rrc_beta=beta,
                    rrc_span=span,
                    oversampling=oversampling,
                )
            else:  # chirp
                f0 = float(eval(self.f_entry.get())) if self.f_entry.get() else 0.0
                f1 = float(eval(self.f1_entry.get())) if self.f1_entry.get() else None
                data = generate_waveform(
                    waveform,
                    fs,
                    f0,
                    samples,
                    f0=f0,
                    f1=f1,
                    oversampling=oversampling,
                )

            if repeats > 1:
                data = np.tile(data, repeats)

            zeros = 0
            if zeros_mode == "same":
                zeros = len(data)
            elif zeros_mode == "half":
                zeros = len(data) // 2
            elif zeros_mode == "quarter":
                zeros = len(data) // 4
            elif zeros_mode == "double":
                zeros = len(data) * 2
            elif zeros_mode == "quadruple":
                zeros = len(data) * 4
            elif zeros_mode == "octuple":
                zeros = len(data) * 8

            if zeros:
                data = np.concatenate([data, np.zeros(zeros, dtype=np.complex64)])

            save_interleaved(self.file_entry.get(), data, amplitude=amp)

            max_abs = np.max(np.abs(data)) if np.any(data) else 1.0
            scale = amp / max_abs if max_abs > 1e-9 else 1.0
            scaled_data = data * scale
            self._display_gen_plots(scaled_data, fs)
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
        cmd = [
            REPLAY_BIN,
            "--args",
            self.tx_args.get(),
            "--rate",
            self.tx_rate.get(),
            "--freq",
            self.tx_freq.get(),
            "--gain",
            self.tx_gain.get(),
            "--nsamps",
            "0",
            "--file",
            self.tx_file.get(),
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
            kwargs={"max_attempts": 10, "delay": 2.0},
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
        if self.rx_rrc_enable.get():
            cmd += [
                "--rrc",
                "--rrc-beta",
                self.rx_rrc_beta_entry.get(),
                "--rrc-span",
                self.rx_rrc_span_entry.get(),
            ]
        if self.rx_channel_2.get():
            cmd += ["--channels", "0", "1"]
        self._cmd_running = True
        if hasattr(self, "rx_stop"):
            self.rx_stop.config(state="normal")
        if hasattr(self, "rx_button"):
            self.rx_button.config(state="disabled")
        threading.Thread(
            target=self._run_rx_cmd, args=(cmd, out_file), daemon=True
        ).start()
        self._process_queue()

    def on_close(self) -> None:
        self.stop_transmit()
        self.stop_receive()
        _save_state(self._get_current_params())
        self.destroy()


def main() -> None:
    app = TransceiverUI()
    app.mainloop()


if __name__ == "__main__":
    main()
