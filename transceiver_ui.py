#!/usr/bin/env python3
"""Simple GUI to generate, transmit and receive signals."""
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import ttk, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

from tx_generator import generate_waveform

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

    def __init__(self, parent, name: str, width: int | None = None,
                 textvariable: tk.StringVar | None = None) -> None:
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
            rm = tk.Button(frame, text="x", command=lambda v=val: self._remove_suggestion(v), width=2)
            rm.pack(side="right")
            frame.pack(side="left", padx=2, pady=1)
            frame.bind("<Button-1>", lambda _e, v=val: self._fill_entry(v))
            lbl.bind("<Button-1>", lambda _e, v=val: self._fill_entry(v))


def save_interleaved(filename: str, data: np.ndarray, amplitude: float = 10000.0) -> None:
    """Save complex64 data as interleaved int16."""
    max_abs = np.max(np.abs(data)) if np.any(data) else 1.0
    scale = amplitude / max_abs if max_abs > 1e-9 else 1.0
    scaled = data * scale
    real_i16 = np.int16(np.round(np.real(scaled)))
    imag_i16 = np.int16(np.round(np.imag(scaled)))
    interleaved = np.empty(real_i16.size + imag_i16.size, dtype=np.int16)
    interleaved[0::2] = np.clip(real_i16, -32768, 32767)
    interleaved[1::2] = np.clip(imag_i16, -32768, 32767)
    interleaved.tofile(filename)


def _reduce_data(data: np.ndarray, max_bytes: int = 1_000_000) -> np.ndarray:
    """Return a downsampled view of *data* if it exceeds *max_bytes*."""
    if data.nbytes <= max_bytes:
        return data
    step = int(np.ceil(data.nbytes / max_bytes))
    return data[::step]


def visualize(data: np.ndarray, fs: float, mode: str, title: str) -> None:
    """Visualize *data* using PyQtGraph."""
    if data.size == 0:
        messagebox.showerror("Error", "No data to visualize")
        return

    data = _reduce_data(data)

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = pg.mkQApp()

    if mode == "Signal":
        win = pg.plot(title=title)
        win.addLegend()
        win.plot(np.real(data), pen=pg.mkPen("b"), name="Real")
        win.plot(np.imag(data), pen=pg.mkPen("r", style=QtCore.Qt.DashLine), name="Imag")
        win.setLabel("bottom", "Sample Index")
        win.setLabel("left", "Amplitude")
        win.showGrid(x=True, y=True)
    elif mode in ("Freq", "Freq Analysis"):
        spec = np.fft.fftshift(np.fft.fft(data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data), d=1/fs))
        win = pg.plot(freqs, 20*np.log10(np.abs(spec) + 1e-9), pen="b", title=f"Spectrum: {title}")
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
        ac = np.correlate(data, data, mode="full")
        lags = np.arange(-len(data) + 1, len(data))
        win = pg.plot(lags, np.abs(ac), pen="b", title=f"Autocorrelation: {title}")
        win.setLabel("bottom", "Lag")
        win.setLabel("left", "Magnitude")
        win.showGrid(x=True, y=True)
    elif mode == "Crosscorr":
        messagebox.showinfo("Info", "Crosscorrelation requires two files.")
        return
    else:
        messagebox.showerror("Error", f"Unknown mode {mode}")
        return

    pg.exec()


def _plot_on_pg(plot: pg.PlotItem, data: np.ndarray, fs: float, mode: str, title: str) -> None:
    """Helper to draw the selected visualization on a PyQtGraph PlotItem."""
    data = _reduce_data(data)
    if mode == "Signal":
        plot.addLegend()
        plot.plot(np.real(data), pen=pg.mkPen("b"), name="Real")
        plot.plot(np.imag(data), pen=pg.mkPen("r", style=QtCore.Qt.DashLine), name="Imag")
        plot.setTitle(title)
        plot.setLabel("bottom", "Sample Index")
        plot.setLabel("left", "Amplitude")
    elif mode in ("Freq", "Freq Analysis"):
        spec = np.fft.fftshift(np.fft.fft(data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data), d=1/fs))
        plot.plot(freqs, 20*np.log10(np.abs(spec) + 1e-9), pen="b")
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
        ac = np.correlate(data, data, mode="full")
        lags = np.arange(-len(data) + 1, len(data))
        plot.plot(lags, np.abs(ac), pen="b")
        plot.setTitle(f"Autocorrelation: {title}")
        plot.setLabel("bottom", "Lag")
        plot.setLabel("left", "Magnitude")
    plot.showGrid(x=True, y=True)


class TransceiverUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Signal Transceiver")
        # define view variables early so callbacks won't fail
        self.rx_view = tk.StringVar(value="Signal")
        self.rate_var = tk.StringVar(value="200e6")
        self.console = None
        self._out_queue = queue.Queue()
        self._cmd_running = False
        self._proc = None
        self.create_widgets()

    def create_widgets(self):
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        # ----- Column 1: Generation -----
        gen_frame = ttk.LabelFrame(self, text="Signal Generation")
        gen_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

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
        wave_box.bind("<<ComboboxSelected>>", lambda _e: self.update_waveform_fields())

        ttk.Label(gen_frame, text="fs").grid(row=1, column=0, sticky="w")
        self.fs_entry = SuggestEntry(gen_frame, "fs_entry",
                                     textvariable=self.rate_var)
        self.fs_entry.grid(row=1, column=1, sticky="ew")

        self.f_label = ttk.Label(gen_frame, text="f")
        self.f_label.grid(row=2, column=0, sticky="w")
        self.f_entry = SuggestEntry(gen_frame, "f_entry")
        self.f_entry.insert(0, "1e6")
        self.f_entry.grid(row=2, column=1, sticky="ew")

        self.f1_label = ttk.Label(gen_frame, text="f1")
        self.f1_entry = SuggestEntry(gen_frame, "f1_entry")
        self.f1_label.grid(row=3, column=0, sticky="w")
        self.f1_entry.grid(row=3, column=1, sticky="ew")

        self.q_label = ttk.Label(gen_frame, text="q")
        self.q_entry = SuggestEntry(gen_frame, "q_entry")
        self.q_entry.insert(0, "1")
        # row placement will be adjusted in update_waveform_fields
        self.q_label.grid(row=2, column=0, sticky="w")
        self.q_entry.grid(row=2, column=1, sticky="ew")

        ttk.Label(gen_frame, text="Samples").grid(row=4, column=0, sticky="w")
        self.samples_entry = SuggestEntry(gen_frame, "samples_entry")
        self.samples_entry.insert(0, "40000")
        self.samples_entry.grid(row=4, column=1, sticky="ew")

        ttk.Label(gen_frame, text="Amplitude").grid(row=5, column=0, sticky="w")
        self.amp_entry = SuggestEntry(gen_frame, "amp_entry")
        self.amp_entry.insert(0, "10000")
        self.amp_entry.grid(row=5, column=1, sticky="ew")

        ttk.Label(gen_frame, text="File").grid(row=6, column=0, sticky="w")
        self.file_entry = SuggestEntry(gen_frame, "file_entry")
        self.file_entry.insert(0, "tx_signal.bin")
        self.file_entry.grid(row=6, column=1, sticky="ew")

        ttk.Button(gen_frame, text="Generate", command=self.generate).grid(row=7, column=0, columnspan=2, pady=5)

        scroll_container = ttk.Frame(gen_frame)
        scroll_container.grid(row=8, column=0, columnspan=2, sticky="nsew")
        scroll_container.columnconfigure(0, weight=1)
        scroll_container.rowconfigure(0, weight=1)

        self.gen_canvas = tk.Canvas(scroll_container)
        self.gen_canvas.grid(row=0, column=0, sticky="nsew")
        self.gen_scroll = ttk.Scrollbar(scroll_container, orient="vertical", command=self.gen_canvas.yview)
        self.gen_scroll.grid(row=0, column=1, sticky="ns")
        self.gen_canvas.configure(yscrollcommand=self.gen_scroll.set)

        # enable mouse wheel scrolling
        self.gen_canvas.bind("<Enter>", self._bind_gen_mousewheel)
        self.gen_canvas.bind("<Leave>", self._unbind_gen_mousewheel)

        self.gen_plots_frame = ttk.Frame(self.gen_canvas)
        self.gen_canvas.create_window((0, 0), window=self.gen_plots_frame, anchor="nw")
        self.gen_plots_frame.bind(
            "<Configure>",
            lambda _e: self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox("all")),
        )
        gen_frame.rowconfigure(8, weight=1)
        self.gen_canvases = []
        self.latest_data = None
        self.latest_fs = 0.0

        self.update_waveform_fields()

        # ----- Presets -----
        preset_frame = ttk.LabelFrame(self, text="Presets")
        preset_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        preset_frame.columnconfigure(0, weight=1)
        self.preset_var = tk.StringVar(value="")
        self.preset_box = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_var,
            values=sorted(_PRESETS.keys()),
            state="readonly",
            width=20,
        )
        self.preset_box.grid(row=0, column=0, padx=5)
        ttk.Button(preset_frame, text="Load", command=self.load_preset).grid(row=0, column=1, padx=5)
        ttk.Button(preset_frame, text="Save", command=self.save_preset).grid(row=0, column=2, padx=5)
        ttk.Button(preset_frame, text="Delete", command=self.delete_preset).grid(row=0, column=3, padx=5)

        # ----- Column 2: Transmit -----
        tx_frame = ttk.LabelFrame(self, text="Transmit")
        tx_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        ttk.Label(tx_frame, text="Args").grid(row=0, column=0, sticky="w")
        self.tx_args = SuggestEntry(tx_frame, "tx_args")
        self.tx_args.insert(0, "addr=192.168.10.2")
        self.tx_args.grid(row=0, column=1, sticky="ew")

        ttk.Label(tx_frame, text="Rate").grid(row=1, column=0, sticky="w")
        self.tx_rate = SuggestEntry(tx_frame, "tx_rate",
                                   textvariable=self.rate_var)
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

        ttk.Button(tx_frame, text="Transmit", command=self.transmit).grid(row=5, column=0, columnspan=2, pady=5)
        self.tx_stop = ttk.Button(tx_frame, text="Stop", command=self.stop_transmit, state="disabled")
        self.tx_stop.grid(row=6, column=0, columnspan=2, pady=(0, 5))

        log_frame = ttk.Frame(tx_frame)
        log_frame.grid(row=7, column=0, columnspan=2, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.tx_log = tk.Text(log_frame, height=10, wrap="none")
        self.tx_log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.tx_log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.tx_log.configure(yscrollcommand=log_scroll.set)
        tx_frame.rowconfigure(7, weight=1)

        # ----- Column 3: Receive -----
        rx_frame = ttk.LabelFrame(self, text="Receive")
        rx_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        ttk.Label(rx_frame, text="Args").grid(row=0, column=0, sticky="w")
        self.rx_args = SuggestEntry(rx_frame, "rx_args")
        self.rx_args.insert(0, "addr=192.168.20.2,clock_source=external")
        self.rx_args.grid(row=0, column=1, sticky="ew")

        ttk.Label(rx_frame, text="Rate").grid(row=1, column=0, sticky="w")
        self.rx_rate = SuggestEntry(rx_frame, "rx_rate",
                                   textvariable=self.rate_var)
        self.rx_rate.grid(row=1, column=1, sticky="ew")

        ttk.Label(rx_frame, text="Freq").grid(row=2, column=0, sticky="w")
        self.rx_freq = SuggestEntry(rx_frame, "rx_freq")
        self.rx_freq.insert(0, "5.18e9")
        self.rx_freq.grid(row=2, column=1, sticky="ew")

        ttk.Label(rx_frame, text="Duration").grid(row=3, column=0, sticky="w")
        self.rx_dur = SuggestEntry(rx_frame, "rx_dur")
        self.rx_dur.insert(0, "0.01")
        self.rx_dur.grid(row=3, column=1, sticky="ew")

        ttk.Label(rx_frame, text="Gain").grid(row=4, column=0, sticky="w")
        self.rx_gain = SuggestEntry(rx_frame, "rx_gain")
        self.rx_gain.insert(0, "80")
        self.rx_gain.grid(row=4, column=1, sticky="ew")

        ttk.Label(rx_frame, text="Output").grid(row=5, column=0, sticky="w")
        self.rx_file = SuggestEntry(rx_frame, "rx_file")
        self.rx_file.insert(0, "rx_signal.bin")
        self.rx_file.grid(row=5, column=1, sticky="ew")

        ttk.Label(rx_frame, text="View").grid(row=6, column=0, sticky="w")
        ttk.Combobox(rx_frame, textvariable=self.rx_view,
                     values=["Signal", "Freq", "InstantFreq", "Autocorr", "Crosscorr"], width=12).grid(row=6, column=1)

        ttk.Button(rx_frame, text="Receive", command=self.receive).grid(row=7, column=0, columnspan=2, pady=5)

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

        if w == "sinus":
            self.f_label.configure(text="f")
            self.f_label.grid(row=2, column=0, sticky="w")
            self.f_entry.grid(row=2, column=1, sticky="ew")
        elif w == "zadoffchu":
            self.q_label.grid(row=2, column=0, sticky="w")
            self.q_entry.grid(row=2, column=1, sticky="ew")
        elif w == "chirp":
            self.f_label.configure(text="f0")
            self.f_label.grid(row=2, column=0, sticky="w")
            self.f_entry.grid(row=2, column=1, sticky="ew")
            self.f1_label.grid(row=3, column=0, sticky="w")
            self.f1_entry.grid(row=3, column=1, sticky="ew")


    def _display_gen_plots(self, data: np.ndarray, fs: float) -> None:
        """Open a PyQtGraph window with different visualizations."""
        self.latest_data = data
        self.latest_fs = fs

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        app = pg.mkQApp()
        win = pg.GraphicsLayoutWidget(title="Generated Signal")

        modes = ["Signal", "Freq", "InstantFreq", "Autocorr"]
        for idx, mode in enumerate(modes):
            plot = win.addPlot(row=idx, col=0)
            _plot_on_pg(plot, data, fs, mode, mode)

        win.show()
        pg.exec()

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

    def _run_cmd(self, cmd: list[str]) -> None:
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
        finally:
            self._cmd_running = False
            self._proc = None
            if hasattr(self, "tx_stop"):
                self.tx_stop.config(state="disabled")

    def _show_fullscreen(self, mode: str) -> None:
        if self.latest_data is None:
            return
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        app = pg.mkQApp()
        win = pg.plot()
        _plot_on_pg(win.getPlotItem(), self.latest_data, self.latest_fs, mode, mode)
        try:
            win.showMaximized()
        except Exception:
            pass
        pg.exec()

    # ----- Preset handling --------------------------------------------------
    def _get_current_params(self) -> dict:
        return {
            "waveform": self.wave_var.get(),
            "fs": self.fs_entry.get(),
            "f": self.f_entry.get(),
            "f1": self.f1_entry.get(),
            "q": self.q_entry.get(),
            "samples": self.samples_entry.get(),
            "amplitude": self.amp_entry.get(),
            "file": self.file_entry.get(),
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
            "rx_file": self.rx_file.get(),
            "rx_view": self.rx_view.get(),
        }

    def load_preset(self) -> None:
        name = self.preset_var.get()
        preset = _PRESETS.get(name)
        if not preset:
            messagebox.showerror("Preset", "No preset selected")
            return
        self.wave_var.set(preset.get("waveform", "sinus"))
        self.update_waveform_fields()
        self.fs_entry.delete(0, tk.END)
        self.fs_entry.insert(0, preset.get("fs", ""))
        self.f_entry.delete(0, tk.END)
        self.f_entry.insert(0, preset.get("f", ""))
        self.f1_entry.delete(0, tk.END)
        self.f1_entry.insert(0, preset.get("f1", ""))
        self.q_entry.delete(0, tk.END)
        self.q_entry.insert(0, preset.get("q", ""))
        self.samples_entry.delete(0, tk.END)
        self.samples_entry.insert(0, preset.get("samples", ""))
        self.amp_entry.delete(0, tk.END)
        self.amp_entry.insert(0, preset.get("amplitude", ""))
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, preset.get("file", ""))
        self.tx_args.delete(0, tk.END)
        self.tx_args.insert(0, preset.get("tx_args", ""))
        self.tx_rate.delete(0, tk.END)
        self.tx_rate.insert(0, preset.get("tx_rate", ""))
        self.tx_freq.delete(0, tk.END)
        self.tx_freq.insert(0, preset.get("tx_freq", ""))
        self.tx_gain.delete(0, tk.END)
        self.tx_gain.insert(0, preset.get("tx_gain", ""))
        self.tx_file.delete(0, tk.END)
        self.tx_file.insert(0, preset.get("tx_file", ""))
        self.rx_args.delete(0, tk.END)
        self.rx_args.insert(0, preset.get("rx_args", ""))
        self.rx_rate.delete(0, tk.END)
        self.rx_rate.insert(0, preset.get("rx_rate", ""))
        self.rx_freq.delete(0, tk.END)
        self.rx_freq.insert(0, preset.get("rx_freq", ""))
        self.rx_dur.delete(0, tk.END)
        self.rx_dur.insert(0, preset.get("rx_dur", ""))
        self.rx_gain.delete(0, tk.END)
        self.rx_gain.insert(0, preset.get("rx_gain", ""))
        self.rx_file.delete(0, tk.END)
        self.rx_file.insert(0, preset.get("rx_file", ""))
        self.rx_view.set(preset.get("rx_view", "Signal"))

    def save_preset(self) -> None:
        name = simpledialog.askstring("Save Preset", "Preset name:")
        if not name:
            return
        _PRESETS[name] = self._get_current_params()
        _save_presets(_PRESETS)
        self.preset_box["values"] = sorted(_PRESETS.keys())
        self.preset_var.set(name)

    def delete_preset(self) -> None:
        name = self.preset_var.get()
        if not name:
            return
        if not messagebox.askyesno("Delete Preset", f"Delete preset '{name}'?"):
            return
        _PRESETS.pop(name, None)
        _save_presets(_PRESETS)
        self.preset_box["values"] = sorted(_PRESETS.keys())
        self.preset_var.set("")


    # ----- Actions -----
    def generate(self):
        try:
            fs = float(eval(self.fs_entry.get()))
            samples = int(self.samples_entry.get())
            amp = float(self.amp_entry.get())
            waveform = self.wave_var.get()

            if waveform == "sinus":
                freq = float(eval(self.f_entry.get())) if self.f_entry.get() else 0.0
                data = generate_waveform(waveform, fs, freq, samples)
            elif waveform == "zadoffchu":
                q = int(self.q_entry.get()) if self.q_entry.get() else 1
                data = generate_waveform(waveform, fs, 0.0, samples, q=q)
            else:  # chirp
                f0 = float(eval(self.f_entry.get())) if self.f_entry.get() else 0.0
                f1 = float(eval(self.f1_entry.get())) if self.f1_entry.get() else None
                data = generate_waveform(waveform, fs, f0, samples, f0=f0, f1=f1)

            save_interleaved(self.file_entry.get(), data, amplitude=amp)
            self._display_gen_plots(data, fs)
        except Exception as exc:
            messagebox.showerror("Generate error", str(exc))

    def transmit(self):
        cmd = ["./rfnoc_replay_samples_from_file",
               "--args", self.tx_args.get(),
               "--rate", self.tx_rate.get(),
               "--freq", self.tx_freq.get(),
               "--gain", self.tx_gain.get(),
               "--nsamps", "0",
               "--file", self.tx_file.get()]
        if hasattr(self, "tx_log"):
            self.tx_log.delete("1.0", tk.END)
        self._cmd_running = True
        if hasattr(self, "tx_stop"):
            self.tx_stop.config(state="normal")
        threading.Thread(target=self._run_cmd, args=(cmd,), daemon=True).start()
        self._process_queue()

    def stop_transmit(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if hasattr(self, "tx_stop"):
            self.tx_stop.config(state="disabled")

    def receive(self):
        out_file = self.rx_file.get()
        cmd = ["./rx_to_file.py",
               "-a", self.rx_args.get(),
               "-f", self.rx_freq.get(),
               "-r", self.rx_rate.get(),
               "-d", self.rx_dur.get(),
               "-g", self.rx_gain.get(),
               "--dram",
               "--output-file", out_file]
        try:
            subprocess.run(cmd, check=True)
        except Exception as exc:
            messagebox.showerror("Receive error", str(exc))
            return
        try:
            subprocess.run(["./rx_convert.py", out_file], check=True)
            conv_file = out_file.replace(".bin", "_conv.bin")
            data = np.fromfile(conv_file, dtype=np.complex64)
            visualize(data, float(eval(self.rx_rate.get())), self.rx_view.get(), "Received")
        except Exception as exc:
            messagebox.showerror("Visualization error", str(exc))


def main() -> None:
    app = TransceiverUI()
    app.mainloop()


if __name__ == "__main__":
    main()
