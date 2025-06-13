#!/usr/bin/env python3
"""Simple GUI to generate, transmit and receive signals."""
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tx_generator import generate_waveform

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


def visualize(data: np.ndarray, fs: float, mode: str, title: str) -> None:
    """Visualize the data using matplotlib."""
    if data.size == 0:
        messagebox.showerror("Error", "No data to visualize")
        return

    if mode == "Signal":
        plt.figure()
        plt.plot(np.real(data), label="Real")
        plt.plot(np.imag(data), label="Imag")
        plt.title(title)
        plt.grid(True)
        plt.legend()
    elif mode in ("Freq", "Freq Analysis"):
        spec = np.fft.fftshift(np.fft.fft(data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data), d=1/fs))
        plt.figure()
        plt.plot(freqs, 20*np.log10(np.abs(spec)+1e-9))
        plt.title(f"Spectrum: {title}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.grid(True)
    elif mode == "InstantFreq":
        phase = np.unwrap(np.angle(data))
        inst = np.diff(phase)
        fi = fs * inst / (2*np.pi)
        t = np.arange(len(fi))/fs
        plt.figure()
        plt.plot(t, fi)
        plt.title(f"Instantaneous Frequency: {title}")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.grid(True)
    elif mode == "Autocorr":
        ac = np.correlate(data, data, mode="full")
        lags = np.arange(-len(data)+1, len(data))
        plt.figure()
        plt.plot(lags, np.abs(ac))
        plt.title(f"Autocorrelation: {title}")
        plt.xlabel("Lag")
        plt.ylabel("Magnitude")
        plt.grid(True)
    elif mode == "Crosscorr":
        messagebox.showinfo("Info", "Crosscorrelation requires two files.")
        return
    else:
        messagebox.showerror("Error", f"Unknown mode {mode}")
        return
    plt.tight_layout()
    plt.show()


def _plot_on_axes(ax, data: np.ndarray, fs: float, mode: str, title: str) -> None:
    """Helper to draw the selected visualization on *ax*."""
    if mode == "Signal":
        ax.plot(np.real(data), label="Real")
        ax.plot(np.imag(data), label="Imag")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    elif mode in ("Freq", "Freq Analysis"):
        spec = np.fft.fftshift(np.fft.fft(data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data), d=1/fs))
        ax.plot(freqs, 20*np.log10(np.abs(spec) + 1e-9))
        ax.set_title(f"Spectrum: {title}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.grid(True)
    elif mode == "InstantFreq":
        phase = np.unwrap(np.angle(data))
        inst = np.diff(phase)
        fi = fs * inst / (2*np.pi)
        t = np.arange(len(fi)) / fs
        ax.plot(t, fi)
        ax.set_title(f"Instantaneous Frequency: {title}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.grid(True)
    elif mode == "Autocorr":
        ac = np.correlate(data, data, mode="full")
        lags = np.arange(-len(data) + 1, len(data))
        ax.plot(lags, np.abs(ac))
        ax.set_title(f"Autocorrelation: {title}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Magnitude")
        ax.grid(True)


def _create_plot_figure(data: np.ndarray, fs: float, mode: str, title: str, size=(4, 3)) -> Figure:
    """Return a matplotlib Figure for the given visualization."""
    fig = Figure(figsize=size)
    ax = fig.add_subplot(111)
    _plot_on_axes(ax, data, fs, mode, title)
    fig.tight_layout()
    return fig


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

    def _clear_gen_plots(self) -> None:
        for canv in self.gen_canvases:
            canv.get_tk_widget().destroy()
        self.gen_canvases.clear()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox("all"))

    def _display_gen_plots(self, data: np.ndarray, fs: float) -> None:
        """Render all visualizations below the Generate button."""
        self.latest_data = data
        self.latest_fs = fs
        self._clear_gen_plots()

        modes = ["Signal", "Freq", "InstantFreq", "Autocorr"]
        for idx, mode in enumerate(modes):
            fig = _create_plot_figure(data, fs, mode, mode)
            canvas = FigureCanvasTkAgg(fig, master=self.gen_plots_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.grid(row=idx, column=0, sticky="nsew", pady=2)
            widget.bind("<Button-1>", lambda _e, m=mode: self._show_fullscreen(m))
            self.gen_canvases.append(canvas)
        self.gen_plots_frame.update_idletasks()
        self.gen_canvas.configure(scrollregion=self.gen_canvas.bbox("all"))

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
            for line in proc.stdout:
                self._out_queue.put(line)
            proc.wait()
            self._out_queue.put(f"[Exited with code {proc.returncode}]\n")
        except Exception as exc:
            self._out_queue.put(f"Error: {exc}\n")
        finally:
            self._cmd_running = False

    def _show_fullscreen(self, mode: str) -> None:
        if self.latest_data is None:
            return
        fig, ax = plt.subplots()
        _plot_on_axes(ax, self.latest_data, self.latest_fs, mode, mode)
        fig.tight_layout()
        try:
            fig.canvas.manager.full_screen_toggle()
        except Exception:
            pass
        plt.show()


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
        self._open_console("Transmit Log")
        self._cmd_running = True
        threading.Thread(target=self._run_cmd, args=(cmd,), daemon=True).start()
        self._process_queue()

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
