#!/usr/bin/env python3
import numpy as np
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import argparse
from transceiver.helpers.plot_colors import PLOT_COLORS

def main():
    parser = argparse.ArgumentParser(
        description="Frequenzspektrum eines RX-Binärfiles anzeigen"
    )
    parser.add_argument("filename", type=str,
                        help="Pfad zur RX-Binärdatei (int16, 2 Kanäle: I und Q)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Anzahl der zu plottenden Samples (Standard: alle)")
    parser.add_argument("--rate", type=float, default=None,
                        help="Abtastrate in Hz (für richtige Frequenzachse)")
    args = parser.parse_args()

    # Datei einlesen
    data = np.fromfile(args.filename, dtype=np.int16)
    if data.size % 2 != 0:
        raise ValueError("Datei enthält ungerade Anzahl von int16-Werten")

    # In (I,Q)-Paare umformen und komplexes Signal bilden
    iq = data.reshape((-1, 2))
    rx = iq[:,0] + 1j*iq[:,1]

    # Optional Samples beschränken
    if args.samples is not None:
        rx = rx[:args.samples]

    # Zu große Datenmengen reduzieren (mehr als 1 MB)
    step = 1
    if rx.nbytes > 1_000_000:
        step = int(np.ceil(rx.nbytes / 1_000_000))
        rx = rx[::step]

    N = rx.size

    # FFT und zentrieren
    spec = np.fft.fftshift(np.fft.fft(rx))
    mag = 20*np.log10(np.abs(spec) / N)

    # Frequenzachse
    if args.rate:
        fs = args.rate / step
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
        xlabel = "Frequenz (Hz)"
    else:
        freqs = np.fft.fftshift(np.fft.fftfreq(N))
        xlabel = "Normierte Frequenz"

    # Plot mit PyQtGraph
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    app = pg.mkQApp()
    win = pg.plot(
        freqs,
        mag,
        pen=pg.mkPen(PLOT_COLORS["freq"]),
        title=f"Frequenzspektrum: {args.filename}",
    )
    win.setWindowTitle(args.filename)
    win.showGrid(x=True, y=True)
    win.setLabel('bottom', xlabel)
    win.setLabel('left', 'Magnitude (dB)')

    pg.exec()

if __name__ == "__main__":
    main()
