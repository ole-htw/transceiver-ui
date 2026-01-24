#!/usr/bin/env python3
import numpy as np
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import argparse
from transceiver.helpers.plot_colors import PLOT_COLORS

def main():
    parser = argparse.ArgumentParser(
        description="Visualisierung eines RX Binärfiles analog zum MATLAB-Skript"
    )
    parser.add_argument("filename", type=str,
                        help="Pfad zur RX Binärdatei")
    parser.add_argument("--samples", type=int, default=None,
                        help="Anzahl der zu plottenden Samples (Standard: Alle Samples)")
    parser.add_argument("--format", type=str, choices=["int16", "complex64"], default="int16",
                        help="Eingabeformat der Datei: 'int16' für sc16 oder 'complex64' für fc32 (default: int16)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Skalierungsfaktor für die Amplitude (default: 1.0)")
    args = parser.parse_args()

    try:
        if args.format == "int16":
            # Einlesen der int16-Daten (sc16)
            data = np.fromfile(args.filename, dtype=np.int16)
            if data.size % 2 != 0:
                print("Die Anzahl der Datenwerte ist ungerade. Die Datei hat vermutlich ein falsches Format.")
                return
            data = data.reshape((-1, 2))
            rx_sig = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)
        else:
            # Einlesen der complex64-Daten (fc32)
            rx_sig = np.fromfile(args.filename, dtype=np.complex64)
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei: {e}")
        return

    # Falls --samples angegeben wurde, nur diese Anzahl an Samples plotten
    if args.samples is not None:
        rx_sig = rx_sig[:args.samples]

    # Skalierung anwenden
    rx_sig = rx_sig * args.scale

    # Zu große Datenmengen reduzieren (mehr als 1 MB)
    if rx_sig.nbytes > 1_000_000:
        step = int(np.ceil(rx_sig.nbytes / 1_000_000))
        rx_sig = rx_sig[::step]

    # PyQtGraph-Konfiguration
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    app = pg.mkQApp()
    win = pg.plot(title='Raw RX waveform')
    win.setWindowTitle(args.filename)
    win.showGrid(x=True, y=True)
    win.setLabel('bottom', 'Sample Index')
    win.setLabel('left', 'Amplitude')
    win.addLegend()

    win.plot(
        np.real(rx_sig),
        pen=pg.mkPen(PLOT_COLORS["real"]),
        name="Realteil",
    )
    win.plot(
        np.imag(rx_sig),
        pen=pg.mkPen(PLOT_COLORS["imag"], style=QtCore.Qt.DashLine),
        name='Imaginärteil')

    pg.exec()

if __name__ == '__main__':
    main()
