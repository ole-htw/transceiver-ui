#!/usr/bin/env python3
import numpy as np
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import argparse
import itertools

def main():
    p = argparse.ArgumentParser(
        description=("Zeigt 1-4 parallele RX-Kanäle aus einer Bin-Datei "
                     "(Layout: I0 Q0 I1 Q1 …, int16)."))
    p.add_argument("filename", type=str, help="Pfad zur Binärdatei")
    p.add_argument("--channels", type=int, choices=range(1,5), default=1,
                   help="Anzahl der Kanäle in der Datei (1-4, Standard 1)")
    p.add_argument("--samples", type=int, default=None,
                   help="Anzahl komplexer Samples, die geplottet werden sollen")
    args = p.parse_args()

    ch        = args.channels
    col_count = 2 * ch              # I- und Q-Werte je Kanal

    # --- Datei laden --------------------------------------------------------
    raw = np.fromfile(args.filename, dtype=np.int16)
    if raw.size % col_count != 0:
        raise ValueError("Dateigröße passt nicht zu channel-Anzahl.")

    data = raw.reshape((-1, col_count))        # [samples , 2*channels]

    # optional kürzen
    if args.samples is not None:
        data = data[:args.samples, :]

    # Zu große Datenmengen reduzieren (mehr als 1 MB)
    if data.nbytes > 1_000_000:
        step = int(np.ceil(data.nbytes / 1_000_000))
        data = data[::step, :]

    # --- Komplexe Arrays je Kanal bilden ------------------------------------
    sigs = []
    for k in range(ch):
        I = data[:, 2*k]
        Q = data[:, 2*k + 1]
        sigs.append(I + 1j*Q)

    # --- Plot mit PyQtGraph -------------------------------------------------
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    app = pg.mkQApp()
    win = pg.GraphicsLayoutWidget(title="RX Waveforms (komplex)")
    win.setWindowTitle(args.filename)

    colours = itertools.cycle(('b', 'r', 'g', 'm'))

    for idx, (sig, color) in enumerate(zip(sigs, colours)):
        p = win.addPlot(row=idx, col=0)
        p.showGrid(x=True, y=True)
        p.addLegend()
        p.plot(np.real(sig), pen=pg.mkPen(color), name=f"Ch {idx} I")
        p.plot(np.imag(sig), pen=pg.mkPen(color, style=QtCore.Qt.DashLine),
               name=f"Ch {idx} Q")
        p.setLabel('left', 'Amplitude')
        if idx == ch - 1:
            p.setLabel('bottom', 'Sample Index')

    pg.exec()

if __name__ == "__main__":
    main()

