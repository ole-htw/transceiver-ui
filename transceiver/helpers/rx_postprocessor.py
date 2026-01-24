#!/usr/bin/env python3
import numpy as np
import argparse
import os
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
from transceiver.helpers.plot_colors import PLOT_COLORS

def load_rx_file(filename):
    """
    Liest eine int16-Binärdatei ein und formt diese zu einem komplexen Signal.
    Es wird davon ausgegangen, dass die Daten abwechselnd als Real- und Imaginärteile vorliegen.
    """
    try:
        data = np.fromfile(filename, dtype=np.int16)
    except Exception as e:
        raise RuntimeError(f"Fehler beim Einlesen der Datei '{filename}': {e}")
    if data.size % 2 != 0:
        raise ValueError("Die Anzahl der int16-Werte ist ungerade – prüfen Sie das Dateiformat!")
    data = data.reshape((-1, 2))
    signal = data[:, 0] + 1j * data[:, 1]
    return signal

def save_rx_file(filename, signal):
    """
    Speichert ein komplexes Signal im int16-Format als interleaved Binärdatei.
    """
    real_part = np.int16(np.round(np.real(signal)))
    imag_part = np.int16(np.round(np.imag(signal)))
    interleaved = np.empty(real_part.size + imag_part.size, dtype=np.int16)
    interleaved[0::2] = real_part
    interleaved[1::2] = imag_part
    interleaved.tofile(filename)

def dc_offset_removal(signal):
    """
    Entfernt den DC-Offset, indem der Mittelwert des Real- und Imaginärteils abgezogen wird.
    Rückgabe: korrigiertes Signal sowie ein Tuple mit (Real-Offset, Imag-Offset)
    """
    real_offset = np.mean(np.real(signal))
    imag_offset = np.mean(np.imag(signal))
    corrected_signal = signal - (real_offset + 1j * imag_offset)
    return corrected_signal, (real_offset, imag_offset)

def global_phase_correction(signal):
    """
    Korrigiert eine globale Phasenverschiebung, indem der Mittelwert des Signals
    berechnet und das Signal um den negativen Winkel dieses Mittelwerts rotiert wird.
    Rückgabe: korrigiertes Signal und der angewandte Winkel in Radiant.
    """
    avg = np.mean(signal)
    phi = np.angle(avg)
    corrected_signal = signal * np.exp(-1j * phi)
    return corrected_signal, phi

def generate_output_filename(infile):
    """
    Erzeugt den Output-Dateinamen aus dem Input-Dateinamen,
    indem der Suffix "-postprocessed" eingefügt wird.
    Beispiel: "rx_input.bin" -> "rx_input-postprocessed.bin"
    """
    base, ext = os.path.splitext(infile)
    return f"{base}-postprocessed{ext}"

def main():
    parser = argparse.ArgumentParser(
        description="Postprocessing für RX-Dateien: Entfernt DC-Offset und optional eine globale Phasenkorrektur."
    )
    parser.add_argument("infile", type=str, help="Pfad zur Eingang RX Binärdatei (int16 interleaved)")
    parser.add_argument("--phase-correction", action="store_true",
                        help="Globale Phasenkorrektur anwenden (rotieren, sodass der Mittelwert rein reell wird)")
    parser.add_argument("--plot", action="store_true", 
                        help="Zeige Plot des Signals vor und nach dem Postprocessing")
    args = parser.parse_args()

    # Automatisch generierten Output-Dateinamen erstellen
    outfile = generate_output_filename(args.infile)
    print(f"Output-Datei wird automatisch gesetzt: '{outfile}'")

    # Signal einlesen
    signal = load_rx_file(args.infile)
    orig_signal = signal.copy()

    # DC-Offset entfernen
    signal, offsets = dc_offset_removal(signal)
    print(f"DC Offset entfernt: Real = {offsets[0]:.2f}, Imag = {offsets[1]:.2f}")

    # Optionale globale Phasenkorrektur
    if args.phase_correction:
        signal, phi = global_phase_correction(signal)
        print(f"Globale Phasenkorrektur: Rotationswinkel = {phi:.2f} rad")

    # Speichern des postprozesseten Signals
    save_rx_file(outfile, signal)
    print(f"Postprocessetes Signal wurde in '{outfile}' gespeichert.")

    # Plot, falls gewünscht
    if args.plot:
        orig_plot = orig_signal
        proc_plot = signal
        if orig_plot.nbytes > 1_000_000 or proc_plot.nbytes > 1_000_000:
            step = int(np.ceil(max(orig_plot.nbytes, proc_plot.nbytes) / 1_000_000))
            orig_plot = orig_plot[::step]
            proc_plot = proc_plot[::step]

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        app = pg.mkQApp()
        win = pg.GraphicsLayoutWidget(title="RX Signal")
        p1 = win.addPlot(title="Originales RX-Signal")
        p1.plot(
            np.real(orig_plot),
            pen=pg.mkPen(PLOT_COLORS["real"]),
            name="Real",
        )
        p1.plot(
            np.imag(orig_plot),
            pen=pg.mkPen(PLOT_COLORS["imag"], style=QtCore.Qt.DashLine),
            name="Imag",
        )
        p1.showGrid(x=True, y=True)
        p1.addLegend()
        win.nextRow()
        p2 = win.addPlot(title="Postprocessetes RX-Signal")
        p2.plot(
            np.real(proc_plot),
            pen=pg.mkPen(PLOT_COLORS["real"]),
            name="Real",
        )
        p2.plot(
            np.imag(proc_plot),
            pen=pg.mkPen(PLOT_COLORS["imag"], style=QtCore.Qt.DashLine),
            name="Imag",
        )
        p2.showGrid(x=True, y=True)
        p2.addLegend()
        pg.exec()

if __name__ == '__main__':
    main()
