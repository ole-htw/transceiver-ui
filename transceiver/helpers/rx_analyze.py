#!/usr/bin/env python3
"""
Dieses Skript lädt eine RX-Datei (int16 interleaved), bereinigt das Signal durch Entfernen
des DC-Offsets, skaliert das Rauschen so, dass dessen RMS ≈ 100 beträgt und berechnet
Kennzahlen wie die durchschnittliche Amplitude des Signals, den Rauschpegel sowie einen
Clipping-Indikator. Der Output-Dateiname wird automatisch als "<Inputname>-postprocessed.<Ext>" erstellt.
Optional können Original- und postprocessetes Signal geplottet werden.
"""

import numpy as np
import argparse
import os
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
from transceiver.helpers.plot_colors import PLOT_COLORS

def load_rx_file(filename):
    """
    Liest eine int16-Binärdatei ein und formt diese zu einem komplexen Signal.
    Es wird angenommen, dass die Daten abwechselnd als Real- und Imaginärteile vorliegen.
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
    # Runden und konvertieren
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

def compute_metrics(signal, clip_threshold=0.9, abs_scale=32767):
    """
    Berechnet folgende Kennzahlen:
      - Bereinigte durchschnittliche Amplitude (nur Samples oberhalb einer Schwelle T)
      - Rausch-RMS (nur Samples unterhalb von T)
      - Clipping-Indikator (Anteil der Samples, die >= clip_threshold * abs_scale sind)
    
    Wir wählen hier als Schwelle T den Median des Betrags, der oft eine gute Trennung zwischen
    Signal und Rauschen liefert.
    """
    abs_signal = np.abs(signal)
    T = np.median(abs_signal)  # Schwelle als Median
    signal_mask = abs_signal > T
    noise_mask = ~signal_mask

    if np.sum(signal_mask) > 0:
        avg_signal = np.mean(abs_signal[signal_mask])
    else:
        avg_signal = 0.0

    if np.sum(noise_mask) > 0:
        noise_rms = np.sqrt(np.mean((abs_signal[noise_mask])**2))
    else:
        noise_rms = 0.0

    clip_val = clip_threshold * abs_scale
    n_clip = np.sum(abs_signal >= clip_val)
    clipping_ratio = n_clip / signal.size

    return avg_signal, noise_rms, clipping_ratio

def generate_output_filename(infile):
    """
    Generiert den Output-Dateinamen aus dem Input-Dateinamen,
    indem der Suffix "-postprocessed" eingefügt wird.
    Beispiel: "rx_input.bin" -> "rx_input-postprocessed.bin"
    """
    base, ext = os.path.splitext(infile)
    return f"{base}-postprocessed{ext}"

def main():
    parser = argparse.ArgumentParser(
        description="Postprocessing von RX-Dateien: Entfernt DC-Offset, skaliert das Rauschen auf RMS ≈ 100, "
                    "berechnet Kennzahlen und speichert das Ergebnis in einer neuen Datei."
    )
    parser.add_argument("infile", type=str, help="Pfad zur Eingang RX Binärdatei (int16 interleaved)")
    parser.add_argument("--phase-correction", action="store_true",
                        help="Globale Phasenkorrektur anwenden")
    parser.add_argument("--plot", action="store_true",
                        help="Plotten von Original- und postprocessetem Signal")
    args = parser.parse_args()

    # Automatisch generierten Output-Dateinamen erstellen
    outfile = generate_output_filename(args.infile)
    print(f"Output-Datei wird automatisch gesetzt: '{outfile}'")

    # Signal laden und speichern als Original
    signal = load_rx_file(args.infile)
    orig_signal = signal.copy()

    # DC-Offset entfernen
    signal, offsets = dc_offset_removal(signal)
    print(f"DC Offset entfernt: Real = {offsets[0]:.2f}, Imag = {offsets[1]:.2f}")

    # Optionale globale Phasenkorrektur
    if args.phase_correction:
        signal, phi = global_phase_correction(signal)
        print(f"Globale Phasenkorrektur: Rotationswinkel = {phi:.2f} rad")

    # Berechne Kennzahlen vor Skalierung
    avg_sig, noise_rms, clipping_ratio = compute_metrics(signal)
    print("Vor Skalierung:")
    print(f"  Durchschnittliche Signal-Amplitude (bereinigt): {avg_sig:.2f}")
    print(f"  Rausch-RMS: {noise_rms:.2f}")
    print(f"  Clipping-Anteil: {clipping_ratio*100:.2f}%")

    # Noise-Skalierung: Ziel ist, dass der Rausch-RMS ≈ 100 wird.
    if noise_rms > 0:
        scale_factor = 100.0 / noise_rms
    else:
        scale_factor = 1.0
        print("Warnung: Kein Rauschpegel erkannt, Skalierungsfaktor wird 1.0 gesetzt.")

    print(f"Angewandter Skalierungsfaktor: {scale_factor:.3f}")
    signal = signal * scale_factor

    # Berechne Kennzahlen nach der Skalierung
    avg_sig_scaled, noise_rms_scaled, clipping_ratio_scaled = compute_metrics(signal)
    print("Nach Skalierung:")
    print(f"  Durchschnittliche Signal-Amplitude (bereinigt): {avg_sig_scaled:.2f}")
    print(f"  Rausch-RMS: {noise_rms_scaled:.2f}")
    print(f"  Clipping-Anteil: {clipping_ratio_scaled*100:.2f}%")

    # Speichere das postprocessete Signal
    save_rx_file(outfile, signal)
    print(f"Postprocessetes Signal wurde in '{outfile}' gespeichert.")

    # Optional: Plot des Original- und postprocesseten Signals
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
        p1 = win.addPlot(title="Originales RX Signal")
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
        p2 = win.addPlot(title="Postprocessetes RX Signal (skaliert)")
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
