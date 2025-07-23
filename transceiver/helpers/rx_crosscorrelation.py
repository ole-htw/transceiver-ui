#!/usr/bin/env python3
import numpy as np
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import argparse
import os # For basename in title
import time # To measure correlation time

def read_signal_file(filename):
    """Liest eine Binärdatei mit interleaved int16 Samples und gibt ein komplexes Signal zurück."""
    try:
        print(f"Lese Datei: {filename}")
        data_int16 = np.fromfile(filename, dtype=np.int16)
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden: {filename}")
        return None
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei {filename}: {e}")
        return None

    if data_int16.size == 0:
        print(f"Fehler: Datei {filename} ist leer.")
        return None
    if data_int16.size % 2 != 0:
        print(f"Warnung: Ungerade Sample-Anzahl in {filename}. Letztes Sample ignoriert.")
        data_int16 = data_int16[:-1]

    # Konvertiere zu komplexem Signal (float64 für Präzision)
    real_part = data_int16[0::2].astype(np.float64)
    imag_part = data_int16[1::2].astype(np.float64)
    signal = real_part + 1j * imag_part
    print(f"Datei {filename} eingelesen: {len(signal)} komplexe Samples.")
    return signal


def find_los_echo(cc: np.ndarray) -> tuple[int | None, int | None]:
    """Return indices of the LOS peak and the first echo."""
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
        echo = int(np.argmax(mag[los + 1:]) + los + 1)
    return los, echo

def main():
    parser = argparse.ArgumentParser(
        description="Berechnet und visualisiert die Kreuzkorrelation zwischen zwei Signalen aus Binärdateien. "
                    "Kürzt das längere Signal auf die Länge des kürzeren Signals."
    )
    parser.add_argument("filename1", type=str,
                        help="Pfad zur 1. Binärdatei (int16, interleaved Real/Imaginär)")
    parser.add_argument("filename2", type=str,
                        help="Pfad zur 2. Binärdatei (int16, interleaved Real/Imaginär)")
    parser.add_argument("--db", action='store_true',
                        help="Zeigt die Magnitude der Kreuzkorrelation in dB an.")
    parser.add_argument("--maxlag", type=int, default=None,
                        help="Maximaler Lag (Verschiebung) der angezeigt wird (z.B. 10000). Standard: Alles anzeigen.")

    args = parser.parse_args()

    # --- Signale einlesen ---
    signal1 = read_signal_file(args.filename1)
    signal2 = read_signal_file(args.filename2)

    # Prüfen, ob beide Signale erfolgreich gelesen wurden
    if signal1 is None or signal2 is None:
        print("Abbruch wegen Fehlern beim Einlesen.")
        return

    N1_orig = len(signal1)
    N2_orig = len(signal2)

    # --- Signale auf gleiche Länge kürzen (auf Länge des kürzeren Signals) ---
    min_len = min(N1_orig, N2_orig)
    if N1_orig != N2_orig:
        print(f"Info: Kürze Signale auf die minimale Länge von {min_len} Samples.")
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]
    else:
        print(f"Info: Beide Signale haben bereits die gleiche Länge ({min_len} Samples).")

    N = min_len # Die neue, gemeinsame Länge

    if N == 0:
         print("Fehler: Signallänge nach Kürzung ist Null.")
         return

    # --- Kreuzkorrelation berechnen ---
    print(f"Berechne Kreuzkorrelation zwischen gekürzten Signalen (Länge {N})...")
    start_time = time.time()
    cross_corr = np.correlate(signal1, signal2, mode='full')
    end_time = time.time()
    print(f"Berechnung dauerte {end_time - start_time:.2f} Sekunden.")

    # Der Output hat die Länge N + N - 1 = 2*N - 1
    # Lags reichen von -(N-1) bis +(N-1)
    lags = np.arange(-(N - 1), N)
    zero_lag_index = N - 1  # Index des Lags 0

    los_idx_full, echo_idx_full = find_los_echo(cross_corr)
    if los_idx_full is not None and echo_idx_full is not None:
        delay_samples = int(lags[echo_idx_full] - lags[los_idx_full])
        print(f"LOS-Echo Abstand: {delay_samples} Samples")

    # --- Daten für den Plot vorbereiten ---
    plot_title = f'Kreuzkorrelation (gekürzt auf {N} Samples)'
    plot_title_suffix = ""
    ylabel_text = 'Kreuzkorrelations-Magnitude'
    plot_data = np.abs(cross_corr)


    # dB-Skala (optional)
    if args.db:
        # Normalisiere auf den maximalen Wert für dB-Darstellung
        max_val = np.max(plot_data)
        if max_val < 1e-9:
            print("Warnung: Maximale Korrelation ist nahe Null. dB-Plot nicht sinnvoll.")
            args.db = False # Deaktiviere dB intern
        else:
            # Füge kleinen Wert für log10 hinzu
            plot_data_db = 20 * np.log10((plot_data / max_val) + 1e-12)
            plot_data = plot_data_db # Überschreibe die zu plottenden Daten
            ylabel_text = 'Normalisierte Kreuzkorr.-Magnitude [dB]'
            plot_title_suffix = ' [dB]'


    # Datenreduktion bei sehr großen Datenmengen (>1 MB)
    if plot_data.nbytes > 1_000_000:
        step = int(np.ceil(plot_data.nbytes / 1_000_000))
        plot_data = plot_data[::step]
        lags = lags[::step]
        if los_idx_full is not None:
            los_idx = los_idx_full // step
        else:
            los_idx = None
        if echo_idx_full is not None:
            echo_idx = echo_idx_full // step
        else:
            echo_idx = None
    else:
        los_idx = los_idx_full
        echo_idx = echo_idx_full

    # --- Plotten mit PyQtGraph ---
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = pg.mkQApp()

    fname1_base = os.path.basename(args.filename1)
    fname2_base = os.path.basename(args.filename2)
    win = pg.plot(lags, plot_data, pen=pg.mkPen("b"))
    win.setWindowTitle(f"Kreuzkorrelation - {fname1_base} vs {fname2_base}")
    win.setLabel("bottom", "Lag / Verschiebung von Signal 2 zu Signal 1 [Samples]")
    win.setLabel("left", ylabel_text)
    if los_idx is not None:
        win.plot([lags[los_idx]], [plot_data[los_idx]], pen=None, symbol="o", symbolBrush="r")
    if echo_idx is not None:
        win.plot([lags[echo_idx]], [plot_data[echo_idx]], pen=None, symbol="o", symbolBrush="g")
    win.showGrid(x=True, y=True)
    win.setTitle(f"{plot_title}: {fname1_base} vs {fname2_base}{plot_title_suffix}")

    pg.exec()

if __name__ == '__main__':
    main()
