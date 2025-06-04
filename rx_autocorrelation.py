#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time # Um die Berechnungszeit zu messen

def main():
    parser = argparse.ArgumentParser(
        description="Berechnet und visualisiert die Autokorrelation eines Signals aus einer Binärdatei."
    )
    parser.add_argument("filename", type=str,
                        help="Pfad zur Binärdatei (int16, interleaved Real/Imaginär)")
    parser.add_argument("--normalize", action='store_true',
                        help="Normalisiert die Autokorrelation, sodass der Peak bei Lag 0 gleich 1 ist.")
    parser.add_argument("--db", action='store_true',
                        help="Zeigt die Magnitude der Autokorrelation in dB an (impliziert Normalisierung).")
    parser.add_argument("--maxlag", type=int, default=None,
                        help="Maximaler Lag (Verschiebung) der angezeigt wird (z.B. 10000). Standard: Alles anzeigen.")

    args = parser.parse_args()

    try:
        print(f"Lese Datei: {args.filename}")
        # Einlesen der int16-Daten
        data_int16 = np.fromfile(args.filename, dtype=np.int16)
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden: {args.filename}")
        return
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei: {e}")
        return

    if data_int16.size == 0:
        print("Fehler: Datei ist leer.")
        return
    # Stelle sicher, dass die Anzahl der Samples gerade ist
    if data_int16.size % 2 != 0:
        print("Warnung: Die Anzahl der Datenwerte ist ungerade. Letztes Sample wird ignoriert.")
        data_int16 = data_int16[:-1] # Entferne das letzte Sample

    # Konvertiere interleaved int16 zu komplexem Signal
    # Konvertiere zu float64 für bessere Präzision bei der Berechnung
    real_part = data_int16[0::2].astype(np.float64)
    imag_part = data_int16[1::2].astype(np.float64)
    signal = real_part + 1j * imag_part
    N = len(signal)
    print(f"Datei eingelesen: {N} komplexe Samples.")

    if N == 0:
        print("Fehler: Kein Signal nach der Konvertierung vorhanden.")
        return

    # --- Autokorrelation berechnen ---
    # np.correlate berechnet die lineare Autokorrelation
    # mode='full' gibt alle Lags von -(N-1) bis +(N-1) aus
    print("Berechne Autokorrelation (mode='full')...")
    start_time = time.time()
    autocorr = np.correlate(signal, signal, mode='full')
    end_time = time.time()
    print(f"Berechnung dauerte {end_time - start_time:.2f} Sekunden.")

    # Der Output von mode='full' hat die Länge 2*N - 1
    # Der Index N-1 im Ergebnisarray entspricht dem Lag 0
    zero_lag_index = N - 1

    # Erstelle einen Vektor der Lags (Verschiebungen) für die X-Achse
    lags = np.arange(-(N - 1), N)

    # --- Daten für den Plot vorbereiten ---
    plot_title = f'Autokorrelation von {args.filename}'

    # Normalisierung (optional, aber für dB notwendig)
    if args.normalize or args.db:
        peak_value = autocorr[zero_lag_index]
        # Prüfe, ob Peak existiert und nicht (nahe) Null ist
        if np.abs(peak_value) < 1e-9:
             print("Warnung: Peak-Wert bei Lag 0 ist nahe Null. Normalisierung/dB-Plot nicht möglich.")
             plot_data = np.abs(autocorr) # Zeige unnormalisierte Magnitude
             ylabel_text = 'Magnitude (Unnormalisiert)'
             args.normalize = False # Deaktiviere Flags intern
             args.db = False
        else:
             # Normalisiere, sodass der Peak bei Lag 0 den Wert 1 hat
             autocorr = autocorr / peak_value
             plot_data = np.abs(autocorr)
             ylabel_text = 'Normalisierte Magnitude'
             if not args.normalize: # Wenn nur --db aktiv war
                 print("Info: Autokorrelation für dB-Plot normalisiert.")
    else:
        # Keine Normalisierung
        plot_data = np.abs(autocorr)
        ylabel_text = 'Magnitude'

    # dB-Skala (optional)
    if args.db:
         # Füge kleinen Wert hinzu, um log(0) zu vermeiden, falls Samples Null sind
         plot_data_db = 20 * np.log10(plot_data + 1e-12)
         plot_data = plot_data_db # Überschreibe die zu plottenden Daten
         ylabel_text = 'Normalisierte Magnitude [dB]'
         plot_title += ' [dB]' # Füge dB zum Titel hinzu


    # --- Plotten ---
    print("Erstelle Plot...")
    fig, ax = plt.subplots(figsize=(12, 7)) # Etwas größeres Fenster
    fig.canvas.manager.set_window_title(f"Autokorrelation - {args.filename}")

    ax.plot(lags, plot_data)

    ax.set_title(plot_title)
    ax.set_xlabel('Lag / Verschiebung [Samples]')
    ax.set_ylabel(ylabel_text)
    ax.grid(True)

    # Optional: X-Achsen-Limit setzen, falls maxlag angegeben wurde
    if args.maxlag is not None and args.maxlag > 0:
        ax.set_xlim(-args.maxlag, args.maxlag)
        print(f"Zeige Lags von {-args.maxlag} bis {args.maxlag}.")
    else:
        ax.set_xlim(lags[0], lags[-1]) # Voller Bereich

    # Optional: Y-Achsen-Limit für dB-Plots
    if args.db:
        # Setze unteres Limit z.B. auf -80 dB oder Minimum+Puffer, oberes auf etwas über 0
        min_db_val = max(np.min(plot_data[plot_data > -np.inf]), -80) # Ignoriere -inf, mindestens -80dB
        ax.set_ylim(bottom=min_db_val - 5, top=5)

    plt.tight_layout() # Optimiert Layout
    print("Plot wird angezeigt. Schließe das Fenster, um das Skript zu beenden.")
    plt.show()

if __name__ == '__main__':
    main()

