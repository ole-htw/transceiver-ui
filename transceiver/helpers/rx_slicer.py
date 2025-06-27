#!/usr/bin/env python3
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Schneidet einen Abschnitt aus einer interleaved int16 RX-Binärdatei aus."
    )
    parser.add_argument("input_file", type=str,
                        help="Pfad zur Eingabe-Binärdatei (interleaved int16 Real,Imag)")
    parser.add_argument("output_file", type=str,
                        help="Pfad zur Ausgabe-Binärdatei")
    parser.add_argument("--start_pct", type=float, required=True,
                        help="Start des Ausschnitts in Prozent der Gesamtlänge (0-100)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--length_samples", type=int,
                       help="Länge des Ausschnitts in Samples")
    group.add_argument("--length_pct", type=float,
                       help="Länge des Ausschnitts in Prozent der Gesamtlänge (0-100)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Lese Rohdaten als int16
    data = np.fromfile(args.input_file, dtype=np.int16)
    if data.size % 2 != 0:
        raise ValueError("Die Eingabedatei enthält eine ungerade Anzahl von int16-Werten.")

    # Forme zu komplexem Signal um
    data = data.reshape(-1, 2)
    sig = data[:,0].astype(np.float32) + 1j * data[:,1].astype(np.float32)
    total_samples = sig.size

    # Bestimme Start-Index
    if not (0 <= args.start_pct <= 100):
        raise ValueError("--start_pct muss zwischen 0 und 100 liegen.")
    start_idx = int(np.floor(args.start_pct / 100.0 * total_samples))

    # Bestimme Länge in Samples
    if args.length_samples is not None:
        length = args.length_samples
    else:
        if not (0 <= args.length_pct <= 100):
            raise ValueError("--length_pct muss zwischen 0 und 100 liegen.")
        length = int(np.floor(args.length_pct / 100.0 * total_samples))

    # Prüfe Grenzen
    end_idx = start_idx + length
    if start_idx < 0 or end_idx > total_samples:
        raise ValueError(f"Ausschnitt [{start_idx}:{end_idx}] liegt außerhalb der Daten (0:{total_samples}).")

    # Schneide Signal
    slice_sig = sig[start_idx:end_idx]

    # Zurück in interleaved int16
    real_i16 = np.int16(np.round(np.real(slice_sig)))
    imag_i16 = np.int16(np.round(np.imag(slice_sig)))
    interleaved = np.empty(real_i16.size + imag_i16.size, dtype=np.int16)
    interleaved[0::2] = np.clip(real_i16, -32768, 32767)
    interleaved[1::2] = np.clip(imag_i16, -32768, 32767)

    # Schreibe Ausgabedatei
    interleaved.tofile(args.output_file)
    print(f"Geschrieben: {args.output_file} — Samples: {length}")

if __name__ == '__main__':
    main()

