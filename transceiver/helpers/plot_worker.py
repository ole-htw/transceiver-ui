#!/usr/bin/env python3
"""Standalone PyQtGraph plotting worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg

from transceiver import __main__ as plot_impl
from transceiver.helpers import rx_convert


def _load_iq(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    file_path = Path(path)
    if file_path.suffix.lower() in {".npy", ".npz"}:
        return rx_convert.load_numpy(file_path, mmap_mode="r")
    return rx_convert.load_iq_file(file_path)


def _parse_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyQtGraph plot worker")
    parser.add_argument(
        "--payload",
        required=True,
        help="Path to JSON payload containing plot parameters.",
    )
    args = parser.parse_args()

    payload = _parse_payload(Path(args.payload))
    mode = payload.get("mode", "")
    title = payload.get("title", "")
    fs = float(payload.get("fs", 0.0))
    data = _load_iq(payload.get("data_file"))
    ref_data = _load_iq(payload.get("ref_file"))
    manual_lags = payload.get("manual_lags") or None
    fullscreen = bool(payload.get("fullscreen", False))
    output_path = payload.get("output_path")

    if data is None or np.size(data) == 0:
        return

    if isinstance(manual_lags, dict):
        manual_state: dict[str, int | None] = {
            "los": manual_lags.get("los"),
            "echo": manual_lags.get("echo"),
        }
    else:
        manual_state = {"los": None, "echo": None}

    def _update_manual(kind: str, lag_value: float) -> None:
        if kind not in ("los", "echo"):
            return
        manual_state[kind] = int(round(lag_value))

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    pg.mkQApp()
    win = pg.plot()
    win.setWindowTitle(str(title))
    plot_impl._plot_on_pg(
        win.getPlotItem(),
        data,
        fs,
        mode,
        title,
        ref_data=ref_data,
        manual_lags=manual_state if manual_lags is not None else None,
        on_los_drag_end=lambda _idx, lag: _update_manual("los", lag),
        on_echo_drag_end=lambda _idx, lag: _update_manual("echo", lag),
    )

    if fullscreen:
        try:
            win.showMaximized()
        except Exception:
            win.show()
    else:
        win.show()

    pg.exec()

    if output_path:
        try:
            with Path(output_path).open("w", encoding="utf-8") as handle:
                json.dump(manual_state, handle)
        except Exception:
            pass


if __name__ == "__main__":
    main()
