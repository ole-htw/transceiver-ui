#!/usr/bin/env python3
"""Standalone PyQtGraph plotting worker."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from multiprocessing import shared_memory

import numpy as np
import pyqtgraph as pg

from transceiver import __main__ as plot_impl
from transceiver.helpers import rx_convert


def _load_iq(
    path: str | None, shm_meta: dict[str, object] | None
) -> tuple[np.ndarray | None, shared_memory.SharedMemory | None]:
    if shm_meta:
        shm_name = shm_meta.get("name")
        shape = shm_meta.get("shape")
        dtype = shm_meta.get("dtype")
        if shm_name and shape and dtype:
            try:
                shm = shared_memory.SharedMemory(name=str(shm_name))
                array = np.ndarray(
                    tuple(shape), dtype=np.dtype(dtype), buffer=shm.buf
                )
                return array, shm
            except (FileNotFoundError, OSError, TypeError, ValueError):
                pass
    if path is None:
        return None, None
    data_path = Path(path)
    mmap_mode = "r" if data_path.suffix.lower() in {".npy", ".npz"} else None
    return rx_convert.load_iq_file(data_path, mmap_mode=mmap_mode), None


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
    reduction_step = int(payload.get("reduction_step", 1))
    data_meta = {
        "name": payload.get("shm_name"),
        "shape": payload.get("shape"),
        "dtype": payload.get("dtype"),
    }
    ref_meta = {
        "name": payload.get("ref_shm_name"),
        "shape": payload.get("ref_shape"),
        "dtype": payload.get("ref_dtype"),
    }
    data, data_shm = _load_iq(payload.get("data_file"), data_meta)
    ref_data, ref_shm = _load_iq(payload.get("ref_file"), ref_meta)
    manual_lags = payload.get("manual_lags") or None
    fullscreen = bool(payload.get("fullscreen", False))
    output_path = payload.get("output_path")

    if data is None or np.size(data) == 0:
        for shm in (data_shm, ref_shm):
            if shm is None:
                continue
            with contextlib.suppress(FileNotFoundError):
                shm.unlink()
            shm.close()
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
        reduce_data=False,
        reduction_step=reduction_step,
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

    for shm in (data_shm, ref_shm):
        if shm is None:
            continue
        with contextlib.suppress(FileNotFoundError):
            shm.unlink()
        shm.close()


if __name__ == "__main__":
    main()
