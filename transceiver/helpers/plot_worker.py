#!/usr/bin/env python3
"""Standalone PyQtGraph plotting worker."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from multiprocessing import shared_memory, Pipe

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

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


def _cleanup_shm(shm: shared_memory.SharedMemory | None) -> None:
    if shm is None:
        return
    with contextlib.suppress(FileNotFoundError):
        shm.unlink()
    shm.close()


def _clear_plot(plot_item: pg.PlotItem) -> None:
    if plot_item.legend is not None:
        legend = plot_item.legend
        plot_item.legend = None
        if legend.scene() is not None:
            legend.scene().removeItem(legend)
        legend.deleteLater()
    plot_item.clear()


def _write_manual_state(path: str | None, manual_state: dict[str, int | None]) -> None:
    if not path:
        return
    try:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(manual_state, handle)
    except Exception:
        pass


def _prepare_payload(payload: dict[str, object]) -> tuple[dict[str, object], dict[str, int | None]]:
    manual_lags = payload.get("manual_lags") or None
    if isinstance(manual_lags, dict):
        manual_state = {
            "los": manual_lags.get("los"),
            "echo": manual_lags.get("echo"),
        }
    else:
        manual_state = {"los": None, "echo": None}
    return payload, manual_state


def worker_loop(conn, initial_payload: dict[str, object] | None = None) -> None:
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    app = pg.mkQApp()
    win = pg.plot()
    plot_item = win.getPlotItem()
    manual_state: dict[str, int | None] = {"los": None, "echo": None}
    output_path: str | None = None

    def _update_manual(kind: str, lag_value: float) -> None:
        if kind not in ("los", "echo"):
            return
        manual_state[kind] = int(round(lag_value))
        _write_manual_state(output_path, manual_state)

    def _apply_payload(payload: dict[str, object]) -> None:
        nonlocal manual_state, output_path
        payload, manual_state = _prepare_payload(payload)
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
            _cleanup_shm(data_shm)
            _cleanup_shm(ref_shm)
            return

        if data_shm is not None:
            data = np.array(data)
            _cleanup_shm(data_shm)
        if ref_shm is not None and ref_data is not None:
            ref_data = np.array(ref_data)
            _cleanup_shm(ref_shm)

        _clear_plot(plot_item)
        win.setWindowTitle(str(title))
        plot_impl._plot_on_pg(
            plot_item,
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
            win.showNormal()
            win.show()

        _write_manual_state(output_path, manual_state)

    def _poll_conn() -> None:
        try:
            while conn.poll():
                message = conn.recv()
                if message is None:
                    app.quit()
                    return
                if isinstance(message, dict) and message.get("command") == "shutdown":
                    app.quit()
                    return
                if isinstance(message, dict) and message.get("command") == "plot":
                    payload = message.get("payload") or {}
                else:
                    payload = message
                if isinstance(payload, dict):
                    _apply_payload(payload)
        except (EOFError, OSError):
            app.quit()

    if initial_payload is not None:
        _apply_payload(initial_payload)

    timer = QtCore.QTimer()
    timer.timeout.connect(_poll_conn)
    timer.start(100)

    pg.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description="PyQtGraph plot worker")
    parser.add_argument(
        "--payload",
        help="Path to JSON payload containing plot parameters.",
    )
    args = parser.parse_args()
    payload = _parse_payload(Path(args.payload)) if args.payload else None
    recv_conn, send_conn = Pipe(duplex=False)
    send_conn.close()
    worker_loop(recv_conn, initial_payload=payload)


if __name__ == "__main__":
    main()
