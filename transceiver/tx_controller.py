"""In-process TX controller for UHD/RFNoC streaming."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import uhd


_LOG = logging.getLogger(__name__)


def _ensure_uhd_frame_sizes(args_str: str, send_frame_size: int = 8000) -> str:
    components = [part for part in args_str.split(",") if part]
    if not any(part.startswith("send_frame_size=") for part in components):
        components.append(f"send_frame_size={send_frame_size}")
    return ",".join(components)


def _extract_addr_key(args_str: str) -> str:
    for part in args_str.split(","):
        part = part.strip()
        if part.startswith("addr="):
            return part.split("=", 1)[1].strip() or args_str
    return args_str or "default"


@dataclass
class _TxParams:
    file_path: str
    repeat: bool
    rate: float
    freq: float
    gain: float
    chan: int


class TxController:
    _instances: dict[str, "TxController"] = {}
    _instances_lock = threading.Lock()

    def __init__(self, args: str) -> None:
        self._args = args
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self.last_end_monotonic = 0.0
        self.last_error: Exception | None = None
        self._log_callback: Callable[[str], None] | None = None

    @classmethod
    def for_args(cls, args: str) -> "TxController":
        key = _extract_addr_key(args)
        with cls._instances_lock:
            controller = cls._instances.get(key)
            if controller is None:
                controller = cls(args)
                cls._instances[key] = controller
            else:
                controller._args = args
            return controller

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start_tx(
        self,
        file_path: str,
        *,
        repeat: bool,
        rate: float,
        freq: float,
        gain: float,
        chan: int = 0,
        log_callback: Callable[[str], None] | None = None,
    ) -> bool:
        with self._lock:
            if self._running:
                self._log("TX start requested but already running.")
                return False
            self._stop_event.clear()
            self.last_error = None
            self._log_callback = log_callback
            params = _TxParams(
                file_path=file_path,
                repeat=repeat,
                rate=rate,
                freq=freq,
                gain=gain,
                chan=chan,
            )
            self._thread = threading.Thread(
                target=self._tx_worker, args=(params,), daemon=True
            )
            self._running = True
            self._thread.start()
            self._log(
                f"TX start: file={file_path} rate={rate} freq={freq} "
                f"gain={gain} chan={chan} repeat={repeat}\n"
            )
            return True

    def stop_tx(self, timeout: float = 5.0) -> bool:
        with self._lock:
            if not self._running:
                return True
            self._stop_event.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self.last_error = TimeoutError("TX stop timed out.")
                self._log("TX stop timed out; worker still running.\n")
                return False
            self._running = False
            self._thread = None
            self.last_end_monotonic = time.monotonic()
        self._log("TX stopped.\n")
        return True

    def _log(self, message: str) -> None:
        _LOG.info(message.rstrip())
        if self._log_callback is not None:
            self._log_callback(message)

    def _tx_worker(self, params: _TxParams) -> None:
        try:
            file_path = Path(params.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"TX file not found: {file_path}")
            samples = np.fromfile(file_path, dtype=np.int16)
            if samples.size < 2:
                raise ValueError("TX file is empty or incomplete.")
            if samples.size % 2 != 0:
                samples = samples[:-1]
            iq = samples.reshape(-1, 2).astype(np.float32)
            complex_samples = (iq[:, 0] + 1j * iq[:, 1]) / 32768.0
            complex_samples = np.ascontiguousarray(
                complex_samples.astype(np.complex64)
            )
            if complex_samples.size == 0:
                raise ValueError("TX file contains no samples.")

            args = _ensure_uhd_frame_sizes(self._args)
            usrp = uhd.usrp.MultiUSRP(args)
            usrp.set_tx_rate(params.rate, params.chan)
            usrp.set_tx_freq(params.freq, params.chan)
            usrp.set_tx_gain(params.gain, params.chan)
            stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
            stream_args.channels = [params.chan]
            tx_stream = usrp.get_tx_stream(stream_args)
            max_samps = tx_stream.get_max_num_samps()
            md = uhd.types.TXMetadata()
            md.start_of_burst = True
            md.end_of_burst = False
            md.has_time_spec = False
            repeat = params.repeat

            while not self._stop_event.is_set():
                for idx in range(0, complex_samples.size, max_samps):
                    if self._stop_event.is_set():
                        break
                    chunk = complex_samples[idx : idx + max_samps]
                    tx_stream.send(chunk, md)
                    if md.start_of_burst:
                        md.start_of_burst = False
                if not repeat:
                    break
            md.end_of_burst = True
            tx_stream.send(np.zeros(0, dtype=np.complex64), md)
        except Exception as exc:
            self.last_error = exc
            self._log(f"TX worker error: {exc}\n")
        finally:
            with self._lock:
                self._running = False
                self._thread = None
                self.last_end_monotonic = time.monotonic()
            self._log("TX worker exit.\n")
