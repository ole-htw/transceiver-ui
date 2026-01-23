"""In-process TX controller for UHD streaming (TX streamer).

Assumptions:
- Input file format: interleaved int16 I/Q samples: I0,Q0,I1,Q1,... (little endian as stored)
- Samples are normalized by /32768.0 to complex64 for fc32 CPU format.

Design goals:
- Cooperative stop via threading.Event
- Clean TX termination via explicit EOB
- Robust send loop that honors partial sends
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import uhd


_LOG = logging.getLogger(__name__)


def _ensure_uhd_frame_sizes(args_str: str, send_frame_size: int = 8000) -> str:
    components = [part.strip() for part in args_str.split(",") if part.strip()]
    if not any(part.startswith("send_frame_size=") for part in components):
        components.append(f"send_frame_size={send_frame_size}")
    return ",".join(components)


def _extract_addr_key(args_str: str) -> str:
    # Keep behavior compatible with your existing singleton logic.
    # If you also use "serial=" etc., extend this extractor accordingly.
    for part in args_str.split(","):
        part = part.strip()
        if part.startswith("addr="):
            return part.split("=", 1)[1].strip() or args_str
    return args_str or "default"


@dataclass(frozen=True)
class _TxParams:
    file_path: str
    repeat: bool
    rate: float
    freq: float
    gain: float
    chan: int

    # Optional device settings
    antenna: Optional[str] = None
    bandwidth: Optional[float] = None

    # Stream formats
    cpu_format: str = "fc32"
    otw_format: str = "sc16"

    # Transport tuning
    send_frame_size: int = 8000

    # Chunking / IO
    max_chunk_samps: Optional[int] = None  # cap to reduce blocking duration
    use_memmap: bool = True

    # Normalization
    scale: float = 32768.0


class TxController:
    _instances: dict[str, "TxController"] = {}
    _instances_lock = threading.Lock()

    def __init__(self, args: str) -> None:
        self._args = args
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self.last_end_monotonic = 0.0
        self.last_error: Optional[BaseException] = None

        self._log_callback: Optional[Callable[[str], None]] = None

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
        antenna: Optional[str] = None,
        bandwidth: Optional[float] = None,
        cpu_format: str = "fc32",
        otw_format: str = "sc16",
        send_frame_size: int = 8000,
        max_chunk_samps: Optional[int] = None,
        use_memmap: bool = True,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> bool:
        with self._lock:
            if self._running:
                self._log("TX start requested but already running.\n")
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
                antenna=antenna,
                bandwidth=bandwidth,
                cpu_format=cpu_format,
                otw_format=otw_format,
                send_frame_size=send_frame_size,
                max_chunk_samps=max_chunk_samps,
                use_memmap=use_memmap,
            )

            t = threading.Thread(target=self._tx_worker, args=(params,), daemon=True)
            self._thread = t
            self._running = True

        t.start()
        self._log(
            f"TX start: file={file_path} rate={rate} freq={freq} gain={gain} "
            f"chan={chan} repeat={repeat} cpu={cpu_format} otw={otw_format}\n"
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
                # Cannot forcibly kill a stuck in-process thread.
                self.last_error = TimeoutError("TX stop timed out; worker still running.")
                self._log("TX stop timed out; worker still running.\n")
                return False

        self._log("TX stopped.\n")
        return True

    def _log(self, message: str) -> None:
        # Avoid holding locks while calling callbacks.
        _LOG.info(message.rstrip(), extra={"ui_forwarded": True})
        cb = self._log_callback
        if cb is not None:
            try:
                cb(message)
            except Exception:
                _LOG.exception("TX log_callback raised an exception (ignored).")

    @staticmethod
    def _open_iq_int16_file(path: Path, *, use_memmap: bool) -> np.ndarray:
        if use_memmap:
            return np.memmap(path, dtype=np.int16, mode="r")
        return np.fromfile(path, dtype=np.int16)

    @staticmethod
    def _convert_i16_to_c64(iq_i16: np.ndarray, scale: float) -> np.ndarray:
        # iq_i16 is shape (2*N,), interleaved I,Q.
        if iq_i16.size % 2 != 0:
            iq_i16 = iq_i16[:-1]
        if iq_i16.size == 0:
            return np.empty(0, dtype=np.complex64)

        # reshape to (N,2) -> float32 -> complex64
        iq_f32 = iq_i16.reshape(-1, 2).astype(np.float32, copy=False)
        c = (iq_f32[:, 0] + 1j * iq_f32[:, 1]) / float(scale)
        return np.ascontiguousarray(c.astype(np.complex64, copy=False))

    @staticmethod
    def _send_all(tx_stream, data: np.ndarray, md: "uhd.types.TXMetadata", stop_event: threading.Event) -> int:
        """Send the entire buffer, honoring partial sends. Returns total sent."""
        total = 0
        # Defensive: avoid busy spinning on repeated 0 sends.
        zero_sends = 0

        while total < len(data) and not stop_event.is_set():
            n = tx_stream.send(data[total:], md)
            if n is None:
                n = 0
            if n <= 0:
                zero_sends += 1
                if zero_sends >= 1000:
                    # ~1s at 1ms sleep; reduces CPU burn and surfaces potential stuck send.
                    time.sleep(0.005)
                    zero_sends = 0
                else:
                    time.sleep(0.001)
                continue

            total += int(n)
            zero_sends = 0

            # SOB only for the very first successful send call.
            if md.start_of_burst:
                md.start_of_burst = False

        return total

    def _tx_worker(self, params: _TxParams) -> None:
        sent_any = False
        try:
            file_path = Path(params.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"TX file not found: {file_path}")

            raw = self._open_iq_int16_file(file_path, use_memmap=params.use_memmap)
            if raw.size < 2:
                raise ValueError("TX file is empty or incomplete (needs at least I/Q).")

            # Number of complex samples in the file
            num_complex = (raw.size // 2)
            if num_complex <= 0:
                raise ValueError("TX file contains no complete I/Q samples.")

            # Build USRP
            args = _ensure_uhd_frame_sizes(self._args, send_frame_size=params.send_frame_size)
            usrp = uhd.usrp.MultiUSRP(args)
            
            usrp.set_master_clock_rate(200e6)

            # Device configuration
            usrp.set_tx_rate(params.rate, params.chan)
            usrp.set_tx_freq(params.freq, params.chan)
            usrp.set_tx_gain(params.gain, params.chan)
            if params.antenna:
                usrp.set_tx_antenna(params.antenna, params.chan)
            if params.bandwidth:
                usrp.set_tx_bandwidth(params.bandwidth, params.chan)

            stream_args = uhd.usrp.StreamArgs(params.cpu_format, params.otw_format)
            stream_args.channels = [params.chan]
            tx_stream = usrp.get_tx_stream(stream_args)

            max_samps = int(tx_stream.get_max_num_samps())
            if max_samps <= 0:
                raise RuntimeError("UHD returned non-positive max num samps for TX stream.")

            if params.max_chunk_samps is not None:
                max_samps = max(1, min(max_samps, int(params.max_chunk_samps)))

            md = uhd.types.TXMetadata()
            md.start_of_burst = True
            md.end_of_burst = False
            md.has_time_spec = False

            repeat = params.repeat

            # Outer loop for repeat mode
            while not self._stop_event.is_set():
                # Iterate through file in chunks of complex samples
                samp_idx = 0
                while samp_idx < num_complex and not self._stop_event.is_set():
                    # Slice interleaved int16: [2*samp_idx : 2*(samp_idx + n)]
                    n = min(max_samps, num_complex - samp_idx)
                    i0 = 2 * samp_idx
                    i1 = 2 * (samp_idx + n)
                    chunk_i16 = np.asarray(raw[i0:i1])  # ensure ndarray view
                    chunk_c64 = self._convert_i16_to_c64(chunk_i16, scale=params.scale)

                    if chunk_c64.size == 0:
                        break

                    sent = self._send_all(tx_stream, chunk_c64, md, self._stop_event)
                    if sent > 0:
                        sent_any = True

                    samp_idx += n

                if not repeat:
                    break

            # Clean end-of-burst (robust, avoid 0-length EOB)
            if sent_any:
                md_eob = uhd.types.TXMetadata()
                md_eob.start_of_burst = False
                md_eob.end_of_burst = True
                md_eob.has_time_spec = False
                tx_stream.send(np.array([0 + 0j], dtype=np.complex64), md_eob)

        except Exception as exc:
            with self._lock:
                self.last_error = exc
            _LOG.exception("TX worker error")
            self._log(f"TX worker error: {exc}\n")

        finally:
            with self._lock:
                self._running = False
                self._thread = None
                self.last_end_monotonic = time.monotonic()
            self._log("TX worker exit.\n")

