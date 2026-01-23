"""
TX controller using RFNoC Replay Block (DRAM) by default.

Überarbeitete Drop-in-Version:
- Behebt RoutingError, indem Replay->TX-Kette über die im FPGA-Image vorhandenen
  statischen Ketten (SEP->DUC->Radio) und dynamische SEP-Routen aufgebaut wird.
- Nutzt – falls im UHD-Python-Binding verfügbar – uhd.rfnoc.connect_through_blocks(),
  welches automatisch den Pfad über statische Kanten + SEP-Crossbar findet.
- Korrigiert die Kanal-Logik: `chan` wird als *globaler TX-Kanalindex* interpretiert
  (sortiert nach Radio-ID, dann Port), z.B. X310: (Radio#0:0, Radio#0:1, Radio#1:0, Radio#1:1).
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

import numpy as np
import uhd

_LOG = logging.getLogger(__name__)


def _ensure_uhd_frame_sizes(args_str: str, send_frame_size: int = 8000) -> str:
    components = [part.strip() for part in args_str.split(",") if part.strip()]
    if not any(part.startswith("send_frame_size=") for part in components):
        components.append(f"send_frame_size={send_frame_size}")
    return ",".join(components)


def _extract_addr_key(args_str: str) -> str:
    # Keep behavior compatible with existing singleton logic.
    for part in args_str.split(","):
        part = part.strip()
        if part.startswith("addr="):
            return part.split("=", 1)[1].strip() or args_str
    return args_str or "default"


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _lcm(a: int, b: int) -> int:
    if a <= 0 or b <= 0:
        return 0
    return abs(a // _gcd(a, b) * b)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_up(x: int, align: int) -> int:
    if align <= 0:
        return x
    return _ceil_div(x, align) * align


def _align_down(x: int, align: int) -> int:
    if align <= 0:
        return x
    return (x // align) * align


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

    # Transport tuning (host->device upload only)
    send_frame_size: int = 8000

    # Chunking / IO
    max_chunk_samps: Optional[int] = None  # upload chunk size in complex samples/items
    use_memmap: bool = True

    # Input file format
    input_file_format: str = "sc16_i16_interleaved"  # or "fc32_complex64"

    # Normalization (only used when converting i16->c64 for fc32 upload, if needed)
    scale: float = 32768.0

    # Replay settings
    replay_block_hint: str = "Replay"
    radio_block_hint: str = "Radio"
    replay_port: int = 0          # we map a single replay port; radio selection uses chan
    replay_offset_bytes: int = 0  # must be 4kB-aligned; default 0


@dataclass(frozen=True)
class _TxLane:
    """
    Repräsentiert eine TX-"Lane" aus den statischen Verbindungen:
      SEP(tx) -> DUC -> Radio(tx_port)
    oder ggf. SEP(tx) -> Radio(tx_port) (selten).
    """
    radio_blockid: str
    radio_port: int
    duc_blockid: Optional[str]
    duc_port: Optional[int]
    sep_tx_blockid: Optional[str]
    sep_tx_port: Optional[int]


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
        cpu_format: str = "fc32",     # kept for drop-in compatibility (ignored for replay default)
        otw_format: str = "sc16",     # kept for drop-in compatibility (ignored for replay default)
        send_frame_size: int = 8000,
        max_chunk_samps: Optional[int] = None,
        use_memmap: bool = True,
        log_callback: Optional[Callable[[str], None]] = None,
        auto_cpu_format: bool = True,  # kept for compatibility (ignored in replay default)
        input_file_format: str = "sc16_i16_interleaved",
    ) -> bool:
        restart_requested = False
        with self._lock:
            if self._running:
                restart_requested = True
                self._log("TX start requested while running; stopping current TX before restart.\n")

        if restart_requested:
            if not self.stop_tx(timeout=5.0):
                self._log("TX restart failed; controller still running.\n")
                return False

        with self._lock:
            if self._running:
                self._log("TX restart failed; controller still running.\n")
                return False

            self._stop_event.clear()
            self.last_error = None
            self._log_callback = log_callback

            if cpu_format.lower() != "fc32" or otw_format.lower() != "sc16":
                self._log(
                    "TX: Hinweis: cpu_format/otw_format werden im Replay-Standardpfad nicht für Live-Streaming verwendet.\n"
                )

            params = _TxParams(
                file_path=file_path,
                repeat=repeat,
                rate=rate,
                freq=freq,
                gain=gain,
                chan=chan,
                antenna=antenna,
                bandwidth=bandwidth,
                send_frame_size=send_frame_size,
                max_chunk_samps=max_chunk_samps,
                use_memmap=use_memmap,
                input_file_format=input_file_format,
                replay_port=0,
                replay_offset_bytes=0,
            )

            t = threading.Thread(target=self._tx_worker_replay, args=(params,), daemon=True)
            self._thread = t
            self._running = True

        t.start()
        self._log(
            f"TX start (Replay): file={file_path} rate={rate} freq={freq} gain={gain} "
            f"chan={chan} repeat={repeat} input={params.input_file_format}\n"
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
                self.last_error = TimeoutError("TX stop timed out; worker still running.")
                self._log("TX stop timed out; worker still running.\n")
                return False

        self._log("TX stopped.\n")
        return True

    def _log(self, message: str) -> None:
        _LOG.info(message.rstrip(), extra={"ui_forwarded": True})
        cb = self._log_callback
        if cb is not None:
            try:
                cb(message)
            except Exception:
                _LOG.exception("TX log_callback raised an exception (ignored).")

    # ----------------------------
    # File open helpers
    # ----------------------------
    @staticmethod
    def _open_sc16_interleaved_i16(path: Path, *, use_memmap: bool) -> np.ndarray:
        if use_memmap:
            return np.memmap(path, dtype=np.int16, mode="r")
        return np.fromfile(path, dtype=np.int16)

    @staticmethod
    def _open_fc32_complex64(path: Path, *, use_memmap: bool) -> np.ndarray:
        if use_memmap:
            return np.memmap(path, dtype=np.complex64, mode="r")
        return np.fromfile(path, dtype=np.complex64)

    @staticmethod
    def _convert_i16_to_c64(iq_i16: np.ndarray, scale: float) -> np.ndarray:
        if iq_i16.size % 2 != 0:
            iq_i16 = iq_i16[:-1]
        if iq_i16.size == 0:
            return np.empty(0, dtype=np.complex64)

        iq_f32 = iq_i16.reshape(-1, 2).astype(np.float32, copy=False)
        c = (iq_f32[:, 0] + 1j * iq_f32[:, 1]) / float(scale)
        return np.ascontiguousarray(c.astype(np.complex64, copy=False))

    # ----------------------------
    # Stream send helpers
    # ----------------------------
    @staticmethod
    def _send_with_optional_timeout(tx_stream, data: np.ndarray, md: "uhd.types.TXMetadata", timeout_s: Optional[float]) -> int:
        # UHD Python bindings sometimes expose send(data, md) and sometimes send(data, md, timeout)
        try:
            if timeout_s is None:
                n = tx_stream.send(data, md)
            else:
                n = tx_stream.send(data, md, timeout_s)
        except TypeError:
            n = tx_stream.send(data, md)
        if n is None:
            return 0
        return int(n)

    @classmethod
    def _send_all(cls, tx_stream, data: np.ndarray, md: "uhd.types.TXMetadata", stop_event: threading.Event) -> int:
        """Send entire buffer, honoring partial sends. Returns total sent (items)."""
        total = 0
        zero_sends = 0

        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)

        # Use a modest timeout (prevents indefinite blocking if stop is requested)
        timeout_s = 0.2

        while total < len(data) and not stop_event.is_set():
            n = cls._send_with_optional_timeout(tx_stream, data[total:], md, timeout_s)
            if n <= 0:
                zero_sends += 1
                # Backoff a bit; we're uploading, not hard real-time.
                time.sleep(0.002 if zero_sends < 50 else 0.01)
                continue

            total += n
            zero_sends = 0

            if md.start_of_burst:
                md.start_of_burst = False

        return total

    # ----------------------------
    # RFNoC graph helpers (Routing/Channel map)
    # ----------------------------
    @staticmethod
    def _as_block_id(maybe_str: str):
        """
        In Python-Bindings ist block_id_t häufig direkt konstruierbar.
        Falls nicht, akzeptiert graph.connect() i.d.R. auch Strings.
        """
        try:
            return uhd.rfnoc.block_id_t(maybe_str)  # type: ignore[attr-defined]
        except Exception:
            return maybe_str

    @staticmethod
    def _static_edges(graph) -> List[object]:
        try:
            return list(graph.enumerate_static_connections())
        except Exception:
            return []

    @staticmethod
    def _edge_get(edge: object, name: str, default=None):
        if hasattr(edge, name):
            try:
                return getattr(edge, name)
            except Exception:
                return default
        return default

    @classmethod
    def _extract_tx_lanes(cls, graph) -> List[_TxLane]:
        """
        Liest aus enumerate_static_connections() die TX-Lanes heraus.
        Erwartete typische statische Ketten (z.B. X3xx Default-Image):
          SEP#i:0 ==> DUC#j:k ==> Radio#r:p
        """
        edges = cls._static_edges(graph)

        # Index nach (dst_blockid, dst_port) für upstream lookup
        by_dst: Dict[Tuple[str, int], List[object]] = {}
        for e in edges:
            dst_blk = cls._edge_get(e, "dst_blockid", "")
            dst_port = int(cls._edge_get(e, "dst_port", 0) or 0)
            by_dst.setdefault((str(dst_blk), dst_port), []).append(e)

        lanes: List[_TxLane] = []

        # 1) DUC -> Radio Kanten sammeln (TX-Input am Radio)
        for e in edges:
            src_blk = str(cls._edge_get(e, "src_blockid", ""))
            dst_blk = str(cls._edge_get(e, "dst_blockid", ""))
            src_port = int(cls._edge_get(e, "src_port", 0) or 0)
            dst_port = int(cls._edge_get(e, "dst_port", 0) or 0)

            if "/Radio#" in dst_blk and "/DUC#" in src_blk:
                duc_blk = src_blk
                duc_port = src_port
                radio_blk = dst_blk
                radio_port = dst_port

                # Upstream: SEP -> DUC (falls vorhanden)
                sep_blk = None
                sep_port = None
                ups = by_dst.get((duc_blk, duc_port), [])
                for u in ups:
                    u_src_blk = str(cls._edge_get(u, "src_blockid", ""))
                    u_src_port = int(cls._edge_get(u, "src_port", 0) or 0)
                    if "/SEP#" in u_src_blk:
                        sep_blk = u_src_blk
                        sep_port = u_src_port
                        break

                lanes.append(
                    _TxLane(
                        radio_blockid=radio_blk,
                        radio_port=radio_port,
                        duc_blockid=duc_blk,
                        duc_port=duc_port,
                        sep_tx_blockid=sep_blk,
                        sep_tx_port=sep_port,
                    )
                )

        # 2) Fallback: direkte SEP -> Radio Kanten (selten)
        if not lanes:
            for e in edges:
                src_blk = str(cls._edge_get(e, "src_blockid", ""))
                dst_blk = str(cls._edge_get(e, "dst_blockid", ""))
                src_port = int(cls._edge_get(e, "src_port", 0) or 0)
                dst_port = int(cls._edge_get(e, "dst_port", 0) or 0)
                if "/Radio#" in dst_blk and "/SEP#" in src_blk:
                    lanes.append(
                        _TxLane(
                            radio_blockid=dst_blk,
                            radio_port=dst_port,
                            duc_blockid=None,
                            duc_port=None,
                            sep_tx_blockid=src_blk,
                            sep_tx_port=src_port,
                        )
                    )

        # stabile Ordnung: erst Radio-ID, dann Port
        lanes.sort(key=lambda x: (x.radio_blockid, x.radio_port))
        return lanes

    @classmethod
    def _resolve_radio_for_chan(cls, graph, chan: int) -> Tuple[object, int, int, Optional[_TxLane]]:
        """
        Liefert (radio_block_id, radio_port, radio_chan_index, lane_info).
        - radio_port ist der Port am Radio-Block.
        - radio_chan_index ist der Index, den radio.set_tx_* typischerweise erwartet (meist == radio_port).
        """
        # Mapping aus echten Radio-Block-Objekten
        radio_ids = graph.find_blocks("Radio")
        radio_by_str = {str(b): b for b in radio_ids}

        lanes = cls._extract_tx_lanes(graph)
        if lanes and 0 <= chan < len(lanes):
            lane = lanes[chan]
            rb = radio_by_str.get(lane.radio_blockid)
            if rb is not None:
                return (rb, lane.radio_port, lane.radio_port, lane)

        # Fallback auf altes Verhalten: "Radio#<chan>" oder erstes Radio
        ids = radio_ids
        if not ids:
            raise RuntimeError("No RFNoC Radio block found in FPGA image.")
        wanted = None
        want_token = f"Radio#{chan}"
        for bid in ids:
            if want_token in str(bid):
                wanted = bid
                break
        # Port 0, channel index 0
        return (wanted or ids[0], 0, 0, None)

    @staticmethod
    def _pick_replay_block_id(graph) -> object:
        ids = graph.find_blocks("Replay")
        if not ids:
            raise RuntimeError("No RFNoC Replay block found in FPGA image.")
        return ids[0]

    @classmethod
    def _fallback_connect_replay_to_radio(
        cls,
        graph,
        *,
        replay_blockid: object,
        replay_port: int,
        radio_blockid: object,
        radio_port: int,
        lane: Optional[_TxLane],
        log: Callable[[str], None],
    ) -> None:
        """
        Falls uhd.rfnoc.connect_through_blocks() im Python-Binding nicht verfügbar ist,
        bauen wir den Pfad manuell:

          Replay(out) -> SEP(replay) -> SEP(tx) -> DUC -> Radio

        Dabei werden die statischen Kanten durch connect() "aktiviert", und
        die SEP->SEP Verbindung ist die dynamische Crossbar-Route.
        """
        edges = cls._static_edges(graph)

        replay_id_str = str(replay_blockid)
        radio_id_str = str(radio_blockid)

        # Replay-Port hängt typischerweise an einem SEP:
        # Suche statisch: Replay#0:p ==> SEP#k:0  (Replay->SEP, output Richtung)
        sep_replay = None
        sep_replay_port = None
        for e in edges:
            src_blk = str(cls._edge_get(e, "src_blockid", ""))
            dst_blk = str(cls._edge_get(e, "dst_blockid", ""))
            src_p = int(cls._edge_get(e, "src_port", 0) or 0)
            dst_p = int(cls._edge_get(e, "dst_port", 0) or 0)
            if src_blk == replay_id_str and src_p == int(replay_port) and "/SEP#" in dst_blk:
                sep_replay = dst_blk
                sep_replay_port = dst_p
                break

        if sep_replay is None:
            raise RuntimeError(
                f"Kann Replay-Port {replay_port} nicht an SEP anbinden (keine statische Replay->SEP Kante gefunden)."
            )

        # TX-SEP aus Lane oder aus statischen Kanten rekonstruieren
        sep_tx = None
        sep_tx_port = None
        duc_id = None
        duc_port = None

        if lane is not None and lane.sep_tx_blockid is not None and lane.sep_tx_port is not None:
            sep_tx = lane.sep_tx_blockid
            sep_tx_port = lane.sep_tx_port
            duc_id = lane.duc_blockid
            duc_port = lane.duc_port
        else:
            # Versuche: DUC->Radio Kante für dieses Radio/Port finden, dann SEP->DUC upstream.
            for e in edges:
                src_blk = str(cls._edge_get(e, "src_blockid", ""))
                dst_blk = str(cls._edge_get(e, "dst_blockid", ""))
                src_p = int(cls._edge_get(e, "src_port", 0) or 0)
                dst_p = int(cls._edge_get(e, "dst_port", 0) or 0)
                if dst_blk == radio_id_str and dst_p == int(radio_port) and "/DUC#" in src_blk:
                    duc_id = src_blk
                    duc_port = src_p
                    break

            if duc_id is not None and duc_port is not None:
                for e in edges:
                    src_blk = str(cls._edge_get(e, "src_blockid", ""))
                    dst_blk = str(cls._edge_get(e, "dst_blockid", ""))
                    src_p = int(cls._edge_get(e, "src_port", 0) or 0)
                    dst_p = int(cls._edge_get(e, "dst_port", 0) or 0)
                    if dst_blk == duc_id and dst_p == int(duc_port) and "/SEP#" in src_blk:
                        sep_tx = src_blk
                        sep_tx_port = src_p
                        break

        if sep_tx is None:
            raise RuntimeError(
                "Kann TX-SEP nicht bestimmen (weder Lane-Info noch SEP->DUC Kante gefunden)."
            )

        # 1) Aktivieren: Replay -> SEP(replay) (statisch)
        graph.connect(replay_blockid, int(replay_port), cls._as_block_id(sep_replay), int(sep_replay_port))

        # 2) Dynamische Route: SEP(replay) -> SEP(tx)
        graph.connect(cls._as_block_id(sep_replay), int(sep_replay_port), cls._as_block_id(sep_tx), int(sep_tx_port))

        # 3) Aktivieren: SEP(tx) -> DUC (statisch), falls vorhanden
        if duc_id is not None and duc_port is not None:
            graph.connect(cls._as_block_id(sep_tx), int(sep_tx_port), cls._as_block_id(duc_id), int(duc_port))
            # 4) Aktivieren: DUC -> Radio (statisch)
            graph.connect(cls._as_block_id(duc_id), int(duc_port), radio_blockid, int(radio_port))
        else:
            # Seltenes Layout: SEP(tx) -> Radio (statisch)
            graph.connect(cls._as_block_id(sep_tx), int(sep_tx_port), radio_blockid, int(radio_port))

        log(
            f"TX (Replay): Fallback-Routing aufgebaut: {replay_id_str}:{replay_port} -> "
            f"{sep_replay}:{sep_replay_port} -> {sep_tx}:{sep_tx_port} -> {radio_id_str}:{radio_port}\n"
        )

    # ----------------------------
    # Worker (Replay-based TX)
    # ----------------------------
    def _tx_worker_replay(self, params: _TxParams) -> None:
        sent_any = False
        graph = None
        try:
            file_path = Path(params.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"TX file not found: {file_path}")

            # Device args (transport tuning for upload)
            args = _ensure_uhd_frame_sizes(self._args, send_frame_size=params.send_frame_size)

            # Create RFNoC session/graph
            graph = uhd.rfnoc.RfnocGraph(args)

            replay_id = self._pick_replay_block_id(graph)

            # Resolve global channel index -> (Radio block, Radio port, chan idx within radio)
            radio_bid, radio_port, radio_chan_idx, lane = self._resolve_radio_for_chan(graph, params.chan)

            replay_port = int(params.replay_port)

            # Get block controllers
            replay = uhd.rfnoc.ReplayBlockControl(graph.get_block(replay_id))
            radio  = uhd.rfnoc.RadioControl(graph.get_block(radio_bid))
            
            



            # Create TX streamer (Host -> Replay record input)
            if params.input_file_format == "sc16_i16_interleaved":
                streamer_cpu = "sc16"
                streamer_otw = "sc16"
            elif params.input_file_format == "fc32_complex64":
                streamer_cpu = "fc32"
                streamer_otw = "sc16"
            else:
                raise ValueError(f"Unsupported input_file_format={params.input_file_format}")

            stream_args = uhd.usrp.StreamArgs(streamer_cpu, streamer_otw)
            tx_stream = graph.create_tx_streamer(1, stream_args)

            # Connect Host TX streamer -> Replay(in)
            graph.connect(tx_stream, 0, replay_id, replay_port)

            # Connect Replay(out) -> Radio(in) via proper path
            connect_through = getattr(uhd.rfnoc, "connect_through_blocks", None)
            if callable(connect_through):
                # connect_through_blocks() findet statische Ketten und erzeugt dynamische SEP-Routen automatisch.
                # Damit funktioniert auch X3xx Default-Image: Replay -> SEP -> SEP -> DUC -> Radio.
                connect_through(graph, replay_id, replay_port, radio_bid, int(radio_port))
                self._log(
                    f"TX (Replay): connect_through_blocks: {str(replay_id)}:{replay_port} -> {str(radio_bid)}:{radio_port}\n"
                )
            else:
                # Fallback: manuelles Routing über SEP(s)
                self._fallback_connect_replay_to_radio(
                    graph,
                    replay_blockid=replay_id,
                    replay_port=replay_port,
                    radio_blockid=radio_bid,
                    radio_port=int(radio_port),
                    lane=lane,
                    log=self._log,
                )

            # Commit graph (property propagation and checks)
            graph.commit()

            # Configure Radio (TX path)
            coerced_rate = float(radio.set_rate(params.rate))
            if abs(coerced_rate - params.rate) > 1e-6:
                self._log(f"TX (Replay): rate coerced {params.rate} -> {coerced_rate}\n")

            def _log_if_coerced(label: str, desired: object, actual: object, tol: float = 1e-6) -> None:
                if isinstance(desired, (int, float)) and isinstance(actual, (int, float)):
                    if abs(actual - desired) > tol:
                        self._log(f"TX (Replay): {label} coerced {desired} -> {actual}\n")
                elif actual != desired:
                    self._log(f"TX (Replay): {label} coerced {desired} -> {actual}\n")

            # TX-spezifische Settings bleiben pro chan
            radio.set_tx_frequency(params.freq, radio_chan_idx)
            if hasattr(radio, "get_tx_frequency"):
                try:
                    actual_freq = float(radio.get_tx_frequency(radio_chan_idx))
                except Exception:
                    actual_freq = None
                if actual_freq is not None:
                    _log_if_coerced("freq", params.freq, actual_freq)
            radio.set_tx_gain(params.gain, radio_chan_idx)
            if hasattr(radio, "get_tx_gain"):
                try:
                    actual_gain = float(radio.get_tx_gain(radio_chan_idx))
                except Exception:
                    actual_gain = None
                if actual_gain is not None:
                    _log_if_coerced("gain", params.gain, actual_gain)
            if params.antenna:
                radio.set_tx_antenna(params.antenna, radio_chan_idx)
                if hasattr(radio, "get_tx_antenna"):
                    try:
                        actual_antenna = radio.get_tx_antenna(radio_chan_idx)
                    except Exception:
                        actual_antenna = None
                    if actual_antenna is not None:
                        _log_if_coerced("antenna", params.antenna, actual_antenna)
            if params.bandwidth:
                radio.set_tx_bandwidth(params.bandwidth, radio_chan_idx)
                if hasattr(radio, "get_tx_bandwidth"):
                    try:
                        actual_bw = float(radio.get_tx_bandwidth(radio_chan_idx))
                    except Exception:
                        actual_bw = None
                    if actual_bw is not None:
                        _log_if_coerced("bandwidth", params.bandwidth, actual_bw)



            # DRAM alignment requirements
            word_size = int(replay.get_word_size())
            rec_item_size = int(replay.get_record_item_size(replay_port))
            play_item_size = int(replay.get_play_item_size(replay_port))

            # offset: enforce 4kB boundary and word alignment
            offset = int(params.replay_offset_bytes)
            offset = _align_up(offset, 4096)
            offset = _align_up(offset, word_size)

            # Determine effective bytes from file (truncate to full items)
            if params.input_file_format == "sc16_i16_interleaved":
                raw_i16 = self._open_sc16_interleaved_i16(file_path, use_memmap=params.use_memmap)
                if raw_i16.size < 2:
                    raise ValueError("TX file is empty or incomplete (needs at least I/Q).")
                if raw_i16.size % 2 != 0:
                    raw_i16 = raw_i16[:-1]
                # each complex sample is 4 bytes for sc16
                num_items = int(raw_i16.size // 2)
                file_bytes_eff = num_items * 4
            else:
                raw_c64 = self._open_fc32_complex64(file_path, use_memmap=params.use_memmap)
                if raw_c64.size <= 0:
                    raise ValueError("TX file contains no samples (complex64).")
                num_items = int(raw_c64.size)
                file_bytes_eff = num_items * 8  # complex64

            # Size alignment: must satisfy word size + playback item size (and record item size to be safe)
            align_bytes = _lcm(word_size, _lcm(rec_item_size, play_item_size))
            if align_bytes <= 0:
                align_bytes = max(word_size, rec_item_size, play_item_size, 1)

            size_aligned = _align_up(file_bytes_eff, align_bytes)
            if size_aligned <= 0:
                raise ValueError("Computed replay size is zero; nothing to transmit.")

            # Ensure DRAM capacity
            mem_size = int(replay.get_mem_size())
            if offset + size_aligned > mem_size:
                raise MemoryError(
                    f"Replay DRAM too small: need {offset + size_aligned} bytes, have {mem_size} bytes."
                )

            # Configure record region
            replay.record(offset, size_aligned, replay_port)

            # Upload (Host -> Replay)
            md = uhd.types.TXMetadata()
            md.start_of_burst = True
            md.end_of_burst = False
            md.has_time_spec = False

            # Determine upload chunk size in items
            chunk_items = int(params.max_chunk_samps) if params.max_chunk_samps else 250_000
            chunk_items = max(1, chunk_items)

            total_items_needed = size_aligned // rec_item_size
            items_sent_total = 0

            if params.input_file_format == "sc16_i16_interleaved":
                raw_i32 = raw_i16.view(np.int32)  # packed I16/Q16
                if raw_i32.size != num_items:
                    raw_i32 = raw_i32[:num_items]

                # Send file content
                idx = 0
                while idx < raw_i32.size and not self._stop_event.is_set():
                    n = min(chunk_items, raw_i32.size - idx)
                    chunk = raw_i32[idx : idx + n]
                    sent = self._send_all(tx_stream, chunk, md, self._stop_event)
                    if sent > 0:
                        sent_any = True
                        items_sent_total += sent
                    if sent < n:
                        idx += sent
                        break
                    idx += n

                # Pad to size_aligned (minimal pad)
                pad_items = total_items_needed - items_sent_total
                if pad_items > 0 and not self._stop_event.is_set():
                    while pad_items > 0 and not self._stop_event.is_set():
                        n = min(chunk_items, pad_items)
                        z = np.zeros(n, dtype=np.int32)
                        sent = self._send_all(tx_stream, z, md, self._stop_event)
                        if sent > 0:
                            sent_any = True
                            items_sent_total += sent
                            pad_items -= sent
                        if sent < n:
                            break

            else:
                # fc32 upload
                raw_c64 = raw_c64.astype(np.complex64, copy=False)
                if not raw_c64.flags["C_CONTIGUOUS"]:
                    raw_c64 = np.ascontiguousarray(raw_c64)

                idx = 0
                while idx < raw_c64.size and not self._stop_event.is_set():
                    n = min(chunk_items, raw_c64.size - idx)
                    chunk = raw_c64[idx : idx + n]
                    sent = self._send_all(tx_stream, chunk, md, self._stop_event)
                    if sent > 0:
                        sent_any = True
                        items_sent_total += sent
                    if sent < n:
                        idx += sent
                        break
                    idx += n

                pad_items = total_items_needed - items_sent_total
                if pad_items > 0 and not self._stop_event.is_set():
                    while pad_items > 0 and not self._stop_event.is_set():
                        n = min(chunk_items, pad_items)
                        z = np.zeros(n, dtype=np.complex64)
                        sent = self._send_all(tx_stream, z, md, self._stop_event)
                        if sent > 0:
                            sent_any = True
                            items_sent_total += sent
                            pad_items -= sent
                        if sent < n:
                            break

            # Mark end-of-burst for the uploader stream
            md.end_of_burst = True
            try:
                if params.input_file_format == "sc16_i16_interleaved":
                    self._send_with_optional_timeout(tx_stream, np.zeros(0, dtype=np.int32), md, 0.1)
                else:
                    self._send_with_optional_timeout(tx_stream, np.zeros(0, dtype=np.complex64), md, 0.1)
            except Exception:
                pass

            if self._stop_event.is_set():
                self._log("TX (Replay): stop requested during upload; exiting.\n")
                return

            # Wait until Replay reports buffer fullness (best effort)
            t0 = time.time()
            while time.time() - t0 < 2.0:
                fullness = int(replay.get_record_fullness(replay_port))
                if fullness >= size_aligned:
                    break
                time.sleep(0.02)

            fullness = int(replay.get_record_fullness(replay_port))
            self._log(f"TX (Replay): upload done. record_fullness={fullness} bytes\n")

            # Start playback from DRAM
            start_time = uhd.types.TimeSpec(0.0)  # immediate
            replay.play(offset, size_aligned, replay_port, start_time, bool(params.repeat))
            self._log(
                f"TX (Replay): playback started. repeat={params.repeat} size={size_aligned} bytes "
                f"radio={str(radio_bid)}:{radio_port}\n"
            )

            # Run until stop (repeat=True) or until estimated end (repeat=False)
            if params.repeat:
                while not self._stop_event.is_set():
                    time.sleep(0.05)
                self._log("TX (Replay): stopping playback...\n")
                replay.stop(replay_port)
            else:
                rate_used = coerced_rate if "coerced_rate" in locals() else params.rate
                items_to_play = size_aligned // play_item_size
                est_seconds = max(0.0, items_to_play / float(rate_used))
                deadline = time.time() + est_seconds + 0.5
                while time.time() < deadline and not self._stop_event.is_set():
                    time.sleep(0.05)
                replay.stop(replay_port)

        except Exception as exc:
            with self._lock:
                self.last_error = exc
            _LOG.exception("TX worker (Replay) error")
            self._log(f"TX worker (Replay) error: {exc}\n")

        finally:
            try:
                if graph is not None:
                    graph.release()
            except Exception:
                pass

            with self._lock:
                self._running = False
                self._thread = None
                self.last_end_monotonic = time.monotonic()
            self._log("TX worker (Replay) exit.\n")
