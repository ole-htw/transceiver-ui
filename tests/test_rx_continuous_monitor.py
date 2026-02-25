import importlib
import sys
import threading
import time


class FakeErrorCodes:
    none = "none"
    overflow = "overflow"
    timeout = "timeout"


class FakeRXMetadata:
    def __init__(self, error_code=FakeErrorCodes.none):
        self.error_code = error_code

    def strerror(self):
        return str(self.error_code)


class FakeReplay:
    def __init__(self, fullness=0, metadata=None):
        self.fullness = fullness
        self.metadata = list(metadata or [])
        self.record_restarts = 0

    def get_record_fullness(self, port):
        return self.fullness

    def record_restart(self, port):
        self.record_restarts += 1

    def get_record_async_metadata(self, timeout=0.1):
        if self.metadata:
            return self.metadata.pop(0)
        return None


class FakeReplayOldSignature(FakeReplay):
    def get_record_async_metadata(self, md, timeout=0.1):
        return False


class FakeEdge:
    def __init__(self, dst_blockid, dst_port):
        self.dst_blockid = dst_blockid
        self.dst_port = dst_port


class FakeBlockID:
    def __init__(self, blockid):
        self.blockid = blockid

    def get_block_name(self):
        return self.blockid.split("#", 1)[0]


class FakeDdcController:
    def __init__(self, ddc_block):
        self.ddc_block = ddc_block

    def set_output_rate(self, rate, chan):
        self.ddc_block.set_output_rate_calls.append((rate, chan))
        return self.ddc_block.return_rate


class FakeDdcBlock:
    def __init__(self, return_rate):
        self.return_rate = return_rate
        self.set_output_rate_calls = []


class FakeRadio:
    def __init__(self, uid):
        self.uid = uid
        self.rate_calls = []
        self.freq_calls = []
        self.gain_calls = []
        self.antenna_calls = []

    def get_unique_id(self):
        return self.uid

    def set_rx_antenna(self, antenna, chan):
        self.antenna_calls.append((antenna, chan))

    def set_rx_frequency(self, freq, chan):
        self.freq_calls.append((freq, chan))

    def set_rx_gain(self, gain, chan):
        self.gain_calls.append((gain, chan))

    def get_rx_frequency(self, chan):
        return self.freq_calls[-1][0]

    def get_rx_gain(self, chan):
        return self.gain_calls[-1][0]

    def get_rx_antenna(self, chan):
        return self.antenna_calls[-1][0]

    def set_rate(self, rate):
        self.rate_calls.append(rate)
        return rate


def load_module_with_stubbed_uhd():
    fake_uhd = type(sys)("uhd")
    fake_uhd.types = type(sys)("types")
    fake_uhd.types.RXMetadata = FakeRXMetadata
    fake_uhd.types.RXMetadataErrorCode = FakeErrorCodes
    fake_uhd.rfnoc = type(sys)("rfnoc")
    fake_uhd.rfnoc.BlockID = FakeBlockID
    sys.modules["uhd"] = fake_uhd
    module = importlib.import_module("transceiver.helpers.rx_continous")
    return importlib.reload(module)


def test_validate_replay_async_api_accepts_timeout_only_signature():
    module = load_module_with_stubbed_uhd()
    replay = FakeReplay()
    module.validate_replay_async_api(replay)


def test_validate_replay_async_api_rejects_old_signature():
    module = load_module_with_stubbed_uhd()
    replay = FakeReplayOldSignature()
    try:
        module.validate_replay_async_api(replay)
    except RuntimeError as exc:
        assert "get_record_async_metadata" in str(exc)
    else:
        raise AssertionError("Expected validate_replay_async_api to raise RuntimeError")


def test_record_monitor_handles_metadata_and_restart():
    module = load_module_with_stubbed_uhd()
    metadata = [FakeRXMetadata(), FakeRXMetadata(FakeErrorCodes.overflow)]
    replay = FakeReplay(fullness=95, metadata=metadata)
    stop_evt = threading.Event()
    lock = threading.Lock()
    thread = threading.Thread(
        target=module.record_monitor,
        args=(stop_evt, replay, 0, 100, 10, lock),
        daemon=True,
    )
    thread.start()
    time.sleep(0.05)
    stop_evt.set()
    thread.join(timeout=1)
    assert replay.record_restarts >= 1


def test_connect_radios_uses_ddc_port_mapping_when_ports_differ():
    module = load_module_with_stubbed_uhd()

    ddc_block = FakeDdcBlock(return_rate=2.5e6)

    class FakeGraph:
        def get_block(self, blockid):
            assert blockid == "DDC#0"
            return ddc_block

    class FakeReplayBlock:
        def get_unique_id(self):
            return "Replay#0"

    radio = FakeRadio("Radio#0")

    module.uhd.rfnoc.connect_through_blocks = lambda *_args: [FakeEdge("DDC#0", 3)]
    module.uhd.rfnoc.DdcBlockControl = FakeDdcController

    rate = module.connect_radios(
        FakeGraph(),
        FakeReplayBlock(),
        [(radio, 0)],
        freqs=[100e6],
        gains=[10],
        antennas=["RX2"],
        rate=2.5e6,
        memory_only=True,
    )

    assert rate == 2.5e6
    assert ddc_block.set_output_rate_calls == [(2.5e6, 3)]
    assert radio.rate_calls == []


def test_connect_radios_non_ddc_path_rejects_invalid_rate_return():
    module = load_module_with_stubbed_uhd()

    class FakeReplayBlock:
        def get_unique_id(self):
            return "Replay#0"

    class InvalidRateRadio(FakeRadio):
        def set_rate(self, rate):
            self.rate_calls.append(rate)
            return float("nan")

    radio = InvalidRateRadio("Radio#1")
    module.uhd.rfnoc.connect_through_blocks = lambda *_args: []
    module.uhd.rfnoc.DdcBlockControl = FakeDdcController

    try:
        module.connect_radios(
            graph=object(),
            replay=FakeReplayBlock(),
            radio_chan_pairs=[(radio, 0)],
            freqs=[100e6],
            gains=[10],
            antennas=["RX2"],
            rate=1e6,
            memory_only=True,
        )
    except RuntimeError as exc:
        assert "Invalid configured rate" in str(exc)
    else:
        raise AssertionError("Expected non-DDC invalid rate to raise RuntimeError")
