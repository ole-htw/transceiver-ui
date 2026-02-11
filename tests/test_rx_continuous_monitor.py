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


def load_module_with_stubbed_uhd():
    fake_uhd = type(sys)("uhd")
    fake_uhd.types = type(sys)("types")
    fake_uhd.types.RXMetadata = FakeRXMetadata
    fake_uhd.types.RXMetadataErrorCode = FakeErrorCodes
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


def test_validate_replay_async_api_handles_uninspectable_method_signature():
    module = load_module_with_stubbed_uhd()
    replay = FakeReplay()

    original_signature = module.inspect.signature

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("no signature found")

    module.inspect.signature = raise_value_error
    try:
        module.validate_replay_async_api(replay)
    finally:
        module.inspect.signature = original_signature


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
