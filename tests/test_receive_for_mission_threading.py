from __future__ import annotations

import sys
import threading
import time
import types


def _install_pyqtgraph_stub() -> None:
    if "pyqtgraph" in sys.modules:
        return

    qt_mod = types.ModuleType("pyqtgraph.Qt")

    class _PgDummy:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, _name: str):
            return self

    class _QtWidgetsDummy:
        QDialog = type("QDialog", (), {})

        def __getattr__(self, name: str):
            return type(name, (), {})

    qt_mod.QtCore = _PgDummy()
    qt_mod.QtGui = _PgDummy()
    qt_mod.QtWidgets = _QtWidgetsDummy()

    pg_mod = types.ModuleType("pyqtgraph")
    pg_mod.Qt = qt_mod
    pg_mod.exporters = types.ModuleType("pyqtgraph.exporters")
    pg_mod.ScatterPlotItem = type("ScatterPlotItem", (), {})
    pg_mod.PlotWidget = type("PlotWidget", (), {})
    pg_mod.InfiniteLine = type("InfiniteLine", (), {})

    def _pg_getattr(_name: str):
        return _PgDummy()

    pg_mod.__getattr__ = _pg_getattr  # type: ignore[attr-defined]

    sys.modules["pyqtgraph"] = pg_mod
    sys.modules["pyqtgraph.Qt"] = qt_mod
    sys.modules["pyqtgraph.exporters"] = pg_mod.exporters


_install_pyqtgraph_stub()
sys.modules.setdefault("uhd", types.ModuleType("uhd"))

from transceiver.__main__ import TransceiverUI


class _DummyWidget:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def configure(self, **kwargs) -> None:
        self.calls.append(kwargs)


def test_receive_for_mission_uses_worker_path_and_waits_for_result() -> None:
    ui = object.__new__(TransceiverUI)
    ui._cmd_running = False
    ui.rx_stop = _DummyWidget()
    ui.rx_button = _DummyWidget()
    ui._ui = lambda callback: callback()
    process_queue_calls: list[str] = []
    ui._process_queue = lambda: process_queue_calls.append("called")
    ui._build_receive_arg_list_for_worker = (
        lambda *, output_file: (["-a", "type=b200"], 2, 2_000_000.0)
    )

    captured: dict[str, object] = {}

    def _fake_start_rx_worker(
        *,
        arg_list,
        channels,
        rate,
        point_context=None,
        on_complete=None,
    ):
        captured["arg_list"] = arg_list
        captured["channels"] = channels
        captured["rate"] = rate
        captured["point_context"] = point_context
        captured["caller_thread"] = threading.current_thread().name

        def _runner():
            time.sleep(0.05)
            if on_complete is not None:
                on_complete({"ok": True, "output_file": "signals/rx/mission/demo.bin"})

        worker = threading.Thread(target=_runner, daemon=True, name="rx-worker-test")
        worker.start()
        return worker

    ui._start_rx_worker = _fake_start_rx_worker

    started = time.perf_counter()
    result = TransceiverUI.receive_for_mission(
        ui,
        output_file="signals/rx/mission/demo.bin",
        point_context=types.SimpleNamespace(global_index=7),
    )
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.045
    assert captured["arg_list"] == ["-a", "type=b200"]
    assert captured["channels"] == 2
    assert captured["rate"] == 2_000_000.0
    assert process_queue_calls == ["called"]
    assert result == {"ok": True, "output_file": "signals/rx/mission/demo.bin"}
    assert ui.rx_stop.calls == [{"state": "normal"}]
    assert ui.rx_button.calls == [{"state": "disabled"}]
