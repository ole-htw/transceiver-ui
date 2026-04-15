import sys
import types

import numpy as np


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

from transceiver.__main__ import (  # noqa: E402
    QtCore,
    _apply_manual_echo_slots,
    _event_key_to_echo_slot,
)


def _set_test_keycodes() -> None:
    qt = QtCore.Qt
    qt.Key_1 = 49
    qt.Key_2 = 50
    qt.Key_3 = 51
    qt.Key_4 = 52
    qt.Key_5 = 53
    qt.Key_9 = 57
    qt.KeypadModifier = 0


def test_event_key_to_echo_slot_maps_digit_keys() -> None:
    _set_test_keycodes()
    assert _event_key_to_echo_slot(49) == 1
    assert _event_key_to_echo_slot(53) == 5
    assert _event_key_to_echo_slot(57) is None


def test_apply_manual_echo_slots_reorders_echoes_by_requested_slot() -> None:
    lags = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
    echo_indices = [1, 2, 3, 4, 5]
    manual_lags = {
        "echo_2": 50,
        "echo_1": 20,
    }
    reordered = _apply_manual_echo_slots(
        lags=lags,
        echo_indices=echo_indices,
        manual_lags=manual_lags,
        slots=5,
    )
    assert reordered[:4] == [2, 5, 1, 3]
