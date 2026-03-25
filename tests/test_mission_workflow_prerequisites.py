from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transceiver import mission_workflow_ui
from transceiver.mission_workflow_ui import MissionWorkflowWindow


class _DummyEntry:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _DummyWindow:
    _has_crosscorr_reference_data = staticmethod(MissionWorkflowWindow._has_crosscorr_reference_data)
    _load_persisted_tx_reference = MissionWorkflowWindow._load_persisted_tx_reference
    _get_crosscorr_reference_for_mission = MissionWorkflowWindow._get_crosscorr_reference_for_mission
    _ensure_transmitter_before_run = MissionWorkflowWindow._ensure_transmitter_before_run
    _is_continuous_active = MissionWorkflowWindow._is_continuous_active
    _runtime_guard_reasons = MissionWorkflowWindow._runtime_guard_reasons

    def __init__(self, master) -> None:
        self.master = master
        self.validation_messages: list[str] = []

    def _append_validation(self, message: str) -> None:
        self.validation_messages.append(message)


def _write_interleaved_iq(path: Path, samples: np.ndarray) -> None:
    interleaved = np.empty(samples.size * 2, dtype=np.int16)
    interleaved[0::2] = np.real(samples).astype(np.int16)
    interleaved[1::2] = np.imag(samples).astype(np.int16)
    interleaved.tofile(path)


def test_get_crosscorr_reference_uses_persisted_tx_file_when_memory_is_empty(tmp_path) -> None:
    samples = np.array([1 + 2j, -3 + 4j, 5 - 6j], dtype=np.complex64)
    tx_path = tmp_path / "persisted_tx.bin"
    _write_interleaved_iq(tx_path, samples)

    class _Master:
        tx_data = np.array([], dtype=np.complex64)
        tx_file = _DummyEntry(str(tx_path))

        @staticmethod
        def _get_crosscorr_reference():
            return np.array([], dtype=np.complex64), "TX"

    window = _DummyWindow(_Master())

    reference = window._get_crosscorr_reference_for_mission()

    assert isinstance(reference, np.ndarray)
    np.testing.assert_array_equal(reference, samples)


def test_get_crosscorr_reference_resolves_zero_suffix_to_base_file(tmp_path) -> None:
    samples = np.array([7 + 1j, 8 + 2j], dtype=np.complex64)
    base_path = tmp_path / "session_tx.bin"
    zeros_path = tmp_path / "session_tx_zeros.bin"
    _write_interleaved_iq(base_path, samples)

    class _Master:
        tx_data = np.array([], dtype=np.complex64)
        tx_file = _DummyEntry(str(zeros_path))

        @staticmethod
        def _get_crosscorr_reference():
            return np.array([], dtype=np.complex64), "TX"

    window = _DummyWindow(_Master())

    reference = window._get_crosscorr_reference_for_mission()

    assert isinstance(reference, np.ndarray)
    np.testing.assert_array_equal(reference, samples)


def test_ensure_transmitter_before_run_returns_true_when_already_active() -> None:
    class _Master:
        @staticmethod
        def is_transmitter_active_for_mission() -> bool:
            return True

    window = _DummyWindow(_Master())
    assert window._ensure_transmitter_before_run() is True


def test_ensure_transmitter_before_run_can_continue_without_tx(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Master:
        @staticmethod
        def is_transmitter_active_for_mission() -> bool:
            return False

    monkeypatch.setattr(mission_workflow_ui.messagebox, "askyesnocancel", lambda *args, **kwargs: False)
    window = _DummyWindow(_Master())

    assert window._ensure_transmitter_before_run() is True
    assert any("ohne aktiven Transmitter" in message for message in window.validation_messages)


def test_ensure_transmitter_before_run_activates_tx(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Master:
        @staticmethod
        def is_transmitter_active_for_mission() -> bool:
            return False

        @staticmethod
        def activate_transmitter_for_mission() -> tuple[bool, str]:
            return True, "playback_started"

    monkeypatch.setattr(mission_workflow_ui.messagebox, "askyesnocancel", lambda *args, **kwargs: True)
    window = _DummyWindow(_Master())

    assert window._ensure_transmitter_before_run() is True
    assert any("playback started" in message for message in window.validation_messages)


def test_ensure_transmitter_before_run_blocks_when_activation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Master:
        @staticmethod
        def is_transmitter_active_for_mission() -> bool:
            return False

        @staticmethod
        def activate_transmitter_for_mission() -> tuple[bool, str]:
            return False, "timeout_waiting_for_playback_started"

    monkeypatch.setattr(mission_workflow_ui.messagebox, "askyesnocancel", lambda *args, **kwargs: True)
    monkeypatch.setattr(mission_workflow_ui.messagebox, "showerror", lambda *args, **kwargs: None)
    window = _DummyWindow(_Master())

    assert window._ensure_transmitter_before_run() is False


def test_runtime_guard_does_not_block_when_only_cmd_running_is_true_for_non_rx() -> None:
    class _Master:
        _cmd_running = True

        @staticmethod
        def is_receive_active_for_mission() -> bool:
            return False

    window = _DummyWindow(_Master())

    assert window._runtime_guard_reasons() == []


def test_runtime_guard_blocks_when_receive_is_active() -> None:
    class _Master:
        _cmd_running = False

        @staticmethod
        def is_receive_active_for_mission() -> bool:
            return True

    window = _DummyWindow(_Master())

    assert window._runtime_guard_reasons() == ["Laufenden RX-Job beenden (Receive ist aktiv)."]
