from __future__ import annotations

from pathlib import Path

import numpy as np

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

    def __init__(self, master) -> None:
        self.master = master


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
