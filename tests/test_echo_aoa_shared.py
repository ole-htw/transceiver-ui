from pathlib import Path

from transceiver.helpers import continuous_processing
from transceiver.helpers import echo_aoa


def test_continuous_processing_uses_shared_echo_aoa_helpers() -> None:
    assert continuous_processing._find_peaks_simple is echo_aoa._find_peaks_simple
    assert (
        continuous_processing._correlate_and_estimate_echo_aoa
        is echo_aoa._correlate_and_estimate_echo_aoa
    )


def test_main_imports_peak_finder_from_shared_module() -> None:
    main_source = Path("transceiver/__main__.py").read_text(encoding="utf-8")
    assert "from .helpers.echo_aoa import _find_peaks_simple" in main_source
    assert "def _find_peaks_simple(" not in main_source
