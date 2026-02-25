from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np

from transceiver.helpers import continuous_processing
from transceiver.helpers.echo_aoa import _correlate_and_estimate_echo_aoa


def test_continuous_processing_uses_shared_echo_aoa_function() -> None:
    assert (
        continuous_processing._correlate_and_estimate_echo_aoa
        is _correlate_and_estimate_echo_aoa
    )


def test_gui_path_imports_shared_echo_aoa_function() -> None:
    main_path = Path(__file__).resolve().parents[1] / "transceiver" / "__main__.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"))

    has_shared_import = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "helpers.echo_aoa"
        and any(alias.name == "_correlate_and_estimate_echo_aoa" for alias in node.names)
        for node in tree.body
    )
    assert has_shared_import

    local_defs = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_find_peaks_simple",
            "_aoa_from_corr_peak",
            "_correlate_and_estimate_echo_aoa",
        }
    }
    assert not local_defs


def test_shared_core_function_returns_peak_results() -> None:
    signature = inspect.signature(_correlate_and_estimate_echo_aoa)
    assert list(signature.parameters) == [
        "rx_data",
        "tx_data",
        "antenna_spacing",
        "wavelength",
        "rel_thresh",
        "min_dist",
        "peak_win",
    ]

    tx = np.zeros(64, dtype=np.complex64)
    tx[5:10] = 1.0 + 0j

    rx1 = np.zeros(64, dtype=np.complex64)
    rx2 = np.zeros(64, dtype=np.complex64)
    rx1[20:25] = 1.0 + 0j
    rx2[20:25] = 1.0 + 0j

    out = _correlate_and_estimate_echo_aoa(
        np.vstack([rx1, rx2]),
        tx,
        antenna_spacing=0.5,
        wavelength=1.0,
        rel_thresh=0.5,
        min_dist=4,
    )

    assert out["results"]
    assert abs(out["results"][0]["theta_deg"]) < 1e-6
