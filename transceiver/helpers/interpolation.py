from __future__ import annotations

import importlib
import importlib.util

import numpy as np


def _normalize_method(method: str | None) -> str:
    value = (method or "").strip()
    if value == "scipy.interpolate.interp1d":
        return "interp1d"
    if value == "scipy.signal.resample_poly":
        return "resample_poly"
    if value in {"interp1d", "resample_poly"}:
        return value
    return "interp1d"


def _parse_factor(text: str | None) -> int:
    raw = (text or "").strip()
    if not raw:
        return 1
    try:
        value = int(round(float(raw)))
    except Exception:
        return 1
    return max(1, value)


def apply_rx_interpolation(
    data: np.ndarray,
    *,
    enabled: bool,
    method: str,
    factor_text: str,
) -> tuple[np.ndarray, int]:
    arr = np.asarray(data)
    if not enabled or arr.size <= 1:
        return arr, 1

    factor = _parse_factor(factor_text)
    if factor <= 1:
        return arr, 1

    normalized = _normalize_method(method)
    if importlib.util.find_spec("scipy") is None:
        return arr, 1

    if normalized == "resample_poly":
        signal = importlib.import_module("scipy.signal")
        result = signal.resample_poly(arr, up=factor, down=1)
        return np.asarray(result), factor

    interpolate = importlib.import_module("scipy.interpolate")
    x = np.arange(arr.size, dtype=np.float64)
    x_new = np.linspace(0.0, float(arr.size - 1), arr.size * factor)
    if np.iscomplexobj(arr):
        real_fn = interpolate.interp1d(x, arr.real, kind="linear")
        imag_fn = interpolate.interp1d(x, arr.imag, kind="linear")
        result = real_fn(x_new) + 1j * imag_fn(x_new)
    else:
        fn = interpolate.interp1d(x, arr, kind="linear")
        result = fn(x_new)
    return np.asarray(result), factor
