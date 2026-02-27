from __future__ import annotations

import importlib
import importlib.util
import logging
from fractions import Fraction

import numpy as np

from .number_parser import parse_number_expr


def _normalize_method(method: str | None) -> str:
    value = (method or "").strip()
    if value == "scipy.interpolate.interp1d":
        return "interp1d"
    if value == "scipy.signal.resample_poly":
        return "resample_poly"
    if value in {"interp1d", "resample_poly"}:
        return value
    return "interp1d"


def _parse_factor_expr(factor_expr: str | None) -> float:
    raw = (factor_expr or "").strip()
    if not raw:
        raise ValueError("Interpolationsfaktor fehlt")
    try:
        factor = float(parse_number_expr(raw))
    except Exception as exc:
        raise ValueError("Ungültiger Interpolationsfaktor") from exc
    if not np.isfinite(factor):
        raise ValueError("Interpolationsfaktor muss endlich sein")
    if factor <= 0.0:
        raise ValueError("Interpolationsfaktor muss größer als 0 sein")
    if factor < 1.0:
        raise ValueError("Interpolationsfaktor muss mindestens 1 sein")
    return factor


def _apply_rx_interpolation(
    signal: np.ndarray,
    fs: float,
    enabled: bool,
    method: str,
    factor_expr: str,
) -> tuple[np.ndarray, float]:
    arr = np.asarray(signal)
    if not enabled or arr.size == 0:
        return arr, float(fs)

    factor = _parse_factor_expr(factor_expr)
    if factor == 1.0:
        return arr, float(fs)

    normalized = _normalize_method(method)
    if importlib.util.find_spec("scipy") is None:
        raise RuntimeError(
            "Interpolation benötigt SciPy, ist aber in dieser Umgebung nicht verfügbar"
        )

    try:
        if normalized == "resample_poly":
            signal_mod = importlib.import_module("scipy.signal")
            ratio = Fraction(factor).limit_denominator(4096)
            if ratio.numerator <= 0 or ratio.denominator <= 0:
                raise ValueError("Ungültiges Interpolationsverhältnis")
            result = signal_mod.resample_poly(
                arr,
                up=ratio.numerator,
                down=ratio.denominator,
            )
            fs_interpolated = float(fs) * ratio.numerator / ratio.denominator
            return np.asarray(result), fs_interpolated

        interpolate = importlib.import_module("scipy.interpolate")
        x = np.arange(arr.size, dtype=np.float64)
        output_len = max(1, int(round(arr.size * factor)))
        x_new = np.linspace(0.0, float(arr.size - 1), output_len)
        if np.iscomplexobj(arr):
            real_fn = interpolate.interp1d(x, arr.real, kind="linear")
            imag_fn = interpolate.interp1d(x, arr.imag, kind="linear")
            result = real_fn(x_new) + 1j * imag_fn(x_new)
        else:
            fn = interpolate.interp1d(x, arr, kind="linear")
            result = fn(x_new)
        return np.asarray(result), float(fs) * factor
    except ValueError:
        raise
    except Exception as exc:
        logging.warning(
            "RX-Interpolation fehlgeschlagen, verwende Originalsignal als Fallback: %s",
            exc,
        )
        return arr, float(fs)


def apply_rx_interpolation(
    data: np.ndarray,
    *,
    enabled: bool,
    method: str,
    factor_text: str,
) -> tuple[np.ndarray, int]:
    """Backward-compatible helper for legacy call sites.

    Returns interpolation scale as integer factor. Non-integer scales round to nearest int.
    """
    interpolated, fs_new = _apply_rx_interpolation(
        data,
        fs=1.0,
        enabled=enabled,
        method=method,
        factor_expr=factor_text,
    )
    scale = int(round(fs_new))
    return interpolated, max(1, scale)


def apply_tx_upsampling(
    signal: np.ndarray,
    fs_in: float,
    fs_target: float,
    method: str = "resample_poly",
) -> tuple[np.ndarray, float]:
    """Upsample TX-Samples auf ``fs_target`` und liefere ``(signal, fs_out)``.

    Standardmäßig wird ``scipy.signal.resample_poly`` mit einem robusten
    rationalen Verhältnis genutzt. Alternativ kann ``method='linear'`` für
    lineare Interpolation (Fallback) gewählt werden.
    """
    arr = np.asarray(signal)
    fs_in = float(fs_in)
    fs_target = float(fs_target)

    if fs_in <= 0.0:
        raise ValueError("fs_in muss > 0 sein")
    if fs_target <= fs_in:
        raise ValueError("fs_target muss strikt größer als fs_in sein")
    if arr.size == 0:
        return arr, fs_target

    normalized = _normalize_method(method)
    if importlib.util.find_spec("scipy") is None:
        raise RuntimeError(
            "Upsampling benötigt SciPy, ist aber in dieser Umgebung nicht verfügbar"
        )

    try:
        if normalized == "resample_poly":
            signal_mod = importlib.import_module("scipy.signal")
            ratio = Fraction(fs_target / fs_in).limit_denominator(4096)
            if ratio.numerator <= 0 or ratio.denominator <= 0:
                raise ValueError("Ungültiges Upsampling-Verhältnis")
            result = signal_mod.resample_poly(
                arr,
                up=ratio.numerator,
                down=ratio.denominator,
            )
            fs_out = fs_in * ratio.numerator / ratio.denominator
            return np.asarray(result), float(fs_out)

        interpolate = importlib.import_module("scipy.interpolate")
        x = np.arange(arr.size, dtype=np.float64)
        factor = fs_target / fs_in
        output_len = max(1, int(round(arr.size * factor)))
        x_new = np.linspace(0.0, float(arr.size - 1), output_len)
        if np.iscomplexobj(arr):
            real_fn = interpolate.interp1d(x, arr.real, kind="linear")
            imag_fn = interpolate.interp1d(x, arr.imag, kind="linear")
            result = real_fn(x_new) + 1j * imag_fn(x_new)
        else:
            fn = interpolate.interp1d(x, arr, kind="linear")
            result = fn(x_new)
        return np.asarray(result), fs_target
    except ValueError:
        raise
    except Exception as exc:
        logging.warning(
            "TX-Upsampling fehlgeschlagen, verwende Originalsignal als Fallback: %s",
            exc,
        )
        return arr, fs_in
