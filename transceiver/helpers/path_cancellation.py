import numpy as np

from .correlation_utils import apply_manual_lags, find_los_echo, lag_overlap, xcorr_fft


def apply_path_cancellation(
    data: np.ndarray,
    ref_data: np.ndarray,
    *,
    manual_lags: dict[str, int | None] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    info: dict[str, object] = {
        "applied": False,
        "k0": None,
        "a0": None,
        "k1": None,
        "corr2_peak": None,
        "delta_k": None,
        "warning": None,
    }
    if data.size == 0 or ref_data.size == 0:
        return data, info

    def _append_warning(message: str) -> None:
        if info["warning"]:
            info["warning"] = f"{info['warning']} {message}"
        else:
            info["warning"] = message

    n = min(len(data), len(ref_data))
    if n == 0:
        return data, info
    cc = xcorr_fft(data[:n], ref_data[:n])
    lags = np.arange(-n + 1, n)
    base_los_idx, _ = find_los_echo(cc)
    los_idx, _ = apply_manual_lags(lags, base_los_idx, None, manual_lags)
    if los_idx is None:
        _append_warning("Pfad-Cancellation nicht möglich (kein LOS-Peak).")
        return data, info
    k0 = int(lags[los_idx])
    r_start, s_start, length = lag_overlap(len(data), len(ref_data), k0)
    if length <= 0:
        _append_warning("Pfad-Cancellation nicht möglich (Segment außerhalb Bounds).")
        return data, info
    if length < len(ref_data):
        _append_warning("Pfad-Cancellation: Fenster gekürzt (Segment außerhalb Bounds).")
    r_seg = data[r_start : r_start + length]
    s_seg = ref_data[s_start : s_start + length]
    denom = np.vdot(s_seg, s_seg)
    if denom == 0:
        _append_warning("Pfad-Cancellation nicht möglich (Referenz ohne Energie).")
        return data, info
    coeff = np.vdot(s_seg, r_seg) / (denom + 1e-12)
    if abs(coeff) < 1e-6:
        _append_warning("Pfad-Cancellation: a0 ist sehr klein.")
    residual = data.copy()
    residual[r_start : r_start + length] = r_seg - coeff * s_seg

    n2 = min(len(residual), len(ref_data))
    if n2 == 0:
        return data, info
    cc2 = xcorr_fft(residual[:n2], ref_data[:n2])
    mag2 = np.abs(cc2)
    k1 = None
    corr2_peak = None
    if mag2.size:
        lags2 = np.arange(-n2 + 1, n2)
        k1_idx = int(np.argmax(mag2))
        k1 = int(lags2[k1_idx])
        corr2_peak = float(mag2[k1_idx])
    base_peak = float(np.max(np.abs(cc))) if cc.size else 0.0
    if corr2_peak is None or corr2_peak < 0.1 * base_peak:
        _append_warning("Echo nach Pfad-Cancellation nicht detektierbar.")
        k1 = None

    delta_k = k1 - k0 if k1 is not None else None
    info.update(
        {
            "applied": True,
            "k0": k0,
            "a0": coeff,
            "k1": k1,
            "corr2_peak": corr2_peak,
            "delta_k": delta_k,
        }
    )
    return residual, info
