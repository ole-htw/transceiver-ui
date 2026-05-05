
"""
zc_echo_estimator.py

Scientific, reproducible echo-time estimation for Zadoff-Chu or other known
complex baseband probe signals.

Core idea
---------
The received signal is correlated with the transmitted reference sequence. The
result is treated as a matched-filter output / channel impulse response estimate.
Echoes are estimated as significant components in the power delay profile (PDP),
optionally supported by shoulder detection and a complex least-squares model fit
using the measured autocorrelation of the reference waveform.

The module is intentionally self-contained and depends only on NumPy.

Typical use
-----------
    cfg = EchoEstimatorConfig(sample_rate_hz=1e6)
    result = estimate_echoes(rx, zc, cfg)
    for echo in result.echoes:
        print(echo.delay_s, echo.kind, echo.cfar_margin_db)

Notes for scientific use
------------------------
- Store EchoEstimatorConfig together with the result.
- Report the sample rate, sequence length, correlation normalization, CFAR
  parameters, the estimated main-lobe width, and calibration delay.
- For absolute time-of-flight, subtract the hardware/calibration delay.
- For unresolved multipath, prefer the model-fit refined delays over raw local
  maxima.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal
import math
import numpy as np


ArrayLikeComplex = np.ndarray


def _as_1d_complex(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr.astype(np.complex128, copy=False)


def _as_1d_float(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array, got shape {arr.shape}.")
    return arr


def db10(x: float | np.ndarray, floor: float = 1e-300) -> float | np.ndarray:
    """Return 10*log10(x) with numerical floor."""
    return 10.0 * np.log10(np.maximum(x, floor))


def db20(x: float | np.ndarray, floor: float = 1e-300) -> float | np.ndarray:
    """Return 20*log10(x) with numerical floor."""
    return 20.0 * np.log10(np.maximum(x, floor))


def xcorr_fft(data: np.ndarray, reference: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    """Full complex cross-correlation using an FFT.

    The returned array has length len(data)+len(reference)-1 and corresponds to
    lags ``[-len(reference)+1, ..., len(data)-1]``.

    It is equivalent to ``np.correlate(data, reference, mode="full")`` for
    complex arrays, i.e. the reference is conjugated.
    """
    data_c = _as_1d_complex(data, "data")
    ref_c = _as_1d_complex(reference, "reference")

    n = data_c.size + ref_c.size - 1
    nfft = 1 << (n - 1).bit_length()

    data_fft = np.fft.fft(data_c, nfft)
    ref_fft = np.fft.fft(ref_c, nfft)
    cc = np.fft.ifft(data_fft * np.conj(ref_fft))

    # Reorder from circular-lag order to full-correlation lag order.
    out = np.concatenate((cc[-(ref_c.size - 1):], cc[: data_c.size]))

    if normalize:
        energy = float(np.sum(np.abs(ref_c) ** 2))
        if energy <= 0.0:
            raise ValueError("reference energy is zero.")
        out = out / energy

    return out


def autocorr_fft(x: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    """Full autocorrelation of a one-dimensional complex waveform."""
    return xcorr_fft(x, x, normalize=normalize)


def correlation_lags(data_len: int, reference_len: int) -> np.ndarray:
    """Return lags for ``xcorr_fft(data, reference)``."""
    if data_len <= 0 or reference_len <= 0:
        raise ValueError("data_len and reference_len must be positive.")
    return np.arange(-(reference_len - 1), data_len, dtype=int)


def robust_baseline_sigma(x: np.ndarray) -> tuple[float, float]:
    """Median baseline and MAD-based Gaussian-equivalent sigma.

    This is robust against a small number of strong peaks and is used only for
    secondary heuristics; the main strict peak detection uses CFAR on the PDP.
    """
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 0.0
    baseline = float(np.median(y))
    mad = float(np.median(np.abs(y - baseline)))
    return baseline, 1.4826 * mad


def smooth_edge(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding and no peak shift."""
    y = np.asarray(x, dtype=float)
    if y.size == 0:
        return y.copy()
    window = max(1, int(window))
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.pad(y, (pad, pad), mode="edge"), kernel, mode="valid")


def parabolic_subsample_refinement(y: np.ndarray, idx: int, *, log_domain: bool = True) -> float:
    """Refine a peak location using a three-point parabolic estimator.

    Parameters
    ----------
    y:
        Magnitude or power sequence.
    idx:
        Integer peak index.
    log_domain:
        If true, fit the parabola to log(y). This is often more stable for
        matched-filter lobes with high dynamic range.

    Returns
    -------
    float
        Refined index in samples. If refinement is not possible, returns idx.
    """
    values = np.asarray(y, dtype=float)
    idx = int(idx)
    if idx <= 0 or idx >= values.size - 1:
        return float(idx)

    work = np.log(values + 1e-300) if log_domain else values
    denom = work[idx - 1] - 2.0 * work[idx] + work[idx + 1]
    if not np.isfinite(denom) or abs(float(denom)) < 1e-30:
        return float(idx)

    delta = 0.5 * (work[idx - 1] - work[idx + 1]) / denom
    if not np.isfinite(delta):
        return float(idx)

    # The 3-point model is local; clipping avoids absurd jumps on shoulders/noise.
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(idx) + delta


def _local_maxima_indices(y: np.ndarray, mask: np.ndarray | None = None) -> list[int]:
    """Return indices of strict or plateau-like local maxima."""
    values = np.asarray(y, dtype=float)
    if values.size < 3:
        return []
    if mask is None:
        active = np.ones(values.size, dtype=bool)
    else:
        active = np.asarray(mask, dtype=bool)
        if active.size != values.size:
            raise ValueError("mask size must match y size.")

    out: list[int] = []
    for i in range(1, values.size - 1):
        if not active[i]:
            continue
        if (
            values[i] >= values[i - 1]
            and values[i] >= values[i + 1]
            and (values[i] > values[i - 1] or values[i] > values[i + 1])
        ):
            out.append(i)
    return out


def _cluster_sorted_indices(indices: list[int], max_gap: int) -> list[list[int]]:
    if not indices:
        return []
    max_gap = max(0, int(max_gap))
    ordered = sorted({int(i) for i in indices})
    clusters: list[list[int]] = [[ordered[0]]]
    for idx in ordered[1:]:
        if idx - clusters[-1][-1] <= max_gap:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return clusters


def suppress_nearby_candidates(
    indices: list[int],
    score: np.ndarray,
    *,
    min_distance: int,
    keep: Literal["strongest", "earliest", "latest"] = "strongest",
) -> list[int]:
    """Merge nearby candidates and keep one representative per cluster."""
    values = np.asarray(score, dtype=float)
    valid = sorted({int(i) for i in indices if 0 <= int(i) < values.size})
    if not valid:
        return []

    def pick(cluster: list[int]) -> int:
        if keep == "earliest":
            return int(min(cluster))
        if keep == "latest":
            return int(max(cluster))
        return int(max(cluster, key=lambda j: float(values[j])))

    return sorted(pick(c) for c in _cluster_sorted_indices(valid, int(min_distance)))


def collapse_unresolved_lobes(
    indices: list[int],
    power_or_mag: np.ndarray,
    *,
    min_distance: int,
    min_valley_drop_rel: float = 0.10,
    min_valley_drop_sigma: float = 0.25,
    smooth_window: int = 3,
    keep: Literal["strongest", "earliest", "latest"] = "strongest",
) -> list[int]:
    """Collapse candidates that are not separated by a significant valley.

    This avoids reporting several samples within the same matched-filter lobe as
    separate physical echoes. The criterion is intentionally reported and
    configurable so that it can be described in a paper.
    """
    values = np.asarray(power_or_mag, dtype=float)
    raw = sorted({int(i) for i in indices if 0 <= int(i) < values.size})
    if not raw:
        return []

    work = smooth_edge(values, smooth_window)
    _baseline, sigma = robust_baseline_sigma(work)
    sigma = max(float(sigma), 1e-300)

    clusters: list[list[int]] = [[raw[0]]]

    for idx in raw[1:]:
        prev = clusters[-1][-1]

        if idx - prev < int(min_distance):
            clusters[-1].append(idx)
            continue

        lo, hi = sorted((prev, idx))
        valley = float(np.min(work[lo: hi + 1]))
        prev_h = float(work[prev])
        curr_h = float(work[idx])
        smaller_peak = max(1e-300, min(prev_h, curr_h))
        valley_drop = smaller_peak - valley
        valley_drop_rel = valley_drop / smaller_peak

        is_resolved = (
            valley_drop_rel >= float(min_valley_drop_rel)
            and valley_drop >= float(min_valley_drop_sigma) * sigma
        )
        if is_resolved:
            clusters.append([idx])
        else:
            clusters[-1].append(idx)

    def pick(cluster: list[int]) -> int:
        if keep == "earliest":
            return int(min(cluster))
        if keep == "latest":
            return int(max(cluster))
        return int(max(cluster, key=lambda j: float(values[j])))

    return sorted(pick(c) for c in clusters)


@dataclass(frozen=True)
class MainLobeMetrics:
    """Measured main-lobe properties of the reference autocorrelation."""

    peak_index: int
    peak_lag_samples: int
    half_power_left_lag: int
    half_power_right_lag: int
    half_power_width_samples: int
    first_below_threshold_left_lag: int
    first_below_threshold_right_lag: int
    guard_half_width_samples: int
    peak_sidelobe_level_db: float | None


def estimate_main_lobe_metrics(
    reference: np.ndarray,
    *,
    half_power_rel: float = 0.5,
    guard_rel_power: float = 1e-3,
) -> MainLobeMetrics:
    """Estimate main-lobe width from the measured reference autocorrelation.

    ``guard_rel_power`` defines where the main lobe is considered sufficiently
    small for CFAR guard cells. With real filtering this is often more useful
    than assuming an ideal one-sample Zadoff-Chu autocorrelation.
    """
    ref = _as_1d_complex(reference, "reference")
    acf = autocorr_fft(ref, normalize=True)
    lags = correlation_lags(ref.size, ref.size)
    power = np.abs(acf) ** 2

    peak_index = int(np.argmax(power))
    peak_power = float(power[peak_index])
    if peak_power <= 0.0:
        raise ValueError("reference autocorrelation has zero peak power.")

    half_thr = float(half_power_rel) * peak_power
    guard_thr = float(guard_rel_power) * peak_power

    left_hp = peak_index
    while left_hp > 0 and power[left_hp] >= half_thr:
        left_hp -= 1
    if power[left_hp] < half_thr and left_hp < peak_index:
        left_hp += 1

    right_hp = peak_index
    while right_hp < power.size - 1 and power[right_hp] >= half_thr:
        right_hp += 1
    if power[right_hp] < half_thr and right_hp > peak_index:
        right_hp -= 1

    left_guard = peak_index
    while left_guard > 0 and power[left_guard] >= guard_thr:
        left_guard -= 1

    right_guard = peak_index
    while right_guard < power.size - 1 and power[right_guard] >= guard_thr:
        right_guard += 1

    guard_half_width = int(max(abs(lags[peak_index] - lags[left_guard]), abs(lags[right_guard] - lags[peak_index]), 1))

    # Estimate peak sidelobe level outside the guard region.
    sidelobe_mask = np.ones(power.size, dtype=bool)
    sidelobe_mask[left_guard: right_guard + 1] = False
    if np.any(sidelobe_mask):
        psl = float(db10(np.max(power[sidelobe_mask]) / peak_power))
    else:
        psl = None

    return MainLobeMetrics(
        peak_index=peak_index,
        peak_lag_samples=int(lags[peak_index]),
        half_power_left_lag=int(lags[left_hp]),
        half_power_right_lag=int(lags[right_hp]),
        half_power_width_samples=int(lags[right_hp] - lags[left_hp] + 1),
        first_below_threshold_left_lag=int(lags[left_guard]),
        first_below_threshold_right_lag=int(lags[right_guard]),
        guard_half_width_samples=guard_half_width,
        peak_sidelobe_level_db=psl,
    )


@dataclass(frozen=True)
class CFARConfig:
    """Configuration for one-dimensional CFAR on the PDP."""

    enabled: bool = True
    mode: Literal["ca", "median"] = "ca"
    train_cells: int = 48
    guard_cells: int | None = None
    pfa: float = 1e-4
    median_scale: float = 12.0
    minimum_noise_power: float = 1e-300

    def resolved_guard_cells(self, default_guard: int) -> int:
        return max(1, int(default_guard if self.guard_cells is None else self.guard_cells))


def cfar_1d(
    power: np.ndarray,
    *,
    train_cells: int,
    guard_cells: int,
    pfa: float = 1e-4,
    mode: Literal["ca", "median"] = "ca",
    median_scale: float = 12.0,
    minimum_noise_power: float = 1e-300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one-dimensional CFAR to a power delay profile.

    Returns
    -------
    detection_mask:
        Boolean vector where the cell under test exceeds the threshold.
    threshold:
        Power threshold. Edge cells that cannot be tested are NaN.
    noise_estimate:
        Local noise estimate before threshold scaling.

    Notes
    -----
    ``mode="ca"`` uses the standard cell-averaging CFAR scaling for exponential
    power noise. ``mode="median"`` is a robust fallback for contaminated
    training windows but does not have the same direct P_FA interpretation.
    """
    p = _as_1d_float(power, "power")
    n = p.size
    train = max(1, int(train_cells))
    guard = max(1, int(guard_cells))
    if not (0.0 < float(pfa) < 1.0):
        raise ValueError("pfa must be in (0, 1).")
    if mode not in ("ca", "median"):
        raise ValueError("mode must be 'ca' or 'median'.")

    det = np.zeros(n, dtype=bool)
    threshold = np.full(n, np.nan, dtype=float)
    noise = np.full(n, np.nan, dtype=float)

    start = train + guard
    stop = n - train - guard
    if stop <= start:
        return det, threshold, noise

    num_train = 2 * train
    ca_alpha = num_train * (float(pfa) ** (-1.0 / num_train) - 1.0)

    for i in range(start, stop):
        left = p[i - guard - train: i - guard]
        right = p[i + guard + 1: i + guard + train + 1]
        training = np.concatenate((left, right))
        training = training[np.isfinite(training)]
        if training.size == 0:
            continue

        if mode == "ca":
            noise_est = float(np.mean(training))
            alpha = ca_alpha
        else:
            noise_est = float(np.median(training))
            alpha = float(median_scale)

        noise_est = max(noise_est, float(minimum_noise_power))
        noise[i] = noise_est
        threshold[i] = alpha * noise_est
        det[i] = bool(p[i] > threshold[i])

    return det, threshold, noise


@dataclass(frozen=True)
class ShoulderConfig:
    """Configuration for flank/shoulder candidate generation.

    Shoulder detection is used for first-path cases in which the earliest
    arrival is not a strict local maximum because a later multipath component is
    stronger.
    """

    enabled: bool = True
    short_window: int = 5
    long_window: int = 31
    slope_window: int = 3
    residual_sigma_factor: float = 2.0
    kink_sigma_factor: float = 1.2
    noise_sigma_factor: float = 0.8
    min_relative_to_strongest_mag: float = 0.015
    min_relative_to_peak_threshold: float = 0.25
    snap_window: int = 3
    echo_onset_max_distance_factor: float = 3.0
    echo_onset_min_rel_to_next_peak: float = 0.15
    echo_onset_max_rel_to_next_peak: float = 0.98


@dataclass(frozen=True)
class ModelFitConfig:
    """Configuration for optional complex multipath model fitting."""

    enabled: bool = True
    max_candidates: int = 12
    fit_padding_main_lobes: float = 2.0
    coordinate_refinement: bool = True
    refinement_radius_samples: float = 0.75
    refinement_grid_points: int = 9
    refinement_iterations: int = 2
    min_path_improvement_db: float = 0.50
    min_amplitude_rel_to_strongest: float = 0.01
    keep_strict_cfar_even_if_model_weak: bool = False


@dataclass(frozen=True)
class EchoEstimatorConfig:
    """Configuration for echo detection and estimation.

    Parameters are in samples unless explicitly marked with seconds/Hz.
    """

    sample_rate_hz: float
    search_lag_min_samples: int | None = 0
    search_lag_max_samples: int | None = None
    repetition_period_samples: int | None = None
    calibration_delay_s: float = 0.0

    max_echoes: int = 8
    min_relative_power_to_strongest: float = 1e-4
    min_peak_distance_samples: int | None = None
    collapse_unresolved: bool = True

    cfar: CFARConfig = CFARConfig()
    shoulder: ShoulderConfig = ShoulderConfig()
    model_fit: ModelFitConfig = ModelFitConfig()

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive.")
        if self.max_echoes <= 0:
            raise ValueError("max_echoes must be positive.")
        if self.min_relative_power_to_strongest < 0.0:
            raise ValueError("min_relative_power_to_strongest must be non-negative.")


@dataclass
class EchoCandidate:
    """One detected echo/path candidate."""

    index: int
    lag_samples: int
    delay_s: float
    kind: Literal["strict_peak", "left_shoulder", "echo_shoulder", "manual"]
    power: float
    magnitude: float
    cfar_threshold: float | None
    cfar_margin_db: float | None
    score: float

    refined_lag_samples: float | None = None
    refined_delay_s: float | None = None
    amplitude: complex | None = None
    amplitude_abs: float | None = None
    amplitude_phase_rad: float | None = None
    model_path_improvement_db: float | None = None
    accepted_by_model: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.amplitude is not None:
            d["amplitude_real"] = float(np.real(self.amplitude))
            d["amplitude_imag"] = float(np.imag(self.amplitude))
        d.pop("amplitude", None)
        return d


@dataclass
class EchoEstimationResult:
    """Complete result object with arrays needed for plotting and validation."""

    config: EchoEstimatorConfig
    lags: np.ndarray
    correlation: np.ndarray
    pdp: np.ndarray
    cfar_threshold: np.ndarray
    cfar_noise: np.ndarray
    search_mask: np.ndarray
    main_lobe: MainLobeMetrics
    candidates: list[EchoCandidate]
    echoes: list[EchoCandidate]
    model_residual_power: float | None
    model_total_power: float | None

    def echoes_table(self) -> list[dict[str, Any]]:
        return [echo.to_dict() for echo in self.echoes]


def _build_search_mask(lags: np.ndarray, cfg: EchoEstimatorConfig, pdp: np.ndarray) -> np.ndarray:
    mask = np.ones(lags.size, dtype=bool)

    if cfg.search_lag_min_samples is not None:
        mask &= lags >= int(cfg.search_lag_min_samples)
    if cfg.search_lag_max_samples is not None:
        mask &= lags <= int(cfg.search_lag_max_samples)

    # If repetitions are used, keep only the half-period group around the
    # strongest peak inside the current lag constraints.
    if cfg.repetition_period_samples is not None and cfg.repetition_period_samples > 1 and np.any(mask):
        tmp = np.where(mask, pdp, -np.inf)
        anchor_idx = int(np.argmax(tmp))
        if np.isfinite(tmp[anchor_idx]):
            half_period = float(cfg.repetition_period_samples) / 2.0
            anchor_lag = float(lags[anchor_idx])
            mask &= np.abs(lags.astype(float) - anchor_lag) <= half_period

    return mask


def _threshold_at(threshold: np.ndarray, idx: int) -> tuple[float | None, float | None]:
    if idx < 0 or idx >= threshold.size:
        return None, None
    th = float(threshold[idx])
    if not np.isfinite(th) or th <= 0.0:
        return None, None
    return th, None


def _candidate_from_index(
    idx: int,
    *,
    kind: Literal["strict_peak", "left_shoulder", "echo_shoulder", "manual"],
    lags: np.ndarray,
    pdp: np.ndarray,
    threshold: np.ndarray,
    sample_rate_hz: float,
    calibration_delay_s: float,
    score: float | None = None,
) -> EchoCandidate:
    idx = int(idx)
    power = float(pdp[idx])
    magnitude = float(math.sqrt(max(power, 0.0)))
    th = float(threshold[idx]) if 0 <= idx < threshold.size and np.isfinite(threshold[idx]) else None
    margin = float(db10(power / th)) if th is not None and th > 0.0 and power > 0.0 else None
    lag = int(lags[idx])
    delay = float(lag / sample_rate_hz - calibration_delay_s)
    return EchoCandidate(
        index=idx,
        lag_samples=lag,
        delay_s=delay,
        kind=kind,
        power=power,
        magnitude=magnitude,
        cfar_threshold=th,
        cfar_margin_db=margin,
        score=float(power if score is None else score),
    )


def _detect_strict_peaks(
    pdp: np.ndarray,
    *,
    search_mask: np.ndarray,
    cfar_mask: np.ndarray,
    min_power: float,
    min_distance: int,
) -> list[int]:
    active = search_mask & cfar_mask & np.isfinite(pdp) & (pdp >= float(min_power))
    peaks = _local_maxima_indices(pdp, active)
    return suppress_nearby_candidates(peaks, pdp, min_distance=min_distance, keep="strongest")


def _detect_left_shoulder_candidates(
    mag: np.ndarray,
    *,
    center_idx: int,
    left_bound: int,
    right_bound: int,
    min_height: float,
    min_peak_distance: int,
    cfg: ShoulderConfig,
) -> list[int]:
    """Detect weak first-path shoulders on the rising flank before center_idx."""
    if not cfg.enabled or mag.size < 9:
        return []

    mag_f = np.asarray(mag, dtype=float)
    center_idx = int(np.clip(center_idx, 0, mag_f.size - 1))
    left_bound = int(np.clip(left_bound, 0, mag_f.size - 1))
    right_bound = int(np.clip(right_bound, 0, mag_f.size - 1))

    search_right = min(center_idx - max(1, int(min_peak_distance)), right_bound)
    if search_right - left_bound < 8:
        return []

    segment = mag_f[left_bound: search_right + 1]
    mag_baseline, mag_sigma = robust_baseline_sigma(segment)

    center_mag = float(mag_f[center_idx])
    height_threshold = max(
        cfg.min_relative_to_strongest_mag * center_mag,
        mag_baseline + max(0.0, cfg.noise_sigma_factor) * mag_sigma,
        cfg.min_relative_to_peak_threshold * float(min_height),
    )

    scale = max(1e-12, mag_baseline + mag_sigma)
    work = np.log1p(mag_f / scale)

    short = smooth_edge(work, cfg.short_window)
    long = smooth_edge(work, cfg.long_window)
    residual = short - long
    slope = np.gradient(short)
    long_slope = np.gradient(long)

    res_segment = residual[left_bound: search_right + 1]
    res_baseline, res_sigma = robust_baseline_sigma(res_segment)
    _slope_baseline, slope_sigma = robust_baseline_sigma(slope[left_bound: search_right + 1])

    res_threshold = cfg.residual_sigma_factor * max(res_sigma, 1e-12)
    kink_threshold = cfg.kink_sigma_factor * max(slope_sigma, 1e-12)
    slope_eps = 0.2 * max(slope_sigma, 1e-12)

    candidates: set[int] = set()
    sw = max(1, int(cfg.slope_window))
    start = max(left_bound + sw, 1)
    stop = min(search_right - sw, mag_f.size - 2)

    for i in range(start, stop + 1):
        if float(mag_f[i]) < height_threshold:
            continue

        is_on_rising_flank = float(long_slope[i]) > slope_eps

        is_residual_hump = (
            is_on_rising_flank
            and residual[i] >= residual[i - 1]
            and residual[i] >= residual[i + 1]
            and residual[i] - res_baseline >= res_threshold
        )

        left_slope = float(np.median(slope[i - sw: i]))
        right_slope = float(np.median(slope[i + 1: i + 1 + sw]))

        is_rising_kink = (
            is_on_rising_flank
            and left_slope > slope_eps
            and (left_slope - right_slope) >= kink_threshold
        )

        if not (is_residual_hump or is_rising_kink):
            continue

        snap = max(0, int(cfg.snap_window))
        snap_left = max(left_bound, i - snap)
        snap_right = min(search_right, i + snap)
        snap_indices = list(range(snap_left, snap_right + 1))
        if snap_indices:
            i = int(max(snap_indices, key=lambda j: float(residual[j])))

        if float(mag_f[i]) >= height_threshold:
            candidates.add(int(i))

    return suppress_nearby_candidates(
        sorted(candidates),
        mag_f,
        min_distance=max(int(min_peak_distance), 2 * int(cfg.snap_window) + 1),
        keep="strongest",
    )


def _detect_echo_shoulder_candidates(
    mag: np.ndarray,
    *,
    strict_peak_indices: list[int],
    los_anchor_idx: int,
    left_bound: int,
    right_bound: int,
    min_height: float,
    min_peak_distance: int,
    cfg: ShoulderConfig,
) -> list[int]:
    """Detect shoulders after the first path and before/around later peaks."""
    if not cfg.enabled or mag.size < 7 or not strict_peak_indices:
        return []

    mag_f = np.asarray(mag, dtype=float)
    left_bound = int(np.clip(left_bound, 0, mag_f.size - 1))
    right_bound = int(np.clip(right_bound, 0, mag_f.size - 1))
    los_anchor_idx = int(np.clip(los_anchor_idx, left_bound, right_bound))

    strict = sorted({int(i) for i in strict_peak_indices if left_bound <= int(i) <= right_bound})
    if not strict:
        return []

    segment = mag_f[left_bound: right_bound + 1]
    mag_baseline, mag_sigma = robust_baseline_sigma(segment)
    scale = max(1e-12, mag_baseline + mag_sigma)
    work = np.log1p(mag_f / scale)

    smooth_mag = smooth_edge(work, cfg.short_window)
    slope = np.gradient(smooth_mag)

    slope_left = max(left_bound, los_anchor_idx + min_peak_distance)
    _slope_baseline, slope_sigma = robust_baseline_sigma(slope[slope_left: right_bound + 1])
    slope_sigma = max(slope_sigma, 1e-12)
    slope_threshold = cfg.kink_sigma_factor * slope_sigma
    slope_eps = 0.2 * slope_sigma

    sw = max(1, int(cfg.slope_window))
    snap = max(0, int(cfg.snap_window))
    onset_max_distance = max(min_peak_distance + 1, int(round(cfg.echo_onset_max_distance_factor * max(1, min_peak_distance))))

    candidates: set[int] = set()

    def slope_pair(j: int) -> tuple[float, float] | None:
        left_slice = slope[max(0, j - sw): j]
        right_slice = slope[j + 1: min(slope.size, j + 1 + sw)]
        if left_slice.size == 0 or right_slice.size == 0:
            return None
        return float(np.median(left_slice)), float(np.median(right_slice))

    def next_strict_peak_after(j: int) -> int | None:
        for p in strict:
            if p > j + min_peak_distance:
                return int(p)
        return None

    def is_valid_rising_onset(j: int) -> bool:
        if not (los_anchor_idx < j < right_bound):
            return False
        pair = slope_pair(j)
        if pair is None:
            return False
        left_s, right_s = pair
        if not (right_s > slope_eps and (right_s - left_s) >= slope_threshold):
            return False
        next_peak = next_strict_peak_after(j)
        if next_peak is None:
            return False
        distance = next_peak - j
        if distance <= min_peak_distance or distance > onset_max_distance:
            return False
        next_h = float(mag_f[next_peak])
        this_h = float(mag_f[j])
        if next_h <= 1e-12:
            return False
        rel = this_h / next_h
        if rel < cfg.echo_onset_min_rel_to_next_peak:
            return False
        if rel > cfg.echo_onset_max_rel_to_next_peak:
            return False
        lo, hi = sorted((j, next_peak))
        valley = float(np.min(mag_f[lo: hi + 1]))
        # If a deep valley follows, this is more likely a separate strict peak
        # boundary than a shoulder leading into that peak.
        if valley < 0.70 * max(1e-12, this_h):
            return False
        return True

    def refine_to_detrended_crest(j: int) -> int:
        next_peak = next_strict_peak_after(j)
        search_lo = max(left_bound, int(j))
        if next_peak is None:
            search_hi = min(right_bound, int(j) + max(3, min_peak_distance))
            baseline_hi = search_hi
        else:
            peak_guard = max(min_peak_distance, 2)
            search_hi = min(right_bound, int(next_peak) - peak_guard)
            baseline_hi = int(next_peak)
        if search_hi <= search_lo:
            return int(j)

        xs = np.arange(search_lo, search_hi + 1)
        denom = max(1, int(baseline_hi) - int(search_lo))
        baseline = (
            mag_f[search_lo]
            + (mag_f[baseline_hi] - mag_f[search_lo]) * (xs - search_lo) / float(denom)
        )
        residual = mag_f[xs] - baseline
        best = int(xs[int(np.argmax(residual))])
        return best

    start = max(left_bound + sw, los_anchor_idx + min_peak_distance + 1, 1)
    stop = min(right_bound - sw, mag_f.size - 2)

    for i in range(start, stop + 1):
        if any(abs(i - p) <= min_peak_distance for p in strict):
            continue
        if float(mag_f[i]) < float(min_height):
            continue
        mag_prominence = float(mag_f[i]) - mag_baseline
        if mag_sigma > 1e-12 and mag_prominence < cfg.noise_sigma_factor * mag_sigma:
            continue

        pair = slope_pair(i)
        if pair is None:
            continue
        left_s, right_s = pair
        mid_s = float(slope[i])

        is_rising_break = left_s > slope_eps and (left_s - right_s) >= slope_threshold
        is_falling_shoulder = (
            left_s < -slope_eps
            and right_s < -slope_eps
            and mid_s > left_s
            and mid_s > right_s
            and (mid_s - max(left_s, right_s)) >= slope_threshold
        )
        is_rising_onset = is_valid_rising_onset(i)

        if not (is_rising_break or is_falling_shoulder or is_rising_onset):
            continue

        snap_left = max(left_bound, i - snap)
        snap_right = min(right_bound, i + snap)
        snap_candidates = [
            j
            for j in range(snap_left, snap_right + 1)
            if all(abs(j - p) > min_peak_distance for p in strict)
        ]

        if snap_candidates:
            if is_falling_shoulder:
                i = int(max(snap_candidates, key=lambda j: float(mag_f[j])))
            elif is_rising_onset:
                onset_candidates = [j for j in snap_candidates if is_valid_rising_onset(j)]
                if onset_candidates:
                    i = int(max(onset_candidates, key=lambda j: slope_pair(j)[1] - slope_pair(j)[0] if slope_pair(j) else -np.inf))
                i = refine_to_detrended_crest(i)
            else:
                i = int(max(snap_candidates, key=lambda j: float(mag_f[j])))
                i = refine_to_detrended_crest(i)

        if any(abs(i - p) <= min_peak_distance for p in strict):
            continue
        if left_bound <= i <= right_bound and float(mag_f[i]) >= float(min_height):
            candidates.add(int(i))

    return suppress_nearby_candidates(
        sorted(candidates),
        mag_f,
        min_distance=max(int(min_peak_distance), 2 * int(cfg.snap_window) + 1),
        keep="strongest",
    )


def _candidate_bounds_from_search_mask(search_mask: np.ndarray) -> tuple[int, int]:
    valid = np.flatnonzero(search_mask)
    if valid.size == 0:
        return 0, search_mask.size - 1
    return int(valid[0]), int(valid[-1])


def _sample_complex_on_lags(lag_axis: np.ndarray, values: np.ndarray, query_lags: np.ndarray) -> np.ndarray:
    """Linearly interpolate complex values on an integer lag axis, zero outside."""
    x = lag_axis.astype(float)
    q = np.asarray(query_lags, dtype=float)
    real = np.interp(q, x, np.real(values), left=0.0, right=0.0)
    imag = np.interp(q, x, np.imag(values), left=0.0, right=0.0)
    return real + 1j * imag


def _design_matrix_from_acf(
    lags_corr: np.ndarray,
    lags_acf: np.ndarray,
    acf: np.ndarray,
    delays_samples: np.ndarray,
) -> np.ndarray:
    """Build columns R_xx[k - tau_l] for candidate delays tau_l."""
    k = lags_corr.astype(float)
    cols = []
    for tau in np.asarray(delays_samples, dtype=float):
        cols.append(_sample_complex_on_lags(lags_acf, acf, k - float(tau)))
    if not cols:
        return np.empty((lags_corr.size, 0), dtype=np.complex128)
    return np.column_stack(cols).astype(np.complex128, copy=False)


def _least_squares_model(
    y: np.ndarray,
    lags_fit: np.ndarray,
    lags_acf: np.ndarray,
    acf: np.ndarray,
    delays_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit complex amplitudes for fixed delays."""
    delays = np.asarray(delays_samples, dtype=float)
    if delays.size == 0:
        residual = np.asarray(y, dtype=np.complex128)
        return np.empty(0, dtype=np.complex128), np.zeros_like(residual), float(np.vdot(residual, residual).real)

    A = _design_matrix_from_acf(lags_fit, lags_acf, acf, delays)
    if A.size == 0:
        residual = np.asarray(y, dtype=np.complex128)
        return np.empty(0, dtype=np.complex128), np.zeros_like(residual), float(np.vdot(residual, residual).real)

    alpha, *_ = np.linalg.lstsq(A, y, rcond=None)
    model = A @ alpha
    residual = y - model
    sse = float(np.vdot(residual, residual).real)
    return alpha, model, sse


def _refine_delays_coordinate_grid(
    y: np.ndarray,
    lags_fit: np.ndarray,
    lags_acf: np.ndarray,
    acf: np.ndarray,
    initial_delays: np.ndarray,
    *,
    radius: float,
    grid_points: int,
    iterations: int,
) -> np.ndarray:
    """Simple deterministic coordinate grid refinement of delays."""
    delays = np.asarray(initial_delays, dtype=float).copy()
    if delays.size == 0:
        return delays

    radius = max(0.0, float(radius))
    grid_points = max(3, int(grid_points))
    iterations = max(0, int(iterations))
    if radius == 0.0 or iterations == 0:
        return delays

    offsets = np.linspace(-radius, radius, grid_points)
    _alpha, _model, best_sse = _least_squares_model(y, lags_fit, lags_acf, acf, delays)

    for _ in range(iterations):
        improved = False
        for m in range(delays.size):
            current = float(delays[m])
            local_best_delay = current
            local_best_sse = best_sse
            for off in offsets:
                trial = delays.copy()
                trial[m] = current + float(off)
                # Keep path order stable, but allow very close hypotheses.
                alpha, model, sse = _least_squares_model(y, lags_fit, lags_acf, acf, trial)
                if sse < local_best_sse:
                    local_best_sse = sse
                    local_best_delay = float(trial[m])
            if local_best_sse < best_sse:
                delays[m] = local_best_delay
                best_sse = local_best_sse
                improved = True
        if not improved:
            break

    return delays


def _apply_complex_model_fit(
    candidates: list[EchoCandidate],
    *,
    correlation: np.ndarray,
    lags: np.ndarray,
    reference: np.ndarray,
    main_lobe: MainLobeMetrics,
    cfg: EchoEstimatorConfig,
) -> tuple[list[EchoCandidate], float | None, float | None]:
    """Fit and validate candidates using a complex autocorrelation model."""
    fit_cfg = cfg.model_fit
    if not fit_cfg.enabled or not candidates:
        # Still provide parabolic refinement for strict peaks.
        for cand in candidates:
            if cand.kind == "strict_peak":
                cand.refined_lag_samples = float(lags[0] + (parabolic_subsample_refinement(np.abs(correlation), cand.index) - 0))
            else:
                cand.refined_lag_samples = float(cand.lag_samples)
            cand.refined_delay_s = float(cand.refined_lag_samples / cfg.sample_rate_hz - cfg.calibration_delay_s)
            cand.accepted_by_model = None
        return candidates, None, None

    # Limit model complexity by score. Keep earliest candidate if it is not among
    # the highest-scoring ones, because first-path detection is often the target.
    ordered_by_score = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected = ordered_by_score[: max(1, fit_cfg.max_candidates)]
    earliest = min(candidates, key=lambda c: c.lag_samples)
    if earliest not in selected:
        selected[-1] = earliest
    selected = sorted({id(c): c for c in selected}.values(), key=lambda c: c.lag_samples)

    delays0 = np.array([float(c.lag_samples) for c in selected], dtype=float)

    ref = _as_1d_complex(reference, "reference")
    acf = autocorr_fft(ref, normalize=True)
    lags_acf = correlation_lags(ref.size, ref.size)

    pad = int(math.ceil(max(1, main_lobe.guard_half_width_samples) * float(fit_cfg.fit_padding_main_lobes)))
    min_lag = int(math.floor(np.min(delays0))) - pad
    max_lag = int(math.ceil(np.max(delays0))) + pad
    fit_mask = (lags >= min_lag) & (lags <= max_lag)
    if not np.any(fit_mask):
        fit_mask = np.ones(lags.size, dtype=bool)

    y = correlation[fit_mask]
    lags_fit = lags[fit_mask]
    total_power = float(np.vdot(y, y).real)

    delays = delays0.copy()
    if fit_cfg.coordinate_refinement and delays.size <= fit_cfg.max_candidates:
        delays = _refine_delays_coordinate_grid(
            y,
            lags_fit,
            lags_acf,
            acf,
            delays,
            radius=fit_cfg.refinement_radius_samples,
            grid_points=fit_cfg.refinement_grid_points,
            iterations=fit_cfg.refinement_iterations,
        )

    alpha, model, sse_full = _least_squares_model(y, lags_fit, lags_acf, acf, delays)

    amp_abs = np.abs(alpha) if alpha.size else np.array([], dtype=float)
    strongest_amp = float(np.max(amp_abs)) if amp_abs.size else 0.0

    # Leave-one-out improvement in dB.
    improvements: list[float | None] = []
    for m in range(delays.size):
        if delays.size == 1:
            improvements.append(float(db10(total_power / max(sse_full, 1e-300))) if total_power > 0 else None)
            continue
        mask = np.ones(delays.size, dtype=bool)
        mask[m] = False
        _alpha_wo, _model_wo, sse_wo = _least_squares_model(y, lags_fit, lags_acf, acf, delays[mask])
        improvements.append(float(db10(max(sse_wo, 1e-300) / max(sse_full, 1e-300))))

    selected_ids = {id(c): idx for idx, c in enumerate(selected)}
    for cand in candidates:
        if id(cand) not in selected_ids:
            cand.accepted_by_model = False
            cand.refined_lag_samples = float(cand.lag_samples)
            cand.refined_delay_s = float(cand.refined_lag_samples / cfg.sample_rate_hz - cfg.calibration_delay_s)
            continue

        m = selected_ids[id(cand)]
        cand.refined_lag_samples = float(delays[m])
        cand.refined_delay_s = float(delays[m] / cfg.sample_rate_hz - cfg.calibration_delay_s)
        cand.amplitude = complex(alpha[m]) if alpha.size else None
        cand.amplitude_abs = float(abs(alpha[m])) if alpha.size else None
        cand.amplitude_phase_rad = float(np.angle(alpha[m])) if alpha.size else None
        cand.model_path_improvement_db = improvements[m]

        amp_ok = (
            strongest_amp <= 0.0
            or (cand.amplitude_abs is not None and cand.amplitude_abs >= fit_cfg.min_amplitude_rel_to_strongest * strongest_amp)
        )
        improvement_ok = (
            cand.model_path_improvement_db is not None
            and cand.model_path_improvement_db >= fit_cfg.min_path_improvement_db
        )
        cfar_strict_ok = (
            fit_cfg.keep_strict_cfar_even_if_model_weak
            and cand.kind == "strict_peak"
            and cand.cfar_margin_db is not None
            and cand.cfar_margin_db >= 0.0
        )

        cand.accepted_by_model = bool((amp_ok and improvement_ok) or cfar_strict_ok)

    return candidates, sse_full, total_power


def estimate_echoes(
    received: np.ndarray,
    reference: np.ndarray,
    config: EchoEstimatorConfig,
) -> EchoEstimationResult:
    """Estimate LOS/echo times from a received waveform and known reference.

    Parameters
    ----------
    received:
        Complex baseband receive samples.
    reference:
        Complex reference waveform, e.g. the transmitted Zadoff-Chu sequence
        including any pulse shaping/resampling that was used in the matched
        filter.
    config:
        Estimator configuration.

    Returns
    -------
    EchoEstimationResult
        Full result, including correlation, PDP, CFAR threshold, candidates and
        accepted echo estimates.

    Method summary
    --------------
    1. FFT cross-correlation normalized by reference energy.
    2. PDP = |correlation|^2.
    3. Main-lobe metrics from measured reference autocorrelation.
    4. CA-CFAR/median-CFAR strict peak detection.
    5. Optional flank shoulder candidates for weak first paths.
    6. Optional complex least-squares fit using shifted autocorrelation columns.
    """
    rx = _as_1d_complex(received, "received")
    ref = _as_1d_complex(reference, "reference")
    cfg = config

    correlation = xcorr_fft(rx, ref, normalize=True)
    lags = correlation_lags(rx.size, ref.size)
    pdp = np.abs(correlation) ** 2

    main_lobe = estimate_main_lobe_metrics(ref)
    guard = cfg.cfar.resolved_guard_cells(main_lobe.guard_half_width_samples)
    min_peak_distance = int(cfg.min_peak_distance_samples) if cfg.min_peak_distance_samples is not None else max(1, main_lobe.guard_half_width_samples)

    search_mask = _build_search_mask(lags, cfg, pdp)

    if cfg.cfar.enabled:
        cfar_mask, cfar_threshold, cfar_noise = cfar_1d(
            pdp,
            train_cells=cfg.cfar.train_cells,
            guard_cells=guard,
            pfa=cfg.cfar.pfa,
            mode=cfg.cfar.mode,
            median_scale=cfg.cfar.median_scale,
            minimum_noise_power=cfg.cfar.minimum_noise_power,
        )
    else:
        cfar_mask = np.ones(pdp.size, dtype=bool)
        cfar_threshold = np.full(pdp.size, np.nan, dtype=float)
        cfar_noise = np.full(pdp.size, np.nan, dtype=float)

    strongest_power = float(np.max(pdp[search_mask])) if np.any(search_mask) else float(np.max(pdp))
    min_power = max(0.0, float(cfg.min_relative_power_to_strongest)) * strongest_power

    strict_indices = _detect_strict_peaks(
        pdp,
        search_mask=search_mask,
        cfar_mask=cfar_mask,
        min_power=min_power,
        min_distance=min_peak_distance,
    )

    # Fallback: if CFAR finds nothing, keep the strongest point in the search
    # region so downstream code can still provide a deterministic estimate.
    if not strict_indices and np.any(search_mask):
        idx = int(np.argmax(np.where(search_mask, pdp, -np.inf)))
        if np.isfinite(pdp[idx]) and pdp[idx] > 0.0:
            strict_indices = [idx]

    if cfg.collapse_unresolved:
        strict_indices = collapse_unresolved_lobes(
            strict_indices,
            pdp,
            min_distance=min_peak_distance,
            min_valley_drop_rel=0.10,
            min_valley_drop_sigma=0.25,
            smooth_window=3,
            keep="strongest",
        )

    left_bound, right_bound = _candidate_bounds_from_search_mask(search_mask)
    center_idx = int(np.argmax(np.where(search_mask, pdp, -np.inf))) if np.any(search_mask) else int(np.argmax(pdp))
    mag = np.abs(correlation)
    min_height = math.sqrt(max(min_power, 0.0))

    shoulder_left: list[int] = []
    shoulder_echo: list[int] = []
    if cfg.shoulder.enabled and strict_indices:
        shoulder_left = _detect_left_shoulder_candidates(
            mag,
            center_idx=center_idx,
            left_bound=left_bound,
            right_bound=right_bound,
            min_height=min_height,
            min_peak_distance=min_peak_distance,
            cfg=cfg.shoulder,
        )

        provisional = sorted({*strict_indices, *shoulder_left})
        los_anchor = min(provisional) if provisional else min(strict_indices)
        shoulder_echo = _detect_echo_shoulder_candidates(
            mag,
            strict_peak_indices=strict_indices,
            los_anchor_idx=los_anchor,
            left_bound=left_bound,
            right_bound=right_bound,
            min_height=min_height,
            min_peak_distance=min_peak_distance,
            cfg=cfg.shoulder,
        )

    strict_set = set(strict_indices)
    left_set = set(shoulder_left)
    echo_set = set(shoulder_echo)

    all_indices = sorted(strict_set | left_set | echo_set)
    candidates: list[EchoCandidate] = []
    for idx in all_indices:
        if idx in strict_set:
            kind: Literal["strict_peak", "left_shoulder", "echo_shoulder", "manual"] = "strict_peak"
            score = float(pdp[idx])
        elif idx in left_set:
            kind = "left_shoulder"
            score = float(pdp[idx]) * 0.95
        else:
            kind = "echo_shoulder"
            score = float(pdp[idx]) * 0.90
        candidates.append(
            _candidate_from_index(
                idx,
                kind=kind,
                lags=lags,
                pdp=pdp,
                threshold=cfar_threshold,
                sample_rate_hz=cfg.sample_rate_hz,
                calibration_delay_s=cfg.calibration_delay_s,
                score=score,
            )
        )

    # Sort by physical delay. This is important because "LOS" is the earliest
    # accepted path, not necessarily the strongest path.
    candidates.sort(key=lambda c: c.lag_samples)

    candidates, model_residual_power, model_total_power = _apply_complex_model_fit(
        candidates,
        correlation=correlation,
        lags=lags,
        reference=ref,
        main_lobe=main_lobe,
        cfg=cfg,
    )

    echoes = [c for c in candidates if c.accepted_by_model is not False]
    echoes.sort(key=lambda c: (c.refined_lag_samples if c.refined_lag_samples is not None else c.lag_samples))

    # Enforce max_echoes after sorting by delay, because the primary scientific
    # output is a time-ordered list of arrivals.
    echoes = echoes[: cfg.max_echoes]

    return EchoEstimationResult(
        config=cfg,
        lags=lags,
        correlation=correlation,
        pdp=pdp,
        cfar_threshold=cfar_threshold,
        cfar_noise=cfar_noise,
        search_mask=search_mask,
        main_lobe=main_lobe,
        candidates=candidates,
        echoes=echoes,
        model_residual_power=model_residual_power,
        model_total_power=model_total_power,
    )


def zadoff_chu_sequence(root: int, length: int, *, cyclic_shift: int = 0) -> np.ndarray:
    """Generate a unit-magnitude Zadoff-Chu sequence.

    Parameters
    ----------
    root:
        ZC root. It should be coprime with length.
    length:
        Sequence length.
    cyclic_shift:
        Optional cyclic shift in samples.

    Returns
    -------
    np.ndarray
        Complex Zadoff-Chu sequence of shape (length,).

    Notes
    -----
    For even length N: x[n] = exp(-j*pi*u*n^2/N)
    For odd length N:  x[n] = exp(-j*pi*u*n*(n+1)/N)
    """
    N = int(length)
    u = int(root)
    if N <= 0:
        raise ValueError("length must be positive.")
    if math.gcd(u, N) != 1:
        raise ValueError("root and length should be coprime for a Zadoff-Chu sequence.")

    n = np.arange(N, dtype=float)
    if N % 2 == 0:
        x = np.exp(-1j * np.pi * u * n * n / N)
    else:
        x = np.exp(-1j * np.pi * u * n * (n + 1.0) / N)

    if cyclic_shift:
        x = np.roll(x, int(cyclic_shift))
    return x.astype(np.complex128)


def simulate_multipath_received(
    reference: np.ndarray,
    *,
    delays_samples: list[float],
    amplitudes: list[complex],
    noise_power: float = 0.0,
    leading_zeros: int = 0,
    trailing_zeros: int = 128,
    seed: int | None = None,
) -> np.ndarray:
    """Small deterministic simulation helper for unit tests and examples.

    Fractional delays are applied by linearly interpolating the reference. This
    is not a replacement for a physical RF channel simulator, but it is useful
    for validating the estimator workflow.
    """
    ref = _as_1d_complex(reference, "reference")
    if len(delays_samples) != len(amplitudes):
        raise ValueError("delays_samples and amplitudes must have the same length.")
    if not delays_samples:
        raise ValueError("at least one path is required.")

    max_delay = int(math.ceil(max(delays_samples)))
    n_out = int(leading_zeros) + max_delay + ref.size + int(trailing_zeros)
    out = np.zeros(n_out, dtype=np.complex128)

    n_ref = np.arange(ref.size, dtype=float)
    for delay, amp in zip(delays_samples, amplitudes):
        delay_total = float(leading_zeros) + float(delay)
        n_out_axis = np.arange(n_out, dtype=float)
        q = n_out_axis - delay_total
        real = np.interp(q, n_ref, np.real(ref), left=0.0, right=0.0)
        imag = np.interp(q, n_ref, np.imag(ref), left=0.0, right=0.0)
        out += complex(amp) * (real + 1j * imag)

    if noise_power > 0.0:
        rng = np.random.default_rng(seed)
        sigma = math.sqrt(noise_power / 2.0)
        noise = sigma * (rng.standard_normal(n_out) + 1j * rng.standard_normal(n_out))
        out += noise

    return out


def _demo() -> None:
    """Run a minimal smoke test when executing the file directly."""
    fs = 1e6
    zc = zadoff_chu_sequence(root=25, length=127)
    rx = simulate_multipath_received(
        zc,
        delays_samples=[40.2, 47.8, 91.0],
        amplitudes=[0.18 + 0.05j, 1.0 + 0.0j, 0.35 - 0.2j],
        noise_power=1e-3,
        leading_zeros=20,
        trailing_zeros=180,
        seed=7,
    )
    cfg = EchoEstimatorConfig(
        sample_rate_hz=fs,
        search_lag_min_samples=0,
        max_echoes=6,
        cfar=CFARConfig(train_cells=24, pfa=1e-4),
        model_fit=ModelFitConfig(enabled=True, refinement_iterations=2),
    )
    result = estimate_echoes(rx, zc, cfg)
    print("Main-lobe guard half-width:", result.main_lobe.guard_half_width_samples)
    print("Detected echoes:")
    for echo in result.echoes:
        print(echo.to_dict())


if __name__ == "__main__":
    _demo()
