from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np


class _RobustScaleEstimate(NamedTuple):
    """Internal robust location/scale estimate."""

    location: float
    sigma: float
    method: str


@dataclass(frozen=True)
class NoiseFloorEstimate:
    """Diagnostic noise-floor estimate for magnitude-domain CIR/correlation data.

    ``baseline`` is the robust background level in the magnitude domain.
    ``sigma`` is an empirical robust scale of background magnitude variations.
    ``threshold`` is the detection level used for significance tests.

    Notes
    -----
    The magnitude of complex Gaussian noise is Rayleigh/Rice distributed, not
    Gaussian.  Therefore ``sigma`` is intentionally documented as an empirical
    robust scale, not as a physical AWGN standard deviation.  If
    ``false_alarm_probability`` is supplied, a Rayleigh tail threshold is also
    computed from background samples and used as an additional lower bound.
    """

    baseline: float
    sigma: float
    threshold: float
    method: str
    n_background: int
    false_alarm_probability: float | None = None
    rayleigh_sigma: float | None = None
    rayleigh_threshold: float | None = None


@dataclass(frozen=True)
class PeakEvidence:
    """Diagnostic evidence for one selected path/peak marker."""

    index: int
    magnitude: float
    prominence: float
    threshold: float
    snr_db: float | None
    method: str


@dataclass(frozen=True)
class CIRPathDetection:
    """Detailed result of magnitude-domain path detection.

    The public legacy functions still return only indices.  This object is the
    scientific/diagnostic interface: it keeps the strongest path separate from
    the first significant path, stores the active search group, the noise-floor
    estimate, and per-marker evidence.  ``noise`` is kept as a backward-friendly
    alias for ``first_path_noise``.
    """

    strongest_idx: int | None
    first_path_idx: int | None
    los_idx: int | None
    echo_indices: list[int]
    group_indices: list[int]
    strict_peak_indices: list[int]
    shoulder_indices: list[int]
    left_bound: int
    right_bound: int
    noise: NoiseFloorEstimate
    first_path_noise: NoiseFloorEstimate
    echo_noise: NoiseFloorEstimate
    evidence: list[PeakEvidence]
    first_path_policy: str


FirstPathPolicy = Literal[
    "earliest_significant",
    "earliest_local_maximum",
    "strongest",
    "legacy",
]


def _as_finite_1d(y: np.ndarray) -> np.ndarray:
    """Return a finite float copy, linearly interpolating non-finite samples."""
    y = np.asarray(y, dtype=float).copy()
    if y.size == 0 or np.all(np.isfinite(y)):
        return y

    finite = np.isfinite(y)
    if not np.any(finite):
        return np.zeros_like(y, dtype=float)

    x = np.arange(y.size)
    y[~finite] = np.interp(x[~finite], x[finite], y[finite])
    return y


def _robust_scale_1d(x: np.ndarray) -> _RobustScaleEstimate:
    """Return a robust 1-D location/scale estimate.

    Preference order:
    1. MAD scaled to a Gaussian-equivalent sigma.
    2. IQR scaled to a Gaussian-equivalent sigma.
    3. Classical standard deviation as a degenerate fallback.

    This avoids hidden dependence on a single large peak or outlier when noise
    floors and slope thresholds are estimated from a peak group.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return _RobustScaleEstimate(location=0.0, sigma=0.0, method="empty")

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad
    method = "mad"

    if sigma <= 1e-12 and x.size >= 4:
        q25, q75 = np.percentile(x, [25.0, 75.0])
        sigma = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
        method = "iqr"

    if sigma <= 1e-12 and x.size >= 2:
        sigma = float(np.std(x))
        method = "std"

    return _RobustScaleEstimate(location=med, sigma=max(0.0, sigma), method=method)


def _robust_sigma_1d(x: np.ndarray) -> float:
    """Return a robust sigma estimate for a 1-D array."""
    return _robust_scale_1d(x).sigma


def _odd_window(value: int, *, minimum: int = 3) -> int:
    """Return an odd integer window >= minimum."""
    value = max(int(minimum), int(value))
    if value % 2 == 0:
        value += 1
    return value


def _smooth_1d_edge(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding, preserving array length."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y.copy()

    window = _odd_window(window, minimum=1)
    if window <= 1:
        return y.copy()

    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    return np.convolve(np.pad(y, pad, mode="edge"), kernel, mode="valid")


def _merge_close_indices(indices: list[int], *, max_gap: int) -> list[list[int]]:
    """Merge sorted indices into runs when consecutive hits are close."""
    if not indices:
        return []

    max_gap = max(1, int(max_gap))
    sorted_indices = sorted({int(i) for i in indices})
    runs: list[list[int]] = [[sorted_indices[0]]]

    for idx in sorted_indices[1:]:
        if idx - runs[-1][-1] <= max_gap:
            runs[-1].append(idx)
        else:
            runs.append([idx])

    return runs


def _local_maxima_compat(
    mag: np.ndarray,
    *,
    left_bound: int,
    right_bound: int,
    min_height: float,
) -> list[int]:
    """Return local maxima with the same plateau behavior as the old component."""
    n = int(mag.size)
    if n < 3:
        return []

    left = max(1, int(left_bound))
    right_exclusive = min(n - 1, int(right_bound) + 1)
    if right_exclusive <= left:
        return []

    min_height = float(min_height)
    return [
        int(i)
        for i in range(left, right_exclusive)
        if (
            mag[i] >= mag[i - 1]
            and mag[i] >= mag[i + 1]
            and (mag[i] > mag[i - 1] or mag[i] > mag[i + 1])
            and mag[i] >= min_height
        )
    ]


def _peak_search_bounds(
    mag: np.ndarray,
    *,
    center_idx: int,
    repetition_period_samples: int | None = None,
) -> tuple[int, int]:
    """Return inclusive [left_bound, right_bound] used for peak search.

    If the repetition period is known, the active group is the half-period
    window around the anchor.  Otherwise, the method searches for the first
    separating valley whose far side contains another dominant lobe.  This is a
    conservative segmentation rule: weaker echoes remain in the same group.
    """
    work_mag = _as_finite_1d(mag)
    if work_mag.size == 0:
        return 0, -1

    center_idx = int(np.clip(center_idx, 0, work_mag.size - 1))
    center_mag = float(work_mag[center_idx])

    if repetition_period_samples is not None and repetition_period_samples > 1:
        half_period = max(1, int(round(repetition_period_samples / 2.0)))
        left_bound = max(0, center_idx - half_period)
        right_bound = min(work_mag.size - 1, center_idx + half_period)
        return left_bound, right_bound

    left_bound = 0
    right_bound = work_mag.size - 1
    main_lobe_threshold = 0.8 * center_mag

    for idx in range(center_idx - 1, 0, -1):
        if work_mag[idx] <= work_mag[idx - 1] and work_mag[idx] <= work_mag[idx + 1]:
            has_new_main_lobe = any(
                work_mag[j] >= work_mag[j - 1]
                and work_mag[j] >= work_mag[j + 1]
                and work_mag[j] >= main_lobe_threshold
                for j in range(idx - 1, 0, -1)
            )
            if has_new_main_lobe:
                left_bound = idx
                break

    for idx in range(center_idx + 1, work_mag.size - 1):
        if work_mag[idx] <= work_mag[idx - 1] and work_mag[idx] <= work_mag[idx + 1]:
            has_new_main_lobe = any(
                work_mag[j] >= work_mag[j - 1]
                and work_mag[j] >= work_mag[j + 1]
                and work_mag[j] >= main_lobe_threshold
                for j in range(idx + 1, work_mag.size - 1)
            )
            if has_new_main_lobe:
                right_bound = idx
                break

    return left_bound, right_bound


def estimate_noise_floor_from_mag(
    mag: np.ndarray,
    *,
    left_bound: int | None = None,
    right_bound: int | None = None,
    exclude_indices: list[int] | None = None,
    exclude_radius: int = 3,
    noise_sigma_factor: float = 3.0,
    false_alarm_probability: float | None = None,
    min_background_fraction: float = 0.20,
) -> NoiseFloorEstimate:
    """Estimate a robust empirical noise floor in the magnitude domain.

    Parameters
    ----------
    mag:
        Magnitude-domain correlation/CIR samples.
    left_bound, right_bound:
        Optional inclusive search region.  Background statistics are preferably
        computed from this region, excluding neighborhoods around known peaks.
    exclude_indices:
        Candidate/path indices to remove from the background sample set.
    exclude_radius:
        Half-width around each excluded index.
    noise_sigma_factor:
        Empirical prominence threshold in units of robust background scale.
    false_alarm_probability:
        Optional per-search false alarm probability.  If supplied, a Rayleigh
        tail threshold is estimated from the background median and used as an
        additional threshold.  This is only an approximation because the input
        is magnitude-domain and may contain correlation sidelobes.
    min_background_fraction:
        If too many samples are excluded, fall back to a less aggressive mask.
    """
    work_mag = _as_finite_1d(mag)
    n = int(work_mag.size)
    if n == 0:
        return NoiseFloorEstimate(
            baseline=0.0,
            sigma=0.0,
            threshold=0.0,
            method="empty",
            n_background=0,
            false_alarm_probability=false_alarm_probability,
        )

    left = 0 if left_bound is None else int(np.clip(left_bound, 0, n - 1))
    right = n - 1 if right_bound is None else int(np.clip(right_bound, 0, n - 1))
    if right < left:
        left, right = 0, n - 1

    segment = work_mag[left : right + 1]
    mask = np.ones(segment.size, dtype=bool)

    if exclude_indices:
        radius = max(0, int(exclude_radius))
        for idx in exclude_indices:
            idx = int(idx)
            if idx < left or idx > right:
                continue
            local = idx - left
            a = max(0, local - radius)
            b = min(segment.size, local + radius + 1)
            mask[a:b] = False

    min_count = max(8, int(np.ceil(float(min_background_fraction) * segment.size)))
    if int(mask.sum()) < min_count:
        # Fall back to all samples in the active segment.  This prevents the
        # threshold from becoming unstable in dense multipath or very short CIRs.
        background = segment
        mask_method = "all_segment"
    else:
        background = segment[mask]
        mask_method = "peak_excluded"

    robust = _robust_scale_1d(background)
    baseline = max(0.0, float(robust.location))
    sigma = max(0.0, float(robust.sigma))

    if sigma <= 1e-12 and background.size >= 2:
        # Magnitude data with a flat floor is common after quantization or after
        # a previous preprocessing step.  Accept positive prominence above the
        # floor instead of silently returning NaN/inf thresholds.
        sigma = 0.0

    empirical_threshold = baseline + max(0.0, float(noise_sigma_factor)) * sigma

    rayleigh_sigma: float | None = None
    rayleigh_threshold: float | None = None
    threshold = empirical_threshold
    method = f"{robust.method}:{mask_method}:empirical"

    if false_alarm_probability is not None and background.size > 0:
        pfa = float(false_alarm_probability)
        if 0.0 < pfa < 1.0:
            # Estimate Rayleigh scale from the median of non-negative magnitude
            # samples.  This is a conservative optional statistical guardrail.
            bg_nonnegative = np.clip(np.asarray(background, dtype=float), 0.0, None)
            median_mag = float(np.median(bg_nonnegative))
            if median_mag > 0.0:
                rayleigh_sigma = median_mag / np.sqrt(2.0 * np.log(2.0))
                # Bonferroni-style per-bin correction inside the active search.
                pfa_bin = max(np.finfo(float).tiny, pfa / max(1, int(segment.size)))
                rayleigh_threshold = float(
                    rayleigh_sigma * np.sqrt(-2.0 * np.log(pfa_bin))
                )
                threshold = max(threshold, rayleigh_threshold)
                method += "+rayleigh_pfa"

    return NoiseFloorEstimate(
        baseline=float(baseline),
        sigma=float(sigma),
        threshold=float(threshold),
        method=method,
        n_background=int(background.size),
        false_alarm_probability=false_alarm_probability,
        rayleigh_sigma=rayleigh_sigma,
        rayleigh_threshold=rayleigh_threshold,
    )


def _prominence_above_noise(mag: np.ndarray, idx: int, noise: NoiseFloorEstimate) -> float:
    idx = int(np.clip(idx, 0, mag.size - 1))
    return max(0.0, float(mag[idx]) - float(noise.baseline))


def _snr_db_from_prominence(prominence: float, noise: NoiseFloorEstimate) -> float | None:
    sigma = float(noise.sigma)
    if sigma <= 1e-12:
        return None
    return float(20.0 * np.log10(max(float(prominence), 1e-300) / sigma))


def _significant_peak_indices(
    mag: np.ndarray,
    candidate_indices: list[int],
    *,
    noise: NoiseFloorEstimate,
    min_prominence: float = 0.0,
) -> list[int]:
    threshold = max(float(noise.threshold), float(noise.baseline) + max(0.0, float(min_prominence)))
    return [
        int(idx)
        for idx in sorted({int(i) for i in candidate_indices})
        if 0 <= int(idx) < mag.size and float(mag[int(idx)]) >= threshold
    ]


def find_shoulder_candidates_from_mag(
    mag: np.ndarray,
    *,
    center_idx: int,
    strict_peak_indices: list[int],
    min_height: float,
    left_bound: int,
    right_bound: int,
    include_left_shoulders: bool = True,
    smooth_window: int = 5,
    slope_window: int = 3,
    slope_threshold_factor: float = 1.0,
    slope_eps_factor: float = 0.2,
    min_peak_distance: int = 2,
    snap_window: int = 2,
    noise_sigma_factor: float = 1.0,
    shoulder_merge_distance: int | None = None,
    residual_window: int | None = None,
    max_shoulders_per_interval: int = 1,
) -> list[int]:
    """Return shoulder-like peak candidates within the active peak group.

    The detector uses two independent pieces of evidence:

    * derivative evidence: a local relief of the signed slope relative to a
      slowly varying slope background;
    * amplitude evidence: a positive residual against a local linear baseline.

    Placement is separated from detection.  Detection marks a shoulder event on
    a monotonic flank; placement then moves the marker toward the most plausible
    hidden echo center using the zero crossing of the slope residual and the
    local maximum of the integrated residual.  The strict local-maxima detector
    remains the primary source of peaks; shoulder candidates are only added when
    ``include_shoulders=True`` in the public functions.
    """
    work_mag = _as_finite_1d(mag)
    n = work_mag.size
    if n < 7:
        return []

    left_bound = int(np.clip(left_bound, 0, n - 1))
    right_bound = int(np.clip(right_bound, 0, n - 1))
    if right_bound - left_bound < 6:
        return []

    center_idx = int(np.clip(center_idx, 0, n - 1))
    smooth_window = _odd_window(smooth_window, minimum=3)
    slope_window = max(1, int(slope_window))
    min_peak_distance = max(1, int(min_peak_distance))
    snap_window = max(0, int(snap_window))

    if residual_window is None:
        residual_window = max(
            snap_window + 1,
            smooth_window,
            2 * slope_window + 1,
        )
    residual_window = max(1, int(residual_window))

    if shoulder_merge_distance is None:
        shoulder_merge_distance = max(
            min_peak_distance + 1,
            smooth_window,
            2 * slope_window + 1,
            residual_window,
        )
    shoulder_merge_distance = max(1, int(shoulder_merge_distance))
    max_shoulders_per_interval = max(1, int(max_shoulders_per_interval))

    smooth_mag = _smooth_1d_edge(work_mag, smooth_window)
    slope = np.gradient(smooth_mag)

    relief_window = _odd_window(
        max(
            3 * smooth_window,
            2 * residual_window + 1,
            4 * slope_window + 3,
        ),
        minimum=3,
    )
    slope_background = _smooth_1d_edge(slope, relief_window)
    slope_relief = slope - slope_background

    strict = sorted(
        {
            int(i)
            for i in strict_peak_indices
            if left_bound <= int(i) <= right_bound
        }
    )
    if len(strict) < 2:
        return []

    strict_set = set(strict)

    def _too_close_to_strict(idx: int) -> bool:
        return any(abs(int(idx) - peak_idx) <= min_peak_distance for peak_idx in strict)

    def _valid_marker_idx(idx: int, segment_left: int, segment_right: int) -> bool:
        idx = int(idx)
        return (
            segment_left <= idx <= segment_right
            and left_bound <= idx <= right_bound
            and idx not in strict_set
            and not _too_close_to_strict(idx)
            and float(work_mag[idx]) >= float(min_height)
        )

    high_freq_residual = work_mag - smooth_mag
    global_mag_noise = _robust_sigma_1d(high_freq_residual[left_bound : right_bound + 1])
    global_mag_span = float(np.ptp(smooth_mag[left_bound : right_bound + 1]))

    residual_floor_global = max(
        0.0,
        float(noise_sigma_factor) * global_mag_noise,
        0.0025 * global_mag_span,
    )

    def _crossing_idx(*, idx0: int, idx1: int, level: float = 0.0) -> int:
        idx0 = int(idx0)
        idx1 = int(idx1)
        y0 = float(slope_relief[idx0] - level)
        y1 = float(slope_relief[idx1] - level)
        denom = y1 - y0

        if abs(denom) <= 1e-12:
            return idx1

        frac = float(np.clip(-y0 / denom, 0.0, 1.0))
        return int(np.clip(round(idx0 + frac * (idx1 - idx0)), 0, n - 1))

    def _slope_zero_center(
        *,
        run: list[int],
        core_idx: int,
        segment_left: int,
        segment_right: int,
        mode: str,
        margin: int,
    ) -> int:
        core_idx = int(core_idx)
        search_span = max(
            residual_window + len(run) + 1,
            shoulder_merge_distance,
            smooth_window + 2 * slope_window + 1,
        )

        segment_left = int(segment_left)
        segment_right = int(segment_right)
        search_left = max(segment_left + margin, left_bound + margin)
        search_right = min(segment_right - margin, right_bound - margin)

        if search_right < search_left:
            return int(np.clip(core_idx, segment_left, segment_right))

        if mode == "descending":
            start = int(np.clip(max(core_idx, run[-1]), search_left, search_right))
            end = int(np.clip(start + search_span, search_left, search_right))

            for j in range(start + 1, end + 1):
                if slope_relief[j - 1] > 0.0 and slope_relief[j] <= 0.0:
                    return _crossing_idx(idx0=j - 1, idx1=j)

            xs = np.arange(start, end + 1, dtype=int)

        elif mode == "ascending":
            start = int(np.clip(min(core_idx, run[0]), search_left, search_right))
            end = int(np.clip(start - search_span, search_left, search_right))

            for j in range(start, end, -1):
                if slope_relief[j - 1] > 0.0 and slope_relief[j] <= 0.0:
                    return _crossing_idx(idx0=j - 1, idx1=j)

            xs = np.arange(end, start + 1, dtype=int)

        else:
            raise ValueError("mode must be 'descending' or 'ascending'")

        if xs.size == 0:
            return int(np.clip(core_idx, segment_left, segment_right))

        best = int(xs[np.argmin(np.abs(slope_relief[xs]))])
        return int(np.clip(best, segment_left, segment_right))

    def _integrated_relief_best(
        *,
        event_left: int,
        event_right: int,
        segment_left: int,
        segment_right: int,
        fallback_idx: int,
    ) -> tuple[int | None, float]:
        segment_left = int(np.clip(segment_left, left_bound, right_bound))
        segment_right = int(np.clip(segment_right, left_bound, right_bound))

        event_left = int(np.clip(event_left, segment_left, segment_right))
        event_right = int(np.clip(event_right, segment_left, segment_right))
        if event_right - event_left < 2:
            return (
                (int(fallback_idx), 0.0)
                if _valid_marker_idx(fallback_idx, segment_left, segment_right)
                else (None, 0.0)
            )

        xs = np.arange(event_left, event_right + 1, dtype=int)
        component = np.cumsum(slope_relief[xs].astype(float))

        if component.size > 1:
            t = np.linspace(0.0, 1.0, component.size)
            component = component - (component[0] + t * (component[-1] - component[0]))

        smooth_component = _smooth_1d_edge(
            component,
            _odd_window(min(smooth_window, max(1, component.size)), minimum=1),
        )

        valid = np.array(
            [_valid_marker_idx(int(x), segment_left, segment_right) for x in xs],
            dtype=bool,
        )

        if not np.any(valid):
            return (
                (int(fallback_idx), 0.0)
                if _valid_marker_idx(fallback_idx, segment_left, segment_right)
                else (None, 0.0)
            )

        xs_valid = xs[valid]
        values_valid = smooth_component[valid]
        best_local = int(np.argmax(values_valid))
        best_idx = int(xs_valid[best_local])

        edge_level = float(np.median(smooth_component))
        gain = float(values_valid[best_local] - edge_level)

        if gain < residual_floor_global and _valid_marker_idx(
            fallback_idx,
            segment_left,
            segment_right,
        ):
            return int(fallback_idx), gain

        return best_idx, gain

    def _linear_residual_snap(
        *,
        center_idx_est: int,
        event_left: int,
        event_right: int,
        segment_left: int,
        segment_right: int,
    ) -> tuple[int | None, float]:
        segment_left = int(np.clip(segment_left, left_bound, right_bound))
        segment_right = int(np.clip(segment_right, left_bound, right_bound))
        event_left = int(np.clip(event_left, segment_left, segment_right))
        event_right = int(np.clip(event_right, segment_left, segment_right))
        center_idx_est = int(np.clip(center_idx_est, event_left, event_right))

        snap_radius = max(snap_window, 1)
        snap_left = max(event_left, center_idx_est - snap_radius)
        snap_right = min(event_right, center_idx_est + snap_radius)
        if snap_right < snap_left:
            return (
                (center_idx_est, 0.0)
                if _valid_marker_idx(center_idx_est, segment_left, segment_right)
                else (None, 0.0)
            )

        xs = np.arange(event_left, event_right + 1, dtype=int)
        if event_right == event_left:
            baseline = np.array([float(smooth_mag[event_left])], dtype=float)
        else:
            t = (xs.astype(float) - float(event_left)) / float(event_right - event_left)
            baseline = (
                float(smooth_mag[event_left])
                + t * (float(smooth_mag[event_right]) - float(smooth_mag[event_left]))
            )

        residual = smooth_mag[xs] - baseline
        snap_xs = np.arange(snap_left, snap_right + 1, dtype=int)
        valid = np.array(
            [_valid_marker_idx(int(x), segment_left, segment_right) for x in snap_xs],
            dtype=bool,
        )

        if not np.any(valid):
            return (
                (center_idx_est, 0.0)
                if _valid_marker_idx(center_idx_est, segment_left, segment_right)
                else (None, 0.0)
            )

        residual_lookup = {int(x): float(residual[int(x) - event_left]) for x in xs}
        values = np.array([residual_lookup[int(x)] for x in snap_xs[valid]], dtype=float)
        best_idx = int(snap_xs[valid][int(np.argmax(values))])
        best_residual = float(np.max(values))
        return best_idx, best_residual

    def _segment_candidates(
        *,
        segment_left: int,
        segment_right: int,
        mode: str,
    ) -> list[tuple[int, float]]:
        segment_left = int(np.clip(segment_left, left_bound, right_bound))
        segment_right = int(np.clip(segment_right, left_bound, right_bound))

        if segment_right - segment_left < max(6, 2 * slope_window + 2):
            return []

        delta = float(smooth_mag[segment_right] - smooth_mag[segment_left])

        if mode == "descending":
            if delta >= 0.0:
                return []
            signed_relief = slope_relief
            expected_slope_sign = -1.0
        elif mode == "ascending":
            if delta <= 0.0:
                return []
            signed_relief = -slope_relief
            expected_slope_sign = 1.0
        else:
            raise ValueError("mode must be 'descending' or 'ascending'")

        segment_len = max(1, segment_right - segment_left)
        trend_slope = abs(delta) / float(segment_len)

        margin = max(1, slope_window, min_peak_distance)
        search_left = max(segment_left + margin, left_bound + margin)
        search_right = min(segment_right - margin, right_bound - margin)
        if search_right < search_left:
            return []

        signed_segment = signed_relief[search_left : search_right + 1]
        slope_segment = slope[search_left : search_right + 1]

        relief_sigma = _robust_sigma_1d(signed_segment)
        slope_sigma = _robust_sigma_1d(slope_segment)

        if relief_sigma <= 1e-12:
            relief_sigma = max(
                1e-12,
                0.01 * float(np.ptp(signed_segment)) if signed_segment.size > 1 else 1e-12,
            )

        if slope_sigma <= 1e-12:
            slope_sigma = max(
                1e-12,
                0.01 * float(np.ptp(slope_segment)) if slope_segment.size > 1 else 1e-12,
            )

        threshold_from_mad = max(0.0, float(slope_threshold_factor)) * relief_sigma
        trend_floor = 0.025 * trend_slope
        trend_cap = 0.35 * trend_slope
        local_threshold = max(
            1e-12,
            trend_floor,
            min(threshold_from_mad, trend_cap),
        )

        sign_eps = max(
            1e-12,
            min(
                max(0.0, float(slope_eps_factor)) * slope_sigma,
                0.35 * trend_slope if trend_slope > 0.0 else np.inf,
            ),
        )

        raw_hits: list[int] = []
        hit_score: dict[int, float] = {}

        for i in range(search_left, search_right + 1):
            i = int(i)
            if _too_close_to_strict(i):
                continue
            if float(work_mag[i]) < float(min_height):
                continue

            score_i = float(signed_relief[i])
            if score_i < local_threshold:
                continue

            left_slice = slope[max(segment_left, i - slope_window) : i]
            right_slice = slope[i + 1 : min(segment_right + 1, i + 1 + slope_window)]
            if left_slice.size == 0 or right_slice.size == 0:
                continue

            left_slope = float(np.median(left_slice))
            right_slope = float(np.median(right_slice))

            if expected_slope_sign > 0.0:
                sign_ok = left_slope >= -sign_eps and right_slope >= -sign_eps
            else:
                sign_ok = left_slope <= sign_eps and right_slope <= sign_eps

            if not sign_ok:
                continue

            raw_hits.append(i)
            hit_score[i] = score_i

        if not raw_hits:
            return []

        candidates: list[tuple[int, float]] = []
        side_band = max(slope_window, smooth_window // 2, 1)

        for run in _merge_close_indices(raw_hits, max_gap=shoulder_merge_distance):
            core_idx = int(max(run, key=lambda idx: hit_score.get(int(idx), 0.0)))
            event_score = float(hit_score.get(core_idx, 0.0))

            left_band = signed_relief[max(segment_left, run[0] - side_band) : run[0]]
            right_band = signed_relief[
                run[-1] + 1 : min(segment_right + 1, run[-1] + 1 + side_band)
            ]

            side_values: list[float] = []
            if left_band.size:
                side_values.append(float(np.median(left_band)))
            if right_band.size:
                side_values.append(float(np.median(right_band)))

            boundary_level = max(side_values) if side_values else 0.0
            event_contrast = event_score - boundary_level

            if event_contrast < 0.12 * local_threshold and event_score < 1.50 * local_threshold:
                continue

            zero_idx = _slope_zero_center(
                run=run,
                core_idx=core_idx,
                segment_left=segment_left,
                segment_right=segment_right,
                mode=mode,
                margin=margin,
            )

            event_left = max(segment_left + margin, min(run[0], zero_idx) - residual_window)
            event_right = min(segment_right - margin, max(run[-1], zero_idx) + residual_window)

            best_idx, component_gain = _integrated_relief_best(
                event_left=event_left,
                event_right=event_right,
                segment_left=segment_left,
                segment_right=segment_right,
                fallback_idx=zero_idx,
            )
            if best_idx is None:
                continue

            max_center_shift = max(residual_window, smooth_window, 2 * slope_window + 1)
            if abs(best_idx - zero_idx) > max_center_shift and _valid_marker_idx(
                zero_idx,
                segment_left,
                segment_right,
            ):
                best_idx = int(zero_idx)

            snap_idx, snap_residual = _linear_residual_snap(
                center_idx_est=best_idx,
                event_left=event_left,
                event_right=event_right,
                segment_left=segment_left,
                segment_right=segment_right,
            )
            if (
                snap_idx is not None
                and abs(snap_idx - best_idx) <= max(1, snap_window)
                and snap_residual >= residual_floor_global
            ):
                best_idx = int(snap_idx)

            if not _valid_marker_idx(best_idx, segment_left, segment_right):
                if _valid_marker_idx(zero_idx, segment_left, segment_right):
                    best_idx = int(zero_idx)
                elif _valid_marker_idx(core_idx, segment_left, segment_right):
                    best_idx = int(core_idx)
                else:
                    continue

            placement_gain = max(
                float(component_gain),
                float(snap_residual),
                residual_floor_global,
                0.01 * global_mag_span,
                1e-12,
            )
            combined_score = (
                event_score
                * max(event_contrast, 0.20 * local_threshold, 1e-12)
                * placement_gain
            )
            candidates.append((int(best_idx), float(combined_score)))

        candidates.sort(key=lambda item: item[1], reverse=True)

        kept: list[tuple[int, float]] = []
        for idx, score in candidates:
            if any(abs(idx - kept_idx) <= shoulder_merge_distance for kept_idx, _ in kept):
                continue
            kept.append((idx, score))

        return kept

    all_candidates: list[tuple[int, float]] = []

    for a, b in zip(strict[:-1], strict[1:]):
        a = int(a)
        b = int(b)
        if b - a < 6:
            continue

        interval_left = max(left_bound, a)
        interval_right = min(right_bound, b)
        if interval_right - interval_left < 6:
            continue

        valley = int(interval_left + np.argmin(smooth_mag[interval_left : interval_right + 1]))

        interval_candidates: list[tuple[int, float]] = []

        interval_candidates.extend(
            _segment_candidates(
                segment_left=interval_left,
                segment_right=valley,
                mode="descending",
            )
        )

        if include_left_shoulders:
            interval_candidates.extend(
                _segment_candidates(
                    segment_left=valley,
                    segment_right=interval_right,
                    mode="ascending",
                )
            )

        interval_candidates.sort(key=lambda item: item[1], reverse=True)

        kept_in_interval: list[tuple[int, float]] = []
        for idx, score in interval_candidates:
            if any(abs(idx - kept_idx) <= shoulder_merge_distance for kept_idx, _ in kept_in_interval):
                continue

            kept_in_interval.append((idx, score))

            if len(kept_in_interval) >= max_shoulders_per_interval:
                break

        all_candidates.extend(kept_in_interval)

    if not all_candidates:
        return []

    all_candidates.sort(key=lambda item: item[1], reverse=True)

    final: list[tuple[int, float]] = []
    for idx, score in all_candidates:
        if _too_close_to_strict(idx):
            continue

        if any(abs(idx - kept_idx) <= shoulder_merge_distance for kept_idx, _ in final):
            continue

        final.append((idx, score))

    return sorted(int(idx) for idx, _ in final)


def _build_peak_evidence(
    mag: np.ndarray,
    indices: list[int],
    *,
    noise: NoiseFloorEstimate,
    echo_noise: NoiseFloorEstimate | None = None,
    shoulder_indices: set[int] | None = None,
    first_path_idx: int | None = None,
    strongest_idx: int | None = None,
) -> list[PeakEvidence]:
    shoulder_indices = shoulder_indices or set()
    evidence: list[PeakEvidence] = []
    for idx in sorted({int(i) for i in indices if 0 <= int(i) < mag.size}):
        noise_for_idx = noise if idx == first_path_idx else (echo_noise or noise)
        prominence = _prominence_above_noise(mag, idx, noise_for_idx)
        labels: list[str] = []
        if idx == strongest_idx:
            labels.append("strongest")
        if idx == first_path_idx:
            labels.append("first_path")
        labels.append("shoulder" if idx in shoulder_indices else "strict_peak")
        evidence.append(
            PeakEvidence(
                index=int(idx),
                magnitude=float(mag[idx]),
                prominence=float(prominence),
                threshold=float(noise_for_idx.threshold),
                snr_db=_snr_db_from_prominence(prominence, noise_for_idx),
                method="+".join(labels),
            )
        )
    return evidence


def detect_cir_paths_from_mag(
    mag: np.ndarray,
    *,
    repetition_period_samples: int | None = None,
    include_shoulders: bool = False,
    first_path_policy: FirstPathPolicy = "earliest_significant",
    min_rel_height: float = 0.0,
    peaks_before: int = 32,
    peaks_after: int = 32,
    noise_sigma_factor: float = 3.0,
    first_path_noise_sigma_factor: float | None = 3.0,
    noise_exclude_radius: int = 3,
    false_alarm_probability: float | None = None,
    min_echo_lag_samples: int = 2,
    max_echoes: int | None = None,
) -> CIRPathDetection:
    """Detect first path/LOS and echoes with diagnostics.

    This is the scientifically preferable interface.  It separates the
    strongest path from the first significant path and returns the noise-floor
    estimate used for significance filtering.  The legacy wrappers below keep
    their historical return shapes.

    Parameters
    ----------
    first_path_policy:
        ``"earliest_significant"`` chooses the earliest local maximum in the
        active group above the robust noise threshold.  ``"strongest"`` and
        ``"legacy"`` choose the strongest path and therefore reproduce the old
        dominant-peak LOS semantics.  ``"earliest_local_maximum"`` uses the
        earliest local maximum in the active group even if the threshold is not
        exceeded; this is useful only for very high-SNR, manually windowed data.
    noise_sigma_factor:
        Echo significance threshold.  The default ``3.0`` is stricter than the
        old permissive echo filter and reduces the chance that random local
        maxima between the first path and the strongest path become echoes.
        For plot-level continuity with the old component, set this to ``0.3``.
    first_path_noise_sigma_factor:
        LOS/first-path threshold.  The default ``3.0`` is deliberately stricter
        than the echo threshold, because choosing the earliest significant path
        with a permissive threshold can otherwise turn early background noise
        into a false LOS marker.  Set this to ``None`` to reuse
        ``noise_sigma_factor`` for both decisions.
    false_alarm_probability:
        Optional Rayleigh-tail guardrail for calibrated/high-stakes detection.
    """
    work_mag = _as_finite_1d(mag)
    n = int(work_mag.size)
    empty_noise = NoiseFloorEstimate(
        baseline=0.0,
        sigma=0.0,
        threshold=0.0,
        method="empty",
        n_background=0,
        false_alarm_probability=false_alarm_probability,
    )
    if n == 0:
        return CIRPathDetection(
            strongest_idx=None,
            first_path_idx=None,
            los_idx=None,
            echo_indices=[],
            group_indices=[],
            strict_peak_indices=[],
            shoulder_indices=[],
            left_bound=0,
            right_bound=-1,
            noise=empty_noise,
            first_path_noise=empty_noise,
            echo_noise=empty_noise,
            evidence=[],
            first_path_policy=str(first_path_policy),
        )

    strongest_idx = int(np.argmax(work_mag))
    left_bound, right_bound = _peak_search_bounds(
        work_mag,
        center_idx=strongest_idx,
        repetition_period_samples=repetition_period_samples,
    )
    if right_bound < left_bound:
        return CIRPathDetection(
            strongest_idx=strongest_idx,
            first_path_idx=strongest_idx,
            los_idx=strongest_idx,
            echo_indices=[],
            group_indices=[strongest_idx],
            strict_peak_indices=[strongest_idx],
            shoulder_indices=[],
            left_bound=left_bound,
            right_bound=right_bound,
            noise=empty_noise,
            first_path_noise=empty_noise,
            echo_noise=empty_noise,
            evidence=[],
            first_path_policy=str(first_path_policy),
        )

    # Keep the primary peak finder compatible: local maxima are identified with
    # the historical plateau rule.  ``min_rel_height`` remains optional, but the
    # default is zero so weak first paths are not suppressed solely because a
    # later reflection is stronger.
    min_height = max(0.0, float(min_rel_height)) * float(work_mag[strongest_idx])
    strict_candidates = _local_maxima_compat(
        work_mag,
        left_bound=left_bound,
        right_bound=right_bound,
        min_height=min_height,
    )
    if strongest_idx not in strict_candidates:
        strict_candidates.append(strongest_idx)
    strict_candidates = sorted({int(i) for i in strict_candidates})

    shoulder_indices: list[int] = []
    if include_shoulders and len(strict_candidates) >= 2:
        shoulder_indices = find_shoulder_candidates_from_mag(
            work_mag,
            center_idx=strongest_idx,
            strict_peak_indices=strict_candidates,
            min_height=min_height,
            left_bound=left_bound,
            right_bound=right_bound,
        )

    all_candidates = sorted({*strict_candidates, *shoulder_indices})

    first_factor = (
        float(noise_sigma_factor)
        if first_path_noise_sigma_factor is None
        else float(first_path_noise_sigma_factor)
    )

    first_path_noise = estimate_noise_floor_from_mag(
        work_mag,
        left_bound=left_bound,
        right_bound=right_bound,
        exclude_indices=all_candidates,
        exclude_radius=noise_exclude_radius,
        noise_sigma_factor=first_factor,
        false_alarm_probability=false_alarm_probability,
    )
    echo_noise = estimate_noise_floor_from_mag(
        work_mag,
        left_bound=left_bound,
        right_bound=right_bound,
        exclude_indices=all_candidates,
        exclude_radius=noise_exclude_radius,
        noise_sigma_factor=noise_sigma_factor,
        false_alarm_probability=false_alarm_probability,
    )

    first_significant = _significant_peak_indices(
        work_mag, all_candidates, noise=first_path_noise
    )
    echo_significant = _significant_peak_indices(
        work_mag, all_candidates, noise=echo_noise
    )

    # Never lose the dominant path.  If the robust threshold is too strict for a
    # very short/dense segment, we fall back to the strongest peak rather than
    # returning no marker.
    if strongest_idx not in first_significant:
        first_significant.append(strongest_idx)
        first_significant.sort()
    if strongest_idx not in echo_significant:
        echo_significant.append(strongest_idx)
        echo_significant.sort()

    if first_path_policy in {"strongest", "legacy"}:
        first_path_idx = strongest_idx
    elif first_path_policy == "earliest_local_maximum":
        first_path_idx = int(min(all_candidates)) if all_candidates else strongest_idx
    elif first_path_policy == "earliest_significant":
        first_path_idx = int(min(first_significant)) if first_significant else strongest_idx
    else:
        raise ValueError(
            "first_path_policy must be 'earliest_significant', "
            "'earliest_local_maximum', 'strongest', or 'legacy'"
        )

    min_echo_gap = max(0, int(min_echo_lag_samples))
    echo_indices = [
        int(idx)
        for idx in echo_significant
        if int(idx) > int(first_path_idx) + min_echo_gap
    ]
    if max_echoes is not None:
        echo_indices = echo_indices[: max(0, int(max_echoes))]

    group_indices = sorted({int(first_path_idx), int(strongest_idx), *echo_indices})
    # Keep diagnostics complete for significant candidates while LOS/echo
    # semantics remain governed by first_path_idx and echo_indices.
    group_indices = sorted({*group_indices, *first_significant})

    evidence = _build_peak_evidence(
        work_mag,
        group_indices,
        noise=first_path_noise,
        echo_noise=echo_noise,
        shoulder_indices=set(shoulder_indices),
        first_path_idx=first_path_idx,
        strongest_idx=strongest_idx,
    )

    return CIRPathDetection(
        strongest_idx=int(strongest_idx),
        first_path_idx=int(first_path_idx),
        los_idx=int(first_path_idx),
        echo_indices=[int(i) for i in echo_indices],
        group_indices=[int(i) for i in group_indices],
        strict_peak_indices=[int(i) for i in strict_candidates],
        shoulder_indices=[int(i) for i in shoulder_indices],
        left_bound=int(left_bound),
        right_bound=int(right_bound),
        noise=first_path_noise,
        first_path_noise=first_path_noise,
        echo_noise=echo_noise,
        evidence=evidence,
        first_path_policy=str(first_path_policy),
    )


def apply_manual_lags(
    lags: np.ndarray,
    los_idx: int | None,
    echo_idx: int | None,
    manual_lags: dict[str, int | None] | None,
) -> tuple[int | None, int | None]:
    """Return marker indices adjusted by manual lag selections."""
    if manual_lags is None or lags.size == 0:
        return los_idx, echo_idx

    finite_lags = _as_finite_1d(lags)
    manual_los = manual_lags.get("los")
    manual_echo = manual_lags.get("echo")
    min_lag = float(finite_lags.min())
    max_lag = float(finite_lags.max())

    if manual_los is not None and min_lag <= manual_los <= max_lag:
        los_idx = int(np.abs(finite_lags - manual_los).argmin())
    if manual_echo is not None and min_lag <= manual_echo <= max_lag:
        echo_idx = int(np.abs(finite_lags - manual_echo).argmin())
    return los_idx, echo_idx


def xcorr_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the full cross-correlation of *a* and *b* using FFT."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("xcorr_fft expects 1-D arrays")
    if len(a) == 0 or len(b) == 0:
        return np.array([], dtype=np.result_type(a, b, complex))

    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    A = np.fft.fft(a, nfft)
    B = np.fft.fft(b, nfft)
    cc = np.fft.ifft(A * np.conj(B))
    return np.concatenate((cc[-(len(b) - 1) :], cc[: len(a)]))


def autocorr_fft(x: np.ndarray) -> np.ndarray:
    """Return the full autocorrelation of *x* using FFT."""
    return xcorr_fft(x, x)


def find_los_echo(
    cc: np.ndarray,
    *,
    repetition_period_samples: int | None = None,
    include_shoulders: bool = False,
    first_path_policy: FirstPathPolicy = "earliest_significant",
    noise_sigma_factor: float = 3.0,
    first_path_noise_sigma_factor: float | None = 3.0,
    false_alarm_probability: float | None = None,
    min_echo_lag_samples: int = 2,
) -> tuple[int | None, int | None]:
    """Return LOS/first-path and first echo indices using one magnitude conversion.

    Drop-in compatibility: existing calls with only ``cc`` and optionally
    ``repetition_period_samples`` still work.  The default semantics are now
    physically clearer: LOS is the earliest significant path in the active peak
    group, not blindly the strongest peak.  Use
    ``first_path_policy='legacy'`` to reproduce the old strongest-peak behavior.
    """
    return find_los_echo_from_mag(
        np.abs(cc),
        repetition_period_samples=repetition_period_samples,
        include_shoulders=include_shoulders,
        first_path_policy=first_path_policy,
        noise_sigma_factor=noise_sigma_factor,
        first_path_noise_sigma_factor=first_path_noise_sigma_factor,
        false_alarm_probability=false_alarm_probability,
        min_echo_lag_samples=min_echo_lag_samples,
    )


def find_los_echo_from_mag(
    mag: np.ndarray,
    *,
    repetition_period_samples: int | None = None,
    include_shoulders: bool = False,
    first_path_policy: FirstPathPolicy = "earliest_significant",
    noise_sigma_factor: float = 3.0,
    first_path_noise_sigma_factor: float | None = 3.0,
    false_alarm_probability: float | None = None,
    min_echo_lag_samples: int = 2,
) -> tuple[int | None, int | None]:
    """Return LOS/first-path and first echo indices from magnitude data.

    Existing callers can keep using the two-index return value.  For scientific
    diagnostics, prefer ``detect_cir_paths_from_mag``.
    """
    detection = detect_cir_paths_from_mag(
        mag,
        repetition_period_samples=repetition_period_samples,
        include_shoulders=include_shoulders,
        first_path_policy=first_path_policy,
        noise_sigma_factor=noise_sigma_factor,
        first_path_noise_sigma_factor=first_path_noise_sigma_factor,
        false_alarm_probability=false_alarm_probability,
        min_echo_lag_samples=min_echo_lag_samples,
        max_echoes=1,
    )
    echo_idx = detection.echo_indices[0] if detection.echo_indices else None
    return detection.los_idx, echo_idx


def filter_echo_indices_by_noise_prominence(
    mag: np.ndarray,
    *,
    los_idx: int | None,
    echo_indices: list[int],
    repetition_period_samples: int | None = None,
    noise_sigma_factor: float = 0.3,
    min_echo_lag_samples: int = 2,
    false_alarm_probability: float | None = None,
    legacy_compatible: bool = True,
) -> list[int]:
    """Keep echo peaks that stand out from global background noise.

    By default this remains compatible with the previous component: global
    median + MAD is used, and if MAD is zero, any positive prominence is
    accepted.  Set ``legacy_compatible=False`` to use the new diagnostic noise
    estimator with IQR/std fallback and optional Rayleigh-PFA guardrail.
    """
    del repetition_period_samples  # grouping is already reflected in candidates

    work_mag = _as_finite_1d(mag)
    if work_mag.size == 0 or not echo_indices:
        return []

    los_idx_int = int(los_idx) if los_idx is not None else None
    cleaned_indices = sorted(
        {
            int(idx)
            for idx in echo_indices
            if (
                0 <= int(idx) < work_mag.size
                and (
                    los_idx_int is None
                    or int(idx) > los_idx_int + max(0, int(min_echo_lag_samples))
                )
            )
        }
    )
    if not cleaned_indices:
        return []

    if legacy_compatible:
        global_baseline = float(np.median(work_mag))
        global_mad = float(np.median(np.abs(work_mag - global_baseline)))
        noise_sigma = 1.4826 * global_mad

        filtered: list[int] = []
        for idx in cleaned_indices:
            prominence = float(work_mag[idx]) - global_baseline
            if prominence <= 0.0:
                continue
            if noise_sigma <= 1e-12:
                filtered.append(int(idx))
                continue
            if prominence >= max(0.0, float(noise_sigma_factor)) * noise_sigma:
                filtered.append(int(idx))
        return filtered

    noise = estimate_noise_floor_from_mag(
        work_mag,
        exclude_indices=cleaned_indices + ([] if los_idx_int is None else [los_idx_int]),
        noise_sigma_factor=noise_sigma_factor,
        false_alarm_probability=false_alarm_probability,
    )
    return _significant_peak_indices(work_mag, cleaned_indices, noise=noise)


def classify_peak_group(
    cc: np.ndarray,
    *,
    peaks_before: int = 3,
    peaks_after: int = 3,
    min_rel_height: float = 0.1,
    repetition_period_samples: int | None = None,
    include_shoulders: bool = False,
    first_path_policy: Literal["selected_group", "earliest_significant", "legacy", "strongest"] = "selected_group",
) -> tuple[int | None, int | None, list[int], list[int]]:
    """Return (highest_idx, los_idx, echo_indices, group_indices)."""
    return classify_peak_group_from_mag(
        np.abs(cc),
        peaks_before=peaks_before,
        peaks_after=peaks_after,
        min_rel_height=min_rel_height,
        repetition_period_samples=repetition_period_samples,
        include_shoulders=include_shoulders,
        first_path_policy=first_path_policy,
    )


def classify_peak_group_from_mag(
    mag: np.ndarray,
    *,
    peaks_before: int = 3,
    peaks_after: int = 3,
    min_rel_height: float = 0.1,
    repetition_period_samples: int | None = None,
    include_shoulders: bool = False,
    first_path_policy: Literal["selected_group", "earliest_significant", "legacy", "strongest"] = "selected_group",
) -> tuple[int | None, int | None, list[int], list[int]]:
    """Return (highest_idx, los_idx, echo_indices, group_indices).

    ``first_path_policy='selected_group'`` preserves the previous classification
    behavior: LOS is the earliest index among the selected group peaks.  Use
    ``'earliest_significant'`` for the physically clearer first-path semantics
    used by ``find_los_echo_from_mag``.
    """
    work_mag = _as_finite_1d(mag)
    if work_mag.size == 0:
        return None, None, [], []

    highest_idx = int(np.argmax(work_mag))

    if first_path_policy in {"earliest_significant", "legacy", "strongest"}:
        detection = detect_cir_paths_from_mag(
            work_mag,
            repetition_period_samples=repetition_period_samples,
            include_shoulders=include_shoulders,
            first_path_policy=("legacy" if first_path_policy in {"legacy", "strongest"} else "earliest_significant"),
            min_rel_height=0.0 if first_path_policy == "earliest_significant" else min_rel_height,
            peaks_before=max(peaks_before, 32),
            peaks_after=max(peaks_after, 32),
            max_echoes=None,
        )
        return (
            detection.strongest_idx,
            detection.los_idx,
            list(detection.echo_indices),
            list(detection.group_indices),
        )

    peak_indices = find_local_maxima_around_peak_from_mag(
        work_mag,
        center_idx=highest_idx,
        peaks_before=peaks_before,
        peaks_after=peaks_after,
        min_rel_height=min_rel_height,
        repetition_period_samples=repetition_period_samples,
        include_shoulders=include_shoulders,
    )
    if not peak_indices:
        peak_indices = [highest_idx]

    group_indices = sorted({int(idx) for idx in peak_indices})
    if highest_idx not in group_indices:
        group_indices.append(highest_idx)
        group_indices.sort()

    los_idx = int(group_indices[0])
    echo_indices = [int(idx) for idx in group_indices[1:]]
    return highest_idx, los_idx, echo_indices, group_indices


def find_local_maxima_around_peak(
    cc: np.ndarray,
    center_idx: int | None = None,
    *,
    peaks_before: int = 3,
    peaks_after: int = 3,
    min_rel_height: float = 0.1,
    repetition_period_samples: int | None = None,
    include_shoulders: bool = False,
) -> list[int]:
    """Return local maxima indices around a center peak (before + after)."""
    return find_local_maxima_around_peak_from_mag(
        np.abs(cc),
        center_idx=center_idx,
        peaks_before=peaks_before,
        peaks_after=peaks_after,
        min_rel_height=min_rel_height,
        repetition_period_samples=repetition_period_samples,
        include_shoulders=include_shoulders,
    )


def find_local_maxima_around_peak_from_mag(
    mag: np.ndarray,
    center_idx: int | None = None,
    *,
    peaks_before: int = 3,
    peaks_after: int = 3,
    min_rel_height: float = 0.1,
    repetition_period_samples: int | None = None,
    include_shoulders: bool = False,
) -> list[int]:
    """Return local maxima indices around a center peak (before + after).

    Compatibility note: for finite input and ``include_shoulders=False`` the
    strict local-maxima rule is intentionally the same as in the original code.
    This is the main guardrail that keeps existing peak detections stable while
    the scientific support code becomes explicit and diagnosable.
    """
    work_mag = _as_finite_1d(mag)
    if work_mag.size < 3:
        return []
    if center_idx is None:
        center_idx = int(np.argmax(work_mag))
    center_idx = int(np.clip(center_idx, 0, work_mag.size - 1))

    center_mag = float(work_mag[center_idx])
    min_height = max(0.0, float(min_rel_height)) * center_mag

    left_bound, right_bound = _peak_search_bounds(
        work_mag,
        center_idx=center_idx,
        repetition_period_samples=repetition_period_samples,
    )
    if right_bound < left_bound:
        return []

    local_maxima = _local_maxima_compat(
        work_mag,
        left_bound=left_bound,
        right_bound=right_bound,
        min_height=min_height,
    )
    if not local_maxima:
        return []

    before = [i for i in local_maxima if i < center_idx]
    after = [i for i in local_maxima if i > center_idx]

    before_count = max(0, int(peaks_before))
    after_count = max(0, int(peaks_after))
    before_sel = before[-before_count:] if before_count > 0 else []
    after_sel = after[:after_count] if after_count > 0 else []
    selected = before_sel + [center_idx] + after_sel
    if not include_shoulders:
        return selected

    shoulder_candidates = find_shoulder_candidates_from_mag(
        work_mag,
        center_idx=center_idx,
        strict_peak_indices=selected,
        min_height=min_height,
        left_bound=left_bound,
        right_bound=right_bound,
    )
    return sorted({*selected, *shoulder_candidates})


def filter_peak_indices_to_period_group(
    lags: np.ndarray,
    peak_indices: list[int],
    anchor_idx: int | None,
    period_samples: int | None,
) -> list[int]:
    """Keep only peaks that belong to the same repetition group as ``anchor_idx``.

    With repeated TX sequences, cross-correlation maxima appear every
    ``period_samples``. This helper keeps only the peak indices inside a
    half-period window around the selected anchor lag, so marker points remain
    on one peak group and do not jump between adjacent repetitions.
    """
    finite_lags = _as_finite_1d(lags)
    if anchor_idx is None or finite_lags.size == 0:
        return [int(i) for i in peak_indices]
    if period_samples is None or period_samples <= 1:
        return [int(i) for i in peak_indices]

    anchor_idx = int(np.clip(anchor_idx, 0, finite_lags.size - 1))
    anchor_lag = float(finite_lags[anchor_idx])
    half_period = float(period_samples) / 2.0

    filtered = []
    for idx in peak_indices:
        idx = int(idx)
        if idx < 0 or idx >= finite_lags.size:
            continue
        if abs(float(finite_lags[idx]) - anchor_lag) <= half_period:
            filtered.append(idx)
    return filtered


def resolve_manual_los_idx(
    lags: np.ndarray,
    base_los_idx: int | None,
    manual_lags: dict[str, int | None] | None,
    *,
    peak_group_indices: list[int] | None = None,
    highest_idx: int | None = None,
    period_samples: int | None = None,
    constrain_to_peak_group: bool = True,
) -> tuple[int | None, bool]:
    """Return LOS idx with optional validation against the active peak group.

    When ``manual_lags['los']`` points outside the currently dominant
    cross-correlation group, the manual LOS is ignored and ``manual_lags['los']``
    is reset to ``None`` so subsequent frames snap back to the current group.
    """
    finite_lags = _as_finite_1d(lags)
    if finite_lags.size == 0:
        return base_los_idx, False
    if manual_lags is None or manual_lags.get("los") is None:
        return base_los_idx, False

    manual_los = float(manual_lags["los"])
    min_lag = float(finite_lags.min())
    max_lag = float(finite_lags.max())
    if manual_los < min_lag or manual_los > max_lag:
        return base_los_idx, False

    manual_idx = int(np.abs(finite_lags - manual_los).argmin())
    allow_manual = True

    if constrain_to_peak_group and peak_group_indices:
        normalized = {int(idx) for idx in peak_group_indices if 0 <= int(idx) < finite_lags.size}
        if normalized and manual_idx not in normalized:
            allow_manual = False

    if (
        constrain_to_peak_group
        and allow_manual
        and highest_idx is not None
        and period_samples is not None
        and period_samples > 1
    ):
        highest_idx = int(np.clip(highest_idx, 0, finite_lags.size - 1))
        highest_lag = float(finite_lags[highest_idx])
        half_period = float(period_samples) / 2.0
        if abs(manual_los - highest_lag) > half_period:
            allow_manual = False

    if not allow_manual:
        manual_lags["los"] = None
        return base_los_idx, True

    return manual_idx, False


def lag_overlap(data_len: int, ref_len: int, lag: int) -> tuple[int, int, int]:
    """Return (data_start, ref_start, length) for a given lag."""
    data_len = int(data_len)
    ref_len = int(ref_len)
    lag = int(lag)

    if lag >= 0:
        r_start = lag
        s_start = 0
        length = min(data_len - r_start, ref_len)
    else:
        r_start = 0
        s_start = -lag
        length = min(data_len, ref_len - s_start)

    return r_start, s_start, max(0, int(length))


def xcorr_fft_energy_normalized(
    a: np.ndarray,
    b: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return overlap-energy-normalized full cross-correlation.

    ``xcorr_fft`` is intentionally left unnormalized for drop-in compatibility.
    This helper is the preferable amplitude-comparison primitive when peak
    heights from different frames, windows, or reference snippets are compared.

    The returned array has the same lag order as ``xcorr_fft`` and
    ``np.correlate(a, b, mode='full')``: lags run from ``-(len(b)-1)`` to
    ``len(a)-1``.  Samples with zero overlap energy are set to zero.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("xcorr_fft_energy_normalized expects 1-D arrays")
    if len(a) == 0 or len(b) == 0:
        return np.array([], dtype=np.result_type(a, b, complex))

    raw = xcorr_fft(a, b)
    out = np.zeros_like(raw, dtype=np.result_type(raw, complex))
    lags = np.arange(-(len(b) - 1), len(a), dtype=int)
    eps = max(0.0, float(eps))

    a_abs2 = np.abs(a) ** 2
    b_abs2 = np.abs(b) ** 2

    for out_idx, lag in enumerate(lags):
        a_start, b_start, length = lag_overlap(len(a), len(b), int(lag))
        if length <= 0:
            continue
        a_energy = float(np.sum(a_abs2[a_start : a_start + length]))
        b_energy = float(np.sum(b_abs2[b_start : b_start + length]))
        denom = float(np.sqrt(max(0.0, a_energy * b_energy)))
        if denom > eps:
            out[out_idx] = raw[out_idx] / denom
    return out
