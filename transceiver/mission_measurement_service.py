from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .measurement_run_executor import PointExecutionContext

_LOS_ECHO_SAMPLE_TO_M = 1.5
REVIEW_REASON_REVIEW_UNAVAILABLE = "review_unavailable"
REVIEW_REASON_MISSING_TX_REFERENCE = "missing_tx_reference"
REVIEW_REASON_NO_DETECTABLE_LOS = "no_detectable_los"
REVIEW_REASON_REVIEW_EXCEPTION = "review_exception"
REVIEW_REASON_OPERATOR_REJECTED = "operator_rejected"

ALLOWED_REVIEW_REASONS = {
    REVIEW_REASON_REVIEW_UNAVAILABLE,
    REVIEW_REASON_MISSING_TX_REFERENCE,
    REVIEW_REASON_NO_DETECTABLE_LOS,
    REVIEW_REASON_REVIEW_EXCEPTION,
    REVIEW_REASON_OPERATOR_REJECTED,
}


def normalize_review_reason(raw_reason: Any, *, default: str = REVIEW_REASON_OPERATOR_REJECTED) -> str:
    if isinstance(raw_reason, str):
        candidate = raw_reason.strip()
        if candidate in ALLOWED_REVIEW_REASONS:
            return candidate
    return default


_REVIEW_REASON_TO_ERROR_CODE = {
    REVIEW_REASON_REVIEW_UNAVAILABLE: "measurement_review_unavailable",
    REVIEW_REASON_MISSING_TX_REFERENCE: "measurement_review_missing_tx_reference",
    REVIEW_REASON_NO_DETECTABLE_LOS: "measurement_review_no_detectable_los",
    REVIEW_REASON_REVIEW_EXCEPTION: "measurement_review_exception",
    REVIEW_REASON_OPERATOR_REJECTED: "measurement_review_operator_rejected",
}


def _coerce_echo_delay_entries(
    *,
    echo_delays: Any,
    echo_indices: Any,
    echo_lags: Any,
    los_lag: Any,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    delays = list(echo_delays) if isinstance(echo_delays, list) else []
    indices = list(echo_indices) if isinstance(echo_indices, list) else []
    lags = list(echo_lags) if isinstance(echo_lags, list) else []
    los_lag_value = int(los_lag) if isinstance(los_lag, (int, float)) else None

    total = max(len(delays), len(indices), len(lags))
    for idx in range(total):
        raw_delay = delays[idx] if idx < len(delays) else None
        raw_echo_index = indices[idx] if idx < len(indices) else None
        raw_echo_lag = lags[idx] if idx < len(lags) else None

        if isinstance(raw_delay, dict):
            entry = dict(raw_delay)
        else:
            delta_lag = int(raw_delay) if isinstance(raw_delay, (int, float)) else None
            if delta_lag is None and los_lag_value is not None and isinstance(raw_echo_lag, (int, float)):
                delta_lag = int(abs(int(raw_echo_lag) - los_lag_value))
            entry = {
                "echo_index": int(raw_echo_index) if isinstance(raw_echo_index, (int, float)) else idx,
                "delta_lag": delta_lag,
            }

        if "distance_m" not in entry and isinstance(entry.get("delta_lag"), (int, float)):
            entry["distance_m"] = round(float(entry["delta_lag"]) * _LOS_ECHO_SAMPLE_TO_M, 3)
        entries.append(entry)
    return entries


class MissionRxMeasurementService:
    """Mission measurement service backed by TransceiverUI RX pipeline."""

    def __init__(
        self,
        *,
        app: Any,
        on_status,
        on_operator_message=None,
        review_measurement=None,
    ) -> None:
        self._app = app
        self._on_status = on_status
        self._on_operator_message = on_operator_message
        self._review_measurement = review_measurement

    def trigger(self, point_context: PointExecutionContext) -> dict[str, Any]:
        self._on_status("measurement", "running")
        timestamp = datetime.now(timezone.utc)
        filename = (
            f"point-{point_context.global_index:04d}-"
            f"{timestamp.strftime('%Y%m%d-%H%M%S')}.bin"
        )
        mission_dir = Path("signals") / "rx" / "mission" / point_context.mission_name
        mission_dir.mkdir(parents=True, exist_ok=True)
        output_file = mission_dir / filename

        try:
            result = self._app.receive_for_mission(
                output_file=str(output_file),
                point_context=point_context,
            )
        except Exception as exc:
            self._on_status("measurement", "failed")
            if self._on_operator_message is not None:
                self._on_operator_message(f"RX-Fehler bei Punkt {point_context.global_index}: {exc}")
            raise RuntimeError("measurement_failed") from exc

        if not result.get("ok"):
            self._on_status("measurement", "failed")
            detail = str(result.get("error") or "Receive fehlgeschlagen")
            if self._on_operator_message is not None:
                self._on_operator_message(f"RX-Fehler bei Punkt {point_context.global_index}: {detail}")
            raise RuntimeError("measurement_failed")

        file_ref = str(result.get("output_file") or output_file)
        review_payload: dict[str, Any] | None = None
        if self._review_measurement is not None:
            approved = False
            review_reason = REVIEW_REASON_OPERATOR_REJECTED
            review_detail = ""
            review_result = self._review_measurement(
                point_context=point_context,
                output_file=file_ref,
            )
            if isinstance(review_result, dict):
                approved = bool(review_result.get("approved"))
                raw_reason = review_result.get("reason")
                raw_detail = review_result.get("detail")
                review_reason = "" if approved else normalize_review_reason(raw_reason)
                if isinstance(raw_detail, str) and raw_detail.strip():
                    review_detail = raw_detail.strip()
                if approved:
                    echo_delay_entries = _coerce_echo_delay_entries(
                        echo_delays=review_result.get("echo_delays"),
                        echo_indices=review_result.get("echo_indices"),
                        echo_lags=review_result.get("echo_lags"),
                        los_lag=review_result.get("los_lag"),
                    )
                    review_payload = {
                        "approved": True,
                        "detail": review_detail,
                        "manual_lags": review_result.get("manual_lags"),
                        "los_idx": review_result.get("los_idx"),
                        "echo_indices": review_result.get("echo_indices"),
                        "los_lag": review_result.get("los_lag"),
                        "echo_lags": review_result.get("echo_lags"),
                        "echo_delays": echo_delay_entries,
                    }
            else:
                approved = bool(review_result)
            if not approved:
                self._on_status("measurement", "failed")
                error_code = _REVIEW_REASON_TO_ERROR_CODE[review_reason]
                if self._on_operator_message is not None:
                    self._on_operator_message(
                        f"Review abgebrochen bei Punkt {point_context.global_index} [{review_reason}]: "
                        f"{review_detail or 'Messung wird nicht persistiert.'}"
                    )
                raise RuntimeError(error_code, review_reason, review_detail)

        self._on_status("measurement", "succeeded")
        payload = {
            "measurement_id": f"{point_context.mission_name}-{point_context.global_index:04d}-{int(timestamp.timestamp())}",
            "file_ref": file_ref,
            "point_id": point_context.point.id,
            "timestamp": timestamp.isoformat(),
            "rx": result,
        }
        if review_payload is not None:
            payload["review"] = review_payload
            payload["los_lag"] = review_payload.get("los_lag")
            payload["echo_lags"] = review_payload.get("echo_lags")
            payload["echo_delays"] = review_payload.get("echo_delays")
        return payload
