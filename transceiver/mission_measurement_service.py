from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .measurement_run_executor import PointExecutionContext


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
        timestamp = datetime.now(UTC)
        filename = (
            f"point-{point_context.global_index:04d}-"
            f"{timestamp.strftime('%Y%m%d-%H%M%S')}.bin"
        )
        mission_dir = Path("signals") / "rx" / "mission" / point_context.mission_name
        mission_dir.mkdir(parents=True, exist_ok=True)
        output_file = mission_dir / filename

        try:
            result = self._app.receive_for_mission(output_file=str(output_file))
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
            review_result = self._review_measurement(
                point_context=point_context,
                output_file=file_ref,
            )
            if isinstance(review_result, dict):
                approved = bool(review_result.get("approved"))
                if approved:
                    review_payload = {
                        "manual_lags": review_result.get("manual_lags"),
                        "los_idx": review_result.get("los_idx"),
                        "echo_indices": review_result.get("echo_indices"),
                        "los_lag": review_result.get("los_lag"),
                        "echo_lags": review_result.get("echo_lags"),
                        "echo_delays": review_result.get("echo_delays"),
                    }
            else:
                approved = bool(review_result)
            if not approved:
                self._on_status("measurement", "failed")
                if self._on_operator_message is not None:
                    self._on_operator_message(
                        f"Review abgebrochen bei Punkt {point_context.global_index}: Messung wird nicht persistiert."
                    )
                raise RuntimeError("measurement_review_rejected")

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
        return payload
