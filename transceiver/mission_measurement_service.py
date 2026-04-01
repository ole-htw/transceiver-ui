from __future__ import annotations

import subprocess
import shlex
import logging
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

LOGGER = logging.getLogger(__name__)


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
        collect_lidar_reference=None,
        enable_lidar_reference: bool = True,
        lidar_topic: str = "/scan",
        lidar_timeout_s: float = 15.0,
        lidar_ros_env_cmd: str = "",
        lidar_ros_setup: str = "",
    ) -> None:
        self._app = app
        self._on_status = on_status
        self._on_operator_message = on_operator_message
        self._review_measurement = review_measurement
        self._lidar_topic = lidar_topic.strip() or "/scan"
        self._lidar_timeout_s = max(1.0, float(lidar_timeout_s))
        self._lidar_ros_env_cmd = lidar_ros_env_cmd.strip()
        self._lidar_ros_setup = lidar_ros_setup.strip()
        self._collect_lidar_reference = collect_lidar_reference or self._capture_lidar_reference
        self._enable_lidar_reference = enable_lidar_reference

    def _capture_lidar_reference(self, output_file: Path) -> dict[str, Any]:
        ros2_command = f"ros2 topic echo {shlex.quote(self._lidar_topic)} --once"
        shell_parts = ["set -eo pipefail"]
        if self._lidar_ros_env_cmd:
            shell_parts.append(self._lidar_ros_env_cmd)
        elif self._lidar_ros_setup:
            shell_parts.append(f"source {shlex.quote(self._lidar_ros_setup)}")
        shell_parts.append(
            "command -v ros2 >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ros2 CLI not found in PATH' >&2; exit 70; }"
        )
        shell_parts.append(ros2_command)
        shell_command = "; ".join(shell_parts)
        command = ["bash", "-lc", shell_command]
        LOGGER.debug("LIDAR reference command (final): %s", shell_command)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self._lidar_timeout_s,
        )
        LOGGER.debug(
            "LIDAR reference result: returncode=%s stdout=%r stderr=%r",
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        payload = {
            "topic": self._lidar_topic,
            "command": shell_command,
            "shell_command": command,
            "output_file": str(output_file),
        }
        output_file.write_text(completed.stdout, encoding="utf-8")
        return payload

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
        lidar_reference_file = mission_dir / f"{output_file.stem}.lidar.scan.txt"

        lidar_reference: dict[str, Any] | None = None
        if self._enable_lidar_reference:
            try:
                lidar_reference = self._collect_lidar_reference(lidar_reference_file)
            except Exception as exc:
                self._on_status("measurement", "failed")
                if self._on_operator_message is not None:
                    self._on_operator_message(
                        f"LIDAR-Referenzmessung fehlgeschlagen bei Punkt {point_context.global_index}: {exc}"
                    )
                raise RuntimeError("lidar_reference_failed") from exc

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
        if isinstance(lidar_reference, dict):
            payload["lidar_reference"] = lidar_reference
        return payload
