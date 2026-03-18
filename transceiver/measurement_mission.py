from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MeasurementPoint:
    id: str | None
    name: str | None
    x: float
    y: float
    z: float = 0.0
    yaw: float | None = None
    qx: float | None = None
    qy: float | None = None
    qz: float | None = None
    qw: float | None = None
    notes: str | None = None
    measurement_profile: str | None = None


@dataclass(frozen=True)
class MeasurementMission:
    name: str
    points: list[MeasurementPoint]
    repeat: int | None = None
    wait_after_arrival_s: float = 0.0
    map_config: MapConfig | None = None


@dataclass(frozen=True)
class MapConfig:
    image: str
    resolution: float
    origin: tuple[float, float, float]
    frame_id: str | None = None
    negate: int | None = None
    occupied_thresh: float | None = None
    free_thresh: float | None = None


def _require_finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"'{field_name}' must be a numeric value")

    parsed: Any = value
    if isinstance(value, str):
        normalized = value.strip().replace(",", ".")
        if not normalized:
            raise ValueError(f"'{field_name}' must be a numeric value")
        try:
            parsed = float(normalized)
        except ValueError as exc:
            raise ValueError(f"'{field_name}' must be a numeric value") from exc

    if not isinstance(parsed, (int, float)):
        raise ValueError(f"'{field_name}' must be a numeric value")

    number = float(parsed)
    if not math.isfinite(number):
        raise ValueError(f"'{field_name}' must be finite")
    return number


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string")
    return value.strip()


def _parse_point(payload: dict[str, Any], index: int) -> MeasurementPoint:
    point_label = f"points[{index}]"
    point_id = payload.get("id")
    point_name = payload.get("name")

    if point_id is None and point_name is None:
        raise ValueError(f"{point_label} must include either 'id' or 'name'")

    if point_id is not None:
        point_id = _require_string(point_id, f"{point_label}.id")
    if point_name is not None:
        point_name = _require_string(point_name, f"{point_label}.name")

    x = _require_finite_number(payload.get("x"), f"{point_label}.x")
    y = _require_finite_number(payload.get("y"), f"{point_label}.y")
    z = _require_finite_number(payload.get("z", 0.0), f"{point_label}.z")

    has_yaw = "yaw" in payload
    has_quaternion = any(axis in payload for axis in ("qx", "qy", "qz", "qw"))

    if has_yaw == has_quaternion:
        raise ValueError(
            f"{point_label} must include exactly one orientation format: "
            "either 'yaw' or quaternion 'qx/qy/qz/qw'"
        )

    yaw: float | None = None
    qx = qy = qz = qw = None

    if has_yaw:
        yaw = _require_finite_number(payload.get("yaw"), f"{point_label}.yaw")
        if yaw < -math.pi or yaw > math.pi:
            raise ValueError(f"{point_label}.yaw must be within [-pi, pi] radians")
    else:
        qx = _require_finite_number(payload.get("qx"), f"{point_label}.qx")
        qy = _require_finite_number(payload.get("qy"), f"{point_label}.qy")
        qz = _require_finite_number(payload.get("qz"), f"{point_label}.qz")
        qw = _require_finite_number(payload.get("qw"), f"{point_label}.qw")
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"{point_label} quaternion must be normalized (norm ~= 1.0)")

    notes = payload.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"{point_label}.notes must be a string when provided")

    measurement_profile = payload.get("measurement_profile")
    if measurement_profile is not None and not isinstance(measurement_profile, str):
        raise ValueError(
            f"{point_label}.measurement_profile must be a string when provided"
        )

    return MeasurementPoint(
        id=point_id,
        name=point_name,
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        qx=qx,
        qy=qy,
        qz=qz,
        qw=qw,
        notes=notes,
        measurement_profile=measurement_profile,
    )


def _parse_map_config(payload: Any) -> MapConfig:
    if not isinstance(payload, dict):
        raise ValueError("'map_config' must be an object when provided")

    image = _require_string(payload.get("image"), "map_config.image")
    resolution = _require_finite_number(payload.get("resolution"), "map_config.resolution")
    if resolution <= 0:
        raise ValueError("'map_config.resolution' must be > 0")

    origin_raw = payload.get("origin")
    if not isinstance(origin_raw, list) or len(origin_raw) != 3:
        raise ValueError("'map_config.origin' must be a list with exactly 3 numbers")

    origin: tuple[float, float, float] = (
        _require_finite_number(origin_raw[0], "map_config.origin[0]"),
        _require_finite_number(origin_raw[1], "map_config.origin[1]"),
        _require_finite_number(origin_raw[2], "map_config.origin[2]"),
    )

    frame_id = payload.get("frame_id")
    if frame_id is not None:
        frame_id = _require_string(frame_id, "map_config.frame_id")

    negate = payload.get("negate")
    if negate is not None:
        if isinstance(negate, bool) or not isinstance(negate, int):
            raise ValueError("'map_config.negate' must be an integer when provided")

    occupied_thresh = payload.get("occupied_thresh")
    if occupied_thresh is not None:
        occupied_thresh = _require_finite_number(
            occupied_thresh, "map_config.occupied_thresh"
        )

    free_thresh = payload.get("free_thresh")
    if free_thresh is not None:
        free_thresh = _require_finite_number(free_thresh, "map_config.free_thresh")

    return MapConfig(
        image=image,
        resolution=resolution,
        origin=origin,
        frame_id=frame_id,
        negate=negate,
        occupied_thresh=occupied_thresh,
        free_thresh=free_thresh,
    )


def measurement_mission_from_dict(payload: dict[str, Any]) -> MeasurementMission:
    if not isinstance(payload, dict):
        raise ValueError("Mission payload must be a JSON/YAML object")

    name = _require_string(payload.get("name"), "name")

    points_raw = payload.get("points")
    if not isinstance(points_raw, list) or not points_raw:
        raise ValueError("'points' must be a non-empty list")

    parsed_points: list[MeasurementPoint] = []
    point_ids: set[str] = set()

    for index, point_raw in enumerate(points_raw):
        if not isinstance(point_raw, dict):
            raise ValueError(f"points[{index}] must be an object")
        point = _parse_point(point_raw, index)
        if point.id:
            if point.id in point_ids:
                raise ValueError(f"Duplicate point id '{point.id}'")
            point_ids.add(point.id)
        parsed_points.append(point)

    repeat = payload.get("repeat")
    if repeat is not None:
        if not isinstance(repeat, int) or repeat < 1:
            raise ValueError("'repeat' must be an integer >= 1 when provided")

    wait_after_arrival_s = _require_finite_number(
        payload.get("wait_after_arrival_s", 0.0), "wait_after_arrival_s"
    )
    if wait_after_arrival_s < 0.0:
        raise ValueError("'wait_after_arrival_s' must be >= 0")

    map_config: MapConfig | None = None
    if "map_config" in payload and payload.get("map_config") is not None:
        map_config = _parse_map_config(payload.get("map_config"))

    return MeasurementMission(
        name=name,
        points=parsed_points,
        repeat=repeat,
        wait_after_arrival_s=wait_after_arrival_s,
        map_config=map_config,
    )


def load_measurement_mission(path: str | Path) -> MeasurementMission:
    mission_path = Path(path)
    raw_text = mission_path.read_text(encoding="utf-8")

    suffix = mission_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(raw_text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError(
                "YAML mission files require PyYAML. "
                "Install with `pip install pyyaml` or use JSON."
            ) from exc
        payload = yaml.safe_load(raw_text)
    else:
        raise ValueError("Unsupported mission file extension. Use .json, .yaml or .yml")

    return measurement_mission_from_dict(payload)
