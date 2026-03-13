from __future__ import annotations

import json

import pytest

from transceiver.measurement_mission import load_measurement_mission


def _valid_payload() -> dict:
    return {
        "name": "file-based-mission",
        "repeat": 2,
        "wait_after_arrival_s": 0.25,
        "points": [
            {"id": "p1", "x": 1.0, "y": 2.0, "yaw": 0.0},
            {"id": "p2", "x": 3.0, "y": 4.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
        ],
    }


def test_loads_valid_json_mission_file(tmp_path) -> None:
    mission_path = tmp_path / "mission.json"
    mission_path.write_text(json.dumps(_valid_payload()), encoding="utf-8")

    mission = load_measurement_mission(mission_path)

    assert mission.name == "file-based-mission"
    assert mission.repeat == 2
    assert len(mission.points) == 2


def test_loads_valid_yaml_mission_file(tmp_path) -> None:
    pytest.importorskip("yaml")
    mission_path = tmp_path / "mission.yaml"
    mission_path.write_text(
        """
name: yaml-mission
points:
  - id: p1
    x: 0.0
    y: 0.0
    yaw: 0.0
""".strip(),
        encoding="utf-8",
    )

    mission = load_measurement_mission(mission_path)

    assert mission.name == "yaml-mission"
    assert len(mission.points) == 1


def test_rejects_invalid_mission_file_content(tmp_path) -> None:
    mission_path = tmp_path / "mission.json"
    mission_path.write_text(json.dumps({"name": "broken", "points": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="'points' must be a non-empty list"):
        load_measurement_mission(mission_path)


def test_rejects_unsupported_extension(tmp_path) -> None:
    mission_path = tmp_path / "mission.txt"
    mission_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported mission file extension"):
        load_measurement_mission(mission_path)
