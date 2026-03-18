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


def test_loads_map_config_from_relative_yaml_file(tmp_path) -> None:
    pytest.importorskip("yaml")
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()

    mission_path = tmp_path / "mission.yaml"
    mission_path.write_text(
        """
name: yaml-mission
points:
  - id: p1
    x: 0.0
    y: 0.0
    yaw: 0.0
map_config: maps/site_a.yaml
""".strip(),
        encoding="utf-8",
    )

    map_config_path = maps_dir / "site_a.yaml"
    map_config_path.write_text(
        """
image: maps/site_a.pgm
resolution: 0.05
origin: [0.0, 0.0, 0.0]
frame_id: map
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
""".strip(),
        encoding="utf-8",
    )

    mission = load_measurement_mission(mission_path)

    assert mission.map_config is not None
    assert mission.map_config.image == "maps/site_a.pgm"
    assert mission.map_config.resolution == 0.05
    assert mission.map_config.origin == (0.0, 0.0, 0.0)
