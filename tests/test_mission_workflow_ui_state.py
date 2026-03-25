import json

from transceiver.mission_workflow_ui import _load_json_dict, _save_json_dict


def test_save_and_load_json_dict_roundtrip(tmp_path) -> None:
    state_file = tmp_path / "mission-workflow-state.json"
    payload = {
        "name": "scan-1",
        "repeat": 2,
        "start_point_index": 0,
        "points": [{"id": "p1", "x": 0.0, "y": 1.0, "z": 0.0, "yaw": 0.0}],
    }

    _save_json_dict(state_file, payload)
    loaded = _load_json_dict(state_file)

    assert loaded == payload


def test_load_json_dict_rejects_non_object_payload(tmp_path) -> None:
    state_file = tmp_path / "mission-workflow-state.json"
    state_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    loaded = _load_json_dict(state_file)

    assert loaded == {}


def test_save_and_load_json_dict_preserves_point_order(tmp_path) -> None:
    state_file = tmp_path / "mission-workflow-state.json"
    payload = {
        "name": "scan-ordered",
        "repeat": 1,
        "start_point_index": 1,
        "points": [
            {"id": "p003", "x": 3.0, "y": 3.0, "z": 0.0, "yaw": 0.0},
            {"id": "p001", "x": 1.0, "y": 1.0, "z": 0.0, "yaw": 0.0},
            {"id": "p002", "x": 2.0, "y": 2.0, "z": 0.0, "yaw": 0.0},
        ],
    }

    _save_json_dict(state_file, payload)
    loaded = _load_json_dict(state_file)

    assert [point["id"] for point in loaded["points"]] == ["p003", "p001", "p002"]


def test_save_and_load_json_dict_preserves_auto_generated_ids(tmp_path) -> None:
    state_file = tmp_path / "mission-workflow-state.json"
    payload = {
        "name": "scan-auto-ids",
        "repeat": 2,
        "start_point_index": 2,
        "points": [
            {"id": "p001", "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            {"id": "p002", "x": 1.0, "y": 1.0, "z": 0.0, "yaw": 0.0},
            {"id": "p003", "x": 2.0, "y": 2.0, "z": 0.0, "yaw": 0.0},
        ],
    }

    _save_json_dict(state_file, payload)
    loaded = _load_json_dict(state_file)

    assert [point["id"] for point in loaded["points"]] == ["p001", "p002", "p003"]


def test_save_and_load_json_dict_preserves_enabled_per_point(tmp_path) -> None:
    state_file = tmp_path / "mission-workflow-state.json"
    payload = {
        "name": "scan-enabled",
        "repeat": 1,
        "start_point_index": 0,
        "points": [
            {"id": "p001", "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "enabled": True},
            {"id": "p002", "x": 1.0, "y": 1.0, "z": 0.0, "yaw": 0.0, "enabled": False},
            {"id": "p003", "x": 2.0, "y": 2.0, "z": 0.0, "yaw": 0.0, "enabled": True},
        ],
    }

    _save_json_dict(state_file, payload)
    loaded = _load_json_dict(state_file)

    assert [point["enabled"] for point in loaded["points"]] == [True, False, True]


def test_save_and_load_json_dict_preserves_rx_antenna_global_position(tmp_path) -> None:
    state_file = tmp_path / "mission-workflow-state.json"
    payload = {
        "name": "scan-rx-antenna",
        "repeat": 1,
        "start_point_index": 0,
        "rx_antenna_global_position": {"x": 12.345, "y": -6.789},
        "points": [{"id": "p001", "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "enabled": True}],
    }

    _save_json_dict(state_file, payload)
    loaded = _load_json_dict(state_file)

    assert loaded["rx_antenna_global_position"] == {"x": 12.345, "y": -6.789}


def test_save_and_load_json_dict_preserves_lidar_reference_enabled_flag(tmp_path) -> None:
    state_file = tmp_path / "mission-workflow-state.json"
    payload = {
        "name": "scan-lidar-toggle",
        "repeat": 1,
        "start_point_index": 0,
        "lidar_reference_enabled": False,
        "points": [{"id": "p001", "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "enabled": True}],
    }

    _save_json_dict(state_file, payload)
    loaded = _load_json_dict(state_file)

    assert loaded["lidar_reference_enabled"] is False
