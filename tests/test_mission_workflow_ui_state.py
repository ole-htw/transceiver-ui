import json

from transceiver.mission_workflow_ui import _load_json_dict, _save_json_dict


def test_save_and_load_json_dict_roundtrip(tmp_path) -> None:
    state_file = tmp_path / "mission-workflow-state.json"
    payload = {
        "name": "scan-1",
        "repeat": 2,
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
