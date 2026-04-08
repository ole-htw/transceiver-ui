from __future__ import annotations

from transceiver.measurement_mission import MeasurementPoint
from transceiver.mission_workflow_ui import MissionWorkflowWindow


def test_format_echo_distances_for_table_returns_only_meter_values_for_first_five_echoes() -> None:
    text = MissionWorkflowWindow._format_echo_distances_for_table(
        [
            {"echo_index": 3, "delta_lag": 24, "distance_m": 36.0},
            {"echo_index": 5, "delta_lag": 42},
            {"echo_index": 6, "delta_lag": 52, "distance_m": 78.25},
            {"echo_index": 7, "delta_lag": 61, "distance_m": 12.5},
            {"echo_index": 8, "delta_lag": 73, "distance_m": 6.0},
            {"echo_index": 9, "delta_lag": 91, "distance_m": 99.0},
        ]
    )
    assert text == ("36", "-", "78.25", "12.5", "6")


def test_yaw_conversion_uses_clockwise_degrees_in_ui() -> None:
    yaw_rad = MissionWorkflowWindow._yaw_cw_degrees_to_internal_radians("90")
    assert yaw_rad == -1.5707963267948966
    assert MissionWorkflowWindow._yaw_internal_radians_to_cw_degrees(yaw_rad) == 90.0


def test_format_start_point_label_uses_index_and_name_without_id() -> None:
    label_without_name = MissionWorkflowWindow._format_start_point_label(
        0,
        MeasurementPoint(id="p001", name="", x=0.0, y=0.0, yaw=0.0),
    )
    label_with_name = MissionWorkflowWindow._format_start_point_label(
        1,
        MeasurementPoint(id="p002", name="Messpunkt B", x=1.0, y=1.0, yaw=0.0),
    )

    assert label_without_name == "1: Punkt 1"
    assert label_with_name == "2: Messpunkt B"


def test_format_one_based_index_converts_zero_based_values_for_ui() -> None:
    assert MissionWorkflowWindow._format_one_based_index(0) == "1"
    assert MissionWorkflowWindow._format_one_based_index(4) == "5"
    assert MissionWorkflowWindow._format_one_based_index(-1) == "-1"


def test_parse_lidar_scan_text_for_overlay_supports_inline_ranges() -> None:
    parsed = MissionWorkflowWindow._parse_lidar_scan_text_for_overlay(
        "angle_min: -1.57\nangle_increment: 0.1\nranges: [1.0, 2.5, inf, nan]\n"
    )
    assert parsed is not None
    assert parsed["angle_min"] == -1.57
    assert parsed["angle_increment"] == 0.1
    assert parsed["ranges"][:2] == [1.0, 2.5]


def test_extract_lidar_ranges_from_scan_text_supports_list_style() -> None:
    values = MissionWorkflowWindow._extract_lidar_ranges_from_scan_text(
        "ranges:\n  - 1.2\n  - inf\n  - 3.4\n"
    )
    assert len(values) == 3
    assert values[0] == 1.2
    assert values[2] == 3.4


class _FakeAdapter:
    def __init__(self, events):
        self.config = object()
        self._events = events

    def navigate_to_point(self, point, *, timeout_s, on_event):
        for event in self._events:
            on_event(event)
        return "succeeded"


class _FakeNavigationEvent:
    def __init__(self, event_type: str, data=None, attempt: int = 1, message: str | None = None) -> None:
        self.type = event_type
        self.data = data
        self.attempt = attempt
        self.message = message


class _FakePoseStreamTransport:
    last_instance = None

    def __init__(self) -> None:
        self.on_event = None
        _FakePoseStreamTransport.last_instance = self

    def start(self, *, config, on_event, expected_frame_id=None) -> None:
        self.on_event = on_event

    def stop(self) -> None:
        return


def test_ui_navigator_uses_pose_stream_for_live_position(monkeypatch) -> None:
    from transceiver import mission_workflow_ui as module

    monkeypatch.setattr(module, "Ros2CliPoseStreamTransport", _FakePoseStreamTransport)
    navigator = module._UiNavigator(
        adapter=_FakeAdapter([_FakeNavigationEvent("feedback", data={"position": {"x": 99.0, "y": 77.0}})]),
        on_status=lambda *_args: None,
        on_operator_message=lambda *_args: None,
    )

    runtime_events = []
    navigator.start_pose_stream(on_runtime_event=runtime_events.append)
    pose_stream = _FakePoseStreamTransport.last_instance
    assert pose_stream is not None
    pose_stream.on_event(
        {
            "type": "pose_stream",
            "event": {"type": "position_update", "position": {"x": 1.5, "y": 2.5, "yaw": 0.1}},
        }
    )

    nav_events = []
    state = navigator.navigate_to_point(object(), timeout_s=1.0, on_navigation_event=nav_events.append)

    assert state == "succeeded"
    position_updates = [event for event in nav_events if event.get("type") == "position_update"]
    assert position_updates
    assert position_updates[-1]["position"]["x"] == 1.5
    assert position_updates[-1]["position"]["y"] == 2.5


def test_ui_navigator_navigation_succeeds_without_feedback_position(monkeypatch) -> None:
    from transceiver import mission_workflow_ui as module

    monkeypatch.setattr(module, "Ros2CliPoseStreamTransport", _FakePoseStreamTransport)
    navigator = module._UiNavigator(
        adapter=_FakeAdapter([_FakeNavigationEvent("feedback", data={"raw": "feedback without pose"})]),
        on_status=lambda *_args: None,
        on_operator_message=lambda *_args: None,
    )

    nav_events = []
    state = navigator.navigate_to_point(object(), timeout_s=1.0, on_navigation_event=nav_events.append)

    assert state == "succeeded"
    position_updates = [event for event in nav_events if event.get("type") == "position_update"]
    assert position_updates
