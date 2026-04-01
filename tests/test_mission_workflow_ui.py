from __future__ import annotations

from transceiver.mission_workflow_ui import MissionWorkflowWindow


def test_format_echo_delays_for_table_uses_structured_measurement_result_entries() -> None:
    text = MissionWorkflowWindow._format_echo_delays_for_table(
        [
            {"echo_index": 3, "delta_lag": 24, "distance_m": 36.0},
            {"echo_index": 5, "delta_lag": 42},
        ]
    )
    assert text == "E3: 24 (36.0m); E5: 42"


def test_yaw_conversion_uses_clockwise_degrees_in_ui() -> None:
    yaw_rad = MissionWorkflowWindow._yaw_cw_degrees_to_internal_radians("90")
    assert yaw_rad == -1.5707963267948966
    assert MissionWorkflowWindow._yaw_internal_radians_to_cw_degrees(yaw_rad) == 90.0


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

    def start(self, *, config, on_event) -> None:
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
