from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from transceiver.measurement_mission import MeasurementMission, MeasurementPoint
from transceiver.navigation_adapter import NavigationPoint
from transceiver.mission_workflow_ui import MissionWorkflowWindow, _compute_bistatic_echo_ellipse_axes


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


def test_format_start_point_label_uses_only_one_based_index() -> None:
    label_without_name = MissionWorkflowWindow._format_start_point_label(
        0,
        MeasurementPoint(id="p001", name="", x=0.0, y=0.0, yaw=0.0),
    )
    label_with_name = MissionWorkflowWindow._format_start_point_label(
        1,
        MeasurementPoint(id="p002", name="Messpunkt B", x=1.0, y=1.0, yaw=0.0),
    )

    assert label_without_name == "1"
    assert label_with_name == "2"


def test_format_one_based_index_converts_zero_based_values_for_ui() -> None:
    assert MissionWorkflowWindow._format_one_based_index(0) == "1"
    assert MissionWorkflowWindow._format_one_based_index(4) == "5"
    assert MissionWorkflowWindow._format_one_based_index(-1) == "-1"


def test_derive_table_status_maps_error_to_failed() -> None:
    assert (
        MissionWorkflowWindow._derive_table_status(
            {"measurement": {"status": "succeeded"}, "error": "navigation_failed.timeout"}
        )
        == "failed"
    )


def test_derive_table_status_uses_measurement_status_without_error() -> None:
    assert MissionWorkflowWindow._derive_table_status({"measurement": {"status": "skipped"}, "error": None}) == "skipped"


def test_compose_table_status_accumulates_status_and_error_text() -> None:
    assert (
        MissionWorkflowWindow._compose_table_status("failed", "navigation_failed.timeout [lidar_missing]: Peaks fehlen")
        == "failed: navigation_failed.timeout [lidar_missing]: Peaks fehlen"
    )


def test_compose_table_status_uses_status_when_error_missing() -> None:
    assert MissionWorkflowWindow._compose_table_status("succeeded", "") == "succeeded"


def test_compose_table_outcome_returns_succeeded_for_clean_success() -> None:
    payload = {
        "navigation": {"state": "succeeded"},
        "measurement": {"status": "succeeded"},
    }
    assert MissionWorkflowWindow._compose_table_outcome(payload, "") == "succeeded"


def test_compose_table_outcome_includes_navigation_and_measurement_for_failures() -> None:
    payload = {
        "navigation": {"state": "aborted"},
        "measurement": {"status": "skipped"},
    }
    assert (
        MissionWorkflowWindow._compose_table_outcome(payload, "navigation_failed.aborted")
        == "navigation aborted, measurement skipped: navigation_failed.aborted"
    )


def test_format_distance_to_rx_for_table_uses_measurement_point_coordinates() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._rx_antenna_global_position = (1.0, 2.0)
    window._mission_points = [
        MeasurementPoint(id="p0", name="P0", x=4.0, y=6.0, yaw=0.0),
    ]

    distance = window._format_distance_to_rx_for_table({"point_index": 0})

    assert distance == "5"


def test_format_distance_to_rx_for_table_returns_dash_without_rx_position() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._rx_antenna_global_position = None
    window._mission_points = [
        MeasurementPoint(id="p0", name="P0", x=4.0, y=6.0, yaw=0.0),
    ]

    distance = window._format_distance_to_rx_for_table({"point_index": 0})

    assert distance == "-"


def test_format_live_distance_to_rx_for_table_uses_live_position_coordinates() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._rx_antenna_global_position = (1.0, 2.0)

    distance = window._format_live_distance_to_rx_for_table(
        {"live_position_at_measurement": {"x": 4.0, "y": 6.0}}
    )

    assert distance == "5"


def test_format_live_distance_to_rx_for_table_returns_dash_without_live_position() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._rx_antenna_global_position = (1.0, 2.0)

    distance = window._format_live_distance_to_rx_for_table({"point_index": 0})

    assert distance == "-"


def test_selected_record_measurement_position_reads_live_coordinates() -> None:
    position = MissionWorkflowWindow._selected_record_measurement_position(
        {"live_position_at_measurement": {"x": 1.25, "y": -3.5}}
    )

    assert position == (1.25, -3.5)


def test_selected_record_measurement_position_returns_none_without_live_coordinates() -> None:
    position = MissionWorkflowWindow._selected_record_measurement_position({"point_index": 0})

    assert position is None


def test_draw_selected_echo_overlay_uses_live_measurement_position() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._selected_result_index = 0
    window._records = [
        {
            "point_index": 0,
            "live_position_at_measurement": {"x": 7.0, "y": -2.0},
            "measurement": {"result": {"echo_delays": [{"distance_m": 3.0}]}},
        }
    ]
    window._rx_antenna_global_position = (1.0, 1.0)
    window._mission_points = [MeasurementPoint(id="p0", name="P0", x=50.0, y=50.0, yaw=0.0)]
    calls: list[dict[str, object]] = []
    window._draw_echo_ellipse_for_overlay = lambda **kwargs: calls.append(kwargs)

    window._draw_selected_echo_overlay()

    assert len(calls) == 1
    assert calls[0]["measurement_position"] == (7.0, -2.0)


def test_draw_selected_echo_overlay_renders_all_selected_results() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._selected_result_index = 0
    window._selected_result_indices = (0, 1)
    window._records = [
        {
            "point_index": 0,
            "live_position_at_measurement": {"x": 7.0, "y": -2.0},
            "measurement": {"result": {"echo_delays": [{"distance_m": 3.0}]}},
        },
        {
            "point_index": 1,
            "live_position_at_measurement": {"x": 8.0, "y": -1.0},
            "measurement": {"result": {"echo_delays": [{"distance_m": 4.0}]}},
        },
    ]
    window._rx_antenna_global_position = (1.0, 1.0)
    window._mission_points = [
        MeasurementPoint(id="p0", name="P0", x=50.0, y=50.0, yaw=0.0),
        MeasurementPoint(id="p1", name="P1", x=60.0, y=60.0, yaw=0.0),
    ]
    calls: list[dict[str, object]] = []
    window._draw_echo_ellipse_for_overlay = lambda **kwargs: calls.append(kwargs)

    window._draw_selected_echo_overlay()

    assert len(calls) == 2
    assert {call["measurement_position"] for call in calls} == {(7.0, -2.0), (8.0, -1.0)}


def test_selected_record_overlay_point_prefers_live_yaw() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._mission_points = [MeasurementPoint(id="p0", name="P0", x=50.0, y=50.0, yaw=0.5)]

    overlay_point = window._selected_record_overlay_point(
        {"point_index": 0, "live_position_at_measurement": {"x": 7.0, "y": -2.0, "yaw": 1.25}},
        measurement_position=(7.0, -2.0),
    )

    assert overlay_point is not None
    assert overlay_point.x == 7.0
    assert overlay_point.y == -2.0
    assert overlay_point.yaw == 1.25


def test_draw_selected_lidar_reference_overlay_uses_live_measurement_position() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._selected_result_index = 0
    window._records = [
        {
            "point_index": 0,
            "live_position_at_measurement": {"x": 7.0, "y": -2.0, "yaw": 0.25},
            "measurement": {"result": {"lidar_reference": {"output_file": "scan.yaml"}}},
        }
    ]
    window._mission_points = [MeasurementPoint(id="p0", name="P0", x=50.0, y=50.0, yaw=0.0)]
    window._load_lidar_scan_for_overlay = lambda _path: {"angle_min": 0.0, "angle_increment": 0.1, "ranges": [1.0]}
    calls: list[dict[str, object]] = []
    window._draw_lidar_scan_overlay_for_point = lambda **kwargs: calls.append(kwargs)

    window._draw_selected_lidar_reference_overlay()

    assert len(calls) == 1
    point = calls[0]["point"]
    assert isinstance(point, MeasurementPoint)
    assert point.x == 7.0
    assert point.y == -2.0


def test_resolve_cmd_vel_topic_uses_namespace_when_present() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._runtime_config = SimpleNamespace(ros2_namespace="robot1")

    assert window._resolve_cmd_vel_topic() == "/robot1/cmd_vel"


def test_build_manual_drive_command_reuses_remote_ssh_transport_builder() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._runtime_config = SimpleNamespace(
        ros2_namespace="robot1",
        robot_host="robot@10.0.0.2",
        goal_acceptance_timeout_s=5.0,
        remote_ros_env_cmd="",
        remote_ros_setup="/opt/ros/jazzy/setup.bash",
        fastdds_profiles_file="",
    )

    command = window._build_manual_drive_command(linear_x=0.15, angular_z=-0.7)

    assert command[0:2] == ["ssh", "-o"]
    assert "robot@10.0.0.2" in command
    remote_cmd = command[-1]
    assert "source /opt/ros/jazzy/setup.bash" in remote_cmd
    assert "ros2 topic pub --once /robot1/cmd_vel geometry_msgs/msg/Twist" in remote_cmd
    assert "{linear: {x: 0.150, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.700}}" in remote_cmd


def test_navigation_point_from_world_position_uses_identity_orientation() -> None:
    point = MissionWorkflowWindow._navigation_point_from_world_position((1.25, -3.5))

    assert point == NavigationPoint(x=1.25, y=-3.5)


def test_navigation_point_from_world_position_converts_yaw_to_quaternion() -> None:
    point = MissionWorkflowWindow._navigation_point_from_world_position((0.5, 2.0), yaw_radians=math.pi / 2.0)

    assert point.x == pytest.approx(0.5)
    assert point.y == pytest.approx(2.0)
    assert point.qz == pytest.approx(math.sqrt(0.5))
    assert point.qw == pytest.approx(math.sqrt(0.5))


def test_on_map_canvas_click_starts_nav2point_pick_preview_when_mode_enabled() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._nav2point_map_pick_mode_enabled = True
    window._preview_pixel_to_world = lambda preview_x, preview_y: (preview_x + 0.5, preview_y - 0.25)
    window._draw_map_preview = lambda: None

    window._on_map_canvas_click(SimpleNamespace(x=10, y=20))

    assert window._pending_nav2point_world_position == (10.5, 19.75)
    assert window._pending_nav2point_yaw_radians == 0.0
    assert window._nav2point_drag_start_preview == (10.0, 20.0)
    assert window._nav2point_drag_active is False


def test_format_position_for_table_uses_one_decimal_for_x_and_y() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._mission_points = [
        MeasurementPoint(id="p0", name="P0", x=3.24, y=-1.01, yaw=0.0),
    ]

    position = window._format_position_for_table({"point_index": 0})

    assert position == "3.2,-1.0"


def test_format_position_for_table_returns_dash_without_known_point() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._mission_points = []

    position = window._format_position_for_table({"point_index": 0})

    assert position == "-"


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


def test_draw_lidar_scan_overlay_deduplicates_dense_endpoints() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._map_image_original = SimpleNamespace(width=lambda: 200, height=lambda: 120)
    window._map_preview_scale = (1.0, 1.0)
    window._map_preview_offset = (0.0, 0.0)
    window._world_to_map_pixel = lambda *, x, y, image_height: (x, y)
    window._is_pixel_inside_map = lambda *_args, **_kwargs: True
    line_calls: list[tuple[float, float, float, float]] = []
    window.map_preview_canvas = SimpleNamespace(
        create_oval=lambda *_args, **_kwargs: None,
        create_line=lambda sx, sy, ex, ey, **_kwargs: line_calls.append((sx, sy, ex, ey)),
    )

    window._draw_lidar_scan_overlay_for_point(
        point=MeasurementPoint(id="p1", name="P1", x=10.0, y=20.0, yaw=0.0),
        scan={"angle_min": 0.0, "angle_increment": 0.0, "ranges": [2.0] * 1800},
    )

    assert len(line_calls) == 1


def test_draw_lidar_scan_overlay_adapts_stride_for_large_scan() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._map_image_original = SimpleNamespace(width=lambda: 3000, height=lambda: 3000)
    window._map_preview_scale = (1.0, 1.0)
    window._map_preview_offset = (0.0, 0.0)
    window._world_to_map_pixel = lambda *, x, y, image_height: (x, y)
    window._is_pixel_inside_map = lambda *_args, **_kwargs: True
    line_calls: list[tuple[float, float, float, float]] = []
    window.map_preview_canvas = SimpleNamespace(
        create_oval=lambda *_args, **_kwargs: None,
        create_line=lambda sx, sy, ex, ey, **_kwargs: line_calls.append((sx, sy, ex, ey)),
    )

    window._draw_lidar_scan_overlay_for_point(
        point=MeasurementPoint(id="p2", name="P2", x=0.0, y=0.0, yaw=0.0),
        scan={"angle_min": 0.0, "angle_increment": 0.01, "ranges": [500.0] * 2000},
    )

    assert len(line_calls) <= 700


def test_build_waypoint_arrow_polygon_points_to_positive_x_for_zero_yaw() -> None:
    points = MissionWorkflowWindow._build_waypoint_arrow_polygon(
        center_x=100.0,
        center_y=50.0,
        yaw_radians=0.0,
        arrow_length=10.0,
        tail_length=4.0,
        tail_width=8.0,
    )

    tip_x, tip_y, left_x, left_y, right_x, right_y = points
    assert (tip_x, tip_y) == (110.0, 50.0)
    assert (left_x, left_y) == (96.0, 54.0)
    assert (right_x, right_y) == (96.0, 46.0)


def test_build_waypoint_arrow_polygon_points_up_for_ninety_degree_yaw() -> None:
    points = MissionWorkflowWindow._build_waypoint_arrow_polygon(
        center_x=100.0,
        center_y=50.0,
        yaw_radians=1.5707963267948966,
        arrow_length=10.0,
        tail_length=4.0,
        tail_width=8.0,
    )

    tip_x, tip_y, left_x, left_y, right_x, right_y = points
    assert round(tip_x, 3) == 100.0
    assert round(tip_y, 3) == 40.0
    assert round(left_x, 3) == 104.0
    assert round(left_y, 3) == 54.0
    assert round(right_x, 3) == 96.0
    assert round(right_y, 3) == 54.0


def test_compute_bistatic_echo_ellipse_axes_uses_sum_path_geometry() -> None:
    axes = _compute_bistatic_echo_ellipse_axes(distance_rx_to_point=10.0, echo_distance_m=6.0)

    assert axes is not None
    c, a, b = axes
    assert c == 5.0
    assert a == 8.0
    assert b == pytest.approx(math.sqrt(39.0))


def test_compute_bistatic_echo_ellipse_points_satisfy_focus_distance_sum() -> None:
    axes = _compute_bistatic_echo_ellipse_axes(distance_rx_to_point=10.0, echo_distance_m=6.0)

    assert axes is not None
    c, a, b = axes
    d = 10.0
    delta = 6.0
    for step in range(0, 65):
        t = (2.0 * math.pi * step) / 64.0
        x = a * math.cos(t)
        y = b * math.sin(t)
        distance_to_focus_1 = math.hypot(x + c, y)
        distance_to_focus_2 = math.hypot(x - c, y)
        assert distance_to_focus_1 + distance_to_focus_2 == pytest.approx(d + delta, abs=1e-9)


def test_compute_bistatic_echo_ellipse_axes_handles_zero_delta_without_crash() -> None:
    axes = _compute_bistatic_echo_ellipse_axes(distance_rx_to_point=10.0, echo_distance_m=0.0)

    assert axes is not None
    c, a, b = axes
    assert c == 5.0
    assert a == 5.0
    assert b == 0.0


def test_compute_bistatic_echo_ellipse_axes_rejects_negative_delta() -> None:
    assert _compute_bistatic_echo_ellipse_axes(distance_rx_to_point=10.0, echo_distance_m=-0.1) is None


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

    monkeypatch.setattr(module, "RosbridgePoseStreamTransport", _FakePoseStreamTransport)
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

    monkeypatch.setattr(module, "RosbridgePoseStreamTransport", _FakePoseStreamTransport)
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


def test_manual_prompt_navigator_prompts_and_returns_succeeded(monkeypatch) -> None:
    from transceiver import mission_workflow_ui as module

    parent = SimpleNamespace(after=lambda _delay, callback: callback())
    status_updates: list[tuple[str, str]] = []
    operator_messages: list[str] = []
    prompts: list[str] = []

    def _askokcancel(title: str, message: str, parent=None) -> bool:
        assert title == "Manuelle Navigation"
        prompts.append(message)
        return True

    monkeypatch.setattr("transceiver.mission_workflow_ui.messagebox.askokcancel", _askokcancel)
    navigator = module._ManualPromptNavigator(
        parent=parent,
        on_status=lambda stage, status: status_updates.append((stage, status)),
        on_operator_message=operator_messages.append,
        start_index=2,
    )

    nav_events: list[dict[str, object]] = []
    result = navigator.navigate_to_point(
        module.NavigationPoint(x=1.234, y=5.678, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        timeout_s=5.0,
        on_navigation_event=nav_events.append,
    )

    assert result == "succeeded"
    assert status_updates == [("navigation", "running"), ("navigation", "succeeded")]
    assert operator_messages == []
    assert prompts == ["Roboter zur Position 2 bringen: 1.234,5.678"]
    assert nav_events == []


def test_manual_prompt_navigator_returns_canceled_on_abort(monkeypatch) -> None:
    from transceiver import mission_workflow_ui as module

    parent = SimpleNamespace(after=lambda _delay, callback: callback())
    status_updates: list[tuple[str, str]] = []
    operator_messages: list[str] = []

    monkeypatch.setattr("transceiver.mission_workflow_ui.messagebox.askokcancel", lambda *_args, **_kwargs: False)
    navigator = module._ManualPromptNavigator(
        parent=parent,
        on_status=lambda stage, status: status_updates.append((stage, status)),
        on_operator_message=operator_messages.append,
        start_index=0,
    )

    result = navigator.navigate_to_point(
        module.NavigationPoint(x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        timeout_s=5.0,
    )

    assert result == "canceled"
    assert status_updates == [("navigation", "running"), ("navigation", "canceled")]
    assert any("Punktindex 0" in msg for msg in operator_messages)


def test_on_map_canvas_click_ignores_click_when_pick_mode_disabled() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._waypoint_map_pick_mode_enabled = False
    window._rx_antenna_map_pick_mode_enabled = False
    window._preview_pixel_to_world = lambda **_kwargs: (1.0, 2.0)
    calls: list[tuple[float, float]] = []
    window._set_rx_antenna_position = lambda *, x, y: calls.append((x, y))
    window._set_rx_antenna_map_pick_mode = lambda _enabled: None
    window._append_validation = lambda _text: None

    window._on_map_canvas_click(SimpleNamespace(x=10, y=20))

    assert calls == []


def test_on_map_canvas_click_sets_position_and_disables_pick_mode() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._waypoint_map_pick_mode_enabled = False
    window._rx_antenna_map_pick_mode_enabled = True
    window._preview_pixel_to_world = lambda **_kwargs: (3.5, -2.5)
    set_calls: list[tuple[float, float]] = []
    mode_calls: list[bool] = []
    window._set_rx_antenna_position = lambda *, x, y: set_calls.append((x, y))
    window._set_rx_antenna_map_pick_mode = lambda enabled: mode_calls.append(enabled)
    window._append_validation = lambda _text: None

    window._on_map_canvas_click(SimpleNamespace(x=10, y=20))

    assert set_calls == [(3.5, -2.5)]
    assert mode_calls == [False]


def test_on_map_canvas_click_starts_waypoint_pick_preview() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._waypoint_map_pick_mode_enabled = True
    window._preview_pixel_to_world = lambda **_kwargs: (1.5, 2.5)
    window._draw_map_preview = lambda: None

    window._on_map_canvas_click(SimpleNamespace(x=15, y=25))

    assert window._pending_waypoint_world_position == (1.5, 2.5)
    assert window._waypoint_drag_start_preview == (15.0, 25.0)
    assert window._pending_waypoint_yaw_radians == 0.0
    assert window._waypoint_drag_active is False


def test_on_map_canvas_drag_updates_pending_waypoint_yaw() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._waypoint_map_pick_mode_enabled = True
    window._pending_waypoint_world_position = (1.0, 1.0)
    window._waypoint_drag_start_preview = (10.0, 10.0)
    window._draw_map_preview = lambda: None

    window._on_map_canvas_drag(SimpleNamespace(x=20, y=10))

    assert window._waypoint_drag_active is True
    assert window._pending_waypoint_yaw_radians == 0.0


def test_on_map_canvas_drag_updates_pending_nav2point_yaw() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._nav2point_map_pick_mode_enabled = True
    window._pending_nav2point_world_position = (1.0, 1.0)
    window._nav2point_drag_start_preview = (10.0, 10.0)
    window._draw_map_preview = lambda: None

    window._on_map_canvas_drag(SimpleNamespace(x=20, y=10))

    assert window._nav2point_drag_active is True
    assert window._pending_nav2point_yaw_radians == 0.0


def test_on_map_canvas_release_creates_waypoint_and_disables_pick_mode() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._waypoint_map_pick_mode_enabled = True
    window._pending_waypoint_world_position = (4.0, -3.0)
    window._waypoint_drag_active = True
    window._pending_waypoint_yaw_radians = 1.2
    add_calls: list[tuple[float, float, float]] = []
    mode_calls: list[bool] = []
    window._add_point_from_values = lambda *, x, y, yaw_internal_radians, name=None: add_calls.append((x, y, yaw_internal_radians))
    window._clear_pending_waypoint_marker = lambda: None
    window._set_waypoint_map_pick_mode = lambda enabled: mode_calls.append(enabled)

    window._on_map_canvas_release(SimpleNamespace(x=4, y=2))

    assert add_calls == [(4.0, -3.0, 1.2)]
    assert mode_calls == [False]


def test_on_map_canvas_release_queues_nav2point_with_dragged_yaw() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._nav2point_map_pick_mode_enabled = True
    window._pending_nav2point_world_position = (4.0, -3.0)
    window._nav2point_drag_active = True
    window._pending_nav2point_yaw_radians = 1.2
    mode_calls: list[bool] = []
    queued: list[tuple[tuple[float, float], float]] = []
    window._set_nav2point_map_pick_mode = lambda enabled: mode_calls.append(enabled)
    window._queue_nav2point = lambda *, world_position, yaw_radians=0.0: queued.append((world_position, yaw_radians))

    window._on_map_canvas_release(SimpleNamespace(x=4, y=2))

    assert mode_calls == [False]
    assert queued == [((4.0, -3.0), 1.2)]


def test_review_measurement_auto_approves_when_manual_review_disabled() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window.manual_review_enabled_var = SimpleNamespace(get=lambda: False)
    observed: dict[str, object] = {}
    window.master = SimpleNamespace(
        review_measurement_for_mission=lambda **kwargs: observed.update(kwargs) or {
            "approved": False,
            "echo_delays": [12],
            "echo_lags": [8],
            "los_lag": -4,
        }
    )

    review_result = window._review_measurement(
        point_context=SimpleNamespace(point=SimpleNamespace(id="p1", name=""), global_index=1),
        output_file="dummy.bin",
    )

    assert observed["auto_approve"] is True
    assert review_result == {
        "approved": True,
        "reason": "",
        "detail": "",
        "echo_delays": [12],
        "echo_lags": [8],
        "los_lag": -4,
    }


def test_confirm_measurement_after_navigation_failure_uses_point_index_in_prompt(monkeypatch) -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    messages: list[str] = []
    asked: dict[str, str] = {}
    window.after = lambda _delay, callback: callback()
    window._append_validation = messages.append

    def _askyesno(title: str, message: str, parent=None) -> bool:
        asked["title"] = title
        asked["message"] = message
        return True

    monkeypatch.setattr("transceiver.mission_workflow_ui.messagebox.askyesno", _askyesno)

    decision = window._confirm_measurement_after_navigation_failure(
        point_context=SimpleNamespace(
            global_index=3,
            point=SimpleNamespace(id="point-007", name="Alpha"),
        ),
        navigation_state="timeout",
    )

    assert decision is True
    assert asked["title"] == "Navigation fehlgeschlagen"
    assert "Punktindex 3" in asked["message"]
    assert "point-007" not in asked["message"]
    assert any("Punktindex 3" in msg for msg in messages)


def test_check_run_prerequisites_skips_review_requirements_when_manual_review_disabled() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window.manual_review_enabled_var = SimpleNamespace(get=lambda: False)
    window._mission_points = [MeasurementPoint(id="p1", name="", x=0.0, y=0.0, yaw=0.0, enabled=True)]
    window._runtime_guard_reasons = lambda: []

    ok, reasons = window._check_run_prerequisites()

    assert ok is True
    assert reasons == []


def test_on_live_pose_stream_switch_changed_persists_and_syncs() -> None:
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    calls: list[str] = []
    window._persist_workflow_state = lambda: calls.append("persist")
    window._sync_live_pose_stream_state = lambda: calls.append("sync")
    window._update_live_label = lambda: calls.append("label")

    window._on_live_pose_stream_switch_changed()

    assert calls == ["persist", "sync", "label"]


def test_measurement_distance_m_returns_hypotenuse_for_two_points() -> None:
    distance = MissionWorkflowWindow._measurement_distance_m((1.0, 2.0), (4.0, 6.0))

    assert distance == 5.0


def test_measurement_distance_m_returns_none_when_points_missing() -> None:
    assert MissionWorkflowWindow._measurement_distance_m((1.0, 2.0), None) is None


def test_set_measurement_map_pick_mode_clears_overlay_when_disabled() -> None:
    class _DummyButton:
        def __init__(self) -> None:
            self.text = ""

        def configure(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            if "text" in kwargs:
                self.text = kwargs["text"]

    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._measurement_map_pick_mode_enabled = True
    window._measurement_start_world_position = (1.0, 2.0)
    window._measurement_end_world_position = (2.0, 3.0)
    window.measurement_map_pick_mode_btn = _DummyButton()
    window._set_waypoint_map_pick_mode = lambda _enabled: None
    window._set_rx_antenna_map_pick_mode = lambda _enabled: None
    window._update_map_canvas_cursor = lambda: None
    window._draw_map_preview = lambda: None

    window._set_measurement_map_pick_mode(False)

    assert window._measurement_map_pick_mode_enabled is False
    assert window._measurement_start_world_position is None
    assert window._measurement_end_world_position is None
    assert window.measurement_map_pick_mode_btn.text == "measurement"


def test_manual_measurement_point_context_prefers_selected_enabled_point() -> None:
    points = [
        MeasurementPoint(id="p1", name="", x=0.0, y=0.0, yaw=0.0, enabled=True),
        MeasurementPoint(id="p2", name="", x=1.0, y=1.0, yaw=0.0, enabled=True),
    ]
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._mission = MeasurementMission(name="m", points=points, repeat=1)
    window._mission_points = points
    window._selected_point_index = 1
    window._records = [{"global_index": 0}]
    window._selected_start_point_index = lambda: 0

    context = window._manual_measurement_point_context()

    assert context is not None
    assert context.point_index == 1
    assert context.point.id == "p2"
    assert context.global_index == 1


def test_manual_measurement_point_context_falls_back_to_selected_active_start_point() -> None:
    points = [
        MeasurementPoint(id="p1", name="", x=0.0, y=0.0, yaw=0.0, enabled=False),
        MeasurementPoint(id="p2", name="", x=1.0, y=1.0, yaw=0.0, enabled=True),
    ]
    window = MissionWorkflowWindow.__new__(MissionWorkflowWindow)
    window._mission = MeasurementMission(name="m", points=points, repeat=1)
    window._mission_points = points
    window._selected_point_index = 0
    window._records = []
    window.start_point_var = SimpleNamespace(get=lambda: "1")

    context = window._manual_measurement_point_context()

    assert context is not None
    assert context.point_index == 1
    assert context.point.id == "p2"
