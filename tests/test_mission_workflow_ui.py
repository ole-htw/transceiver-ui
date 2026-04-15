from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from transceiver.measurement_mission import MeasurementPoint
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
