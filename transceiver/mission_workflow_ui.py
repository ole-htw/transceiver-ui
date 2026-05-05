from __future__ import annotations

import threading
import time
import zipfile
import subprocess
import shlex
from dataclasses import replace
from datetime import datetime
import math
import re
from fractions import Fraction
from pathlib import Path
import json
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from typing import Any, Callable

import customtkinter as ctk
import numpy as np

from .app_config import MissionRuntimeConfig
from .measurement_mission import MapConfig, MeasurementMission, MeasurementPoint, measurement_mission_from_dict
from .measurement_run_executor import (
    JsonRunLogStore,
    MeasurementRunExecutor,
    MeasurementRunExecutorConfig,
    PointExecutionContext,
)
from .mission_measurement_service import (
    MissionRxMeasurementService,
    _coerce_echo_delay_entries,
    REVIEW_REASON_OPERATOR_REJECTED,
    REVIEW_REASON_REVIEW_EXCEPTION,
    REVIEW_REASON_REVIEW_UNAVAILABLE,
    normalize_review_reason,
)
from .navigation_adapter import (
    NavigationAdapter,
    NavigationAdapterConfig,
    NavigationEvent,
    NavigationPoint,
    Ros2CliNavigationTransport,
    RosbridgePoseStreamTransport,
    TerminalNavigationState,
)
from .window_utils import configure_child_window

MISSION_WORKFLOW_STATE_FILE = Path(__file__).with_name("mission_workflow_state.json")
LIVE_LABEL_TICKER_INTERVAL_MS = 250
LIVE_PREVIEW_TARGET_FPS = 25
LIVE_PREVIEW_FALLBACK_REDRAW_AFTER_S = 1.0
AUTO_STOP_CONTINUOUS_BEFORE_RUN = True
ECHO_OVERLAY_COLORS = ("#00796B", "#FFB300", "#8E24AA", "#00ACC1", "#F4511E")
ECHO_HEADING_MARKERS = ("🟢", "🟠", "🟣", "🔵", "🟤")
LIDAR_OVERLAY_MAX_DRAWN_BEAMS = 450
LIDAR_OVERLAY_CELL_SIZE_PX = 3.0
LIDAR_OVERLAY_MAX_BEAMS_PER_CELL = 1
MEASUREMENT_START_LIVE_POSITION_WAIT_TIMEOUT_S = 1.6
MEASUREMENT_START_LIVE_POSITION_WAIT_INTERVAL_S = 0.1
LIVE_ECHO_CACHE_POSITION_DELTA_M = 0.015
LIVE_ECHO_CACHE_DISTANCE_DELTA_M = 0.02
LIVE_ECHO_SAMPLING_NORMAL = (24, 32, 48)
LIVE_ECHO_SAMPLING_REDUCED = (16, 24, 32)
MULTI_SELECTION_PROBABILITY_SIGMA_M = 1.5
MULTI_SELECTION_PROBABILITY_GRID_STEP_PX = 10
MULTI_SELECTION_PROBABILITY_MIN_NORMALIZED = 0.05
MULTI_SELECTION_PROBABILITY_MIN_HEAT = 0.2
MULTI_SELECTION_PROBABILITY_USE_STIPPLE = False
MULTI_SELECTION_PROBABILITY_DEBUG_LOG = True



def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_json_dict(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


def _map_executor_state_to_ui_text(state: str, completion_substatus: str | None = None) -> str:
    mapping = {
        "completed": "Abgeschlossen",
        "interrupted": "Unterbrochen",
        "failed": "Fehlgeschlagen",
        "cancelled": "Abgebrochen",
    }
    base = mapping.get(state, state)
    if state == "completed" and completion_substatus:
        return f"{base} ({completion_substatus})"
    return base


def _operator_error_code(event_type: str, detail: str) -> str:
    normalized = detail.lower()
    if event_type == "connection_error" and "dds payload" in normalized:
        return "navigation_failed.connection_error.dds_payload_mismatch"
    if event_type == "connection_error":
        return "navigation_failed.connection_error"
    if event_type == "timeout":
        return "navigation_failed.timeout"
    if event_type == "aborted":
        return "navigation_failed.aborted"
    return f"navigation_failed.{event_type}"


def _compute_bistatic_echo_ellipse_axes(
    *,
    distance_rx_to_point: float,
    echo_distance_m: float,
) -> tuple[float, float, float] | None:
    if (
        not math.isfinite(distance_rx_to_point)
        or not math.isfinite(echo_distance_m)
        or distance_rx_to_point < 0.0
        or echo_distance_m < 0.0
    ):
        return None
    semi_focal_distance = distance_rx_to_point / 2.0
    semi_major_axis = (distance_rx_to_point + echo_distance_m) / 2.0
    semi_minor_squared = semi_major_axis * semi_major_axis - semi_focal_distance * semi_focal_distance
    if semi_minor_squared < 0.0 and abs(semi_minor_squared) < 1e-12:
        semi_minor_squared = 0.0
    if semi_minor_squared < 0.0:
        return None
    semi_minor_axis = math.sqrt(semi_minor_squared)
    return (semi_focal_distance, semi_major_axis, semi_minor_axis)


class _UiNavigator:
    def __init__(self, *, adapter: NavigationAdapter, on_status, on_operator_message) -> None:
        self._adapter = adapter
        self._on_status = on_status
        self._on_operator_message = on_operator_message
        self._pose_stream = RosbridgePoseStreamTransport()
        self._latest_pose_lock = threading.Lock()
        self._latest_pose_from_stream: dict[str, Any] | None = None

    def start_pose_stream(
        self,
        *,
        on_runtime_event: Callable[[dict[str, Any]], None],
        expected_frame_id: str | None = None,
    ) -> None:
        def _on_pose_stream_event(payload: dict[str, Any]) -> None:
            if payload.get("type") == "pose_stream":
                event = payload.get("event")
                if isinstance(event, dict) and event.get("type") == "position_update":
                    position = event.get("position")
                    with self._latest_pose_lock:
                        self._latest_pose_from_stream = position if isinstance(position, dict) else None
            on_runtime_event(payload)

        self._pose_stream.start(
            config=self._adapter.config,
            on_event=_on_pose_stream_event,
            expected_frame_id=expected_frame_id,
        )

    def stop_pose_stream(self) -> None:
        self._pose_stream.stop()

    def navigate_to_point(  # type: ignore[no-untyped-def]
        self,
        point,
        *,
        timeout_s: float,
        on_navigation_event=None,
    ):
        self._on_status("navigation", "running")
        latest_position_lock = threading.Lock()
        with self._latest_pose_lock:
            latest_position: dict[str, Any] | None = (
                dict(self._latest_pose_from_stream)
                if isinstance(self._latest_pose_from_stream, dict)
                else None
            )
        polling_active = threading.Event()
        polling_active.set()

        def _emit_position_update() -> None:
            if on_navigation_event is None:
                return
            with latest_position_lock:
                position_copy = dict(latest_position) if isinstance(latest_position, dict) else None
            on_navigation_event(
                {
                    "type": "position_update",
                    "position": position_copy,
                }
            )

        def _position_polling_loop() -> None:
            while polling_active.is_set():
                _emit_position_update()
                time.sleep(0.5)

        polling_thread = threading.Thread(target=_position_polling_loop, daemon=True)
        polling_thread.start()

        def _on_event(event: NavigationEvent) -> None:
            nonlocal latest_position
            if event.type in {"connection_error", "aborted", "timeout"}:
                detail = event.message or 'ohne Details'
                error_code = _operator_error_code(event.type, detail)
                if event.type == "connection_error":
                    detail = (
                        f"{detail} | ROS-Umgebung prüfen (TRANSCEIVER_REMOTE_ROS_ENV_CMD / TRANSCEIVER_REMOTE_ROS_SETUP)"
                    )
                self._on_operator_message(
                    f"Navigation {event.type} (Versuch {event.attempt}) [{error_code}]: {detail}"
                )
            if event.type == "succeeded":
                self._on_status("navigation", "succeeded")
            if event.type in {"feedback", "position"}:
                with self._latest_pose_lock:
                    stream_position = self._latest_pose_from_stream
                with latest_position_lock:
                    latest_position = dict(stream_position) if isinstance(stream_position, dict) else None

        state = self._adapter.navigate_to_point(point, timeout_s=timeout_s, on_event=_on_event)
        polling_active.clear()
        polling_thread.join(timeout=1.0)
        _emit_position_update()
        if state != "succeeded":
            self._on_status("navigation", state)
        return state

    def cancel_current_goal(self) -> None:
        self._adapter.cancel_current_goal()


class _ManualPromptNavigator:
    def __init__(
        self,
        *,
        parent: tk.Misc,
        on_status: Callable[[str, str], None],
        on_operator_message: Callable[[str], None],
        start_index: int,
    ) -> None:
        self._parent = parent
        self._on_status = on_status
        self._on_operator_message = on_operator_message
        self._next_global_index = max(0, start_index)
        self._index_lock = threading.Lock()
        self._cancel_requested = threading.Event()

    def navigate_to_point(
        self,
        point: NavigationPoint,
        *,
        timeout_s: float,
        on_navigation_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> TerminalNavigationState:
        del timeout_s
        if self._cancel_requested.is_set():
            return "canceled"
        with self._index_lock:
            current_index = self._next_global_index
            self._next_global_index += 1
        self._on_status("navigation", "running")
        decision_ready = threading.Event()
        decision: dict[str, bool] = {"ok": False}

        def _ask_operator() -> None:
            if self._cancel_requested.is_set():
                decision_ready.set()
                return
            prompt = (
                f"Roboter zur Position {current_index} bringen: "
                f"{point.x:.3f},{point.y:.3f}"
            )
            decision["ok"] = bool(
                messagebox.askokcancel(
                    "Manuelle Navigation",
                    prompt,
                    parent=self._parent,
                )
            )
            decision_ready.set()

        try:
            self._parent.after(0, _ask_operator)
        except tk.TclError:
            return "aborted"
        while not decision_ready.wait(timeout=0.1):
            if self._cancel_requested.is_set():
                return "canceled"
        if self._cancel_requested.is_set():
            return "canceled"
        if not decision["ok"]:
            self._on_status("navigation", "canceled")
            self._on_operator_message(
                f"⚠️ Manuelle Navigation für Punktindex {current_index} wurde abgebrochen."
            )
            return "canceled"
        self._on_status("navigation", "succeeded")
        return "succeeded"

    def cancel_current_goal(self) -> None:
        self._cancel_requested.set()


class MissionWorkflowWindow(ctk.CTkToplevel):
    def __init__(self, parent: ctk.CTk) -> None:
        super().__init__(parent)
        configure_child_window(
            self,
            parent=parent,
            on_close=self._on_window_close,
        )
        self.title("Mission Workflow")
        self.geometry("1100x700")
        self.minsize(980, 640)

        self._mission: MeasurementMission | None = None
        self._executor: MeasurementRunExecutor | None = None
        self._navigator: _UiNavigator | None = None
        self._run_thread: threading.Thread | None = None
        self._nav2point_thread: threading.Thread | None = None
        self._manual_measurement_thread: threading.Thread | None = None
        self._records: list[dict[str, Any]] = []
        self._run_started_at: float | None = None
        self._run_log_dir: Path | None = None
        self._runtime_config = MissionRuntimeConfig.from_env()
        self._workflow_state_file = MISSION_WORKFLOW_STATE_FILE
        self._selected_map_config: MapConfig | None = None
        self._selected_map_config_file: str | None = None
        self._is_restoring_workflow_state = False
        self._live_label_ticker_job: str | None = None
        self._live_label_ticker_active = False
        self.lidar_reference_enabled_var = tk.BooleanVar(value=True)
        self.manual_review_enabled_var = tk.BooleanVar(value=True)
        self.test_run_enabled_var = tk.BooleanVar(value=False)
        self.manual_navigation_enabled_var = tk.BooleanVar(value=False)
        self.live_pose_stream_enabled_var = tk.BooleanVar(value=False)
        self.live_preview_enabled_var = tk.BooleanVar(value=False)
        self._live_pose_stream_active = False

        self._build_ui()
        self._restore_workflow_state()
        self._sync_live_pose_stream_state()
        self._sync_live_preview_state()
        self.after_idle(self._stabilize_initial_geometry)
        self.after_idle(self._open_maximized)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(5, weight=0)

        workflow = ctk.CTkFrame(self)
        workflow.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        for col in range(6):
            workflow.columnconfigure(col, weight=0)
        workflow.columnconfigure(5, weight=1)

        ctk.CTkLabel(workflow, text="1) Missionsparameter").grid(row=0, column=0, padx=8, pady=8)
        ctk.CTkLabel(workflow, text="Name").grid(row=0, column=1, padx=(8, 2))
        self.mission_name_var = tk.StringVar(value="mission-ui")
        ctk.CTkEntry(workflow, textvariable=self.mission_name_var, width=190).grid(row=0, column=2, padx=(0, 8))
        ctk.CTkLabel(workflow, text="Wiederholungen").grid(row=0, column=3, padx=(8, 2))
        self.repeat_var = tk.StringVar(value="1")
        ctk.CTkEntry(workflow, textvariable=self.repeat_var, width=70).grid(row=0, column=4, padx=(0, 8))
        ctk.CTkButton(workflow, text="Mission validieren", command=self._validate_selected).grid(row=0, column=5, padx=(0, 8), sticky="w")

        points_editor = ctk.CTkFrame(self)
        points_editor.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        for col in range(11):
            points_editor.columnconfigure(col, weight=0)
        points_editor.columnconfigure(10, weight=1)

        ctk.CTkLabel(points_editor, text="2) Messpunkt anlegen").grid(row=0, column=0, padx=8, pady=8)
        self.point_name_var = tk.StringVar(value="")
        self.point_x_var = tk.StringVar(value="0.0")
        self.point_y_var = tk.StringVar(value="0.0")
        self.point_yaw_var = tk.StringVar(value="0°")

        self._labeled_entry(points_editor, row=0, column=1, label="Name", variable=self.point_name_var, width=120)
        self._labeled_entry(points_editor, row=0, column=2, label="X", variable=self.point_x_var, width=90)
        self._labeled_entry(points_editor, row=0, column=3, label="Y", variable=self.point_y_var, width=90)
        self._labeled_entry(
            points_editor,
            row=0,
            column=4,
            label="Yaw",
            variable=self.point_yaw_var,
            width=90,
            validatecommand=(self.register(self._validate_yaw_input), "%P"),
        )
        ctk.CTkButton(points_editor, text="Punkt hinzufügen", command=self._add_point).grid(row=0, column=5, padx=(8, 3))
        ctk.CTkButton(points_editor, text="Auswahl entfernen", command=self._remove_selected_point).grid(row=0, column=6, padx=3)
        ctk.CTkButton(points_editor, text="▲", width=36, command=self._move_selected_point_up).grid(row=0, column=7, padx=3)
        ctk.CTkButton(points_editor, text="▼", width=36, command=self._move_selected_point_down).grid(row=0, column=8, padx=(3, 8), sticky="w")
        ctk.CTkButton(
            points_editor,
            text="Aktivieren/Deaktivieren",
            command=self._toggle_selected_point_enabled,
        ).grid(row=0, column=9, padx=(3, 8), sticky="w")
        self.waypoint_map_pick_mode_btn = ctk.CTkButton(
            points_editor,
            text="🖱️",
            command=self._toggle_waypoint_map_pick_mode,
            width=42,
        )
        self.waypoint_map_pick_mode_btn.grid(row=0, column=10, padx=(3, 0), sticky="w")

        map_controls_row = ctk.CTkFrame(self, fg_color="transparent")
        map_controls_row.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 6))
        map_controls_row.columnconfigure(0, weight=3)
        map_controls_row.columnconfigure(1, weight=2)
        map_controls_row.rowconfigure(0, weight=1)

        map_frame = ctk.CTkFrame(map_controls_row)
        map_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        map_frame.columnconfigure(0, weight=1)
        map_frame.rowconfigure(2, weight=1)

        ctk.CTkLabel(map_frame, text="4) Karte").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        map_top_controls = ctk.CTkFrame(map_frame, fg_color="transparent")
        map_top_controls.grid(row=0, column=0, sticky="e", padx=8, pady=(8, 2))
        ctk.CTkButton(map_top_controls, text="Map-Config wählen", command=self._select_map_config_file).grid(
            row=0, column=0, sticky="w"
        )
        self.measurement_map_pick_mode_btn = ctk.CTkButton(
            map_top_controls,
            text="measurement",
            command=self._toggle_measurement_map_pick_mode,
            width=110,
        )
        self.measurement_map_pick_mode_btn.grid(row=0, column=1, padx=(6, 0), sticky="w")
        self.nav2point_map_pick_mode_btn = ctk.CTkButton(
            map_top_controls,
            text="nav2point",
            command=self._toggle_nav2point_map_pick_mode,
            width=110,
        )
        self.nav2point_map_pick_mode_btn.grid(row=0, column=2, padx=(6, 0), sticky="w")
        self.map_status_var = tk.StringVar(value="Karte nicht konfiguriert.")
        ctk.CTkLabel(map_frame, textvariable=self.map_status_var, anchor="w").grid(
            row=1, column=0, sticky="ew", padx=8, pady=(0, 6)
        )
        self.map_preview_canvas = tk.Canvas(
            map_frame,
            bg="#1d1f23",
            highlightthickness=0,
            height=260,
        )
        self.map_preview_canvas.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.map_preview_canvas.bind("<Configure>", self._on_map_canvas_resize)
        self.map_preview_canvas.bind("<Button-1>", self._on_map_canvas_click)
        self.map_preview_canvas.bind("<B1-Motion>", self._on_map_canvas_drag)
        self.map_preview_canvas.bind("<ButtonRelease-1>", self._on_map_canvas_release)
        self._map_image_original: tk.PhotoImage | None = None
        self._map_image_preview: tk.PhotoImage | None = None
        self._map_preview_scale: tuple[float, float] = (1.0, 1.0)
        self._map_preview_offset: tuple[float, float] = (0.0, 0.0)
        self._map_canvas_image_id: int | None = None
        self._map_marker_ids: list[int] = []
        self._live_overlay_item_ids: dict[str, Any] = {
            "echo_slots": {},
            "marker": None,
            "heading": None,
            "position_info_box": None,
            "position_info_text": None,
        }
        self._static_map_layer_signature: tuple[Any, ...] | None = None
        self._map_image_size: tuple[int, int] | None = None
        self._live_position: dict[str, Any] | None = None
        self._live_position_received_at: float | None = None
        self._live_redraw_pending = False
        self._live_redraw_job: str | None = None
        self._live_marker_redraw_pending = False
        self._live_marker_redraw_job: str | None = None
        self._last_live_redraw_ts: float | None = None
        self._live_position_at_measurement_start: dict[str, Any] | None = None
        self._measurement_start_live_position_event = threading.Event()
        self._selected_point_index: int | None = None
        self._selected_result_index: int | None = None
        self._selected_result_indices: tuple[int, ...] = ()
        self._lidar_reference_scan_cache: dict[str, dict[str, Any] | None] = {}
        self._ellipse_unit_circle_cache: dict[int, tuple[tuple[float, float], ...]] = {}
        self._live_echo_geometry_cache: dict[str, dict[str, Any]] = {}
        self._last_live_diagnosis_key: str | None = None
        self._emit_live_diagnostics_to_validation = True
        self._rx_antenna_global_position: tuple[float, float] | None = None
        self._rx_antenna_map_pick_mode_enabled = False
        self._waypoint_map_pick_mode_enabled = False
        self._waypoint_drag_start_preview: tuple[float, float] | None = None
        self._measurement_map_pick_mode_enabled = False
        self._nav2point_map_pick_mode_enabled = False
        self._measurement_start_world_position: tuple[float, float] | None = None
        self._measurement_end_world_position: tuple[float, float] | None = None
        self._manual_drive_lock = threading.Lock()
        self._pending_nav2point_world_position: tuple[float, float] | None = None
        self._pending_nav2point_yaw_radians = 0.0
        self._nav2point_drag_start_preview: tuple[float, float] | None = None
        self._nav2point_drag_active = False
        self._pending_waypoint_world_position: tuple[float, float] | None = None
        self._pending_waypoint_yaw_radians = 0.0
        self._waypoint_drag_active = False
        self.rx_antenna_x_var = tk.StringVar(value="")
        self.rx_antenna_y_var = tk.StringVar(value="")

        rx_position_controls = ctk.CTkFrame(map_frame, fg_color="transparent")
        rx_position_controls.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        rx_position_controls.columnconfigure(8, weight=1)
        ctk.CTkLabel(rx_position_controls, text="RX-Antenne global").grid(row=0, column=0, padx=(0, 4), sticky="w")
        ctk.CTkLabel(rx_position_controls, text="X").grid(row=0, column=1, padx=(6, 2), sticky="w")
        ctk.CTkEntry(rx_position_controls, textvariable=self.rx_antenna_x_var, width=95).grid(
            row=0, column=2, padx=(0, 4), sticky="w"
        )
        ctk.CTkLabel(rx_position_controls, text="Y").grid(row=0, column=3, padx=(6, 2), sticky="w")
        ctk.CTkEntry(rx_position_controls, textvariable=self.rx_antenna_y_var, width=95).grid(
            row=0, column=4, padx=(0, 6), sticky="w"
        )
        ctk.CTkButton(
            rx_position_controls,
            text="Übernehmen",
            command=self._apply_rx_antenna_position_from_inputs,
            width=90,
        ).grid(row=0, column=5, padx=3, sticky="w")
        ctk.CTkButton(
            rx_position_controls,
            text="Löschen",
            command=self._clear_rx_antenna_position,
            width=80,
        ).grid(row=0, column=6, padx=(3, 0), sticky="w")
        self.rx_antenna_map_pick_mode_btn = ctk.CTkButton(
            rx_position_controls,
            text="🖱️",
            command=self._toggle_rx_antenna_map_pick_mode,
            width=42,
        )
        self.rx_antenna_map_pick_mode_btn.grid(row=0, column=7, padx=(3, 0), sticky="w")

        side_panel = ctk.CTkFrame(map_controls_row)
        side_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(0, weight=1)
        side_panel.rowconfigure(1, weight=1)
        side_panel.rowconfigure(2, weight=0)

        points_table_frame = ctk.CTkFrame(side_panel)
        points_table_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 6))
        points_table_frame.columnconfigure(0, weight=1)
        points_table_frame.rowconfigure(1, weight=1)
        ctk.CTkLabel(points_table_frame, text="3) Wegpunkte").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        point_columns = ("idx", "x", "y", "yaw")
        self.points_table = ttk.Treeview(points_table_frame, columns=point_columns, show="headings", height=6)
        self.points_table.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        for key, title in {
            "idx": "#",
            "x": "X",
            "y": "Y",
            "yaw": "Yaw",
        }.items():
            self.points_table.heading(key, text=title)
            self.points_table.column(key, stretch=True, width=95)
        self.points_table.tag_configure("active", foreground="#000000")
        self.points_table.tag_configure("inactive", foreground="#808080")
        self.points_table.bind("<<TreeviewSelect>>", self._on_points_table_select)
        self.points_table.bind("<Double-1>", self._on_points_table_double_click)

        terminal_frame = ctk.CTkFrame(side_panel)
        terminal_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=(0, 6))
        terminal_frame.columnconfigure(0, weight=1)
        terminal_frame.rowconfigure(1, weight=1)
        ctk.CTkLabel(terminal_frame, text="Terminal-Ausgabe").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        self.validation_box = ctk.CTkTextbox(terminal_frame, height=120)
        self.validation_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.validation_box.insert("1.0", "3) Validierungsergebnis erscheint hier.\n")
        self.validation_box.configure(state="disabled")

        controls = ctk.CTkFrame(side_panel)
        controls.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        for col in range(5):
            controls.columnconfigure(col, weight=0)
        controls.columnconfigure(4, weight=1)
        controls.rowconfigure(3, weight=1)

        ctk.CTkLabel(controls, text="5) Laufsteuerung").grid(row=0, column=0, columnspan=5, sticky="w", padx=8, pady=(8, 4))
        self.review_ready_var = tk.StringVar(value="Review: nicht geprüft")
        ctk.CTkLabel(controls, textvariable=self.review_ready_var, anchor="w").grid(
            row=0, column=4, padx=(8, 8), pady=(8, 4), sticky="e"
        )
        self.start_btn = ctk.CTkButton(controls, text="Start", command=self._start_run)
        self.start_btn.grid(row=1, column=0, padx=(8, 3), pady=(0, 4), sticky="w")
        self.manual_measurement_btn = ctk.CTkButton(
            controls,
            text="Manuelle Messung",
            command=self._start_manual_measurement,
        )
        self.manual_measurement_btn.grid(row=1, column=1, padx=(3, 3), pady=(0, 4), sticky="w")
        ctk.CTkLabel(controls, text="Start ab Punkt").grid(row=1, column=2, padx=(10, 2), pady=(0, 4), sticky="e")
        self.start_point_var = tk.StringVar(value="1")
        self.start_point_combo = ctk.CTkComboBox(
            controls,
            width=150,
            values=["1"],
            variable=self.start_point_var,
            state="readonly",
            command=lambda _value: self._persist_workflow_state(),
        )
        self.start_point_combo.grid(row=1, column=3, columnspan=2, padx=(0, 8), pady=(0, 4), sticky="w")
        self.reverse_point_order_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            controls,
            text="Reihenfolge umdrehen",
            variable=self.reverse_point_order_var,
            command=self._persist_workflow_state,
        ).grid(row=2, column=3, padx=(10, 8), pady=3, sticky="w")
        self.pause_btn = ctk.CTkButton(controls, text="Pause", command=self._pause_run, state="disabled")
        self.pause_btn.grid(row=2, column=0, padx=(8, 3), pady=3, sticky="w")
        self.resume_btn = ctk.CTkButton(controls, text="Fortsetzen", command=self._resume_run, state="disabled")
        self.resume_btn.grid(row=2, column=1, padx=3, pady=3, sticky="w")
        self.stop_btn = ctk.CTkButton(controls, text="Stop", command=self._stop_run, state="disabled")
        self.stop_btn.grid(row=2, column=2, padx=3, pady=3, sticky="w")
        ctk.CTkButton(controls, text="Run-Logs exportieren", command=self._export_logs).grid(
            row=2, column=4, padx=(10, 8), pady=3, sticky="w"
        )
        ctk.CTkButton(controls, text="Importieren", command=self._import_logs).grid(
            row=2, column=5, padx=(0, 8), pady=3, sticky="w"
        )
        ctk.CTkCheckBox(
            controls,
            text="LIDAR-Referenzmessung aktiv",
            variable=self.lidar_reference_enabled_var,
            command=self._persist_workflow_state,
        ).grid(row=3, column=0, columnspan=2, padx=(8, 3), pady=(0, 4), sticky="w")
        ctk.CTkCheckBox(
            controls,
            text="Manuelle Prüfung",
            variable=self.manual_review_enabled_var,
            command=self._on_manual_review_toggle_changed,
        ).grid(row=3, column=2, padx=(8, 3), pady=(0, 4), sticky="w")
        ctk.CTkCheckBox(
            controls,
            text="Testlauf (ohne Messung)",
            variable=self.test_run_enabled_var,
            command=self._on_test_run_toggle_changed,
        ).grid(row=3, column=3, padx=(8, 3), pady=(0, 4), sticky="w")
        ctk.CTkCheckBox(
            controls,
            text="Manuelle Navigation",
            variable=self.manual_navigation_enabled_var,
            command=self._persist_workflow_state,
        ).grid(row=3, column=4, padx=(8, 3), pady=(0, 4), sticky="w")
        ctk.CTkSwitch(
            controls,
            text="Live-Position aktivieren",
            variable=self.live_pose_stream_enabled_var,
            command=self._on_live_pose_stream_switch_changed,
        ).grid(row=4, column=0, columnspan=2, padx=(8, 3), pady=(0, 4), sticky="w")
        ctk.CTkSwitch(
            controls,
            text="Live-Preview aktivieren",
            variable=self.live_preview_enabled_var,
            command=self._on_live_preview_switch_changed,
        ).grid(row=4, column=2, columnspan=2, padx=(8, 3), pady=(0, 4), sticky="w")

        self.live_var = tk.StringVar(value="Punkt: - | Navigation: idle | Messung: idle | Verbleibend: - | Live-Status: Karte nicht geladen")
        ctk.CTkLabel(controls, textvariable=self.live_var, anchor="w", justify="left").grid(
            row=5, column=0, columnspan=5, sticky="nsew", padx=8, pady=(4, 8)
        )

        table_frame = ctk.CTkFrame(self)
        table_frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=(0, 10))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = (
            "measurement_idx",
            "idx",
            "live_position",
            "live_distance_to_rx_m",
            "echo_1_m",
            "echo_2_m",
            "echo_3_m",
            "echo_4_m",
            "echo_5_m",
            "review_action",
            "status",
        )
        self.results_table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=14,
            selectmode="extended",
        )
        self.results_table.grid(row=0, column=0, sticky="nsew")
        headings = {
            "measurement_idx": "Messung",
            "idx": "Punktindex",
            "live_position": "Position",
            "live_distance_to_rx_m": "Abstand",
            "echo_1_m": f"{ECHO_HEADING_MARKERS[0]} E1",
            "echo_2_m": f"{ECHO_HEADING_MARKERS[1]} E2",
            "echo_3_m": f"{ECHO_HEADING_MARKERS[2]} E3",
            "echo_4_m": f"{ECHO_HEADING_MARKERS[3]} E4",
            "echo_5_m": f"{ECHO_HEADING_MARKERS[4]} E5",
            "review_action": "Review",
            "status": "Status",
        }
        for key, title in headings.items():
            self.results_table.heading(key, text=title)
            self.results_table.column(key, stretch=True, width=110)
        self.results_table.column("measurement_idx", width=80)
        self.results_table.column("live_position", width=100)
        self.results_table.column("live_distance_to_rx_m", width=90)
        self.results_table.column("echo_1_m", width=80)
        self.results_table.column("echo_2_m", width=80)
        self.results_table.column("echo_3_m", width=80)
        self.results_table.column("echo_4_m", width=80)
        self.results_table.column("echo_5_m", width=80)
        self.results_table.column("review_action", width=90, stretch=False)
        self.results_table.column("status", width=320)

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.results_table.configure(yscrollcommand=scroll.set)
        self.results_table.bind("<<TreeviewSelect>>", self._on_results_table_select)
        self.results_table.bind("<Button-1>", self._on_results_table_click, add="+")
        self.results_selection_diagnostics_var = tk.StringVar(value="Auswahl: 0 Zeilen")
        ctk.CTkLabel(
            table_frame,
            textvariable=self.results_selection_diagnostics_var,
            anchor="w",
            justify="left",
        ).grid(row=1, column=0, sticky="ew", padx=(2, 0), pady=(4, 0))
        ctk.CTkButton(
            table_frame,
            text="Ergebnisliste leeren",
            command=self._clear_results_table,
        ).grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(4, 0))

        self._mission_points: list[MeasurementPoint] = []
        self.mission_name_var.trace_add("write", lambda *_args: self._persist_workflow_state())
        self.repeat_var.trace_add("write", lambda *_args: self._persist_workflow_state())
        self._refresh_start_point_options()
        self._refresh_map_section()
        self._refresh_review_ready_indicator()

    def _stabilize_initial_geometry(self) -> None:
        """Ensure all control rows are visible right after opening the window."""
        self.update_idletasks()
        required_width = max(self.winfo_reqwidth(), 980)
        required_height = max(self.winfo_reqheight(), 640)
        self.geometry(f"{required_width}x{required_height}")

    def _open_maximized(self) -> None:
        """Open the workflow window maximized across supported window managers."""
        try:
            self.state("zoomed")
            return
        except tk.TclError:
            pass

        try:
            self.attributes("-zoomed", True)
            return
        except tk.TclError:
            pass

        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}+0+0")

    @staticmethod
    def _labeled_entry(
        parent: ctk.CTkFrame,
        *,
        row: int,
        column: int,
        label: str,
        variable: tk.StringVar,
        width: int,
        validatecommand: tuple[str, str] | None = None,
    ) -> None:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=column, padx=3, pady=3)
        ctk.CTkLabel(wrap, text=label).pack(side="top", anchor="w")
        entry_kwargs: dict[str, Any] = {"textvariable": variable, "width": width}
        if validatecommand is not None:
            entry_kwargs["validate"] = "key"
            entry_kwargs["validatecommand"] = validatecommand
        ctk.CTkEntry(wrap, **entry_kwargs).pack(side="top")

    def _append_validation(self, text: str) -> None:
        if not self._is_validation_box_available():
            return
        self.validation_box.configure(state="normal")
        self.validation_box.insert("end", text + "\n")
        self.validation_box.see("end")
        self.validation_box.configure(state="disabled")

    def _set_validation_text(self, text: str) -> None:
        if not self._is_validation_box_available():
            return
        self.validation_box.configure(state="normal")
        self.validation_box.delete("1.0", "end")
        if not text.endswith("\n"):
            text = f"{text}\n"
        self.validation_box.insert("1.0", text)
        self.validation_box.see("end")
        self.validation_box.configure(state="disabled")

    def _is_validation_box_available(self) -> bool:
        try:
            return bool(self.validation_box.winfo_exists())
        except tk.TclError:
            return False

    @staticmethod
    def _yaw_cw_degrees_to_internal_radians(yaw_cw_degrees: Any) -> float:
        if isinstance(yaw_cw_degrees, str):
            yaw_deg = float(yaw_cw_degrees.strip().removesuffix("°").strip())
        else:
            yaw_deg = float(yaw_cw_degrees)
        return math.radians(-yaw_deg)

    @staticmethod
    def _yaw_internal_radians_to_cw_degrees(yaw_radians: float) -> float:
        return -math.degrees(yaw_radians)

    @staticmethod
    def _validate_yaw_input(proposed: str) -> bool:
        if proposed in {"", "-", "°", "-°"}:
            return True
        return bool(re.fullmatch(r"-?\d+°?", proposed))

    @staticmethod
    def _format_yaw_degrees(yaw_radians: float | None) -> str:
        if yaw_radians is None:
            return "-"
        yaw_degrees = int(round(MissionWorkflowWindow._yaw_internal_radians_to_cw_degrees(yaw_radians)))
        if yaw_degrees == 0:
            yaw_degrees = 0
        return f"{yaw_degrees}°"

    def _refresh_map_section(self) -> None:
        self._map_image_original = None
        self._map_image_preview = None
        self._map_preview_scale = (1.0, 1.0)
        self._map_preview_offset = (0.0, 0.0)
        self._map_canvas_image_id = None
        self._map_marker_ids = []
        self._live_overlay_item_ids = {
            "echo_slots": {},
            "marker": None,
            "heading": None,
            "position_info_box": None,
            "position_info_text": None,
        }
        self._static_map_layer_signature = None
        self._invalidate_live_echo_geometry_cache()
        self._map_image_size = None
        self._live_position = None
        self._live_position_received_at = None

        map_config = self._selected_map_config
        if map_config is None:
            self.map_status_var.set("Karte nicht konfiguriert (map_config fehlt).")
            self._render_map_placeholder("Kein Kartenbild konfiguriert.")
            self._update_live_label()
            return

        image_path = Path(map_config.image).expanduser()
        if not image_path.is_absolute():
            image_path = (Path.cwd() / image_path).resolve()

        if not image_path.exists():
            self.map_status_var.set(f"Kartenbild nicht gefunden: {image_path}")
            self._render_map_placeholder("Kartenbild fehlt.\nBitte map_config.image prüfen.")
            self._update_live_label()
            return

        try:
            photo = tk.PhotoImage(file=str(image_path))
        except Exception:
            self.map_status_var.set(f"Kartenbild ungültig oder nicht lesbar: {image_path}")
            self._render_map_placeholder("Kartenbild konnte nicht geladen werden.")
            self._update_live_label()
            return

        self._map_image_original = photo
        self._map_image_preview = photo
        self._map_image_size = (photo.width(), photo.height())
        self.map_status_var.set(
            f"Karte geladen: {image_path.name} ({photo.width()}x{photo.height()} px)"
        )
        self._draw_map_preview()
        self._update_live_label()

    def _on_map_canvas_resize(self, _event: tk.Event) -> None:
        if self._map_image_original is None:
            return
        self._draw_map_preview()

    def _on_map_canvas_click(self, event: tk.Event) -> None:
        if getattr(self, "_nav2point_map_pick_mode_enabled", False):
            world_position = self._preview_pixel_to_world(preview_x=float(event.x), preview_y=float(event.y))
            if world_position is None:
                return
            self._pending_nav2point_world_position = world_position
            self._pending_nav2point_yaw_radians = 0.0
            self._nav2point_drag_start_preview = (float(event.x), float(event.y))
            self._nav2point_drag_active = False
            self._draw_map_preview()
            return
        if getattr(self, "_measurement_map_pick_mode_enabled", False):
            world_position = self._preview_pixel_to_world(preview_x=float(event.x), preview_y=float(event.y))
            if world_position is None:
                return
            if self._measurement_start_world_position is None or self._measurement_end_world_position is not None:
                self._measurement_start_world_position = world_position
                self._measurement_end_world_position = None
            else:
                self._measurement_end_world_position = world_position
            self._draw_map_preview()
            return
        if self._waypoint_map_pick_mode_enabled:
            world_position = self._preview_pixel_to_world(preview_x=float(event.x), preview_y=float(event.y))
            if world_position is None:
                return
            self._pending_waypoint_world_position = world_position
            self._waypoint_drag_start_preview = (float(event.x), float(event.y))
            self._pending_waypoint_yaw_radians = 0.0
            self._waypoint_drag_active = False
            self._draw_map_preview()
            return
        if not self._rx_antenna_map_pick_mode_enabled:
            return
        world_position = self._preview_pixel_to_world(preview_x=float(event.x), preview_y=float(event.y))
        if world_position is None:
            return
        self._set_rx_antenna_position(x=world_position[0], y=world_position[1])
        self._set_rx_antenna_map_pick_mode(False)
        self._append_validation(
            f"✅ RX-Antenne auf Karte gesetzt: x={world_position[0]:.3f}, y={world_position[1]:.3f}"
        )

    def _on_map_canvas_drag(self, event: tk.Event) -> None:
        if getattr(self, "_nav2point_map_pick_mode_enabled", False):
            if self._pending_nav2point_world_position is None or self._nav2point_drag_start_preview is None:
                return
            start_x, start_y = self._nav2point_drag_start_preview
            delta_x = float(event.x) - start_x
            delta_y = float(event.y) - start_y
            if abs(delta_x) < 2.0 and abs(delta_y) < 2.0:
                return
            self._nav2point_drag_active = True
            self._pending_nav2point_yaw_radians = math.atan2(-delta_y, delta_x)
            self._draw_map_preview()
            return
        if not self._waypoint_map_pick_mode_enabled:
            return
        if self._pending_waypoint_world_position is None or self._waypoint_drag_start_preview is None:
            return
        start_x, start_y = self._waypoint_drag_start_preview
        delta_x = float(event.x) - start_x
        delta_y = float(event.y) - start_y
        if abs(delta_x) < 2.0 and abs(delta_y) < 2.0:
            return
        self._waypoint_drag_active = True
        self._pending_waypoint_yaw_radians = math.atan2(-delta_y, delta_x)
        self._draw_map_preview()

    def _on_map_canvas_release(self, _event: tk.Event) -> None:
        if getattr(self, "_nav2point_map_pick_mode_enabled", False):
            world_position = self._pending_nav2point_world_position
            if world_position is None:
                return
            yaw_radians = self._pending_nav2point_yaw_radians if self._nav2point_drag_active else 0.0
            self._set_nav2point_map_pick_mode(False)
            self._queue_nav2point(world_position=world_position, yaw_radians=yaw_radians)
            return
        if not self._waypoint_map_pick_mode_enabled:
            return
        world_position = self._pending_waypoint_world_position
        if world_position is None:
            return
        yaw_radians = self._pending_waypoint_yaw_radians if self._waypoint_drag_active else 0.0
        self._add_point_from_values(x=world_position[0], y=world_position[1], yaw_internal_radians=yaw_radians)
        self._clear_pending_waypoint_marker()
        self._set_waypoint_map_pick_mode(False)

    def _clear_pending_waypoint_marker(self) -> None:
        self._waypoint_drag_start_preview = None
        self._pending_waypoint_world_position = None
        self._pending_waypoint_yaw_radians = 0.0
        self._waypoint_drag_active = False

    def _clear_pending_nav2point_marker(self) -> None:
        self._pending_nav2point_world_position = None
        self._pending_nav2point_yaw_radians = 0.0
        self._nav2point_drag_start_preview = None
        self._nav2point_drag_active = False

    def _draw_pending_waypoint_marker(self) -> None:
        world_position = self._pending_waypoint_world_position
        original = self._map_image_original
        if world_position is None or original is None:
            return
        map_pixel = self._world_to_map_pixel(x=world_position[0], y=world_position[1], image_height=original.height())
        if map_pixel is None:
            return
        if not self._is_pixel_inside_map(map_pixel[0], map_pixel[1], width=original.width(), height=original.height()):
            return
        scale_x, scale_y = self._map_preview_scale
        offset_x, offset_y = self._map_preview_offset
        px = map_pixel[0] * scale_x + offset_x
        py = map_pixel[1] * scale_y + offset_y
        marker_points = self._build_waypoint_arrow_polygon(
            center_x=px,
            center_y=py,
            yaw_radians=float(self._pending_waypoint_yaw_radians),
            arrow_length=10.0,
            tail_length=4.0,
            tail_width=8.0,
        )
        self.map_preview_canvas.create_polygon(
            marker_points,
            fill="#7ee6a8",
            outline="#0d1016",
            width=1,
            dash=(3, 2),
        )

    def _draw_pending_nav2point_marker(self) -> None:
        world_position = self._pending_nav2point_world_position
        original = self._map_image_original
        if world_position is None or original is None:
            return
        map_pixel = self._world_to_map_pixel(x=world_position[0], y=world_position[1], image_height=original.height())
        if map_pixel is None:
            return
        if not self._is_pixel_inside_map(map_pixel[0], map_pixel[1], width=original.width(), height=original.height()):
            return
        scale_x, scale_y = self._map_preview_scale
        offset_x, offset_y = self._map_preview_offset
        px = map_pixel[0] * scale_x + offset_x
        py = map_pixel[1] * scale_y + offset_y
        marker_points = self._build_waypoint_arrow_polygon(
            center_x=px,
            center_y=py,
            yaw_radians=float(self._pending_nav2point_yaw_radians),
            arrow_length=10.0,
            tail_length=4.0,
            tail_width=8.0,
        )
        self.map_preview_canvas.create_polygon(
            marker_points,
            fill="#ffd166",
            outline="#3a2c00",
            width=1,
            dash=(3, 2),
        )

    def _set_rx_antenna_map_pick_mode(self, enabled: bool) -> None:
        self._rx_antenna_map_pick_mode_enabled = enabled
        if enabled:
            self._set_waypoint_map_pick_mode(False)
            self._set_measurement_map_pick_mode(False)
            self._set_nav2point_map_pick_mode(False)
        button_text = "✕" if enabled else "🖱️"
        self.rx_antenna_map_pick_mode_btn.configure(text=button_text)
        self._update_map_canvas_cursor()

    def _toggle_rx_antenna_map_pick_mode(self) -> None:
        self._set_rx_antenna_map_pick_mode(not self._rx_antenna_map_pick_mode_enabled)

    def _set_waypoint_map_pick_mode(self, enabled: bool) -> None:
        self._waypoint_map_pick_mode_enabled = enabled
        if enabled:
            self._rx_antenna_map_pick_mode_enabled = False
            self.rx_antenna_map_pick_mode_btn.configure(text="🖱️")
            self._measurement_map_pick_mode_enabled = False
            self._measurement_start_world_position = None
            self._measurement_end_world_position = None
            measurement_button = getattr(self, "measurement_map_pick_mode_btn", None)
            if measurement_button is not None:
                measurement_button.configure(text="measurement")
            self._nav2point_map_pick_mode_enabled = False
            nav2point_button = getattr(self, "nav2point_map_pick_mode_btn", None)
            if nav2point_button is not None:
                nav2point_button.configure(text="nav2point")
        else:
            self._clear_pending_waypoint_marker()
        self.waypoint_map_pick_mode_btn.configure(text="✕" if enabled else "🖱️")
        self._update_map_canvas_cursor()
        self._draw_map_preview()

    def _toggle_waypoint_map_pick_mode(self) -> None:
        self._set_waypoint_map_pick_mode(not self._waypoint_map_pick_mode_enabled)

    def _set_measurement_map_pick_mode(self, enabled: bool) -> None:
        self._measurement_map_pick_mode_enabled = enabled
        if enabled:
            self._set_waypoint_map_pick_mode(False)
            self._set_rx_antenna_map_pick_mode(False)
            self._set_nav2point_map_pick_mode(False)
        else:
            self._measurement_start_world_position = None
            self._measurement_end_world_position = None
        measurement_button = getattr(self, "measurement_map_pick_mode_btn", None)
        if measurement_button is not None:
            measurement_button.configure(text="✕" if enabled else "measurement")
        self._update_map_canvas_cursor()
        self._draw_map_preview()

    def _toggle_measurement_map_pick_mode(self) -> None:
        self._set_measurement_map_pick_mode(not self._measurement_map_pick_mode_enabled)

    def _set_nav2point_map_pick_mode(self, enabled: bool) -> None:
        self._nav2point_map_pick_mode_enabled = enabled
        if enabled:
            self._set_waypoint_map_pick_mode(False)
            self._set_rx_antenna_map_pick_mode(False)
            self._set_measurement_map_pick_mode(False)
        else:
            self._clear_pending_nav2point_marker()
        nav2point_button = getattr(self, "nav2point_map_pick_mode_btn", None)
        if nav2point_button is not None:
            nav2point_button.configure(text="✕" if enabled else "nav2point")
        self._update_map_canvas_cursor()
        self._draw_map_preview()

    def _toggle_nav2point_map_pick_mode(self) -> None:
        self._set_nav2point_map_pick_mode(not self._nav2point_map_pick_mode_enabled)

    @staticmethod
    def _navigation_point_from_world_position(
        world_position: tuple[float, float], *, yaw_radians: float = 0.0
    ) -> NavigationPoint:
        half_yaw = float(yaw_radians) / 2.0
        return NavigationPoint(
            x=float(world_position[0]),
            y=float(world_position[1]),
            qz=math.sin(half_yaw),
            qw=math.cos(half_yaw),
        )

    def _queue_nav2point(self, *, world_position: tuple[float, float], yaw_radians: float = 0.0) -> None:
        if self._run_thread and self._run_thread.is_alive():
            self._append_validation("⚠️ nav2point ist während eines aktiven Runs deaktiviert.")
            return
        if self._nav2point_thread and self._nav2point_thread.is_alive():
            self._append_validation("⚠️ nav2point läuft bereits. Bitte aktuellen Zielpunkt zuerst stoppen oder abwarten.")
            return
        target = self._navigation_point_from_world_position(world_position, yaw_radians=yaw_radians)

        def _worker() -> None:
            state: TerminalNavigationState = "aborted"
            try:
                navigator = self._ensure_navigator()
                state = navigator.navigate_to_point(
                    target,
                    timeout_s=float(self._runtime_config.goal_reached_timeout_s),
                )
            finally:
                self.after(0, self._refresh_stop_button_state)
            if state == "succeeded":
                self.after(
                    0,
                    lambda: self._append_validation(
                        f"✅ nav2point erreicht: x={target.x:.3f}, y={target.y:.3f}"
                    ),
                )
                return
            self.after(
                0,
                lambda: self._append_validation(
                    f"⚠️ nav2point fehlgeschlagen ({state}): x={target.x:.3f}, y={target.y:.3f}"
                ),
            )

        self._nav2point_thread = threading.Thread(target=_worker, daemon=True)
        self._nav2point_thread.start()
        self._refresh_stop_button_state()

    @staticmethod
    def _measurement_distance_m(
        start_world_position: tuple[float, float] | None,
        end_world_position: tuple[float, float] | None,
    ) -> float | None:
        if start_world_position is None or end_world_position is None:
            return None
        return math.hypot(end_world_position[0] - start_world_position[0], end_world_position[1] - start_world_position[1])

    def _draw_measurement_overlay(self) -> None:
        original = self._map_image_original
        if original is None:
            return
        start = self._measurement_start_world_position
        end = self._measurement_end_world_position
        if start is None or end is None:
            return

        start_pixel = self._world_to_map_pixel(x=start[0], y=start[1], image_height=original.height())
        end_pixel = self._world_to_map_pixel(x=end[0], y=end[1], image_height=original.height())
        if start_pixel is None or end_pixel is None:
            return

        if not self._is_pixel_inside_map(start_pixel[0], start_pixel[1], width=original.width(), height=original.height()):
            return
        if not self._is_pixel_inside_map(end_pixel[0], end_pixel[1], width=original.width(), height=original.height()):
            return

        scale_x, scale_y = self._map_preview_scale
        offset_x, offset_y = self._map_preview_offset
        start_preview = (start_pixel[0] * scale_x + offset_x, start_pixel[1] * scale_y + offset_y)
        end_preview = (end_pixel[0] * scale_x + offset_x, end_pixel[1] * scale_y + offset_y)

        self.map_preview_canvas.create_line(
            start_preview[0],
            start_preview[1],
            end_preview[0],
            end_preview[1],
            fill="#ffcc66",
            width=2,
            dash=(4, 2),
        )
        for px, py in (start_preview, end_preview):
            self.map_preview_canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#ffcc66", outline="#231f16", width=1)

        distance_m = self._measurement_distance_m(start, end)
        if distance_m is None:
            return
        label_x = (start_preview[0] + end_preview[0]) / 2.0
        label_y = (start_preview[1] + end_preview[1]) / 2.0 - 10.0
        self.map_preview_canvas.create_text(
            label_x,
            label_y,
            text=f"{distance_m:.2f} m",
            fill="#ffde9a",
            font=("TkDefaultFont", 10, "bold"),
        )

    def _update_map_canvas_cursor(self) -> None:
        pick_mode_active = (
            self._rx_antenna_map_pick_mode_enabled
            or self._waypoint_map_pick_mode_enabled
            or getattr(self, "_measurement_map_pick_mode_enabled", False)
            or getattr(self, "_nav2point_map_pick_mode_enabled", False)
        )
        self.map_preview_canvas.configure(cursor="crosshair" if pick_mode_active else "")

    @staticmethod
    def _resize_photo_to_contain(photo: tk.PhotoImage, *, target_width: int, target_height: int) -> tk.PhotoImage:
        if target_width <= 1 or target_height <= 1:
            return photo

        width = photo.width()
        height = photo.height()
        if width <= 0 or height <= 0:
            return photo

        # Keep the whole map visible in the available canvas area while preserving
        # aspect ratio. The image may be scaled above 100% which can look pixelated,
        # but prevents clipping/cropping.
        scale = min(target_width / width, target_height / height)
        if abs(scale - 1.0) < 0.01:
            return photo

        ratio = Fraction(scale).limit_denominator(32)
        preview = photo.zoom(ratio.numerator, ratio.numerator).subsample(ratio.denominator, ratio.denominator)
        return preview

    def _select_map_config_file(self) -> None:
        selected_file = filedialog.askopenfilename(
            title="Map-Config auswählen",
            parent=self,
            filetypes=[
                ("Map-Dateien", "*.yaml *.yml *.json"),
                ("YAML", "*.yaml *.yml"),
                ("JSON", "*.json"),
                ("Alle Dateien", "*.*"),
            ],
        )
        if not selected_file:
            return

        map_config_path = Path(selected_file).expanduser().resolve()
        try:
            map_config = self._load_map_config_from_file(map_config_path)
        except Exception as exc:
            messagebox.showwarning("Map-Config ungültig", str(exc), parent=self)
            return

        self._selected_map_config = map_config
        self._selected_map_config_file = str(map_config_path)
        if self._mission is not None:
            self._mission = replace(self._mission, map_config=map_config)
        self._persist_workflow_state()
        self._refresh_map_section()
        self._append_validation(f"✅ Map-Config geladen: {map_config_path.name}")

    def _load_map_config_from_file(self, map_config_path: Path) -> MapConfig:
        parsed_mission = measurement_mission_from_dict(
            {
                "name": "map-config-check",
                "points": [{"id": "map-check", "x": 0.0, "y": 0.0, "yaw": 0.0}],
                "map_config": str(map_config_path),
            }
        )
        map_config = parsed_mission.map_config
        if map_config is None:
            raise ValueError("Map-Config konnte nicht gelesen werden.")

        image_path = Path(map_config.image).expanduser()
        if not image_path.is_absolute():
            image_path = (map_config_path.parent / image_path).resolve()
        return replace(map_config, image=str(image_path))

    def _render_map_placeholder(self, text: str) -> None:
        self.map_preview_canvas.delete("all")
        self._live_overlay_item_ids = {
            "echo_slots": {},
            "marker": None,
            "heading": None,
            "position_info_box": None,
            "position_info_text": None,
        }
        self._static_map_layer_signature = None
        self._invalidate_live_echo_geometry_cache()
        self.map_preview_canvas.create_text(
            20,
            20,
            text=text,
            fill="#e6e6e6",
            anchor="nw",
            justify="left",
        )

    def _draw_map_preview(self) -> None:
        original = self._map_image_original
        if original is None:
            return
        if not self.winfo_exists() or not self.map_preview_canvas.winfo_exists():
            return
        try:
            canvas_width = max(1, self.map_preview_canvas.winfo_width())
            canvas_height = max(1, self.map_preview_canvas.winfo_height())
        except tk.TclError:
            return
        static_signature = self._build_static_map_layer_signature(canvas_width=canvas_width, canvas_height=canvas_height)
        if static_signature != self._static_map_layer_signature:
            self._draw_static_map_layer(canvas_width=canvas_width, canvas_height=canvas_height)
            self._static_map_layer_signature = static_signature
        self._draw_live_overlay_layer()
        self._last_live_redraw_ts = time.time()

    def _build_static_map_layer_signature(self, *, canvas_width: int, canvas_height: int) -> tuple[Any, ...]:
        mission_points_signature = tuple((point.id, point.x, point.y, point.yaw) for point in self._mission_points)
        return (
            canvas_width,
            canvas_height,
            id(self._map_image_original),
            mission_points_signature,
            self._selected_point_index,
            self._selected_result_index,
            self._rx_antenna_global_position,
            self._measurement_start_world_position,
            self._measurement_end_world_position,
            self._pending_nav2point_world_position,
            self._pending_nav2point_yaw_radians,
            self._pending_waypoint_world_position,
            self._pending_waypoint_yaw_radians,
        )

    def _draw_static_map_layer(self, *, canvas_width: int, canvas_height: int) -> None:
        original = self._map_image_original
        if original is None:
            return
        preview = self._resize_photo_to_contain(original, target_width=canvas_width, target_height=canvas_height)
        offset_x = (canvas_width - preview.width()) / 2.0
        offset_y = (canvas_height - preview.height()) / 2.0

        self._map_image_preview = preview
        self._map_preview_scale = (preview.width() / original.width(), preview.height() / original.height())
        self._map_preview_offset = (offset_x, offset_y)

        self.map_preview_canvas.delete("all")
        self._live_overlay_item_ids = {
            "echo_slots": {},
            "marker": None,
            "heading": None,
            "position_info_box": None,
            "position_info_text": None,
        }
        self._invalidate_live_echo_geometry_cache()
        self._map_canvas_image_id = self.map_preview_canvas.create_image(offset_x, offset_y, anchor="nw", image=preview)
        self._draw_mission_markers()
        self._draw_pending_nav2point_marker()
        self._draw_pending_waypoint_marker()
        self._draw_rx_antenna_marker()
        self._draw_measurement_overlay()
        self._draw_selected_echo_overlay()
        self._draw_selected_lidar_reference_overlay()

    def _draw_live_overlay_layer(self) -> None:
        self._draw_live_echo_preview_overlay()
        self._draw_live_marker()
        self._draw_live_position_info_overlay()

    def _clear_live_overlay_layer(
        self,
        *,
        components: tuple[str, ...] = ("echo_slots", "marker", "heading", "position_info_box", "position_info_text"),
    ) -> None:
        if not self._live_overlay_item_ids:
            return
        if "echo_slots" in components:
            echo_slots = self._live_overlay_item_ids.get("echo_slots")
            if isinstance(echo_slots, dict):
                for slot_name, item_id in list(echo_slots.items()):
                    if not isinstance(item_id, int):
                        continue
                    try:
                        self.map_preview_canvas.delete(item_id)
                    except tk.TclError:
                        pass
                    echo_slots.pop(slot_name, None)
            self._live_overlay_item_ids["echo_slots"] = {}
            self._invalidate_live_echo_geometry_cache()
        for key in ("marker", "heading", "position_info_box", "position_info_text"):
            if key not in components:
                continue
            item_id = self._live_overlay_item_ids.get(key)
            if not isinstance(item_id, int):
                self._live_overlay_item_ids[key] = None
                continue
            try:
                self.map_preview_canvas.delete(item_id)
            except tk.TclError:
                pass
            self._live_overlay_item_ids[key] = None

    def _draw_rx_antenna_marker(self) -> None:
        position = self._rx_antenna_global_position
        original = self._map_image_original
        if position is None or original is None:
            return
        map_pixel = self._world_to_map_pixel(x=position[0], y=position[1], image_height=original.height())
        if map_pixel is None:
            return
        if not self._is_pixel_inside_map(map_pixel[0], map_pixel[1], width=original.width(), height=original.height()):
            return
        scale_x, scale_y = self._map_preview_scale
        offset_x, offset_y = self._map_preview_offset
        px = map_pixel[0] * scale_x + offset_x
        py = map_pixel[1] * scale_y + offset_y
        cross_size = 7
        self.map_preview_canvas.create_line(px - cross_size, py, px + cross_size, py, fill="#42a5f5", width=2)
        self.map_preview_canvas.create_line(px, py - cross_size, px, py + cross_size, fill="#42a5f5", width=2)
        self.map_preview_canvas.create_oval(
            px - 3,
            py - 3,
            px + 3,
            py + 3,
            fill="#90caf9",
            outline="#1565c0",
            width=1,
        )

    def _draw_mission_markers(self) -> None:
        self._map_marker_ids = []
        mission = self._mission
        preview = self._map_image_preview
        original = self._map_image_original
        if mission is None or mission.map_config is None or preview is None or original is None:
            return

        scale_x, scale_y = self._map_preview_scale
        offset_x, offset_y = self._map_preview_offset

        for index, point in enumerate(self._mission_points):
            pixel_coordinates = self._world_to_preview_pixel(
                x=point.x,
                y=point.y,
                image_height=original.height(),
                scale_x=scale_x,
                scale_y=scale_y,
            )
            if pixel_coordinates is None:
                continue
            px, py = pixel_coordinates
            px += offset_x
            py += offset_y
            marker_points = self._build_waypoint_arrow_polygon(
                center_x=px,
                center_y=py,
                yaw_radians=float(point.yaw),
                arrow_length=10.0,
                tail_length=4.0,
                tail_width=8.0,
            )
            marker_id = self.map_preview_canvas.create_polygon(
                marker_points,
                fill="#00d26a",
                outline="#0d1016",
                width=1,
            )
            self._map_marker_ids.append(marker_id)
            if index == self._selected_point_index:
                self._highlight_marker(marker_id)

    @staticmethod
    def _build_waypoint_arrow_polygon(
        *,
        center_x: float,
        center_y: float,
        yaw_radians: float,
        arrow_length: float,
        tail_length: float,
        tail_width: float,
    ) -> tuple[float, float, float, float, float, float]:
        heading_x = math.cos(yaw_radians)
        heading_y = -math.sin(yaw_radians)
        perpendicular_x = math.sin(yaw_radians)
        perpendicular_y = math.cos(yaw_radians)

        tip_x = center_x + heading_x * arrow_length
        tip_y = center_y + heading_y * arrow_length
        tail_center_x = center_x - heading_x * tail_length
        tail_center_y = center_y - heading_y * tail_length
        half_width = tail_width / 2.0
        left_x = tail_center_x + perpendicular_x * half_width
        left_y = tail_center_y + perpendicular_y * half_width
        right_x = tail_center_x - perpendicular_x * half_width
        right_y = tail_center_y - perpendicular_y * half_width
        return (tip_x, tip_y, left_x, left_y, right_x, right_y)

    def _world_to_preview_pixel(
        self,
        *,
        x: float,
        y: float,
        image_height: int,
        scale_x: float,
        scale_y: float,
    ) -> tuple[float, float] | None:
        mission = self._mission
        if mission is None or mission.map_config is None:
            return None

        origin_x, origin_y, _origin_yaw = mission.map_config.origin
        resolution = mission.map_config.resolution
        if resolution <= 0:
            return None

        map_pixel_x = (x - origin_x) / resolution
        map_pixel_y = float(image_height) - ((y - origin_y) / resolution)
        return (map_pixel_x * scale_x, map_pixel_y * scale_y)

    def _world_to_map_pixel(self, *, x: float, y: float, image_height: int) -> tuple[float, float] | None:
        mission = self._mission
        if mission is None or mission.map_config is None:
            return None
        resolution = mission.map_config.resolution
        if resolution <= 0:
            return None
        origin_x, origin_y, _origin_yaw = mission.map_config.origin
        map_pixel_x = (x - origin_x) / resolution
        map_pixel_y = float(image_height) - ((y - origin_y) / resolution)
        return (map_pixel_x, map_pixel_y)

    def _preview_pixel_to_world(self, *, preview_x: float, preview_y: float) -> tuple[float, float] | None:
        mission = getattr(self, "_mission", None)
        original = getattr(self, "_map_image_original", None)
        if mission is None or mission.map_config is None or original is None:
            return None
        scale_x, scale_y = self._map_preview_scale
        if scale_x <= 0.0 or scale_y <= 0.0:
            return None
        offset_x, offset_y = self._map_preview_offset
        map_pixel_x = (preview_x - offset_x) / scale_x
        map_pixel_y = (preview_y - offset_y) / scale_y
        if not self._is_pixel_inside_map(map_pixel_x, map_pixel_y, width=original.width(), height=original.height()):
            return None
        resolution = mission.map_config.resolution
        if resolution <= 0.0:
            return None
        origin_x, origin_y, _origin_yaw = mission.map_config.origin
        world_x = origin_x + map_pixel_x * resolution
        world_y = origin_y + (float(original.height()) - map_pixel_y) * resolution
        return (world_x, world_y)

    @staticmethod
    def _is_pixel_inside_map(pixel_x: float, pixel_y: float, *, width: int, height: int) -> bool:
        return 0.0 <= pixel_x <= float(width) and 0.0 <= pixel_y <= float(height)

    def _expected_live_frame_id(self) -> str:
        mission = self._mission
        if mission is None or mission.map_config is None:
            return "map"
        frame_id = mission.map_config.frame_id
        if isinstance(frame_id, str) and frame_id.strip():
            return frame_id
        return "map"

    def _live_pose_diagnosis(self) -> tuple[str, str]:
        stale_threshold_s = 1.8
        if not bool(self.live_pose_stream_enabled_var.get()):
            return ("stream_disabled", "Live-Pose deaktiviert")
        if self._map_image_size is None:
            return ("map_not_loaded", "Karte nicht geladen")

        position = self._live_position
        if not isinstance(position, dict):
            if self._live_position_received_at is None:
                return ("no_position_never_received", "Live-Pose noch nie empfangen")
            return ("position_temporarily_unavailable", "Live-Pose temporär ausgefallen")

        if self._live_position_received_at is not None:
            pose_age_s = time.time() - self._live_position_received_at
            if pose_age_s > stale_threshold_s:
                return ("stale_position", "Live-Pose veraltet")

        expected_frame_id = self._expected_live_frame_id()
        received_frame_id = position.get("frame_id")
        if (
            isinstance(received_frame_id, str)
            and received_frame_id.strip()
            and received_frame_id != expected_frame_id
        ):
            return (
                "frame_mismatch",
                "Frame-Mismatch: "
                f"erwarteter Frame={expected_frame_id}, empfangen={received_frame_id}",
            )

        x_value = position.get("x")
        y_value = position.get("y")
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            return ("invalid_numeric_values", "Ungültige numerische Werte in Live-Pose")

        parsed_x = float(x_value)
        parsed_y = float(y_value)
        if not math.isfinite(parsed_x) or not math.isfinite(parsed_y):
            return ("invalid_numeric_values", "Ungültige numerische Werte in Live-Pose")

        width, height = self._map_image_size
        map_pixel = self._world_to_map_pixel(
            x=parsed_x,
            y=parsed_y,
            image_height=height,
        )
        if map_pixel is None:
            return ("map_not_loaded", "Karte nicht geladen")
        if not self._is_pixel_inside_map(map_pixel[0], map_pixel[1], width=width, height=height):
            return ("outside_map", "Koordinaten außerhalb Kartenbereich")
        return ("ok", "Live-Pose verfügbar")

    def _announce_live_diagnosis_if_changed(self, diagnosis_key: str, diagnosis_text: str) -> None:
        if diagnosis_key == self._last_live_diagnosis_key:
            return
        self._last_live_diagnosis_key = diagnosis_key
        if not self._emit_live_diagnostics_to_validation:
            return
        self._append_validation(f"ℹ️ Live-Diagnose geändert: {diagnosis_text}")

    def _on_points_table_select(self, _event: tk.Event) -> None:
        selected = self.points_table.selection()
        if not selected:
            self._selected_point_index = None
            self._draw_map_preview()
            return
        selected_index = self.points_table.index(selected[0])
        self._selected_point_index = selected_index if selected_index >= 0 else None
        self._draw_map_preview()


    def _on_results_table_click(self, event: tk.Event) -> str | None:
        row_id = self.results_table.identify_row(event.y)
        identify_column = getattr(self.results_table, "identify_column", None)
        column_id = identify_column(event.x) if callable(identify_column) else ""
        review_column_id = "#10"
        try:
            columns = tuple(self.results_table.cget("columns"))
        except Exception:
            columns = ()
        if columns:
            try:
                review_column_id = f"#{columns.index('review_action') + 1}"
            except ValueError:
                pass
        if row_id and column_id == review_column_id:
            row_index = self.results_table.index(row_id)
            self._open_review_for_result_row(row_index)
            return "break"
        if row_id:
            return None
        region = self.results_table.identify("region", event.x, event.y)
        if region in {"heading", "separator"}:
            return None
        self._update_results_selection_diagnostics()
        return "break"

    def _open_review_for_result_row(self, row_index: int) -> None:
        if row_index < 0 or row_index >= len(self._records):
            return
        record = self._records[row_index]
        if not isinstance(record, dict):
            return
        measurement = record.get("measurement")
        if not isinstance(measurement, dict):
            return
        result_payload = measurement.get("result")
        if not isinstance(result_payload, dict):
            return
        output_file = result_payload.get("output_file")
        if (not isinstance(output_file, str) or not output_file.strip()) and isinstance(result_payload.get("file_ref"), str):
            output_file = result_payload.get("file_ref")
        if not isinstance(output_file, str) or not output_file.strip():
            rx_payload = result_payload.get("rx")
            if isinstance(rx_payload, dict):
                output_file = rx_payload.get("output_file")
        if not isinstance(output_file, str) or not output_file.strip():
            self._append_validation("⚠️ Review kann nicht geöffnet werden: output_file fehlt.")
            return
        review_fn = getattr(self.master, "review_measurement_for_mission", None)
        if not callable(review_fn):
            self._append_validation("⚠️ Review kann nicht geöffnet werden: Review-Funktion nicht verfügbar.")
            return
        point_label = f"Punktindex {self._format_one_based_index(record.get('global_index'))}"
        review_prefill = self._build_review_prefill_from_result(result_payload)
        try:
            review_outcome = review_fn(
                point_label=point_label,
                output_file=output_file,
                initial_review=review_prefill,
            )
        except Exception as exc:
            self._append_validation(f"⚠️ Review konnte nicht geöffnet werden: {exc}")
            return
        if not isinstance(review_outcome, dict):
            return
        if not bool(review_outcome.get("approved")):
            self._append_validation("ℹ️ Review nicht freigegeben; Messresultat bleibt unverändert.")
            return

        normalized_echo_delays = self._normalize_review_echo_delays(review_outcome)
        if normalized_echo_delays:
            review_outcome["echo_delays"] = normalized_echo_delays

        for key in (
            "manual_lags",
            "los_idx",
            "echo_indices",
            "los_lag",
            "echo_lags",
            "echo_delays",
            "interpolation_enabled",
            "interpolation_factor",
        ):
            if key in review_outcome:
                result_payload[key] = review_outcome.get(key)

        review_payload = result_payload.get("review")
        if not isinstance(review_payload, dict):
            review_payload = {}
            result_payload["review"] = review_payload
        for key in (
            "manual_lags",
            "los_idx",
            "echo_indices",
            "los_lag",
            "echo_lags",
            "echo_delays",
            "interpolation_enabled",
            "interpolation_factor",
        ):
            if key in review_outcome:
                review_payload[key] = review_outcome.get(key)

        self._persist_workflow_state()
        if 0 <= row_index < len(self.results_table.get_children()):
            error_text = record.get("error") or ""
            combined_status = self._compose_table_outcome(record, error_text)
            self.results_table.item(
                self.results_table.get_children()[row_index],
                values=(
                    self._format_one_based_index(record.get("global_index")),
                    self._format_one_based_index(record.get("point_index")),
                    self._format_live_position_for_table(record),
                    self._format_live_distance_to_rx_for_table(record),
                    *self._format_echo_distances_for_table(result_payload.get("echo_delays")),
                    "Review",
                    combined_status,
                ),
            )
        self._update_results_selection_diagnostics()
        self._draw_map_preview()

    @staticmethod
    def _normalize_review_echo_delays(review_outcome: dict[str, Any]) -> list[dict[str, Any]]:
        raw_echo_delays = review_outcome.get("echo_delays")
        has_structured_echo_delays = isinstance(raw_echo_delays, list) and any(
            isinstance(entry, dict) for entry in raw_echo_delays
        )
        return _coerce_echo_delay_entries(
            echo_delays=raw_echo_delays,
            echo_indices=[] if has_structured_echo_delays else review_outcome.get("echo_indices"),
            echo_lags=[] if has_structured_echo_delays else review_outcome.get("echo_lags"),
            los_lag=review_outcome.get("los_lag"),
            interpolation_enabled=review_outcome.get("interpolation_enabled"),
            interpolation_factor=review_outcome.get("interpolation_factor"),
        )

    @staticmethod
    def _build_review_prefill_from_result(result_payload: dict[str, Any]) -> dict[str, Any]:
        prefill: dict[str, Any] = {}
        los_lag = result_payload.get("los_lag")
        if isinstance(los_lag, (int, float)):
            prefill["los_lag"] = int(round(float(los_lag)))
        echo_lags = result_payload.get("echo_lags")
        if isinstance(echo_lags, (list, tuple)):
            parsed_echo_lags = [int(round(float(value))) for value in echo_lags if isinstance(value, (int, float))]
            if parsed_echo_lags:
                prefill["echo_lags"] = parsed_echo_lags
        manual_lags = result_payload.get("manual_lags")
        if isinstance(manual_lags, dict):
            parsed_manual: dict[str, int] = {}
            for key in ("los", "echo"):
                value = manual_lags.get(key)
                if isinstance(value, (int, float)):
                    parsed_manual[key] = int(round(float(value)))
            if parsed_manual:
                prefill["manual_lags"] = parsed_manual
        interpolation_enabled = result_payload.get("interpolation_enabled")
        if interpolation_enabled is None:
            review_payload = result_payload.get("review")
            if isinstance(review_payload, dict):
                interpolation_enabled = review_payload.get("interpolation_enabled")
        if interpolation_enabled is not None:
            prefill["interpolation_enabled"] = bool(interpolation_enabled)

        interpolation_factor = result_payload.get("interpolation_factor")
        if interpolation_factor is None:
            review_payload = result_payload.get("review")
            if isinstance(review_payload, dict):
                interpolation_factor = review_payload.get("interpolation_factor")
        if interpolation_factor is not None:
            prefill["interpolation_factor"] = interpolation_factor
        return prefill

    def _on_results_table_select(self, _event: tk.Event) -> None:
        selected = self.results_table.selection()
        if not selected:
            self._selected_result_index = None
            self._selected_result_indices = ()
            self._update_results_selection_diagnostics()
            self._draw_map_preview()
            return
        selected_indices = tuple(
            sorted(
                idx
                for idx in (self.results_table.index(item_id) for item_id in selected)
                if idx >= 0
            )
        )
        self._selected_result_indices = selected_indices
        self._selected_result_index = selected_indices[0] if selected_indices else None
        self._update_results_selection_diagnostics()
        self._draw_map_preview()

    def _update_results_selection_diagnostics(self) -> None:
        diagnostics_var = getattr(self, "results_selection_diagnostics_var", None)
        if diagnostics_var is None:
            return
        selected_count = len(getattr(self, "_selected_result_indices", ()))
        diagnostics_var.set(f"Auswahl: {selected_count} Zeilen")

    def _invalidate_live_echo_geometry_cache(self) -> None:
        self._live_echo_geometry_cache = {}

    @staticmethod
    def _live_echo_sampling_levels(*, reduced: bool) -> tuple[int, int, int]:
        return LIVE_ECHO_SAMPLING_REDUCED if reduced else LIVE_ECHO_SAMPLING_NORMAL

    def _should_reduce_live_echo_sampling(self) -> bool:
        last_redraw_ts = self._last_live_redraw_ts
        if last_redraw_ts is None:
            return False
        target_period_s = 1.0 / max(1, LIVE_PREVIEW_TARGET_FPS)
        redraw_gap_s = time.time() - last_redraw_ts
        return redraw_gap_s < (target_period_s * 0.8)

    def _build_live_echo_cache_key(
        self,
        *,
        rx_position: tuple[float, float],
        measurement_position: tuple[float, float],
        echo_distance_m: float,
        resolution: float,
        image_height: int,
    ) -> tuple[float, ...]:
        scale_x, scale_y = self._map_preview_scale
        offset_x, offset_y = self._map_preview_offset
        return (
            float(rx_position[0]),
            float(rx_position[1]),
            float(measurement_position[0]),
            float(measurement_position[1]),
            float(echo_distance_m),
            float(scale_x),
            float(scale_y),
            float(offset_x),
            float(offset_y),
            float(resolution),
            float(image_height),
        )

    def _can_reuse_live_echo_cache(self, previous_key: tuple[float, ...], new_key: tuple[float, ...]) -> bool:
        if len(previous_key) != len(new_key) or len(new_key) != 11:
            return False
        if previous_key[5:] != new_key[5:]:
            return False
        position_deltas = (
            abs(previous_key[0] - new_key[0]),
            abs(previous_key[1] - new_key[1]),
            abs(previous_key[2] - new_key[2]),
            abs(previous_key[3] - new_key[3]),
        )
        if any(delta > LIVE_ECHO_CACHE_POSITION_DELTA_M for delta in position_deltas):
            return False
        if abs(previous_key[4] - new_key[4]) > LIVE_ECHO_CACHE_DISTANCE_DELTA_M:
            return False
        return True

    def _build_live_echo_overlay_preview_points_cached(
        self,
        *,
        slot_name: str,
        rx_position: tuple[float, float],
        measurement_position: tuple[float, float],
        echo_distance_m: float,
        reduced_sampling: bool,
    ) -> tuple[list[float] | None, int]:
        mission = getattr(self, "_mission", None)
        original = getattr(self, "_map_image_original", None)
        if mission is None or mission.map_config is None or original is None:
            return (None, 1)
        resolution = mission.map_config.resolution
        if not math.isfinite(resolution) or resolution <= 0.0:
            return (None, 1)
        cache_key = self._build_live_echo_cache_key(
            rx_position=rx_position,
            measurement_position=measurement_position,
            echo_distance_m=echo_distance_m,
            resolution=float(resolution),
            image_height=original.height(),
        )
        cached = self._live_echo_geometry_cache.get(slot_name)
        if isinstance(cached, dict):
            previous_key = cached.get("key")
            previous_points = cached.get("points")
            previous_line_width = cached.get("line_width")
            if (
                isinstance(previous_key, tuple)
                and isinstance(previous_points, list)
                and isinstance(previous_line_width, int)
                and self._can_reuse_live_echo_cache(previous_key, cache_key)
            ):
                return (list(previous_points), previous_line_width)
        preview_points, line_width = self._build_echo_overlay_preview_points(
            rx_position=rx_position,
            measurement_position=measurement_position,
            echo_distance_m=echo_distance_m,
            sample_levels=self._live_echo_sampling_levels(reduced=reduced_sampling),
        )
        if preview_points is None:
            self._live_echo_geometry_cache.pop(slot_name, None)
            return (None, line_width)
        self._live_echo_geometry_cache[slot_name] = {
            "key": cache_key,
            "points": list(preview_points),
            "line_width": int(line_width),
        }
        return (preview_points, line_width)

    def _draw_selected_echo_overlay(self) -> None:
        rx_position = self._rx_antenna_global_position
        if rx_position is None:
            return
        selected_records = self._selected_record_payloads()
        if len(selected_records) > 1 and self._draw_selected_echo_probability_overlay(
            rx_position=rx_position,
            records=selected_records,
        ):
            return
        for record in selected_records:
            measurement_position = self._selected_record_measurement_position(record)
            if measurement_position is None:
                continue
            measurement = record.get("measurement")
            if not isinstance(measurement, dict):
                continue
            result = measurement.get("result")
            if not isinstance(result, dict):
                continue
            echo_distances = self._extract_echo_distances(result.get("echo_delays"), limit=len(ECHO_OVERLAY_COLORS))
            if not echo_distances:
                continue
            for echo_index, echo_distance in enumerate(echo_distances):
                color = ECHO_OVERLAY_COLORS[echo_index % len(ECHO_OVERLAY_COLORS)]
                self._draw_echo_ellipse_for_overlay(
                    rx_position=rx_position,
                    measurement_position=measurement_position,
                    echo_distance_m=echo_distance,
                    color=color,
                )

    def _draw_selected_echo_probability_overlay(
        self,
        *,
        rx_position: tuple[float, float],
        records: list[dict[str, Any]],
    ) -> bool:
        mission = getattr(self, "_mission", None)
        original = getattr(self, "_map_image_original", None)
        if mission is None or mission.map_config is None or original is None:
            return False
        resolution = mission.map_config.resolution
        if not math.isfinite(resolution) or resolution <= 0.0:
            return False
        sigma_sq = MULTI_SELECTION_PROBABILITY_SIGMA_M * MULTI_SELECTION_PROBABILITY_SIGMA_M
        if sigma_sq <= 0.0 or not math.isfinite(sigma_sq):
            return False
        rx_x, rx_y = rx_position
        candidates: list[tuple[tuple[float, float], float]] = []
        for record in records:
            measurement_position = self._selected_record_measurement_position(record)
            if measurement_position is None:
                continue
            measurement = record.get("measurement")
            if not isinstance(measurement, dict):
                continue
            result = measurement.get("result")
            if not isinstance(result, dict):
                continue
            echo_distances = self._extract_echo_distances(result.get("echo_delays"), limit=1)
            if not echo_distances:
                continue
            point_x, point_y = measurement_position
            rho_i = math.hypot(point_x - rx_x, point_y - rx_y) + echo_distances[0]
            if not math.isfinite(rho_i) or rho_i <= 0.0:
                continue
            candidates.append((measurement_position, rho_i))
        if not candidates:
            return False
        step_px = max(4, int(MULTI_SELECTION_PROBABILITY_GRID_STEP_PX))
        canvas_width = max(1, self.map_preview_canvas.winfo_width())
        canvas_height = max(1, self.map_preview_canvas.winfo_height())
        values: list[tuple[float, float, float]] = []
        max_value = 0.0
        for py in range(0, canvas_height, step_px):
            for px in range(0, canvas_width, step_px):
                world_pos = self._preview_pixel_to_world(preview_x=px + (step_px / 2.0), preview_y=py + (step_px / 2.0))
                if world_pos is None:
                    continue
                world_x, world_y = world_pos
                value = 0.0
                for (s_x, s_y), rho_i in candidates:
                    residual = math.hypot(world_x - s_x, world_y - s_y) + math.hypot(world_x - rx_x, world_y - rx_y) - rho_i
                    value += math.exp(-((residual * residual) / (2.0 * sigma_sq)))
                if value <= 0.0 or not math.isfinite(value):
                    continue
                max_value = max(max_value, value)
                values.append((float(px), float(py), value))
        if max_value <= 0.0:
            return False
        total_cells = len(values)
        drawn_cells = 0
        for px, py, value in values:
            normalized = value / max_value
            if normalized < MULTI_SELECTION_PROBABILITY_MIN_NORMALIZED:
                continue
            heat = min(1.0, max(MULTI_SELECTION_PROBABILITY_MIN_HEAT, normalized))
            red = int(round(255 * heat))
            green = int(round(180 * (1.0 - heat)))
            blue = int(round(48 * (1.0 - heat)))
            rectangle_kwargs: dict[str, Any] = {
                "fill": f"#{red:02x}{green:02x}{blue:02x}",
                "outline": "",
            }
            if MULTI_SELECTION_PROBABILITY_USE_STIPPLE:
                rectangle_kwargs["stipple"] = "gray50"
            self.map_preview_canvas.create_rectangle(
                px,
                py,
                px + step_px,
                py + step_px,
                **rectangle_kwargs,
            )
            drawn_cells += 1
        if MULTI_SELECTION_PROBABILITY_DEBUG_LOG:
            self._append_validation(
                "ℹ️ Echo-Heatmap: "
                f"{drawn_cells}/{total_cells} Zellen gezeichnet "
                f"(min_norm={MULTI_SELECTION_PROBABILITY_MIN_NORMALIZED:.2f}, "
                f"min_heat={MULTI_SELECTION_PROBABILITY_MIN_HEAT:.2f}, "
                f"stipple={'an' if MULTI_SELECTION_PROBABILITY_USE_STIPPLE else 'aus'})."
            )
        return drawn_cells > 0

    def _draw_live_echo_preview_overlay(self) -> None:
        if not bool(self.live_preview_enabled_var.get()):
            self._clear_live_overlay_layer(components=("echo_slots",))
            return
        rx_position = self._rx_antenna_global_position
        if rx_position is None:
            return
        live_position = self._copy_valid_live_position()
        if live_position is None:
            return
        x_value = live_position.get("x")
        y_value = live_position.get("y")
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            return
        measurement_position = (float(x_value), float(y_value))
        if not math.isfinite(measurement_position[0]) or not math.isfinite(measurement_position[1]):
            return
        echo_distances = self._get_live_preview_echo_distances(limit=len(ECHO_OVERLAY_COLORS))
        if not echo_distances:
            self._clear_live_overlay_layer(components=("echo_slots",))
            return
        echo_slots = self._live_overlay_item_ids.get("echo_slots")
        if not isinstance(echo_slots, dict):
            echo_slots = {}
            self._live_overlay_item_ids["echo_slots"] = echo_slots
        reduced_sampling = self._should_reduce_live_echo_sampling()
        active_slot_names: set[str] = set()
        for echo_index, echo_distance in enumerate(echo_distances):
            slot_name = f"echo_{echo_index}"
            active_slot_names.add(slot_name)
            color = ECHO_OVERLAY_COLORS[echo_index % len(ECHO_OVERLAY_COLORS)]
            preview_points, line_width = self._build_live_echo_overlay_preview_points_cached(
                slot_name=slot_name,
                rx_position=rx_position,
                measurement_position=measurement_position,
                echo_distance_m=echo_distance,
                reduced_sampling=reduced_sampling,
            )
            if preview_points is None:
                existing_item_id = echo_slots.get(slot_name)
                if isinstance(existing_item_id, int):
                    try:
                        self.map_preview_canvas.itemconfigure(existing_item_id, state="hidden")
                    except tk.TclError:
                        try:
                            self.map_preview_canvas.delete(existing_item_id)
                        except tk.TclError:
                            pass
                        echo_slots.pop(slot_name, None)
                continue
            existing_item_id = echo_slots.get(slot_name)
            if not isinstance(existing_item_id, int):
                try:
                    created_item_id = self.map_preview_canvas.create_line(
                        *preview_points,
                        fill=color,
                        width=line_width,
                        smooth=True,
                        dash=(4, 4),
                    )
                except tk.TclError:
                    continue
                echo_slots[slot_name] = int(created_item_id)
                continue
            try:
                self.map_preview_canvas.coords(existing_item_id, *preview_points)
                self.map_preview_canvas.itemconfigure(
                    existing_item_id,
                    fill=color,
                    width=line_width,
                    dash=(4, 4),
                    state="normal",
                )
            except tk.TclError:
                try:
                    created_item_id = self.map_preview_canvas.create_line(
                        *preview_points,
                        fill=color,
                        width=line_width,
                        smooth=True,
                        dash=(4, 4),
                    )
                except tk.TclError:
                    continue
                echo_slots[slot_name] = int(created_item_id)
        obsolete_slot_names = [slot_name for slot_name in echo_slots if slot_name not in active_slot_names]
        for slot_name in obsolete_slot_names:
            item_id = echo_slots.pop(slot_name, None)
            self._live_echo_geometry_cache.pop(slot_name, None)
            if not isinstance(item_id, int):
                continue
            try:
                self.map_preview_canvas.delete(item_id)
            except tk.TclError:
                pass

    def _selected_record_point(self, record: dict[str, Any] | None) -> MeasurementPoint | None:
        if record is None:
            return None
        point_index = record.get("point_index")
        if not isinstance(point_index, int) or point_index < 0 or point_index >= len(self._mission_points):
            return None
        return self._mission_points[point_index]

    @staticmethod
    def _selected_record_measurement_position(record: dict[str, Any]) -> tuple[float, float] | None:
        live_position = record.get("live_position_at_measurement")
        if not isinstance(live_position, dict):
            return None
        x_value = live_position.get("x")
        y_value = live_position.get("y")
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            return None
        x = float(x_value)
        y = float(y_value)
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return (x, y)

    def _selected_record_overlay_point(
        self,
        record: dict[str, Any],
        *,
        measurement_position: tuple[float, float],
    ) -> MeasurementPoint | None:
        x, y = measurement_position
        live_position = record.get("live_position_at_measurement")
        live_yaw = live_position.get("yaw") if isinstance(live_position, dict) else None
        yaw: float | None = None
        if isinstance(live_yaw, (int, float)) and math.isfinite(float(live_yaw)):
            yaw = float(live_yaw)
        else:
            point = self._selected_record_point(record)
            if point is not None and isinstance(point.yaw, (int, float)) and math.isfinite(float(point.yaw)):
                yaw = float(point.yaw)
        if yaw is None:
            return None
        return MeasurementPoint(id=None, name=None, x=x, y=y, yaw=yaw)

    @staticmethod
    def _extract_echo_distances(value: Any, *, limit: int) -> list[float]:
        if not isinstance(value, list) or limit <= 0:
            return []
        distances: list[float] = []
        for item in value[:limit]:
            if not isinstance(item, dict):
                continue
            distance_m = item.get("distance_m")
            if not isinstance(distance_m, (int, float)):
                continue
            numeric = float(distance_m)
            if not math.isfinite(numeric) or numeric <= 0.0:
                continue
            distances.append(numeric)
        return distances

    def _build_echo_overlay_preview_points(
        self,
        *,
        rx_position: tuple[float, float],
        measurement_position: tuple[float, float],
        echo_distance_m: float,
        sample_levels: tuple[int, int, int] = LIVE_ECHO_SAMPLING_NORMAL,
    ) -> tuple[list[float] | None, int]:
        mission = self._mission
        original = self._map_image_original
        if mission is None or mission.map_config is None or original is None:
            return (None, 1)
        resolution = mission.map_config.resolution
        if not math.isfinite(resolution) or resolution <= 0.0:
            return (None, 1)
        rx_x, rx_y = rx_position
        point_x, point_y = measurement_position
        if (
            not math.isfinite(rx_x)
            or not math.isfinite(rx_y)
            or not math.isfinite(point_x)
            or not math.isfinite(point_y)
            or not math.isfinite(echo_distance_m)
            or echo_distance_m < 0.0
        ):
            return (None, 1)
        distance_rx_to_point = math.hypot(point_x - rx_x, point_y - rx_y)
        ellipse_axes = _compute_bistatic_echo_ellipse_axes(
            distance_rx_to_point=distance_rx_to_point,
            echo_distance_m=echo_distance_m,
        )
        if ellipse_axes is None:
            return (None, 1)
        semi_focal_distance, semi_major_axis, semi_minor_axis = ellipse_axes
        if semi_minor_axis <= 0.0:
            return (None, 1)
        image_height = original.height()
        scale_x, scale_y = self._map_preview_scale
        offset_x, offset_y = self._map_preview_offset
        preview_scale_factor = (abs(scale_x) + abs(scale_y)) / 2.0
        if not math.isfinite(preview_scale_factor) or preview_scale_factor <= 0.0:
            preview_scale_factor = 1.0
        ellipse_size_px = (semi_major_axis / resolution) * preview_scale_factor
        small_samples, medium_samples, large_samples = sample_levels
        if ellipse_size_px < 40.0:
            samples = small_samples
        elif ellipse_size_px < 130.0:
            samples = medium_samples
        else:
            samples = large_samples
        unit_circle_points = self._ellipse_unit_circle_points(samples=samples)
        center_x = (rx_x + point_x) / 2.0
        center_y = (rx_y + point_y) / 2.0
        angle = math.atan2(point_y - rx_y, point_x - rx_x)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        preview_points: list[float] = []
        for unit_cos, unit_sin in unit_circle_points:
            local_x = semi_major_axis * unit_cos
            local_y = semi_minor_axis * unit_sin
            world_x = center_x + local_x * cos_angle - local_y * sin_angle
            world_y = center_y + local_x * sin_angle + local_y * cos_angle
            map_pixel = self._world_to_map_pixel(x=world_x, y=world_y, image_height=image_height)
            if map_pixel is None:
                continue
            preview_points.extend(
                (
                    map_pixel[0] * scale_x + offset_x,
                    map_pixel[1] * scale_y + offset_y,
                )
        )
        if len(preview_points) < 6:
            return (None, 1)
        line_width = max(1, int(round((echo_distance_m / resolution) * preview_scale_factor * 0.03)))
        return (preview_points, line_width)

    def _draw_echo_ellipse_for_overlay(
        self,
        *,
        rx_position: tuple[float, float],
        measurement_position: tuple[float, float],
        echo_distance_m: float,
        color: str,
    ) -> int | None:
        preview_points, line_width = self._build_echo_overlay_preview_points(
            rx_position=rx_position,
            measurement_position=measurement_position,
            echo_distance_m=echo_distance_m,
        )
        if preview_points is None:
            return None
        return int(
            self.map_preview_canvas.create_line(
                *preview_points,
                fill=color,
                width=line_width,
                smooth=True,
                dash=(4, 4),
            )
        )

    def _ellipse_unit_circle_points(self, *, samples: int) -> tuple[tuple[float, float], ...]:
        if samples < 3:
            samples = 3
        cached = self._ellipse_unit_circle_cache.get(samples)
        if cached is not None:
            return cached
        points = tuple(
            (
                math.cos((2.0 * math.pi * idx) / samples),
                math.sin((2.0 * math.pi * idx) / samples),
            )
            for idx in range(samples + 1)
        )
        self._ellipse_unit_circle_cache[samples] = points
        return points

    def _draw_selected_lidar_reference_overlay(self) -> None:
        record = self._selected_record_payload()
        if record is None:
            return
        measurement_position = self._selected_record_measurement_position(record)
        if measurement_position is None:
            return
        measurement = record.get("measurement")
        if not isinstance(measurement, dict):
            return
        result = measurement.get("result")
        if not isinstance(result, dict):
            return
        lidar_reference = result.get("lidar_reference")
        if not isinstance(lidar_reference, dict):
            return
        lidar_file = lidar_reference.get("output_file")
        if not isinstance(lidar_file, str) or not lidar_file.strip():
            return
        scan = self._load_lidar_scan_for_overlay(lidar_file)
        if scan is None:
            return
        overlay_point = self._selected_record_overlay_point(record, measurement_position=measurement_position)
        if overlay_point is None:
            return
        self._draw_lidar_scan_overlay_for_point(point=overlay_point, scan=scan)

    def _selected_record_payload(self) -> dict[str, Any] | None:
        selected_idx = self._selected_result_index
        if selected_idx is None:
            return None
        if selected_idx < 0 or selected_idx >= len(self._records):
            return None
        payload = self._records[selected_idx]
        return payload if isinstance(payload, dict) else None

    def _selected_record_payloads(self) -> list[dict[str, Any]]:
        selected_indices = getattr(self, "_selected_result_indices", ())
        if not selected_indices:
            selected_payload = self._selected_record_payload()
            return [selected_payload] if selected_payload is not None else []

        payloads: list[dict[str, Any]] = []
        for selected_idx in selected_indices:
            if selected_idx < 0 or selected_idx >= len(self._records):
                continue
            payload = self._records[selected_idx]
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    def _load_lidar_scan_for_overlay(self, lidar_file: str) -> dict[str, Any] | None:
        if lidar_file in self._lidar_reference_scan_cache:
            return self._lidar_reference_scan_cache[lidar_file]
        try:
            text = Path(lidar_file).read_text(encoding="utf-8")
        except Exception:
            self._lidar_reference_scan_cache[lidar_file] = None
            return None
        parsed = self._parse_lidar_scan_text_for_overlay(text)
        self._lidar_reference_scan_cache[lidar_file] = parsed
        return parsed

    @staticmethod
    def _parse_lidar_scan_text_for_overlay(raw_text: str) -> dict[str, Any] | None:
        angle_min_match = re.search(r"angle_min:\s*([-+0-9.eE]+)", raw_text)
        angle_inc_match = re.search(r"angle_increment:\s*([-+0-9.eE]+)", raw_text)
        if angle_min_match is None or angle_inc_match is None:
            return None
        try:
            angle_min = float(angle_min_match.group(1))
            angle_increment = float(angle_inc_match.group(1))
        except Exception:
            return None
        ranges = MissionWorkflowWindow._extract_lidar_ranges_from_scan_text(raw_text)
        if not ranges:
            return None
        return {
            "angle_min": angle_min,
            "angle_increment": angle_increment,
            "ranges": ranges,
        }

    @staticmethod
    def _extract_lidar_ranges_from_scan_text(raw_text: str) -> list[float]:
        block_match = re.search(r"ranges:\s*(\[[\s\S]*?\]|(?:\n(?:\s*-\s*[^\n]+))+)", raw_text)
        if block_match is None:
            return []
        tokens = re.findall(r"[-+0-9.eE]+|inf|-inf|nan", block_match.group(1))
        parsed: list[float] = []
        for token in tokens:
            normalized = token.lower()
            if normalized in {"inf", "+inf"}:
                value = float("inf")
            elif normalized == "-inf":
                value = float("-inf")
            elif normalized == "nan":
                value = float("nan")
            else:
                try:
                    value = float(token)
                except Exception:
                    continue
            parsed.append(value)
        return parsed

    def _draw_lidar_scan_overlay_for_point(self, *, point: MeasurementPoint, scan: dict[str, Any]) -> None:
        original = self._map_image_original
        if original is None:
            return
        start_map_pixel = self._world_to_map_pixel(x=point.x, y=point.y, image_height=original.height())
        if start_map_pixel is None:
            return
        if not self._is_pixel_inside_map(start_map_pixel[0], start_map_pixel[1], width=original.width(), height=original.height()):
            return
        scale_x, scale_y = self._map_preview_scale
        offset_x, offset_y = self._map_preview_offset
        start_x = start_map_pixel[0] * scale_x + offset_x
        start_y = start_map_pixel[1] * scale_y + offset_y
        self.map_preview_canvas.create_oval(
            start_x - 5,
            start_y - 5,
            start_x + 5,
            start_y + 5,
            fill="#90caf9",
            outline="#1565c0",
            width=1,
        )
        angle_min = float(scan["angle_min"])
        angle_increment = float(scan["angle_increment"])
        ranges = scan["ranges"]
        finite_positive_beam_count = sum(
            1 for distance in ranges if isinstance(distance, (int, float)) and math.isfinite(distance) and distance > 0.0
        )
        beam_stride = max(1, finite_positive_beam_count // LIDAR_OVERLAY_MAX_DRAWN_BEAMS)
        density_factor = max(1.0, math.sqrt(finite_positive_beam_count / float(LIDAR_OVERLAY_MAX_DRAWN_BEAMS)))
        effective_cell_size_px = LIDAR_OVERLAY_CELL_SIZE_PX * density_factor
        drawn_beam_cells: dict[tuple[int, int], int] = {}
        for idx, distance in enumerate(ranges):
            if not math.isfinite(distance) or distance <= 0.0:
                continue
            if idx % beam_stride != 0:
                continue
            beam_angle = point.yaw + angle_min + idx * angle_increment
            end_world_x = point.x + math.cos(beam_angle) * distance
            end_world_y = point.y + math.sin(beam_angle) * distance
            end_map_pixel = self._world_to_map_pixel(x=end_world_x, y=end_world_y, image_height=original.height())
            if end_map_pixel is None:
                continue
            end_x = end_map_pixel[0] * scale_x + offset_x
            end_y = end_map_pixel[1] * scale_y + offset_y
            beam_cell = (
                int((end_x - start_x) / effective_cell_size_px),
                int((end_y - start_y) / effective_cell_size_px),
            )
            current_cell_count = drawn_beam_cells.get(beam_cell, 0)
            if current_cell_count >= LIDAR_OVERLAY_MAX_BEAMS_PER_CELL:
                continue
            drawn_beam_cells[beam_cell] = current_cell_count + 1
            self.map_preview_canvas.create_line(
                start_x,
                start_y,
                end_x,
                end_y,
                fill="#4fc3f7",
                width=1,
                stipple="gray25",
            )

    def _draw_live_marker(self) -> None:
        mission = self._mission
        preview = self._map_image_preview
        original = self._map_image_original
        position = self._live_position
        if (
            mission is None
            or mission.map_config is None
            or preview is None
            or original is None
            or position is None
        ):
            self._clear_live_overlay_layer(components=("marker", "heading"))
            return
        expected_frame_id = self._expected_live_frame_id()
        received_frame_id = position.get("frame_id")
        if (
            isinstance(received_frame_id, str)
            and received_frame_id.strip()
            and received_frame_id != expected_frame_id
        ):
            self._clear_live_overlay_layer(components=("marker", "heading"))
            return
        x_value = position.get("x")
        y_value = position.get("y")
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            self._clear_live_overlay_layer(components=("marker", "heading"))
            return

        scale_x, scale_y = self._map_preview_scale
        offset_x, offset_y = self._map_preview_offset
        map_pixel = self._world_to_map_pixel(
            x=float(x_value),
            y=float(y_value),
            image_height=original.height(),
        )
        if map_pixel is None:
            self._clear_live_overlay_layer(components=("marker", "heading"))
            return
        if not self._is_pixel_inside_map(
            map_pixel[0],
            map_pixel[1],
            width=original.width(),
            height=original.height(),
        ):
            self._clear_live_overlay_layer(components=("marker", "heading"))
            return
        pixel_coordinates = (map_pixel[0] * scale_x + offset_x, map_pixel[1] * scale_y + offset_y)
        px, py = pixel_coordinates
        radius = 6
        marker_item_id = self._live_overlay_item_ids.get("marker")
        if not isinstance(marker_item_id, int):
            try:
                marker_item_id = int(
                    self.map_preview_canvas.create_oval(
                        px - radius,
                        py - radius,
                        px + radius,
                        py + radius,
                        fill="#ff4d6d",
                        outline="#ffffff",
                        width=1,
                    )
                )
            except tk.TclError:
                marker_item_id = None
            self._live_overlay_item_ids["marker"] = marker_item_id
        if isinstance(marker_item_id, int):
            try:
                self.map_preview_canvas.coords(marker_item_id, px - radius, py - radius, px + radius, py + radius)
                self.map_preview_canvas.itemconfigure(
                    marker_item_id,
                    fill="#ff4d6d",
                    outline="#ffffff",
                    width=1,
                    state="normal",
                )
            except tk.TclError:
                try:
                    marker_item_id = int(
                        self.map_preview_canvas.create_oval(
                            px - radius,
                            py - radius,
                            px + radius,
                            py + radius,
                            fill="#ff4d6d",
                            outline="#ffffff",
                            width=1,
                        )
                    )
                except tk.TclError:
                    marker_item_id = None
                self._live_overlay_item_ids["marker"] = marker_item_id
        yaw_value = position.get("yaw")
        if isinstance(yaw_value, (int, float)):
            heading_length = 14
            end_x = px + math.cos(float(yaw_value)) * heading_length
            end_y = py - math.sin(float(yaw_value)) * heading_length
            heading_item_id = self._live_overlay_item_ids.get("heading")
            if not isinstance(heading_item_id, int):
                try:
                    heading_item_id = int(
                        self.map_preview_canvas.create_line(
                            px,
                            py,
                            end_x,
                            end_y,
                            fill="#ffccd5",
                            width=2,
                            arrow=tk.LAST,
                        )
                    )
                except tk.TclError:
                    heading_item_id = None
                self._live_overlay_item_ids["heading"] = heading_item_id
            if isinstance(heading_item_id, int):
                try:
                    self.map_preview_canvas.coords(heading_item_id, px, py, end_x, end_y)
                    self.map_preview_canvas.itemconfigure(
                        heading_item_id,
                        fill="#ffccd5",
                        width=2,
                        arrow=tk.LAST,
                        state="normal",
                    )
                except tk.TclError:
                    try:
                        heading_item_id = int(
                            self.map_preview_canvas.create_line(
                                px,
                                py,
                                end_x,
                                end_y,
                                fill="#ffccd5",
                                width=2,
                                arrow=tk.LAST,
                            )
                        )
                    except tk.TclError:
                        heading_item_id = None
                    self._live_overlay_item_ids["heading"] = heading_item_id
            return
        self._clear_live_overlay_layer(components=("heading",))


    def _draw_live_position_info_overlay(self) -> None:
        if not bool(self.live_pose_stream_enabled_var.get()):
            self._clear_live_overlay_layer(components=("position_info_box", "position_info_text"))
            return
        position = self._copy_valid_live_position()
        if position is None:
            self._clear_live_overlay_layer(components=("position_info_box", "position_info_text"))
            return
        x_value = float(position["x"])
        y_value = float(position["y"])
        distance_text = "-"
        if self._rx_antenna_global_position is not None:
            rx_x, rx_y = self._rx_antenna_global_position
            distance_text = f"{math.hypot(x_value - rx_x, y_value - rx_y):.2f} m"

        overlay_text = f"Live: x={x_value:.2f}, y={y_value:.2f}\nAbstand zu RX: {distance_text}"
        try:
            canvas_width = max(1, self.map_preview_canvas.winfo_width())
            canvas_height = max(1, self.map_preview_canvas.winfo_height())
        except tk.TclError:
            return
        margin = 10
        anchor_x = canvas_width - margin
        anchor_y = canvas_height - margin

        text_item_id = self._live_overlay_item_ids.get("position_info_text")
        if not isinstance(text_item_id, int):
            text_item_id = int(self.map_preview_canvas.create_text(anchor_x, anchor_y, anchor="se"))
            self._live_overlay_item_ids["position_info_text"] = text_item_id
        self.map_preview_canvas.coords(text_item_id, anchor_x, anchor_y)
        self.map_preview_canvas.itemconfigure(
            text_item_id,
            text=overlay_text,
            fill="#eaf2ff",
            font=("TkDefaultFont", 10, "bold"),
            justify="right",
            state="normal",
        )

        bbox = self.map_preview_canvas.bbox(text_item_id)
        if bbox is None:
            self._clear_live_overlay_layer(components=("position_info_box",))
            return
        x1, y1, x2, y2 = bbox
        padding = 6
        box_item_id = self._live_overlay_item_ids.get("position_info_box")
        if not isinstance(box_item_id, int):
            box_item_id = int(self.map_preview_canvas.create_rectangle(x1, y1, x2, y2))
            self._live_overlay_item_ids["position_info_box"] = box_item_id
        self.map_preview_canvas.coords(box_item_id, x1 - padding, y1 - padding, x2 + padding, y2 + padding)
        self.map_preview_canvas.itemconfigure(
            box_item_id,
            fill="#0b1220",
            outline="#91a4c5",
            width=1,
            state="normal",
        )
        self.map_preview_canvas.tag_lower(box_item_id, text_item_id)

    def _highlight_marker(self, marker_id: int) -> None:
        self.map_preview_canvas.itemconfigure(
            marker_id,
            fill="#ffd54f",
            outline="#ff8f00",
            width=2,
        )

    def _generate_unique_point_id(self) -> str:
        used_ids = {point.id for point in self._mission_points if point.id}
        highest_suffix = 0
        id_pattern = re.compile(r"^p(\d+)$")

        for point_id in used_ids:
            match = id_pattern.match(point_id)
            if match:
                highest_suffix = max(highest_suffix, int(match.group(1)))

        candidate_index = highest_suffix + 1
        while True:
            point_id = f"p{candidate_index:03d}"
            if point_id not in used_ids:
                return point_id
            candidate_index += 1

    def _add_point(self) -> None:
        try:
            yaw_internal_radians = self._yaw_cw_degrees_to_internal_radians(self.point_yaw_var.get().strip())
        except Exception:
            messagebox.showwarning(
                "Messpunkt ungültig",
                "Yaw muss eine ganze Zahl in Grad sein (z. B. 90°).",
                parent=self,
            )
            return
        self.point_yaw_var.set(self._format_yaw_degrees(yaw_internal_radians))
        self._add_point_from_values(
            x=self.point_x_var.get().strip(),
            y=self.point_y_var.get().strip(),
            yaw_internal_radians=yaw_internal_radians,
            name=self.point_name_var.get().strip() or None,
        )

    def _add_point_from_values(
        self,
        *,
        x: float | str,
        y: float | str,
        yaw_internal_radians: float,
        name: str | None = None,
    ) -> None:
        point_payload = {
            "id": self._generate_unique_point_id(),
            "name": name,
            "x": x,
            "y": y,
            "z": 0.0,
            "yaw": yaw_internal_radians,
            "enabled": True,
        }
        try:
            mission = measurement_mission_from_dict(
                {"name": "point-check", "points": [point_payload], "repeat": 1}
            )
        except Exception as exc:
            messagebox.showwarning("Messpunkt ungültig", str(exc), parent=self)
            return

        point = mission.points[0]
        self._mission_points.append(point)
        self._sync_validated_mission_points()
        self._refresh_points_table()
        self._persist_workflow_state()
        self._append_validation(
            f"✅ Punkt hinzugefügt: {point.id or point.name} "
            f"(x={point.x:.2f}, y={point.y:.2f}, z={point.z:.2f}, yaw={self._format_yaw_degrees(point.yaw)})"
        )

    def _remove_selected_point(self) -> None:
        selected = self.points_table.selection()
        if not selected:
            return
        index = self.points_table.index(selected[0])
        if index < 0 or index >= len(self._mission_points):
            return
        removed = self._mission_points.pop(index)
        self._sync_validated_mission_points()
        self._refresh_points_table()
        self._persist_workflow_state()
        self._append_validation(f"ℹ️ Punkt entfernt: {removed.id or removed.name}")

    def _move_selected_point_up(self) -> None:
        selected = self.points_table.selection()
        if not selected:
            return
        index = self.points_table.index(selected[0])
        if index <= 0 or index >= len(self._mission_points):
            return
        self._mission_points[index - 1], self._mission_points[index] = (
            self._mission_points[index],
            self._mission_points[index - 1],
        )
        self._sync_validated_mission_points()
        self._refresh_points_table()
        self._persist_workflow_state()
        moved_index = index - 1
        table_children = self.points_table.get_children()
        moved_item = table_children[moved_index]
        self.points_table.selection_set(moved_item)
        self.points_table.focus(moved_item)
        self.points_table.see(moved_item)
        self._append_validation("ℹ️ Punkt nach oben verschoben.")

    def _move_selected_point_down(self) -> None:
        selected = self.points_table.selection()
        if not selected:
            return
        index = self.points_table.index(selected[0])
        if index < 0 or index >= len(self._mission_points) - 1:
            return
        self._mission_points[index], self._mission_points[index + 1] = (
            self._mission_points[index + 1],
            self._mission_points[index],
        )
        self._sync_validated_mission_points()
        self._refresh_points_table()
        self._persist_workflow_state()
        moved_index = index + 1
        table_children = self.points_table.get_children()
        moved_item = table_children[moved_index]
        self.points_table.selection_set(moved_item)
        self.points_table.focus(moved_item)
        self.points_table.see(moved_item)
        self._append_validation("ℹ️ Punkt nach unten verschoben.")

    def _toggle_selected_point_enabled(self) -> None:
        selected = self.points_table.selection()
        if not selected:
            return
        index = self.points_table.index(selected[0])
        self._toggle_point_enabled(index)

    def _on_points_table_double_click(self, event: tk.Event) -> None:
        region = self.points_table.identify_region(event.x, event.y)
        if region != "cell":
            return
        row_id = self.points_table.identify_row(event.y)
        if not row_id:
            return
        index = self.points_table.index(row_id)
        self._toggle_point_enabled(index)

    def _toggle_point_enabled(self, index: int) -> None:
        if index < 0 or index >= len(self._mission_points):
            return
        point = self._mission_points[index]
        self._mission_points[index] = replace(point, enabled=not point.enabled)
        self._sync_validated_mission_points()
        self._refresh_points_table()
        self._persist_workflow_state()

    def _sync_validated_mission_points(self) -> None:
        if self._mission is None:
            return
        self._mission = replace(self._mission, points=list(self._mission_points))

    def _refresh_points_table(self) -> None:
        self.points_table.delete(*self.points_table.get_children())
        for idx, point in enumerate(self._mission_points):
            self.points_table.insert(
                "",
                "end",
                values=(
                    idx + 1,
                    f"{point.x:.3f}",
                    f"{point.y:.3f}",
                    self._format_yaw_degrees(point.yaw),
                ),
                tags=("active",) if point.enabled else ("inactive",),
            )
        self._draw_map_preview()
        self._refresh_start_point_options()

    def _refresh_start_point_options(self) -> None:
        labels = [
            self._format_start_point_label(active_index, point)
            for active_index, (_mission_index, point) in enumerate(self._active_start_points())
        ]
        if not labels:
            labels = ["1"]
        self.start_point_combo.configure(values=labels)
        current_value = self.start_point_var.get()
        if current_value in labels:
            return
        self.start_point_var.set(labels[0])

    @staticmethod
    def _format_start_point_label(index: int, point: MeasurementPoint) -> str:
        _ = point
        return str(index + 1)

    def _active_start_points(self) -> list[tuple[int, MeasurementPoint]]:
        return [
            (mission_index, point)
            for mission_index, point in enumerate(self._mission_points)
            if point.enabled
        ]

    def _selected_start_point_index(self) -> int:
        selected = self.start_point_var.get().strip()
        match = re.match(r"^(\d+)", selected)
        if not match:
            return 0
        parsed_index = int(match.group(1)) - 1
        if parsed_index < 0:
            return 0
        active_points = self._active_start_points()
        if active_points and parsed_index >= len(active_points):
            return 0
        return parsed_index

    @staticmethod
    def _serialize_point(point: MeasurementPoint) -> dict[str, Any]:
        point_payload: dict[str, Any] = {
            "id": point.id,
            "name": point.name,
            "x": point.x,
            "y": point.y,
            "z": point.z,
            "enabled": point.enabled,
        }

        if point.notes is not None:
            point_payload["notes"] = point.notes
        if point.measurement_profile is not None:
            point_payload["measurement_profile"] = point.measurement_profile

        if point.yaw is not None:
            point_payload["yaw"] = point.yaw
            return point_payload

        quaternion_values = {
            "qx": point.qx,
            "qy": point.qy,
            "qz": point.qz,
            "qw": point.qw,
        }
        if all(value is not None for value in quaternion_values.values()):
            point_payload.update(quaternion_values)
        return point_payload

    def _validate_selected(self) -> None:
        if not self._mission_points:
            self._set_validation_text("Bitte zuerst mindestens einen Messpunkt anlegen.")
            return
        try:
            repeat_raw = self.repeat_var.get().strip()
            repeat = int(repeat_raw)
            mission = MeasurementMission(
                name=self.mission_name_var.get().strip(),
                points=list(self._mission_points),
                repeat=repeat,
                wait_after_arrival_s=0.0,
                map_config=self._selected_map_config,
            )
            serialized_points = [self._serialize_point(point) for point in mission.points]
            measurement_mission_from_dict(
                {
                    "name": mission.name,
                    "repeat": mission.repeat,
                    "wait_after_arrival_s": mission.wait_after_arrival_s,
                    "points": serialized_points,
                    "map_config": {
                        "image": mission.map_config.image,
                        "resolution": mission.map_config.resolution,
                        "origin": list(mission.map_config.origin),
                        "frame_id": mission.map_config.frame_id,
                        "negate": mission.map_config.negate,
                        "occupied_thresh": mission.map_config.occupied_thresh,
                        "free_thresh": mission.map_config.free_thresh,
                    }
                    if mission.map_config is not None
                    else None,
                }
            )
        except Exception as exc:
            self._mission = None
            self._refresh_map_section()
            self._set_validation_text(f"❌ Validierung fehlgeschlagen\nDetails: {exc}")
            return

        self._mission = mission
        self._persist_workflow_state()
        self._refresh_map_section()
        repeats = mission.repeat or 1
        total_points = len(mission.points) * repeats
        validation_lines = [
            f"✅ Mission valide: {mission.name}\n"
            f"Punkte pro Zyklus: {len(mission.points)} | Wiederholungen: {repeats} | Gesamtpunkte: {total_points}"
        ]
        runtime_reasons = self._runtime_guard_reasons()
        if runtime_reasons:
            validation_lines.append("⚠️ Start derzeit blockiert:")
            validation_lines.extend(f"  • {reason}" for reason in runtime_reasons)
        else:
            validation_lines.append("✅ Startfreigabe: Kein Continuous-Modus oder RX-Job aktiv.")

        if self._map_image_size is None:
            validation_lines.append(
                "ℹ️ Karte nicht geladen: Live-Pose und Karten-Grenzprüfung sind derzeit nicht verfügbar."
            )
        else:
            width, height = self._map_image_size
            outside_points: list[str] = []
            for index, point in enumerate(mission.points, start=1):
                map_pixel = self._world_to_map_pixel(x=point.x, y=point.y, image_height=height)
                if map_pixel is None:
                    continue
                if not self._is_pixel_inside_map(map_pixel[0], map_pixel[1], width=width, height=height):
                    point_label = point.name or point.id or f"Punkt {index}"
                    outside_points.append(point_label)
            if outside_points:
                joined = ", ".join(outside_points[:5])
                suffix = " …" if len(outside_points) > 5 else ""
                validation_lines.append(
                    f"⚠️ Koordinaten außerhalb der Karte: {joined}{suffix}. Bitte Punkte korrigieren."
                )

        self._set_validation_text("\n".join(validation_lines))
        self._refresh_review_ready_indicator()

    def _build_workflow_state_payload(self) -> dict[str, Any]:
        repeat_raw = self.repeat_var.get().strip()
        repeat: int | str
        try:
            repeat = int(repeat_raw)
        except ValueError:
            repeat = repeat_raw
        return {
            "name": self.mission_name_var.get().strip(),
            "repeat": repeat,
            "points": [self._serialize_point(point) for point in self._mission_points],
            "start_point_index": self._selected_start_point_index(),
            "map_config_file": self._selected_map_config_file,
            "rx_antenna_global_position": self._serialize_rx_antenna_global_position(),
            "lidar_reference_enabled": bool(self.lidar_reference_enabled_var.get()),
            "manual_review_enabled": bool(self.manual_review_enabled_var.get()),
            "test_run_enabled": bool(self.test_run_enabled_var.get()),
            "manual_navigation_enabled": bool(self.manual_navigation_enabled_var.get()),
            "reverse_point_order": bool(self.reverse_point_order_var.get()),
            "live_pose_stream_enabled": bool(self.live_pose_stream_enabled_var.get()),
            "live_preview_enabled": bool(self.live_preview_enabled_var.get()),
            "records": self._records,
        }

    def _serialize_rx_antenna_global_position(self) -> dict[str, float] | None:
        position = self._rx_antenna_global_position
        if position is None:
            return None
        return {"x": position[0], "y": position[1]}

    @staticmethod
    def _parse_rx_antenna_global_position(payload: Any) -> tuple[float, float] | None:
        if not isinstance(payload, dict):
            return None
        x_value = payload.get("x")
        y_value = payload.get("y")
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            return None
        parsed_x = float(x_value)
        parsed_y = float(y_value)
        if not math.isfinite(parsed_x) or not math.isfinite(parsed_y):
            return None
        return (parsed_x, parsed_y)

    def _persist_workflow_state(self) -> None:
        if self._is_restoring_workflow_state:
            return
        payload = self._build_workflow_state_payload()
        _save_json_dict(self._workflow_state_file, payload)

    def _restore_workflow_state(self) -> None:
        payload = _load_json_dict(self._workflow_state_file)
        if not payload:
            return

        self._is_restoring_workflow_state = True
        try:
            map_config_file = payload.get("map_config_file")
            loaded_map_config: MapConfig | None = None
            if isinstance(map_config_file, str) and map_config_file.strip():
                loaded_map_config = self._load_map_config_from_file(Path(map_config_file))
            mission = measurement_mission_from_dict(
                {
                    "name": str(payload.get("name") or "mission-ui"),
                    "repeat": payload.get("repeat", 1),
                    "wait_after_arrival_s": 0.0,
                    "points": payload.get("points", []),
                    "map_config": {
                        "image": loaded_map_config.image,
                        "resolution": loaded_map_config.resolution,
                        "origin": list(loaded_map_config.origin),
                        "frame_id": loaded_map_config.frame_id,
                        "negate": loaded_map_config.negate,
                        "occupied_thresh": loaded_map_config.occupied_thresh,
                        "free_thresh": loaded_map_config.free_thresh,
                    }
                    if loaded_map_config is not None
                    else None,
                }
            )
        except Exception:
            self._append_validation(
                "⚠️ Persistierter Workflow konnte nicht geladen werden (ungültige Daten)."
            )
            return

        try:
            self.mission_name_var.set(mission.name)
            self.repeat_var.set(str(mission.repeat or 1))
            self._mission_points = list(mission.points)
            self._mission = mission
            self._selected_map_config = mission.map_config
            self._selected_map_config_file = payload.get("map_config_file")
            rx_position = self._parse_rx_antenna_global_position(payload.get("rx_antenna_global_position"))
            if rx_position is None:
                self._clear_rx_antenna_position(persist=False)
            else:
                self._set_rx_antenna_position(x=rx_position[0], y=rx_position[1], persist=False)
            self.lidar_reference_enabled_var.set(bool(payload.get("lidar_reference_enabled", True)))
            self.manual_review_enabled_var.set(bool(payload.get("manual_review_enabled", True)))
            self.test_run_enabled_var.set(bool(payload.get("test_run_enabled", False)))
            self.manual_navigation_enabled_var.set(bool(payload.get("manual_navigation_enabled", False)))
            self.reverse_point_order_var.set(bool(payload.get("reverse_point_order", False)))
            self.live_pose_stream_enabled_var.set(bool(payload.get("live_pose_stream_enabled", False)))
            self.live_preview_enabled_var.set(bool(payload.get("live_preview_enabled", False)))
            self._refresh_points_table()
            self._refresh_map_section()
            persisted_start_point = payload.get("start_point_index")
            active_points = self._active_start_points()
            if isinstance(persisted_start_point, int) and 0 <= persisted_start_point < len(active_points):
                _mission_index, selected_point = active_points[persisted_start_point]
                self.start_point_var.set(
                    self._format_start_point_label(
                        persisted_start_point,
                        selected_point,
                    )
                )
            repeats = mission.repeat or 1
            total_points = len(mission.points) * repeats
            self._set_validation_text(
                f"✅ Persistierter Workflow geladen: {mission.name}\n"
                f"Punkte pro Zyklus: {len(mission.points)} | Wiederholungen: {repeats} | Gesamtpunkte: {total_points}"
            )
            self._refresh_review_ready_indicator()
            persisted_records = payload.get("records")
            if isinstance(persisted_records, list):
                self._clear_results_table()
                for persisted_payload in persisted_records:
                    if isinstance(persisted_payload, dict):
                        self._on_record(dict(persisted_payload))
                if persisted_records:
                    self._append_validation(
                        f"✅ Persistierte Ergebnisliste geladen: {len(self._records)} Messpunkte"
                    )
        finally:
            self._is_restoring_workflow_state = False

    def _set_rx_antenna_position(self, *, x: float, y: float, persist: bool = True) -> None:
        self._rx_antenna_global_position = (x, y)
        self.rx_antenna_x_var.set(f"{x:.3f}")
        self.rx_antenna_y_var.set(f"{y:.3f}")
        self._draw_map_preview()
        if persist:
            self._persist_workflow_state()

    def _clear_rx_antenna_position(self, persist: bool = True) -> None:
        self._rx_antenna_global_position = None
        self.rx_antenna_x_var.set("")
        self.rx_antenna_y_var.set("")
        self._draw_map_preview()
        if persist:
            self._persist_workflow_state()

    def _apply_rx_antenna_position_from_inputs(self) -> None:
        try:
            parsed_x = float(self.rx_antenna_x_var.get().strip().replace(",", "."))
            parsed_y = float(self.rx_antenna_y_var.get().strip().replace(",", "."))
        except ValueError:
            messagebox.showwarning(
                "RX-Antenne",
                "RX-Antennenposition muss numerisch sein (X/Y).",
                parent=self,
            )
            return
        if not math.isfinite(parsed_x) or not math.isfinite(parsed_y):
            messagebox.showwarning(
                "RX-Antenne",
                "RX-Antennenposition enthält ungültige Zahlen.",
                parent=self,
            )
            return
        self._set_rx_antenna_position(x=parsed_x, y=parsed_y)
        self._append_validation(f"✅ RX-Antenne gesetzt: x={parsed_x:.3f}, y={parsed_y:.3f}")

    def _start_run(self) -> None:
        if self._mission is None:
            messagebox.showwarning(
                "Mission",
                "Bitte zuerst eine gültige Mission anlegen und validieren.",
                parent=self,
            )
            return
        if self._run_thread and self._run_thread.is_alive():
            return
        if AUTO_STOP_CONTINUOUS_BEFORE_RUN and self._is_continuous_active():
            stop_continuous = getattr(self.master, "stop_continuous", None)
            if callable(stop_continuous):
                self._append_validation("ℹ️ Continuous-Modus aktiv – versuche vor Run-Start automatisch zu stoppen.")
                try:
                    stop_continuous()
                except Exception as exc:
                    messagebox.showerror(
                        "Run-Start blockiert",
                        "Continuous-Modus konnte nicht automatisch gestoppt werden.\n"
                        "Bitte Continuous-Modus zuerst stoppen und erneut starten.\n\n"
                        f"Details: {exc}",
                        parent=self,
                    )
                    self._append_validation(
                        "❌ Run-Start blockiert: Continuous-Modus zuerst stoppen (Auto-Stop fehlgeschlagen)."
                    )
                    self._refresh_review_ready_indicator(prerequisites_ok=False)
                    return
        ready, reasons = self._check_run_prerequisites()
        self._refresh_review_ready_indicator(prerequisites_ok=ready)
        if not ready:
            details = "\n".join(f"• {reason}" for reason in reasons)
            messagebox.showwarning(
                "Run-Start blockiert",
                "Der Run kann nicht gestartet werden, da Voraussetzungen fehlen:\n\n"
                f"{details}\n\nBitte Voraussetzungen erfüllen und erneut starten.",
                parent=self,
            )
            self._append_validation("❌ Run-Start blockiert: " + " | ".join(reasons))
            return
        test_run_enabled = self._is_test_run_enabled()
        if not test_run_enabled and not self._ensure_transmitter_before_run():
            return
        if test_run_enabled:
            self._append_validation("ℹ️ Testlauf aktiv: Wegpunkte werden ohne Messung angefahren.")
        start_point_index = self._selected_start_point_index()

        # Ergebnisliste absichtlich nicht automatisch löschen:
        # Historische Run-Ergebnisse sollen im Workflow persistent bleiben
        # und nur über "Ergebnisliste leeren" entfernt werden.
        self._run_started_at = time.time()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._run_log_dir = Path("signals") / "mission-runs" / ts
        store = JsonRunLogStore(self._run_log_dir)
        interrupted = store.mark_interrupted_runs()
        if interrupted:
            self._append_validation(
                f"⚠️ {interrupted} verwaiste Run(s) wurden als interrupted markiert."
            )

        def _persist(payload: dict[str, Any]) -> None:
            self._attach_result_table_snapshot(payload)
            self.after(0, self._on_record, payload)

        self._sync_live_pose_stream_state()
        if bool(self.manual_navigation_enabled_var.get()):
            navigator = _ManualPromptNavigator(
                parent=self,
                on_status=self._on_stage_update,
                on_operator_message=self._append_validation,
                start_index=start_point_index,
            )
        else:
            navigator = self._ensure_navigator()

        self._executor = MeasurementRunExecutor(
            mission=self._mission,
            navigator=navigator,
            measurement_service=MissionRxMeasurementService(
                app=self.master,
                on_status=self._on_stage_update,
                on_operator_message=self._append_validation,
                review_measurement=self._review_measurement,
                enable_lidar_reference=bool(self.lidar_reference_enabled_var.get()),
                lidar_topic=self._runtime_config.lidar_topic,
                lidar_timeout_s=self._runtime_config.lidar_reference_timeout_s,
                robot_host=self._runtime_config.robot_host,
                remote_ros_env_cmd=self._runtime_config.remote_ros_env_cmd,
                remote_ros_setup=self._runtime_config.remote_ros_setup,
                fastdds_profiles_file=self._runtime_config.fastdds_profiles_file,
            ),
            persist_result=_persist,
            run_log_store=store,
            on_runtime_event=self._on_executor_runtime_event,
            config=MeasurementRunExecutorConfig(
                on_point_error="stop",
                goal_reached_timeout_s=self._runtime_config.goal_reached_timeout_s,
                navigation_retry_attempts=self._runtime_config.navigation_retry_attempts,
                start_point_index=start_point_index,
                reverse_point_order=bool(self.reverse_point_order_var.get()),
                enable_measurements=not test_run_enabled,
                confirm_measurement_after_navigation_failure=self._confirm_measurement_after_navigation_failure,
            ),
        )

        self._sync_live_pose_stream_state()
        self._start_live_label_ticker()
        self._set_run_buttons(running=True, paused=False)
        self._run_thread = threading.Thread(target=self._run_executor_thread, daemon=True)
        self._run_thread.start()

    def _start_manual_measurement(self) -> None:
        if self._mission is None:
            messagebox.showwarning(
                "Mission",
                "Bitte zuerst eine gültige Mission anlegen und validieren.",
                parent=self,
            )
            return
        if self._run_thread and self._run_thread.is_alive():
            messagebox.showwarning(
                "Manuelle Messung",
                "Während eines laufenden Runs ist keine manuelle Messung möglich.",
                parent=self,
            )
            return
        if self._manual_measurement_thread and self._manual_measurement_thread.is_alive():
            return
        if not self._ensure_transmitter_before_run():
            return
        self._sync_live_pose_stream_state()
        self._start_live_label_ticker()
        self._set_run_buttons(running=True, paused=False)
        self._manual_measurement_thread = threading.Thread(
            target=self._run_manual_measurement_thread,
            daemon=True,
        )
        self._manual_measurement_thread.start()

    def _manual_measurement_point_context(self) -> PointExecutionContext | None:
        if self._mission is None:
            return None
        points = [point for point in self._mission.points if point.enabled]
        if not points:
            return None
        selected_index = self._selected_point_index
        selected_point: MeasurementPoint | None = None
        selected_point_index = 0
        if isinstance(selected_index, int) and 0 <= selected_index < len(self._mission.points):
            candidate = self._mission.points[selected_index]
            if candidate.enabled:
                selected_point = candidate
                selected_point_index = selected_index
        if selected_point is None:
            active_points = self._active_start_points()
            selected_active_index = self._selected_start_point_index()
            if selected_active_index < 0 or selected_active_index >= len(active_points):
                selected_active_index = 0
            selected_point_index, selected_point = active_points[selected_active_index]
        return PointExecutionContext(
            mission_name=self._mission.name,
            cycle=0,
            point_index=selected_point_index,
            global_index=len(self._records),
            point=selected_point,
        )

    def _run_manual_measurement_thread(self) -> None:
        point_context = self._manual_measurement_point_context()
        if point_context is None:
            self.after(
                0,
                lambda: messagebox.showwarning(
                    "Manuelle Messung",
                    "Die Mission enthält keine aktiven Punkte.",
                    parent=self,
                ),
            )
            self.after(0, self._on_manual_measurement_finished)
            return
        measurement_service = MissionRxMeasurementService(
            app=self.master,
            on_status=self._on_stage_update,
            on_operator_message=self._append_validation,
            review_measurement=self._review_measurement,
            enable_lidar_reference=bool(self.lidar_reference_enabled_var.get()),
            lidar_topic=self._runtime_config.lidar_topic,
            lidar_timeout_s=self._runtime_config.lidar_reference_timeout_s,
            robot_host=self._runtime_config.robot_host,
            remote_ros_env_cmd=self._runtime_config.remote_ros_env_cmd,
            remote_ros_setup=self._runtime_config.remote_ros_setup,
            fastdds_profiles_file=self._runtime_config.fastdds_profiles_file,
        )
        payload: dict[str, Any] = {
            "global_index": point_context.global_index,
            "point_index": point_context.point_index,
            "point": self._serialize_point(point_context.point),
            "navigation": {"state": "manual"},
            "measurement": {"status": "failed", "result": {}},
            "error": None,
        }
        try:
            measurement_result = measurement_service.trigger(point_context)
            payload["measurement"] = {
                "status": "succeeded",
                "result": measurement_result,
            }
        except Exception as exc:
            payload["measurement"] = {"status": "failed", "result": {}}
            payload["error"] = str(exc)
        self.after(0, self._on_record, payload)
        self.after(0, self._on_manual_measurement_finished)

    def _on_manual_measurement_finished(self) -> None:
        self._set_run_buttons(running=False, paused=False)
        self._manual_measurement_thread = None
        self._update_live_label()

    def _ensure_transmitter_before_run(self) -> bool:
        is_active_fn = getattr(self.master, "is_transmitter_active_for_mission", None)
        transmitter_active = bool(is_active_fn()) if callable(is_active_fn) else bool(getattr(self.master, "_tx_running", False))
        if transmitter_active:
            return True

        decision = messagebox.askyesnocancel(
            "Transmitter inaktiv",
            "Der Transmitter ist aktuell nicht aktiv.\n\n"
            "Ja: Transmitter aktivieren und auf 'TX (Replay): playback started.' warten.\n"
            "Nein: Mission trotzdem ohne aktiven Transmitter starten.\n"
            "Abbrechen: Run-Start abbrechen.",
            parent=self,
        )
        if decision is None:
            self._append_validation("ℹ️ Run-Start abgebrochen: Transmitter-Entscheidung abgebrochen.")
            return False
        if decision is False:
            self._append_validation("⚠️ Mission startet ohne aktiven Transmitter (Operator-Entscheidung).")
            return True

        activate_fn = getattr(self.master, "activate_transmitter_for_mission", None)
        if not callable(activate_fn):
            messagebox.showerror(
                "Transmitter-Start fehlgeschlagen",
                "Transmitter konnte nicht automatisch aktiviert werden (Funktion nicht verfügbar).",
                parent=self,
            )
            self._append_validation("❌ Run-Start blockiert: TX-Aktivierung ist im Hauptfenster nicht verfügbar.")
            return False

        self._append_validation("ℹ️ Aktiviere Transmitter für Missionsstart ...")
        ok, detail = activate_fn()
        if not ok:
            messagebox.showerror(
                "Transmitter-Start fehlgeschlagen",
                "Transmitter konnte nicht rechtzeitig aktiviert werden.\n\n"
                f"Details: {detail}",
                parent=self,
            )
            self._append_validation(f"❌ Run-Start blockiert: TX-Aktivierung fehlgeschlagen ({detail}).")
            return False
        self._append_validation("✅ Transmitter aktiv ('TX (Replay): playback started.'). Mission startet.")
        return True

    def _start_live_label_ticker(self) -> None:
        if self._live_label_ticker_active:
            return
        self._live_label_ticker_active = True
        self._schedule_live_label_ticker()

    def _schedule_live_label_ticker(self) -> None:
        if not self._live_label_ticker_active:
            return
        self._live_label_ticker_job = self.after(LIVE_LABEL_TICKER_INTERVAL_MS, self._run_live_label_ticker)

    def _run_live_label_ticker(self) -> None:
        self._live_label_ticker_job = None
        if not self._live_label_ticker_active:
            return
        if bool(self.live_preview_enabled_var.get()):
            now = time.time()
            pose_age_s = None
            if self._live_position_received_at is not None:
                pose_age_s = now - self._live_position_received_at
            last_redraw_age_s = None
            if self._last_live_redraw_ts is not None:
                last_redraw_age_s = now - self._last_live_redraw_ts
            if (
                pose_age_s is not None
                and pose_age_s >= LIVE_PREVIEW_FALLBACK_REDRAW_AFTER_S
                and (last_redraw_age_s is None or last_redraw_age_s >= LIVE_PREVIEW_FALLBACK_REDRAW_AFTER_S)
            ):
                self.request_live_redraw()
        self._update_live_label()
        self._schedule_live_label_ticker()

    def _stop_live_label_ticker(self) -> None:
        self._live_label_ticker_active = False
        if self._live_label_ticker_job is None:
            return
        try:
            self.after_cancel(self._live_label_ticker_job)
        except Exception:
            pass
        self._live_label_ticker_job = None

    def request_live_redraw(self, *, include_echo: bool = True) -> None:
        if not include_echo:
            self.request_live_marker_redraw()
            return
        if self._live_redraw_pending or not bool(self.live_preview_enabled_var.get()):
            return
        delay_ms = 0
        now = time.time()
        if self._last_live_redraw_ts is not None:
            min_frame_interval_s = 1.0 / float(LIVE_PREVIEW_TARGET_FPS)
            elapsed_s = now - self._last_live_redraw_ts
            if elapsed_s < min_frame_interval_s:
                delay_ms = max(1, int((min_frame_interval_s - elapsed_s) * 1000.0))
        self._live_redraw_pending = True
        self._live_redraw_job = self.after(delay_ms, self._run_live_redraw)

    def _run_live_redraw(self) -> None:
        self._live_redraw_job = None
        self._live_redraw_pending = False
        if not bool(self.live_preview_enabled_var.get()):
            return
        self._draw_live_overlay_layer()
        self._last_live_redraw_ts = time.time()

    def request_live_marker_redraw(self) -> None:
        if self._live_marker_redraw_pending:
            return
        self._live_marker_redraw_pending = True
        self._live_marker_redraw_job = self.after(0, self._run_live_marker_redraw)

    def _run_live_marker_redraw(self) -> None:
        self._live_marker_redraw_job = None
        self._live_marker_redraw_pending = False
        self._draw_live_marker()

    def _cancel_live_redraw(self) -> None:
        self._live_redraw_pending = False
        if self._live_redraw_job is None:
            pass
        else:
            try:
                self.after_cancel(self._live_redraw_job)
            except Exception:
                pass
            self._live_redraw_job = None
        self._live_marker_redraw_pending = False
        if self._live_marker_redraw_job is None:
            return
        try:
            self.after_cancel(self._live_marker_redraw_job)
        except Exception:
            pass
        self._live_marker_redraw_job = None

    def _review_measurement(self, *, point_context, output_file: str) -> dict[str, object]:  # type: ignore[no-untyped-def]
        manual_review_enabled = bool(self.manual_review_enabled_var.get())
        point_label = f"Punktindex {point_context.global_index}"
        review_fn = getattr(self.master, "review_measurement_for_mission", None)
        if not manual_review_enabled and callable(review_fn):
            try:
                review_result = review_fn(
                    point_label=point_label,
                    output_file=output_file,
                    auto_approve=True,
                )
            except TypeError:
                review_result = review_fn(point_label=point_label, output_file=output_file)
            except Exception:
                review_result = {"approved": True, "reason": "", "detail": ""}
            if isinstance(review_result, dict):
                result_payload = dict(review_result)
                result_payload["approved"] = True
                result_payload["reason"] = ""
                detail = result_payload.get("detail")
                result_payload["detail"] = detail.strip() if isinstance(detail, str) else ""
                return result_payload
            return {"approved": True, "reason": "", "detail": ""}
        if not manual_review_enabled:
            return {"approved": True, "reason": "", "detail": ""}
        if not callable(review_fn):
            detail = "Mission-Review nicht verfügbar; Messung wird verworfen."
            self._append_validation(f"⚠️ {detail}")
            return {
                "approved": False,
                "reason": REVIEW_REASON_REVIEW_UNAVAILABLE,
                "detail": detail,
            }
        try:
            review_result = review_fn(point_label=point_label, output_file=output_file)
        except Exception as exc:
            detail = f"Review-Aufruf fehlgeschlagen: {exc}"
            self._append_validation(f"⚠️ {detail}")
            return {
                "approved": False,
                "reason": REVIEW_REASON_REVIEW_EXCEPTION,
                "detail": detail,
            }

        if not isinstance(review_result, dict):
            approved = bool(review_result)
            return {
                "approved": approved,
                "reason": REVIEW_REASON_OPERATOR_REJECTED if not approved else "",
                "detail": "",
            }

        approved = bool(review_result.get("approved"))
        reason = review_result.get("reason")
        detail = review_result.get("detail")
        reason_text = normalize_review_reason(reason, default=REVIEW_REASON_OPERATOR_REJECTED) if not approved else ""
        detail_text = detail.strip() if isinstance(detail, str) else ""
        result_payload = dict(review_result)
        result_payload["approved"] = approved
        result_payload["reason"] = reason_text
        result_payload["detail"] = detail_text
        return result_payload

    def _confirm_measurement_after_navigation_failure(self, point_context, navigation_state: str) -> bool | str:  # type: ignore[no-untyped-def]
        decision_ready = threading.Event()
        decision = {"action": "skip"}

        def _ask_operator() -> None:
            detail = (
                f"Navigation zu Punktindex {point_context.global_index} ist fehlgeschlagen "
                f"({navigation_state}).\n\n"
                "Ja: Messung an aktueller Position durchführen\n"
                "Nein: Messung überspringen\n"
                "Abbrechen: Navigation erneut versuchen"
            )
            response = messagebox.askyesnocancel(
                "Navigation fehlgeschlagen",
                detail,
                parent=self,
            )
            if response is None:
                decision["action"] = "retry_navigation"
            elif response:
                decision["action"] = "measure"
            else:
                decision["action"] = "skip"
            decision_ready.set()

        try:
            self.after(0, _ask_operator)
        except tk.TclError:
            return False
        decision_ready.wait()
        action = decision["action"]
        if action == "retry_navigation":
            self._append_validation(
                f"⚠️ Navigation zu Punktindex {point_context.global_index} fehlgeschlagen ({navigation_state}); Navigation wird erneut versucht."
            )
            return "retry_navigation"
        if action == "measure":
            self._append_validation(
                f"⚠️ Navigation zu Punktindex {point_context.global_index} fehlgeschlagen ({navigation_state}); Messung wird trotzdem ausgeführt."
            )
            return True

        self._append_validation(
            f"⚠️ Navigation zu Punktindex {point_context.global_index} fehlgeschlagen ({navigation_state}); Messung wird übersprungen."
        )
        return False

    def _get_crosscorr_reference_for_mission(self) -> Any:
        get_reference = getattr(self.master, "_get_crosscorr_reference", None)
        if callable(get_reference):
            try:
                reference_payload = get_reference()
            except Exception:
                reference_payload = None
            if isinstance(reference_payload, tuple) and reference_payload:
                reference_payload = reference_payload[0]
            if self._has_crosscorr_reference_data(reference_payload):
                return reference_payload

        persisted_reference = self._load_persisted_tx_reference()
        if self._has_crosscorr_reference_data(persisted_reference):
            return persisted_reference
        return getattr(self.master, "tx_data", None)

    def _load_persisted_tx_reference(self) -> np.ndarray:
        tx_file_widget = getattr(self.master, "tx_file", None)
        tx_file_getter = getattr(tx_file_widget, "get", None)
        if not callable(tx_file_getter):
            return np.array([], dtype=np.complex64)
        tx_file = str(tx_file_getter() or "").strip()
        if not tx_file:
            return np.array([], dtype=np.complex64)
        candidates = [Path(tx_file)]
        tx_path = candidates[0]
        suffix = "_zeros"
        if tx_path.stem.endswith(suffix):
            candidates.append(tx_path.with_name(f"{tx_path.stem[: -len(suffix)]}{tx_path.suffix}"))
        for path in candidates:
            try:
                raw = np.fromfile(path, dtype=np.int16)
            except Exception:
                continue
            if raw.size < 2:
                continue
            if raw.size % 2 != 0:
                raw = raw[:-1]
            if raw.size == 0:
                continue
            iq = raw.reshape(-1, 2).astype(np.float32)
            return iq[:, 0] + 1j * iq[:, 1]
        return np.array([], dtype=np.complex64)

    @staticmethod
    def _has_crosscorr_reference_data(reference_data: Any) -> bool:
        if reference_data is None:
            return False
        size = getattr(reference_data, "size", None)
        if isinstance(size, int):
            return size > 0
        try:
            return len(reference_data) > 0  # type: ignore[arg-type]
        except Exception:
            return False

    def _check_run_prerequisites(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if not any(point.enabled for point in self._mission_points):
            reasons.append("Keine aktiven Wegpunkte vorhanden. Bitte mindestens einen Punkt aktivieren.")
        reasons.extend(self._runtime_guard_reasons())
        if not self._is_test_run_enabled() and bool(self.manual_review_enabled_var.get()):
            review_fn = getattr(self.master, "review_measurement_for_mission", None)
            if not callable(review_fn):
                reasons.append(
                    "Review-Funktion ist nicht verfügbar (self.master.review_measurement_for_mission ist nicht callable)."
                )
            reference_data = self._get_crosscorr_reference_for_mission()
            if not self._has_crosscorr_reference_data(reference_data):
                reasons.append(
                    "TX-Referenzdaten für Crosscorrelation fehlen. Bitte TX laden (gleiche Quelle wie _get_crosscorr_reference)."
                )
        return len(reasons) == 0, reasons

    def _create_navigation_adapter(self) -> NavigationAdapter:
        return NavigationAdapter(
            config=NavigationAdapterConfig(
                robot_host=self._runtime_config.robot_host,
                ros2_namespace=self._runtime_config.ros2_namespace,
                ros2_action_name=self._runtime_config.ros2_action_name,
                remote_ros_env_cmd=self._runtime_config.remote_ros_env_cmd,
                remote_ros_setup=self._runtime_config.remote_ros_setup,
                fastdds_profiles_file=self._runtime_config.fastdds_profiles_file,
                goal_acceptance_timeout_s=self._runtime_config.goal_acceptance_timeout_s,
                goal_reached_timeout_s=self._runtime_config.goal_reached_timeout_s,
                retry_attempts=self._runtime_config.navigation_retry_attempts,
            )
        )

    def _ensure_navigator(self) -> _UiNavigator:
        if self._navigator is None:
            self._navigator = _UiNavigator(
                adapter=self._create_navigation_adapter(),
                on_status=self._on_stage_update,
                on_operator_message=self._append_validation,
            )
        return self._navigator


    def _on_manual_review_toggle_changed(self) -> None:
        self._persist_workflow_state()
        self._refresh_review_ready_indicator()

    def _on_test_run_toggle_changed(self) -> None:
        self._persist_workflow_state()
        self._refresh_review_ready_indicator()

    def _on_live_pose_stream_switch_changed(self) -> None:
        self._persist_workflow_state()
        self._sync_live_pose_stream_state()
        self._update_live_label()
        if hasattr(self, "_map_image_original"):
            self._draw_map_preview()

    def _on_live_preview_switch_changed(self) -> None:
        self._persist_workflow_state()
        self._sync_live_preview_state()
        if hasattr(self, "_map_image_original"):
            self._draw_map_preview()

    def _sync_live_pose_stream_state(self) -> None:
        should_run = bool(self.live_pose_stream_enabled_var.get())
        if should_run:
            navigator = self._ensure_navigator()
            if not self._live_pose_stream_active:
                navigator.start_pose_stream(
                    on_runtime_event=self._on_executor_runtime_event,
                    expected_frame_id=self._expected_live_frame_id(),
                )
                self._live_pose_stream_active = True
            return
        if self._navigator is not None and self._live_pose_stream_active:
            self._navigator.stop_pose_stream()
        self._live_pose_stream_active = False

    def _sync_live_preview_state(self) -> None:
        if bool(self.live_preview_enabled_var.get()):
            self._sync_live_pose_stream_state()
            start_continuous = getattr(self.master, "start_continuous", None)
            if not self._is_continuous_active() and callable(start_continuous):
                try:
                    start_continuous()
                    self._append_validation("ℹ️ Live-Preview aktiv: Continuous-Modus wurde gestartet.")
                except Exception as exc:
                    self._append_validation(f"⚠️ Live-Preview: Continuous-Start fehlgeschlagen ({exc}).")
            return
        self._cancel_live_redraw()
        stop_continuous = getattr(self.master, "stop_continuous", None)
        if self._is_continuous_active() and callable(stop_continuous):
            try:
                stop_continuous()
                self._append_validation("ℹ️ Live-Preview deaktiviert: Continuous-Modus wurde gestoppt.")
            except Exception as exc:
                self._append_validation(f"⚠️ Live-Preview: Continuous-Stop fehlgeschlagen ({exc}).")

    def _resolve_cmd_vel_topic(self) -> str:
        namespace = self._runtime_config.ros2_namespace.strip("/")
        return f"/{namespace}/cmd_vel" if namespace else "/cmd_vel"

    def _build_manual_drive_command(self, *, linear_x: float, angular_z: float) -> list[str]:
        topic = self._resolve_cmd_vel_topic()
        payload = (
            "{linear: {x: %s, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: %s}}"
            % (format(float(linear_x), ".3f"), format(float(angular_z), ".3f"))
        )
        remote_command = " ".join(
            [
                "command -v ros2 >/dev/null 2>&1 || { echo 'TRANSCEIVER_ENV_CHECK_FAILED: ros2 CLI not found in PATH' >&2; exit 70; }",
                "&&",
                "ros2",
                "topic",
                "pub",
                "--once",
                shlex.quote(topic),
                "geometry_msgs/msg/Twist",
                shlex.quote(payload),
            ]
        )
        return Ros2CliNavigationTransport._build_remote_ssh_command(
            robot_host=self._runtime_config.robot_host,
            connect_timeout_s=self._runtime_config.goal_acceptance_timeout_s,
            remote_ros_env_cmd=self._runtime_config.remote_ros_env_cmd.strip(),
            remote_ros_setup=self._runtime_config.remote_ros_setup.strip(),
            fastdds_profiles_file=self._runtime_config.fastdds_profiles_file.strip(),
            remote_command=remote_command,
            diagnostics_label=f"manual_cmd_vel topic={topic}",
        )

    def _queue_manual_drive(self, *, linear_x: float, angular_z: float, label: str) -> None:
        if self._run_thread and self._run_thread.is_alive():
            self._append_validation("⚠️ Manuelles Verfahren ist während eines aktiven Runs deaktiviert.")
            return

        def _worker() -> None:
            if not self._manual_drive_lock.acquire(blocking=False):
                self.after(0, lambda: self._append_validation("ℹ️ Manuelles Verfahren läuft bereits."))
                return
            try:
                command = self._build_manual_drive_command(linear_x=linear_x, angular_z=angular_z)
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=max(3.0, float(self._runtime_config.goal_acceptance_timeout_s) + 1.0),
                )
                stderr = (result.stderr or "").strip().splitlines()
                stdout = (result.stdout or "").strip().splitlines()
                if result.returncode == 0:
                    self.after(0, lambda: self._append_validation(f"ℹ️ Roboter manuell {label} verfahren."))
                    return
                tail = "; ".join((stderr or stdout)[-2:]) if (stderr or stdout) else "ohne Fehlermeldung"
                self.after(
                    0,
                    lambda: self._append_validation(
                        f"⚠️ Manuelles Verfahren ({label}) fehlgeschlagen (rc={result.returncode}): {tail}"
                    ),
                )
            except Exception as exc:
                self.after(0, lambda: self._append_validation(f"⚠️ Manuelles Verfahren ({label}) fehlgeschlagen ({exc})."))
            finally:
                self._manual_drive_lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    def _get_live_preview_echo_distances(self, *, limit: int) -> list[float]:
        if limit <= 0:
            return []
        getter = getattr(self.master, "get_live_echo_distances_for_mission_preview", None)
        if not callable(getter):
            return []
        try:
            payload = getter(limit=limit)
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        distances: list[float] = []
        for value in payload[:limit]:
            if not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0.0:
                continue
            distances.append(numeric)
        return distances

    def _is_continuous_active(self) -> bool:
        cont_thread = getattr(self.master, "_cont_thread", None)
        return bool(cont_thread is not None and cont_thread.is_alive())

    def _is_test_run_enabled(self) -> bool:
        var = getattr(self, "test_run_enabled_var", None)
        getter = getattr(var, "get", None)
        return bool(getter()) if callable(getter) else False

    def _runtime_guard_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self._is_continuous_active():
            reasons.append("Continuous-Modus zuerst stoppen (laufender Continuous-Thread erkannt).")
        is_receive_active = getattr(self.master, "is_receive_active_for_mission", None)
        if callable(is_receive_active):
            if bool(is_receive_active()):
                reasons.append("Laufenden RX-Job beenden (Receive ist aktiv).")
        elif bool(getattr(self.master, "_cmd_running", False)):
            reasons.append("Laufenden RX-Job beenden (_cmd_running ist aktiv).")
        return reasons

    def _refresh_review_ready_indicator(self, prerequisites_ok: bool | None = None) -> None:
        if prerequisites_ok is None:
            prerequisites_ok, _ = self._check_run_prerequisites()
        if self._is_test_run_enabled():
            self.review_ready_var.set("Review: Testlauf ✅")
            return
        if not bool(self.manual_review_enabled_var.get()):
            self.review_ready_var.set("Review: automatisch ✅")
            return
        if prerequisites_ok:
            self.review_ready_var.set("Review: bereit ✅")
        else:
            self.review_ready_var.set("Review: nicht bereit ❌")

    def _run_executor_thread(self) -> None:
        assert self._executor is not None
        try:
            state = self._executor.start()
        except Exception as exc:
            self.after(0, lambda: self._append_validation(f"❌ Run-Thread fehlgeschlagen: {exc}"))
            state = "failed"
        self.after(0, lambda: self._on_run_finished(state))

    def _pause_run(self) -> None:
        if not self._executor:
            return
        try:
            self._executor.pause()
            self._set_run_buttons(running=True, paused=True)
        except RuntimeError as exc:
            messagebox.showwarning("Pause", str(exc), parent=self)

    def _resume_run(self) -> None:
        if not self._executor:
            return
        try:
            self._executor.resume()
            self._set_run_buttons(running=True, paused=False)
        except RuntimeError as exc:
            messagebox.showwarning("Fortsetzen", str(exc), parent=self)

    def _stop_run(self) -> None:
        if self._nav2point_thread and self._nav2point_thread.is_alive() and self._executor is None:
            if self._navigator is None:
                self._append_validation("⚠️ Stop angefordert, aber kein aktives nav2point-Ziel gefunden.")
                self._refresh_stop_button_state()
                return
            self._navigator.cancel_current_goal()
            self._append_validation("Stop angefordert: nav2point-Cancel wurde an die Navigation gesendet.")
            return
        if not self._executor:
            return
        try:
            self._executor.stop()
        except RuntimeError as exc:
            messagebox.showwarning("Stop", str(exc), parent=self)
            return
        self._append_validation(
            "Stop angefordert: warte auf serverseitige Cancel-Bestätigung für das laufende Goal."
        )

    def _set_run_buttons(self, *, running: bool, paused: bool) -> None:
        self.start_btn.configure(state="disabled" if running else "normal")
        manual_measurement_btn = getattr(self, "manual_measurement_btn", None)
        if manual_measurement_btn is not None:
            manual_measurement_btn.configure(state="disabled" if running else "normal")
        self.pause_btn.configure(state="normal" if running and not paused else "disabled")
        self.resume_btn.configure(state="normal" if running and paused else "disabled")
        self._refresh_stop_button_state()

    def _refresh_stop_button_state(self) -> None:
        run_active = bool(self._run_thread is not None and self._run_thread.is_alive())
        manual_measurement_active = bool(
            self._manual_measurement_thread is not None and self._manual_measurement_thread.is_alive()
        )
        nav2point_active = bool(self._nav2point_thread is not None and self._nav2point_thread.is_alive())
        self.stop_btn.configure(state="normal" if run_active or manual_measurement_active or nav2point_active else "disabled")

    def _on_stage_update(self, stage: str, status: str) -> None:
        if stage == "measurement" and status == "measurement_complete":
            self._live_position_at_measurement_start = self._wait_for_valid_live_position_for_measurement_start()
        self.after(0, lambda: self._update_live_label(stage=stage, status=status))

    def _on_executor_runtime_event(self, payload: dict[str, Any]) -> None:
        try:
            self.after(0, lambda: self._handle_executor_runtime_event(payload))
        except tk.TclError:
            return

    def _handle_executor_runtime_event(self, payload: dict[str, Any]) -> None:
        try:
            payload_type = payload.get("type")
            if payload_type == "navigation":
                event = payload.get("event")
                if not isinstance(event, dict):
                    return
                if event.get("type") != "position_update":
                    return
                self._apply_live_position_update(event.get("position"))
                self.request_live_marker_redraw()
                self.request_live_redraw()
                self._update_live_label()
                return
            if payload_type != "pose_stream":
                return
            event = payload.get("event")
            if not isinstance(event, dict):
                return
            event_type = str(event.get("type") or "")
            if event_type == "position_update":
                self._apply_live_position_update(event.get("position"))
                self.request_live_marker_redraw()
                self.request_live_redraw()
                self._update_live_label()
                return
            if event_type == "stream_connected":
                attempt = event.get("attempt")
                self._append_validation(f"ℹ️ Live-Stream verbunden (Versuch {attempt}).")
                self._update_live_label()
                return
            if event_type == "stream_reconnect_wait":
                attempt = event.get("attempt")
                backoff_s = event.get("backoff_s")
                self._append_validation(
                    f"⚠️ Live-Stream getrennt, neuer Verbindungsversuch {attempt} in {backoff_s}s."
                )
                self._update_live_label()
                return
            if event_type == "stream_error":
                detail = str(event.get("message") or "ohne Details")
                attempt = event.get("attempt")
                self._append_validation(
                    f"⚠️ Live-Stream Fehler (Versuch {attempt}): {detail}"
                )
                self._update_live_label()
        except tk.TclError:
            return

    def _apply_live_position_update(self, position: Any) -> None:
        now = time.time()
        self._live_position = position if isinstance(position, dict) else None
        if self._live_position is not None:
            self._live_position_received_at = now
            if self._copy_valid_live_position() is not None:
                self._measurement_start_live_position_event.set()

    def _update_live_label(self, *, stage: str | None = None, status: str | None = None) -> None:
        total = len(self._mission.points) * (self._mission.repeat or 1) if self._mission else 0
        done = len(self._records)
        current_idx = min(done + 1, total) if total > 0 else 0

        nav_status = "idle"
        meas_status = "idle"
        if self._records:
            last = self._records[-1]
            nav_status = str(last.get("navigation", {}).get("state") or nav_status)
            meas_status = str(last.get("measurement", {}).get("status") or meas_status)
        if stage == "navigation" and status:
            nav_status = status
        if stage == "measurement" and status:
            meas_status = status

        remaining = max(0, total - done)
        eta = "-"
        if done > 0 and self._run_started_at is not None:
            elapsed = max(0.001, time.time() - self._run_started_at)
            eta_s = int((elapsed / done) * remaining)
            eta = f"{eta_s}s"

        diagnosis_key, diagnosis_text = self._live_pose_diagnosis()
        self._announce_live_diagnosis_if_changed(diagnosis_key, diagnosis_text)
        pose_age_text = "-"
        if self._live_position_received_at is not None:
            pose_age_s = max(0.0, time.time() - self._live_position_received_at)
            pose_age_text = f"{pose_age_s:.1f}s"
        self.live_var.set(
            f"Punktindex: {current_idx}/{total} | "
            f"Navigation: {nav_status} | Messung: {meas_status} | "
            f"Verbleibend: {remaining} | ETA: {eta} | "
            f"Pose-Alter: {pose_age_text} | Live-Status: {diagnosis_text}"
        )

    def _on_record(self, payload: dict[str, Any]) -> None:
        self._attach_result_table_snapshot(payload)
        self._records.append(payload)
        meas = payload.get("measurement", {})
        result = meas.get("result", {}) if isinstance(meas.get("result"), dict) else {}
        review = result.get("review", {}) if isinstance(result.get("review"), dict) else {}
        review_reason = review.get("reason") if isinstance(review.get("reason"), str) else ""
        review_detail = review.get("detail") if isinstance(review.get("detail"), str) else ""
        error_text = payload.get("error") or ""
        if review_reason:
            error_text = f"{error_text} [{review_reason}]" if error_text else review_reason
        if review_detail:
            error_text = f"{error_text}: {review_detail}" if error_text else review_detail
        combined_status = self._compose_table_outcome(payload, error_text)
        echo_distances = self._format_echo_distances_for_table(result.get("echo_delays"))
        live_position_text = self._format_live_position_for_table(payload)
        live_distance_to_rx = self._format_live_distance_to_rx_for_table(payload)
        self.results_table.insert(
            "",
            "end",
            values=(
                self._format_one_based_index(payload.get("global_index")),
                self._format_one_based_index(payload.get("point_index")),
                live_position_text,
                live_distance_to_rx,
                *echo_distances,
                "Review",
                combined_status,
            ),
        )
        self._update_live_label()

    def _attach_result_table_snapshot(self, payload: dict[str, Any]) -> None:
        if "live_position_at_measurement" not in payload:
            payload["live_position_at_measurement"] = self._live_position_at_measurement_start
            self._live_position_at_measurement_start = None
        payload["map_name"] = self._current_map_name()
        payload["rx_antenna_global_position"] = self._serialize_rx_antenna_global_position()
        payload["result_table"] = {
            "position": self._format_live_position_for_table(payload),
            "abstand": self._format_live_distance_to_rx_for_table(payload),
        }

    def _current_map_name(self) -> str | None:
        map_config_file = self._selected_map_config_file
        if isinstance(map_config_file, str) and map_config_file.strip():
            return Path(map_config_file).name
        if self._selected_map_config is None:
            return None
        return Path(self._selected_map_config.image).name

    @staticmethod
    def _positions_differ(
        left: tuple[float, float] | None,
        right: tuple[float, float] | None,
        *,
        tolerance: float = 1e-6,
    ) -> bool:
        if left is None and right is None:
            return False
        if left is None or right is None:
            return True
        return not (math.isclose(left[0], right[0], abs_tol=tolerance) and math.isclose(left[1], right[1], abs_tol=tolerance))

    @staticmethod
    def _compose_table_outcome(payload: dict[str, Any], error_text: str) -> str:
        nav = payload.get("navigation")
        nav_state = nav.get("state") if isinstance(nav, dict) else None
        nav_status = str(nav_state) if isinstance(nav_state, str) and nav_state.strip() else "-"
        measurement = payload.get("measurement")
        measurement_state = measurement.get("status") if isinstance(measurement, dict) else None
        measurement_status = (
            str(measurement_state)
            if isinstance(measurement_state, str) and measurement_state.strip()
            else "-"
        )
        if nav_status == "succeeded" and measurement_status == "succeeded" and not error_text.strip():
            return "succeeded"

        base = f"navigation {nav_status}, measurement {measurement_status}"
        details = error_text.strip()
        if details:
            return f"{base}: {details}"
        return base

    @staticmethod
    def _derive_table_status(payload: dict[str, Any]) -> str:
        if payload.get("error") is not None:
            return "failed"

        measurement = payload.get("measurement")
        measurement_status = measurement.get("status") if isinstance(measurement, dict) else None
        if measurement_status in {"succeeded", "failed", "skipped"}:
            return str(measurement_status)
        return "succeeded"

    @staticmethod
    def _compose_table_status(status_text: str, error_text: str) -> str:
        status_value = status_text.strip() if isinstance(status_text, str) else ""
        error_value = error_text.strip() if isinstance(error_text, str) else ""
        if status_value and error_value:
            return f"{status_value}: {error_value}"
        return status_value or error_value or "-"

    @staticmethod
    def _format_one_based_index(value: Any) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return str(value + 1) if value >= 0 else str(value)
        return str(value) if value is not None else ""

    @staticmethod
    def _format_echo_distances_for_table(value: Any, *, limit: int = 5) -> tuple[str, ...]:
        if not isinstance(value, list) or limit <= 0:
            return tuple("-" for _ in range(max(0, limit)))

        def _format_distance(distance_value: Any) -> str:
            if not isinstance(distance_value, (int, float)):
                return "-"
            numeric = float(distance_value)
            if numeric.is_integer():
                return str(int(numeric))
            return f"{numeric:.2f}".rstrip("0").rstrip(".")

        formatted = [
            _format_distance(item.get("distance_m")) if isinstance(item, dict) else "-"
            for item in value[:limit]
        ]
        if len(formatted) < limit:
            formatted.extend("-" for _ in range(limit - len(formatted)))
        return tuple(formatted)

    def _format_distance_to_rx_for_table(self, payload: dict[str, Any]) -> str:
        rx_position = self._rx_antenna_global_position
        if rx_position is None:
            return "-"
        point = self._selected_record_point(payload)
        if point is None:
            return "-"
        distance_m = math.hypot(point.x - rx_position[0], point.y - rx_position[1])
        if not math.isfinite(distance_m):
            return "-"
        return f"{distance_m:.2f}".rstrip("0").rstrip(".")

    def _format_live_distance_to_rx_for_table(self, payload: dict[str, Any]) -> str:
        rx_position = self._rx_antenna_global_position
        if rx_position is None:
            return "-"
        position = payload.get("live_position_at_measurement")
        if not isinstance(position, dict):
            return "-"
        x_value = position.get("x")
        y_value = position.get("y")
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            return "-"
        x = float(x_value)
        y = float(y_value)
        if not math.isfinite(x) or not math.isfinite(y):
            return "-"
        distance_m = math.hypot(x - rx_position[0], y - rx_position[1])
        if not math.isfinite(distance_m):
            return "-"
        return f"{distance_m:.2f}".rstrip("0").rstrip(".")

    def _format_position_for_table(self, payload: dict[str, Any]) -> str:
        point = self._selected_record_point(payload)
        if point is None:
            return "-"
        return f"{point.x:.1f},{point.y:.1f}"

    def _format_live_position_for_table(self, payload: dict[str, Any]) -> str:
        position = payload.get("live_position_at_measurement")
        if not isinstance(position, dict):
            return "-"
        x_value = position.get("x")
        y_value = position.get("y")
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            return "-"
        x = float(x_value)
        y = float(y_value)
        if not math.isfinite(x) or not math.isfinite(y):
            return "-"
        return f"{x:.1f},{y:.1f}"

    def _copy_live_position(self) -> dict[str, Any] | None:
        if not isinstance(self._live_position, dict):
            return None
        return dict(self._live_position)

    def _copy_valid_live_position(self) -> dict[str, Any] | None:
        position = self._copy_live_position()
        if not isinstance(position, dict):
            return None
        x_value = position.get("x")
        y_value = position.get("y")
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            return None
        x = float(x_value)
        y = float(y_value)
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return position

    def _wait_for_valid_live_position_for_measurement_start(self) -> dict[str, Any] | None:
        valid_position = self._copy_valid_live_position()
        if valid_position is not None:
            return valid_position
        self._measurement_start_live_position_event.clear()
        deadline = time.time() + MEASUREMENT_START_LIVE_POSITION_WAIT_TIMEOUT_S
        while time.time() < deadline:
            wait_timeout = min(
                MEASUREMENT_START_LIVE_POSITION_WAIT_INTERVAL_S,
                max(0.0, deadline - time.time()),
            )
            if wait_timeout <= 0.0:
                break
            self._measurement_start_live_position_event.wait(timeout=wait_timeout)
            valid_position = self._copy_valid_live_position()
            if valid_position is not None:
                return valid_position
            self._measurement_start_live_position_event.clear()
        return self._copy_valid_live_position()

    def _on_run_finished(self, state: str) -> None:
        self._stop_live_label_ticker()
        self._sync_live_pose_stream_state()
        self._sync_live_preview_state()
        self._set_run_buttons(running=False, paused=False)
        completion_substatus = None
        if self._run_log_dir is not None:
            summary_path = self._run_log_dir / "run-summary.json"
            if summary_path.exists():
                try:
                    import json

                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    completion_substatus = summary.get("completion_substatus")
                except Exception:
                    completion_substatus = None
        self._append_validation(
            f"Run beendet: {_map_executor_state_to_ui_text(state, completion_substatus)}"
        )
        if self._records:
            last_error = self._records[-1].get("error")
            if last_error:
                self._append_validation(f"Operator-Hinweis: {last_error}")
        self._update_live_label()

    def _on_window_close(self) -> None:
        self._stop_live_label_ticker()
        self._cancel_live_redraw()
        self.live_preview_enabled_var.set(False)
        self._sync_live_preview_state()
        if self._navigator is not None:
            self._navigator.stop_pose_stream()
            self._navigator = None
        self._live_pose_stream_active = False
        self.destroy()


    def _clear_results_table(self) -> None:
        self.results_table.delete(*self.results_table.get_children())
        self._records = []
        self._selected_result_index = None
        self._selected_result_indices = ()
        self._update_results_selection_diagnostics()

    def _export_logs(self) -> None:
        if self._run_log_dir is None or not self._run_log_dir.exists():
            messagebox.showinfo("Export", "Noch keine Run-Logs vorhanden.", parent=self)
            return

        destination = filedialog.asksaveasfilename(
            title="Run-Logs exportieren",
            parent=self,
            defaultextension=".zip",
            initialfile=f"{self._run_log_dir.name}.zip",
            filetypes=[("ZIP", "*.zip")],
        )
        if not destination:
            return

        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(self._run_log_dir.glob("*.json")):
                zf.write(path, arcname=path.name)
        messagebox.showinfo("Export", f"Exportiert nach:\n{destination}", parent=self)

    def _import_logs(self) -> None:
        source = filedialog.askopenfilename(
            title="Run-Logs importieren",
            parent=self,
            filetypes=[("ZIP", "*.zip")],
        )
        if not source:
            return

        try:
            with zipfile.ZipFile(source, "r") as zf:
                json_members = sorted(
                    member
                    for member in zf.namelist()
                    if member.endswith(".json") and "/" not in member.strip("/")
                )
                point_members = [member for member in json_members if member != "run-summary.json"]
                if not point_members:
                    messagebox.showerror("Import", "Keine Punkt-Logs im ZIP gefunden.", parent=self)
                    return

                imported_records: list[dict[str, Any]] = []
                for member in point_members:
                    payload_raw = zf.read(member).decode("utf-8")
                    payload = json.loads(payload_raw)
                    if isinstance(payload, dict):
                        imported_records.append(payload)
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
            messagebox.showerror("Import", f"Import fehlgeschlagen:\n{exc}", parent=self)
            return

        if not imported_records:
            messagebox.showerror("Import", "Keine gültigen Punkt-Logs gefunden.", parent=self)
            return

        imported_map_name = next(
            (
                str(payload.get("map_name")).strip()
                for payload in imported_records
                if isinstance(payload.get("map_name"), str) and str(payload.get("map_name")).strip()
            ),
            "",
        )
        current_map_name = self._current_map_name() or ""
        if imported_map_name and current_map_name and imported_map_name != current_map_name:
            messagebox.showwarning(
                "Import",
                (
                    "Abweichender Map-Name im Import erkannt.\n"
                    f"Import: {imported_map_name}\n"
                    f"Aktuell: {current_map_name}"
                ),
                parent=self,
            )

        imported_rx_position = next(
            (
                parsed
                for payload in imported_records
                for parsed in [self._parse_rx_antenna_global_position(payload.get("rx_antenna_global_position"))]
                if parsed is not None
            ),
            None,
        )
        if self._positions_differ(imported_rx_position, self._rx_antenna_global_position):
            update_rx_position = messagebox.askyesno(
                "Import",
                (
                    "RX-Position aus Import weicht von der aktuellen Position ab.\n"
                    "Soll die RX-Position auf den Importwert aktualisiert werden?"
                ),
                parent=self,
            )
            if update_rx_position and imported_rx_position is not None:
                self._set_rx_antenna_position(
                    x=imported_rx_position[0],
                    y=imported_rx_position[1],
                )

        self._clear_results_table()
        for payload in imported_records:
            self._on_record(payload)

        self._append_validation(
            f"✅ Run-Logs importiert: {len(imported_records)} Messpunkte aus {Path(source).name}"
        )
