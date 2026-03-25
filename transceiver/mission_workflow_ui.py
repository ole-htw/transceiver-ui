from __future__ import annotations

import threading
import time
import zipfile
from dataclasses import replace
from datetime import datetime
import math
from fractions import Fraction
from pathlib import Path
import json
import re
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
)
from .mission_measurement_service import (
    MissionRxMeasurementService,
    REVIEW_REASON_OPERATOR_REJECTED,
    REVIEW_REASON_REVIEW_EXCEPTION,
    REVIEW_REASON_REVIEW_UNAVAILABLE,
    normalize_review_reason,
)
from .navigation_adapter import (
    NavigationAdapter,
    NavigationAdapterConfig,
    NavigationEvent,
    Ros2CliPoseStreamTransport,
)

MISSION_WORKFLOW_STATE_FILE = Path(__file__).with_name("mission_workflow_state.json")
LIVE_LABEL_TICKER_INTERVAL_MS = 250
AUTO_STOP_CONTINUOUS_BEFORE_RUN = True


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

class _UiNavigator:
    def __init__(self, *, adapter: NavigationAdapter, on_status, on_operator_message) -> None:
        self._adapter = adapter
        self._on_status = on_status
        self._on_operator_message = on_operator_message
        self._pose_stream = Ros2CliPoseStreamTransport()

    def start_pose_stream(
        self,
        *,
        on_runtime_event: Callable[[dict[str, Any]], None],
    ) -> None:
        self._pose_stream.start(config=self._adapter.config, on_event=on_runtime_event)

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
        latest_position: dict[str, Any] | None = None
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
                payload = event.data if isinstance(event.data, dict) else {}
                position = payload if event.type == "position" else payload.get("position")
                with latest_position_lock:
                    latest_position = position if isinstance(position, dict) else None

        state = self._adapter.navigate_to_point(point, timeout_s=timeout_s, on_event=_on_event)
        polling_active.clear()
        polling_thread.join(timeout=1.0)
        _emit_position_update()
        if state != "succeeded":
            self._on_status("navigation", state)
        return state

    def cancel_current_goal(self) -> None:
        self._adapter.cancel_current_goal()


class MissionWorkflowWindow(ctk.CTkToplevel):
    def __init__(self, parent: ctk.CTk) -> None:
        super().__init__(parent)
        self.title("Mission Workflow")
        self.geometry("1100x700")
        self.minsize(980, 640)

        self._mission: MeasurementMission | None = None
        self._executor: MeasurementRunExecutor | None = None
        self._navigator: _UiNavigator | None = None
        self._run_thread: threading.Thread | None = None
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

        self._build_ui()
        self._restore_workflow_state()
        self.after_idle(self._stabilize_initial_geometry)
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

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
        self.point_z_var = tk.StringVar(value="0.0")
        self.point_yaw_var = tk.StringVar(value="0.0")

        self._labeled_entry(points_editor, row=0, column=1, label="Name", variable=self.point_name_var, width=120)
        self._labeled_entry(points_editor, row=0, column=2, label="X", variable=self.point_x_var, width=90)
        self._labeled_entry(points_editor, row=0, column=3, label="Y", variable=self.point_y_var, width=90)
        self._labeled_entry(points_editor, row=0, column=4, label="Z", variable=self.point_z_var, width=90)
        self._labeled_entry(points_editor, row=0, column=5, label="Yaw", variable=self.point_yaw_var, width=90)
        ctk.CTkButton(points_editor, text="Punkt hinzufügen", command=self._add_point).grid(row=0, column=6, padx=(8, 3))
        ctk.CTkButton(points_editor, text="Auswahl entfernen", command=self._remove_selected_point).grid(row=0, column=7, padx=3)
        ctk.CTkButton(points_editor, text="▲", width=36, command=self._move_selected_point_up).grid(row=0, column=8, padx=3)
        ctk.CTkButton(points_editor, text="▼", width=36, command=self._move_selected_point_down).grid(row=0, column=9, padx=(3, 8), sticky="w")
        ctk.CTkButton(
            points_editor,
            text="Aktivieren/Deaktivieren",
            command=self._toggle_selected_point_enabled,
        ).grid(row=0, column=10, padx=(3, 8), sticky="w")

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
        ctk.CTkButton(map_frame, text="Map-Config wählen", command=self._select_map_config_file).grid(
            row=0, column=0, sticky="e", padx=8, pady=(8, 2)
        )
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
        self._map_image_original: tk.PhotoImage | None = None
        self._map_image_preview: tk.PhotoImage | None = None
        self._map_preview_scale: tuple[float, float] = (1.0, 1.0)
        self._map_preview_offset: tuple[float, float] = (0.0, 0.0)
        self._map_canvas_image_id: int | None = None
        self._map_marker_ids: list[int] = []
        self._map_image_size: tuple[int, int] | None = None
        self._live_position: dict[str, Any] | None = None
        self._live_position_received_at: float | None = None
        self._selected_point_index: int | None = None
        self._last_live_diagnosis_key: str | None = None
        self._emit_live_diagnostics_to_validation = True

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
        point_columns = ("idx", "enabled", "id", "name", "x", "y", "z", "yaw")
        self.points_table = ttk.Treeview(points_table_frame, columns=point_columns, show="headings", height=6)
        self.points_table.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        for key, title in {
            "idx": "#",
            "enabled": "Aktiv",
            "id": "ID",
            "name": "Name",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "yaw": "Yaw",
        }.items():
            self.points_table.heading(key, text=title)
            self.points_table.column(key, stretch=True, width=85 if key == "enabled" else 95)
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
        for col in range(4):
            controls.columnconfigure(col, weight=0)
        controls.columnconfigure(3, weight=1)
        controls.rowconfigure(3, weight=1)

        ctk.CTkLabel(controls, text="5) Laufsteuerung").grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 4))
        self.review_ready_var = tk.StringVar(value="Review: nicht geprüft")
        ctk.CTkLabel(controls, textvariable=self.review_ready_var, anchor="w").grid(
            row=0, column=3, padx=(8, 8), pady=(8, 4), sticky="e"
        )
        self.start_btn = ctk.CTkButton(controls, text="Start", command=self._start_run)
        self.start_btn.grid(row=1, column=0, padx=(8, 3), pady=(0, 4), sticky="w")
        ctk.CTkLabel(controls, text="Start ab Punkt").grid(row=1, column=1, padx=(10, 2), pady=(0, 4), sticky="e")
        self.start_point_var = tk.StringVar(value="1")
        self.start_point_combo = ctk.CTkComboBox(
            controls,
            width=150,
            values=["1"],
            variable=self.start_point_var,
            state="readonly",
            command=lambda _value: self._persist_workflow_state(),
        )
        self.start_point_combo.grid(row=1, column=2, columnspan=2, padx=(0, 8), pady=(0, 4), sticky="w")
        self.pause_btn = ctk.CTkButton(controls, text="Pause", command=self._pause_run, state="disabled")
        self.pause_btn.grid(row=2, column=0, padx=(8, 3), pady=3, sticky="w")
        self.resume_btn = ctk.CTkButton(controls, text="Fortsetzen", command=self._resume_run, state="disabled")
        self.resume_btn.grid(row=2, column=1, padx=3, pady=3, sticky="w")
        self.stop_btn = ctk.CTkButton(controls, text="Stop", command=self._stop_run, state="disabled")
        self.stop_btn.grid(row=2, column=2, padx=3, pady=3, sticky="w")
        ctk.CTkButton(controls, text="Run-Logs exportieren", command=self._export_logs).grid(
            row=2, column=3, padx=(10, 8), pady=3, sticky="w"
        )

        self.live_var = tk.StringVar(value="Punkt: - | Navigation: idle | Messung: idle | Verbleibend: - | Live-Status: Karte nicht geladen")
        ctk.CTkLabel(controls, textvariable=self.live_var, anchor="w", justify="left").grid(
            row=3, column=0, columnspan=4, sticky="nsew", padx=8, pady=(4, 8)
        )

        table_frame = ctk.CTkFrame(self)
        table_frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=(0, 10))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = ("idx", "point", "nav", "measurement", "echo_delays", "status", "error")
        self.results_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        self.results_table.grid(row=0, column=0, sticky="nsew")
        headings = {
            "idx": "Punktindex",
            "point": "Punkt",
            "nav": "Navigation",
            "measurement": "Messung",
            "echo_delays": "LOS->Echo",
            "status": "Gesamtstatus",
            "error": "Fehler",
        }
        for key, title in headings.items():
            self.results_table.heading(key, text=title)
            self.results_table.column(key, stretch=True, width=110)
        self.results_table.column("echo_delays", width=190)
        self.results_table.column("error", width=260)

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.results_table.configure(yscrollcommand=scroll.set)

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

    @staticmethod
    def _labeled_entry(
        parent: ctk.CTkFrame,
        *,
        row: int,
        column: int,
        label: str,
        variable: tk.StringVar,
        width: int,
    ) -> None:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=column, padx=3, pady=3)
        ctk.CTkLabel(wrap, text=label).pack(side="top", anchor="w")
        ctk.CTkEntry(wrap, textvariable=variable, width=width).pack(side="top")

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

    def _refresh_map_section(self) -> None:
        self._map_image_original = None
        self._map_image_preview = None
        self._map_preview_scale = (1.0, 1.0)
        self._map_preview_offset = (0.0, 0.0)
        self._map_canvas_image_id = None
        self._map_marker_ids = []
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

    @staticmethod
    def _resize_photo_to_cover(photo: tk.PhotoImage, *, target_width: int, target_height: int) -> tk.PhotoImage:
        if target_width <= 1 or target_height <= 1:
            return photo

        width = photo.width()
        height = photo.height()
        if width <= 0 or height <= 0:
            return photo

        # Fill the available canvas area with the map while preserving aspect ratio.
        # This intentionally allows clipping in one axis so there is no free space
        # in both axes at the same time.
        scale = max(target_width / width, target_height / height)
        if abs(scale - 1.0) < 0.01:
            return photo

        ratio = Fraction(scale).limit_denominator(32)
        preview = photo.zoom(ratio.numerator, ratio.numerator).subsample(ratio.denominator, ratio.denominator)
        return preview

    def _select_map_config_file(self) -> None:
        selected_file = filedialog.askopenfilename(
            title="Map-Config auswählen",
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
            messagebox.showwarning("Map-Config ungültig", str(exc))
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
        preview = self._resize_photo_to_cover(original, target_width=canvas_width, target_height=canvas_height)
        offset_x = (canvas_width - preview.width()) / 2.0
        offset_y = (canvas_height - preview.height()) / 2.0

        self._map_image_preview = preview
        self._map_preview_scale = (preview.width() / original.width(), preview.height() / original.height())
        self._map_preview_offset = (offset_x, offset_y)

        self.map_preview_canvas.delete("all")
        self._map_canvas_image_id = self.map_preview_canvas.create_image(offset_x, offset_y, anchor="nw", image=preview)
        self._draw_mission_markers()
        self._draw_live_marker()

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
            marker_id = self.map_preview_canvas.create_oval(
                px - 4,
                py - 4,
                px + 4,
                py + 4,
                fill="#00d26a",
                outline="#0d1016",
                width=1,
            )
            self._map_marker_ids.append(marker_id)
            if index == self._selected_point_index:
                self._highlight_marker(marker_id)

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
            return
        expected_frame_id = self._expected_live_frame_id()
        received_frame_id = position.get("frame_id")
        if (
            isinstance(received_frame_id, str)
            and received_frame_id.strip()
            and received_frame_id != expected_frame_id
        ):
            return
        x_value = position.get("x")
        y_value = position.get("y")
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            return

        scale_x, scale_y = self._map_preview_scale
        offset_x, offset_y = self._map_preview_offset
        map_pixel = self._world_to_map_pixel(
            x=float(x_value),
            y=float(y_value),
            image_height=original.height(),
        )
        if map_pixel is None:
            return
        if not self._is_pixel_inside_map(
            map_pixel[0],
            map_pixel[1],
            width=original.width(),
            height=original.height(),
        ):
            return
        pixel_coordinates = (map_pixel[0] * scale_x + offset_x, map_pixel[1] * scale_y + offset_y)
        px, py = pixel_coordinates
        radius = 6
        self.map_preview_canvas.create_oval(
            px - radius,
            py - radius,
            px + radius,
            py + radius,
            fill="#ff4d6d",
            outline="#ffffff",
            width=1,
        )
        yaw_value = position.get("yaw")
        if isinstance(yaw_value, (int, float)):
            heading_length = 14
            end_x = px + math.cos(float(yaw_value)) * heading_length
            end_y = py - math.sin(float(yaw_value)) * heading_length
            self.map_preview_canvas.create_line(
                px,
                py,
                end_x,
                end_y,
                fill="#ffccd5",
                width=2,
                arrow=tk.LAST,
            )

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
        point_payload = {
            "id": self._generate_unique_point_id(),
            "name": self.point_name_var.get().strip() or None,
            "x": self.point_x_var.get().strip(),
            "y": self.point_y_var.get().strip(),
            "z": self.point_z_var.get().strip() or 0.0,
            "yaw": self.point_yaw_var.get().strip(),
            "enabled": True,
        }
        try:
            mission = measurement_mission_from_dict(
                {"name": "point-check", "points": [point_payload], "repeat": 1}
            )
        except Exception as exc:
            messagebox.showwarning("Messpunkt ungültig", str(exc))
            return

        point = mission.points[0]
        self._mission_points.append(point)
        self._sync_validated_mission_points()
        self._refresh_points_table()
        self._persist_workflow_state()
        self._append_validation(
            f"✅ Punkt hinzugefügt: {point.id or point.name} (x={point.x:.2f}, y={point.y:.2f}, z={point.z:.2f}, yaw={point.yaw})"
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
        column_id = self.points_table.identify_column(event.x)
        if column_id != "#2":
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
                    idx,
                    "✓" if point.enabled else "✗",
                    point.id or "-",
                    point.name or "-",
                    f"{point.x:.3f}",
                    f"{point.y:.3f}",
                    f"{point.z:.3f}",
                    "-" if point.yaw is None else f"{point.yaw:.3f}",
                ),
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
        display_name = point.name or point.id or f"Punkt {index + 1}"
        return f"{index + 1}: {display_name}"

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
        }

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
        finally:
            self._is_restoring_workflow_state = False

    def _start_run(self) -> None:
        if self._mission is None:
            messagebox.showwarning("Mission", "Bitte zuerst eine gültige Mission anlegen und validieren.")
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
            )
            self._append_validation("❌ Run-Start blockiert: " + " | ".join(reasons))
            return
        if not self._ensure_transmitter_before_run():
            return
        start_point_index = self._selected_start_point_index()

        self.results_table.delete(*self.results_table.get_children())
        self._records = []
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
            self.after(0, self._on_record, payload)

        nav_adapter = NavigationAdapter(
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

        self._navigator = _UiNavigator(
            adapter=nav_adapter,
            on_status=self._on_stage_update,
            on_operator_message=self._append_validation,
        )

        self._executor = MeasurementRunExecutor(
            mission=self._mission,
            navigator=self._navigator,
            measurement_service=MissionRxMeasurementService(
                app=self.master,
                on_status=self._on_stage_update,
                on_operator_message=self._append_validation,
                review_measurement=self._review_measurement,
            ),
            persist_result=_persist,
            run_log_store=store,
            on_runtime_event=self._on_executor_runtime_event,
            config=MeasurementRunExecutorConfig(
                on_point_error="stop",
                goal_reached_timeout_s=self._runtime_config.goal_reached_timeout_s,
                navigation_retry_attempts=self._runtime_config.navigation_retry_attempts,
                start_point_index=start_point_index,
            ),
        )

        self._navigator.start_pose_stream(on_runtime_event=self._on_executor_runtime_event)
        self._start_live_label_ticker()
        self._set_run_buttons(running=True, paused=False)
        self._run_thread = threading.Thread(target=self._run_executor_thread, daemon=True)
        self._run_thread.start()

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

    def _review_measurement(self, *, point_context, output_file: str) -> dict[str, object]:  # type: ignore[no-untyped-def]
        point_id = point_context.point.id or point_context.point.name or f"point-{point_context.global_index}"
        point_label = f"Punkt {point_context.global_index} ({point_id})"
        review_fn = getattr(self.master, "review_measurement_for_mission", None)
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

    def _is_continuous_active(self) -> bool:
        cont_thread = getattr(self.master, "_cont_thread", None)
        return bool(cont_thread is not None and cont_thread.is_alive())

    def _runtime_guard_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self._is_continuous_active():
            reasons.append("Continuous-Modus zuerst stoppen (laufender Continuous-Thread erkannt).")
        if bool(getattr(self.master, "_cmd_running", False)):
            reasons.append("Laufenden RX-Job beenden (_cmd_running ist aktiv).")
        return reasons

    def _refresh_review_ready_indicator(self, prerequisites_ok: bool | None = None) -> None:
        if prerequisites_ok is None:
            prerequisites_ok, _ = self._check_run_prerequisites()
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
            messagebox.showwarning("Pause", str(exc))

    def _resume_run(self) -> None:
        if not self._executor:
            return
        try:
            self._executor.resume()
            self._set_run_buttons(running=True, paused=False)
        except RuntimeError as exc:
            messagebox.showwarning("Fortsetzen", str(exc))

    def _stop_run(self) -> None:
        if not self._executor:
            return
        try:
            self._executor.stop()
        except RuntimeError as exc:
            messagebox.showwarning("Stop", str(exc))
            return
        self._append_validation(
            "Stop angefordert: warte auf serverseitige Cancel-Bestätigung für das laufende Goal."
        )

    def _set_run_buttons(self, *, running: bool, paused: bool) -> None:
        self.start_btn.configure(state="disabled" if running else "normal")
        self.pause_btn.configure(state="normal" if running and not paused else "disabled")
        self.resume_btn.configure(state="normal" if running and paused else "disabled")
        self.stop_btn.configure(state="normal" if running else "disabled")

    def _on_stage_update(self, stage: str, status: str) -> None:
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
                self._draw_map_preview()
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
                self._draw_map_preview()
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

    def _update_live_label(self, *, stage: str | None = None, status: str | None = None) -> None:
        total = len(self._mission.points) * (self._mission.repeat or 1) if self._mission else 0
        done = len(self._records)
        current_idx = done if done < total else max(0, total - 1)

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
            f"Punktindex: {current_idx}/{max(total - 1, 0)} | "
            f"Navigation: {nav_status} | Messung: {meas_status} | "
            f"Verbleibend: {remaining} | ETA: {eta} | "
            f"Pose-Alter: {pose_age_text} | Live-Status: {diagnosis_text}"
        )

    def _on_record(self, payload: dict[str, Any]) -> None:
        self._records.append(payload)
        point = payload.get("point", {})
        nav = payload.get("navigation", {})
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
        echo_delays_text = self._format_echo_delays_for_table(result.get("echo_delays"))
        self.results_table.insert(
            "",
            "end",
            values=(
                payload.get("global_index", ""),
                point.get("id") or point.get("name") or "-",
                nav.get("state", "-"),
                meas.get("status", "-"),
                echo_delays_text,
                "ok" if payload.get("error") is None else "fehler",
                error_text,
            ),
        )
        self._update_live_label()

    @staticmethod
    def _format_echo_delays_for_table(value: Any) -> str:
        if not isinstance(value, list) or not value:
            return "-"
        rows: list[str] = []
        for idx, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                rows.append(str(item))
                continue
            echo_idx = item.get("echo_index", idx - 1)
            delta_lag = item.get("delta_lag")
            distance = item.get("distance_m")
            base = f"E{echo_idx}: {delta_lag if delta_lag is not None else '--'}"
            if isinstance(distance, (int, float)):
                base += f" ({float(distance):.1f}m)"
            rows.append(base)
        return "; ".join(rows)

    def _on_run_finished(self, state: str) -> None:
        self._stop_live_label_ticker()
        if self._navigator is not None:
            self._navigator.stop_pose_stream()
            self._navigator = None
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
        if self._navigator is not None:
            self._navigator.stop_pose_stream()
            self._navigator = None
        self.destroy()

    def _export_logs(self) -> None:
        if self._run_log_dir is None or not self._run_log_dir.exists():
            messagebox.showinfo("Export", "Noch keine Run-Logs vorhanden.")
            return

        destination = filedialog.asksaveasfilename(
            title="Run-Logs exportieren",
            defaultextension=".zip",
            initialfile=f"{self._run_log_dir.name}.zip",
            filetypes=[("ZIP", "*.zip")],
        )
        if not destination:
            return

        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(self._run_log_dir.glob("*.json")):
                zf.write(path, arcname=path.name)
        messagebox.showinfo("Export", f"Exportiert nach:\n{destination}")
