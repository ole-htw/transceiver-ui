from __future__ import annotations

import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from typing import Any

import customtkinter as ctk

from .app_config import MissionRuntimeConfig
from .measurement_mission import MeasurementMission, MeasurementPoint, measurement_mission_from_dict
from .measurement_run_executor import (
    JsonRunLogStore,
    MeasurementRunExecutor,
    MeasurementRunExecutorConfig,
    PointExecutionContext,
)
from .navigation_adapter import NavigationAdapter, NavigationAdapterConfig, NavigationEvent



class _UiNavigator:
    def __init__(self, *, adapter: NavigationAdapter, on_status, on_operator_message) -> None:
        self._adapter = adapter
        self._on_status = on_status
        self._on_operator_message = on_operator_message

    def navigate_to_point(self, point, *, timeout_s: float):  # type: ignore[no-untyped-def]
        self._on_status("navigation", "running")

        def _on_event(event: NavigationEvent) -> None:
            if event.type in {"connection_error", "aborted", "timeout"}:
                detail = event.message or 'ohne Details'
                if event.type == "connection_error":
                    detail = (
                        f"{detail} | ROS-Umgebung prüfen (TRANSCEIVER_REMOTE_ROS_ENV_CMD / TRANSCEIVER_REMOTE_ROS_SETUP)"
                    )
                self._on_operator_message(
                    f"Navigation {event.type} (Versuch {event.attempt}): {detail}"
                )
            if event.type == "succeeded":
                self._on_status("navigation", "succeeded")

        state = self._adapter.navigate_to_point(point, timeout_s=timeout_s, on_event=_on_event)
        if state != "succeeded":
            self._on_status("navigation", state)
        return state

    def cancel_current_goal(self) -> None:
        self._adapter.cancel_current_goal()


class _UiMeasurementService:
    def __init__(self, on_status) -> None:
        self._on_status = on_status

    def trigger(self, point_context: PointExecutionContext) -> dict[str, Any]:
        self._on_status("measurement", "running")
        time.sleep(0.25)
        self._on_status("measurement", "succeeded")
        return {
            "measurement_id": f"m-{point_context.global_index:04d}",
            "file_ref": f"signals/rx/point-{point_context.global_index:04d}.bin",
            "point_id": point_context.point.id,
        }


class MissionWorkflowWindow(ctk.CTkToplevel):
    def __init__(self, parent: ctk.CTk) -> None:
        super().__init__(parent)
        self.title("Mission Workflow")
        self.geometry("1100x700")
        self.minsize(980, 640)

        self._mission: MeasurementMission | None = None
        self._executor: MeasurementRunExecutor | None = None
        self._run_thread: threading.Thread | None = None
        self._records: list[dict[str, Any]] = []
        self._run_started_at: float | None = None
        self._run_log_dir: Path | None = None
        self._runtime_config = MissionRuntimeConfig.from_env()

        self._build_ui()
        self.after_idle(self._stabilize_initial_geometry)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)

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
        for col in range(9):
            points_editor.columnconfigure(col, weight=0)
        points_editor.columnconfigure(8, weight=1)

        ctk.CTkLabel(points_editor, text="2) Messpunkt anlegen").grid(row=0, column=0, padx=8, pady=8)
        self.point_id_var = tk.StringVar(value="")
        self.point_name_var = tk.StringVar(value="")
        self.point_x_var = tk.StringVar(value="0.0")
        self.point_y_var = tk.StringVar(value="0.0")
        self.point_z_var = tk.StringVar(value="0.0")
        self.point_yaw_var = tk.StringVar(value="0.0")

        self._labeled_entry(points_editor, row=0, column=1, label="ID", variable=self.point_id_var, width=80)
        self._labeled_entry(points_editor, row=0, column=2, label="Name", variable=self.point_name_var, width=120)
        self._labeled_entry(points_editor, row=0, column=3, label="X", variable=self.point_x_var, width=90)
        self._labeled_entry(points_editor, row=0, column=4, label="Y", variable=self.point_y_var, width=90)
        self._labeled_entry(points_editor, row=0, column=5, label="Z", variable=self.point_z_var, width=90)
        self._labeled_entry(points_editor, row=0, column=6, label="Yaw", variable=self.point_yaw_var, width=90)
        ctk.CTkButton(points_editor, text="Punkt hinzufügen", command=self._add_point).grid(row=0, column=7, padx=(8, 3))
        ctk.CTkButton(points_editor, text="Auswahl entfernen", command=self._remove_selected_point).grid(row=0, column=8, padx=(3, 8), sticky="w")

        points_table_frame = ctk.CTkFrame(self)
        points_table_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))
        points_table_frame.columnconfigure(0, weight=1)
        point_columns = ("idx", "id", "name", "x", "y", "z", "yaw")
        self.points_table = ttk.Treeview(points_table_frame, columns=point_columns, show="headings", height=5)
        self.points_table.grid(row=0, column=0, sticky="ew")
        for key, title in {
            "idx": "#",
            "id": "ID",
            "name": "Name",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "yaw": "Yaw",
        }.items():
            self.points_table.heading(key, text=title)
            self.points_table.column(key, stretch=True, width=95)

        self.validation_box = ctk.CTkTextbox(self, height=110)
        self.validation_box.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 6))
        self.validation_box.insert("1.0", "3) Validierungsergebnis erscheint hier.\n")
        self.validation_box.configure(state="disabled")

        controls = ctk.CTkFrame(self)
        controls.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 6))
        for col in range(8):
            controls.columnconfigure(col, weight=0)
        controls.columnconfigure(7, weight=1)

        ctk.CTkLabel(controls, text="4) Laufsteuerung").grid(row=0, column=0, padx=8, pady=8)
        self.start_btn = ctk.CTkButton(controls, text="Start", command=self._start_run)
        self.start_btn.grid(row=0, column=1, padx=3)
        self.pause_btn = ctk.CTkButton(controls, text="Pause", command=self._pause_run, state="disabled")
        self.pause_btn.grid(row=0, column=2, padx=3)
        self.resume_btn = ctk.CTkButton(controls, text="Fortsetzen", command=self._resume_run, state="disabled")
        self.resume_btn.grid(row=0, column=3, padx=3)
        self.stop_btn = ctk.CTkButton(controls, text="Stop", command=self._stop_run, state="disabled")
        self.stop_btn.grid(row=0, column=4, padx=3)
        ctk.CTkButton(controls, text="Run-Logs exportieren", command=self._export_logs).grid(row=0, column=5, padx=(10, 3))

        self.live_var = tk.StringVar(value="Punkt: - | Navigation: idle | Messung: idle | Verbleibend: -")
        ctk.CTkLabel(controls, textvariable=self.live_var, anchor="w").grid(row=0, column=7, sticky="ew", padx=8)

        table_frame = ctk.CTkFrame(self)
        table_frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=(0, 10))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = ("idx", "point", "nav", "measurement", "status", "error")
        self.results_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        self.results_table.grid(row=0, column=0, sticky="nsew")
        headings = {
            "idx": "Punktindex",
            "point": "Punkt",
            "nav": "Navigation",
            "measurement": "Messung",
            "status": "Gesamtstatus",
            "error": "Fehler",
        }
        for key, title in headings.items():
            self.results_table.heading(key, text=title)
            self.results_table.column(key, stretch=True, width=110)
        self.results_table.column("error", width=260)

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.results_table.configure(yscrollcommand=scroll.set)

        self._mission_points: list[MeasurementPoint] = []

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
        self.validation_box.configure(state="normal")
        self.validation_box.insert("end", text + "\n")
        self.validation_box.see("end")
        self.validation_box.configure(state="disabled")

    def _set_validation_text(self, text: str) -> None:
        self.validation_box.configure(state="normal")
        self.validation_box.delete("1.0", "end")
        self.validation_box.insert("1.0", text)
        self.validation_box.configure(state="disabled")

    def _add_point(self) -> None:
        point_payload = {
            "id": self.point_id_var.get().strip() or None,
            "name": self.point_name_var.get().strip() or None,
            "x": self.point_x_var.get().strip(),
            "y": self.point_y_var.get().strip(),
            "z": self.point_z_var.get().strip() or 0.0,
            "yaw": self.point_yaw_var.get().strip(),
        }
        if point_payload["id"] is None and point_payload["name"] is None:
            messagebox.showwarning("Messpunkt", "Bitte mindestens ID oder Name setzen.")
            return
        try:
            mission = measurement_mission_from_dict(
                {"name": "point-check", "points": [point_payload], "repeat": 1}
            )
        except Exception as exc:
            messagebox.showwarning("Messpunkt ungültig", str(exc))
            return

        point = mission.points[0]
        self._mission_points.append(point)
        self._refresh_points_table()
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
        self._refresh_points_table()
        self._append_validation(f"ℹ️ Punkt entfernt: {removed.id or removed.name}")

    def _refresh_points_table(self) -> None:
        self.points_table.delete(*self.points_table.get_children())
        for idx, point in enumerate(self._mission_points):
            self.points_table.insert(
                "",
                "end",
                values=(
                    idx,
                    point.id or "-",
                    point.name or "-",
                    f"{point.x:.3f}",
                    f"{point.y:.3f}",
                    f"{point.z:.3f}",
                    "-" if point.yaw is None else f"{point.yaw:.3f}",
                ),
            )

    @staticmethod
    def _serialize_point(point: MeasurementPoint) -> dict[str, Any]:
        point_payload: dict[str, Any] = {
            "id": point.id,
            "name": point.name,
            "x": point.x,
            "y": point.y,
            "z": point.z,
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
            )
            serialized_points = [self._serialize_point(point) for point in mission.points]
            measurement_mission_from_dict(
                {
                    "name": mission.name,
                    "repeat": mission.repeat,
                    "wait_after_arrival_s": mission.wait_after_arrival_s,
                    "points": serialized_points,
                }
            )
        except Exception as exc:
            self._mission = None
            self._set_validation_text(f"❌ Validierung fehlgeschlagen\nDetails: {exc}")
            return

        self._mission = mission
        repeats = mission.repeat or 1
        total_points = len(mission.points) * repeats
        self._set_validation_text(
            f"✅ Mission valide: {mission.name}\n"
            f"Punkte pro Zyklus: {len(mission.points)} | Wiederholungen: {repeats} | Gesamtpunkte: {total_points}"
        )

    def _start_run(self) -> None:
        if self._mission is None:
            messagebox.showwarning("Mission", "Bitte zuerst eine gültige Mission anlegen und validieren.")
            return
        if self._run_thread and self._run_thread.is_alive():
            return

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

        self._executor = MeasurementRunExecutor(
            mission=self._mission,
            navigator=_UiNavigator(
                adapter=nav_adapter,
                on_status=self._on_stage_update,
                on_operator_message=self._append_validation,
            ),
            measurement_service=_UiMeasurementService(self._on_stage_update),
            persist_result=_persist,
            run_log_store=store,
            config=MeasurementRunExecutorConfig(
                on_point_error="continue",
                goal_reached_timeout_s=self._runtime_config.goal_reached_timeout_s,
                navigation_retry_attempts=self._runtime_config.navigation_retry_attempts,
            ),
        )

        self._set_run_buttons(running=True, paused=False)
        self._run_thread = threading.Thread(target=self._run_executor_thread, daemon=True)
        self._run_thread.start()

    def _run_executor_thread(self) -> None:
        assert self._executor is not None
        state = self._executor.start()
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
        self._append_validation("Stop angefordert: laufendes Goal wird abgebrochen.")

    def _set_run_buttons(self, *, running: bool, paused: bool) -> None:
        self.start_btn.configure(state="disabled" if running else "normal")
        self.pause_btn.configure(state="normal" if running and not paused else "disabled")
        self.resume_btn.configure(state="normal" if running and paused else "disabled")
        self.stop_btn.configure(state="normal" if running else "disabled")

    def _on_stage_update(self, stage: str, status: str) -> None:
        self.after(0, lambda: self._update_live_label(stage=stage, status=status))

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

        self.live_var.set(
            f"Punktindex: {current_idx}/{max(total - 1, 0)} | "
            f"Navigation: {nav_status} | Messung: {meas_status} | "
            f"Verbleibend: {remaining} | ETA: {eta}"
        )

    def _on_record(self, payload: dict[str, Any]) -> None:
        self._records.append(payload)
        point = payload.get("point", {})
        nav = payload.get("navigation", {})
        meas = payload.get("measurement", {})
        self.results_table.insert(
            "",
            "end",
            values=(
                payload.get("global_index", ""),
                point.get("id") or point.get("name") or "-",
                nav.get("state", "-"),
                meas.get("status", "-"),
                "ok" if payload.get("error") is None else "fehler",
                payload.get("error") or "",
            ),
        )
        self._update_live_label()

    def _on_run_finished(self, state: str) -> None:
        self._set_run_buttons(running=False, paused=False)
        self._append_validation(f"Run beendet: {state}")
        if self._records:
            last_error = self._records[-1].get("error")
            if last_error:
                self._append_validation(f"Operator-Hinweis: {last_error}")
        self._update_live_label()

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
