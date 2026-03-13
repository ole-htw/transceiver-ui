from __future__ import annotations

import re
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
from .measurement_mission import MeasurementMission, load_measurement_mission
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
                self._on_operator_message(
                    f"Navigation {event.type} (Versuch {event.attempt}): {event.message or 'ohne Details'}"
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

        self._mission_file: Path | None = None
        self._mission: MeasurementMission | None = None
        self._executor: MeasurementRunExecutor | None = None
        self._run_thread: threading.Thread | None = None
        self._records: list[dict[str, Any]] = []
        self._run_started_at: float | None = None
        self._run_log_dir: Path | None = None
        self._runtime_config = MissionRuntimeConfig.from_env()

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        workflow = ctk.CTkFrame(self)
        workflow.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        workflow.columnconfigure(1, weight=1)

        ctk.CTkLabel(workflow, text="1) Missionsdatei").grid(row=0, column=0, padx=8, pady=8)
        self.file_var = tk.StringVar(value="Keine Datei geladen")
        ctk.CTkLabel(workflow, textvariable=self.file_var, anchor="w").grid(row=0, column=1, sticky="ew", padx=8)
        ctk.CTkButton(workflow, text="Laden/Auswählen", command=self._choose_mission_file).grid(row=0, column=2, padx=8)
        ctk.CTkButton(workflow, text="Validieren", command=self._validate_selected).grid(row=0, column=3, padx=(0, 8))

        self.validation_box = ctk.CTkTextbox(self, height=110)
        self.validation_box.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        self.validation_box.insert("1.0", "2) Validierungsergebnis erscheint hier.\n")
        self.validation_box.configure(state="disabled")

        controls = ctk.CTkFrame(self)
        controls.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))
        for col in range(8):
            controls.columnconfigure(col, weight=0)
        controls.columnconfigure(7, weight=1)

        ctk.CTkLabel(controls, text="3) Laufsteuerung").grid(row=0, column=0, padx=8, pady=8)
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
        table_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))
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

    def _choose_mission_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Missionsdatei auswählen",
            filetypes=[("Mission", "*.json *.yaml *.yml"), ("All files", "*.*")],
        )
        if not filename:
            return
        self._mission_file = Path(filename)
        self.file_var.set(str(self._mission_file))
        self._validate_selected()

    def _validate_selected(self) -> None:
        if self._mission_file is None:
            self._set_validation_text("Bitte zuerst eine Missionsdatei laden.")
            return
        try:
            mission = load_measurement_mission(self._mission_file)
        except Exception as exc:
            details = self._build_validation_error_details(self._mission_file, exc)
            self._mission = None
            self._set_validation_text(details)
            return

        self._mission = mission
        repeats = mission.repeat or 1
        total_points = len(mission.points) * repeats
        self._set_validation_text(
            f"✅ Mission valide: {mission.name}\n"
            f"Punkte pro Zyklus: {len(mission.points)} | Wiederholungen: {repeats} | Gesamtpunkte: {total_points}"
        )

    def _build_validation_error_details(self, path: Path, exc: Exception) -> str:
        msg = str(exc)
        lines = ["❌ Validierung fehlgeschlagen", f"Datei: {path}"]

        line_no = getattr(exc, "lineno", None)
        if line_no:
            lines.append(f"Zeilenfehler: Zeile {line_no}")

        match = re.search(r"points\[(\d+)\]", msg)
        if match:
            point_idx = int(match.group(1))
            approx_line = self._approx_line_for_point(path, point_idx)
            if approx_line is not None:
                lines.append(f"Punktfehler: points[{point_idx}] (ca. Zeile {approx_line})")
            else:
                lines.append(f"Punktfehler: points[{point_idx}]")

        lines.append(f"Details: {msg}")
        return "\n".join(lines)

    @staticmethod
    def _approx_line_for_point(path: Path, point_idx: int) -> int | None:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None
        candidates: list[int] = []
        for idx, line in enumerate(text.splitlines(), start=1):
            if "- id:" in line or '"id"' in line or "- name:" in line or '"name"' in line:
                candidates.append(idx)
        if point_idx < len(candidates):
            return candidates[point_idx]
        return None

    def _start_run(self) -> None:
        if self._mission is None:
            messagebox.showwarning("Mission", "Bitte zuerst eine gültige Mission laden und validieren.")
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
