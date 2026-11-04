"""PhysioPose AI's single-computer Python demo UI."""
import threading
import time
import queue
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from instructor import Instructor
from clinical_angle import (
    assess_rep_checks,
    camera_angle_to_flexion_degrees,
)
from movement import KneeExtensionAnalyzer, SquatAnalyzer
from session_report import build_report, save_report
from vision import PoseTracker
from voice_agent import HandsFreeVoice
from exercise_catalog import EXERCISES, get_exercise

BG, GREEN, INK, MUTED = "#f4f3ee", "#286b53", "#20352e", "#738079"


class PhysioPoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PhysioPose AI — MVP demo")
        self.root.geometry("1150x800")
        self.root.minsize(850, 650)
        self.root.configure(bg=BG)
        self.tracker = None
        self.voice = None
        self.voice_input_status = "off"
        self.voice_output_status = "off"
        self.voice_ai_status = "chat ready"
        self.voice_ready_announced = False
        self.voice_language = "en"
        self.voice_paused = False
        self.live_activity = {
            "movement_phase": "not_started",
            "tracking_status": "camera_off",
            "tracking_guidance": "Start the camera to detect movement.",
            "repetitions": 0,
        }
        self.analyzer = KneeExtensionAnalyzer()
        self.exercise_type = "knee_extension"
        self.target_repetitions = 8
        self.instructor = Instructor()
        self.prescribed_hold_seconds = 1.0
        self.minimum_cycle_seconds = 2.0
        self.chat = []
        self.events = []
        self.symptom_reports = []
        self.voice_session_feedback = {}
        self.rep_cue_queue = queue.Queue(maxsize=1)
        self.rep_cue_lock = threading.Lock()
        self.rep_cue_worker = None
        self.angles = []
        self.angle_trace = []
        self.last_trace_bucket = -1
        self.started_at = None
        self.camera_started_at = None
        self.tracking_stable_since = None
        self.tracking_acquired = False
        self.elapsed = 0
        self.confidence = 0
        self.angle = None
        self.guided = tk.BooleanVar(value=True)
        self.speech_model_choice = tk.StringVar(value="Gemini Live · English (streaming)")
        self.voice_backend_active = None
        self._header()
        self._show_checkin()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _header(self):
        bar = tk.Frame(self.root, bg="#faf9f5", padx=20, height=56)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="P", bg=GREEN, fg="white", font=("Segoe UI", 14, "bold"), width=3).pack(side="left", pady=9)
        tk.Label(bar, text="PhysioPose AI", bg="#faf9f5", fg=INK, font=("Segoe UI", 14, "bold")).pack(side="left", padx=9)
        ttk.Button(bar, text="Instructor settings", command=self._settings).pack(side="right", pady=10)

    def _page(self):
        if hasattr(self, "page"):
            self.page.destroy()
        self.page = tk.Frame(self.root, bg=BG, padx=24, pady=20)
        self.page.pack(fill="both", expand=True)

    def _eyebrow(self, parent, text):
        tk.Label(parent, text=text.upper(), bg=parent.cget("bg"), fg="#739080", font=("Segoe UI", 9, "bold")).pack(anchor="w")

    def _field(self, parent, title, widget):
        frame = tk.Frame(parent, bg=parent.cget("bg"))
        frame.pack(fill="x", pady=5)
        tk.Label(frame, text=title, bg=parent.cget("bg"), fg=INK, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 3))
        widget.pack(fill="x")

    def _exercise(self):
        return get_exercise(self.exercise_type)

    def _selected_exercise(self):
        label = self.exercise_choice.get() if hasattr(self, "exercise_choice") else "Seated knee extension"
        return "squat" if label == "Squat" else "knee_extension"

    def _update_exercise_preview(self, *_args):
        self.exercise_type = self._selected_exercise()
        item = self._exercise()
        if hasattr(self, "exercise_description") and self.exercise_description.winfo_exists():
            joints = ", ".join(item["required_joints"])
            self.exercise_description.configure(text=f"{item['checkin_instructions']}\n\nCamera tracking requires {len(item['required_joints'])} joints: {joints}.")

    def _show_checkin(self):
        self._page()
        self._eyebrow(self.page, "YOUR HOME EXERCISE SESSION")
        tk.Label(self.page, text="Choose your exercise", bg=BG, fg=INK, font=("Segoe UI", 28, "bold")).pack(anchor="w", pady=(5, 2))
        tk.Label(self.page, text="A simple demo of camera-based movement tracking and an AI instructor.", bg=BG, fg=MUTED, font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 16))
        card = tk.LabelFrame(self.page, text="Before we begin", bg="white", fg=INK, padx=16, pady=12, font=("Segoe UI", 11, "bold"))
        card.pack(fill="x")
        self.pre_feeling = tk.StringVar(value="Good")
        self.pre_pain = tk.StringVar(value="No")
        self.pre_symptoms = tk.StringVar()
        self._field(card, "How are you feeling today?", ttk.Combobox(card, textvariable=self.pre_feeling, values=["Good", "Okay", "Not great"], state="readonly"))
        self._field(card, "Any pain today or yesterday?", ttk.Combobox(card, textvariable=self.pre_pain, values=["No", "Mild", "Moderate", "Severe"], state="readonly"))
        self._field(card, "Any new symptoms or concerns?", ttk.Entry(card, textvariable=self.pre_symptoms))
        exercise = tk.LabelFrame(self.page, text="Your prescribed movement", bg="white", fg=INK, padx=16, pady=12, font=("Segoe UI", 11, "bold"))
        exercise.pack(fill="x", pady=14)
        names = {"knee_extension": "Seated knee extension", "squat": "Squat"}
        default_name = names.get(self.exercise_type, names["knee_extension"])
        self.exercise_choice = tk.StringVar(value=default_name)
        _field_label = ttk.Combobox(exercise, textvariable=self.exercise_choice, values=list(names.values()), state="readonly")
        _field_label.pack(fill="x", pady=(0, 8))
        _field_label.bind("<<ComboboxSelected>>", self._update_exercise_preview)
        self.exercise_description = tk.Label(exercise, text="", bg="white", fg=INK, wraplength=850, justify="left")
        self.exercise_description.pack(anchor="w")
        tk.Label(exercise, text="1 demo set · 8 repetitions · stay in your comfortable prescribed range", bg="white", fg=GREEN, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=9)
        self._update_exercise_preview()
        tk.Label(self.page, text="Stop if pain increases. This demo does not diagnose or change your plan.", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")
        ttk.Button(self.page, text="Continue to camera setup →", command=self._show_setup).pack(anchor="w", pady=15)

    def _show_setup(self):
        self.pre = {"feeling": self.pre_feeling.get(), "pain": self.pre_pain.get(), "symptoms": self.pre_symptoms.get().strip()}
        self.exercise_type = self._selected_exercise()
        self.target_repetitions = self._exercise()["target_repetitions"]
        self._page()
        self._eyebrow(self.page, "SESSION PREPARATION")
        tk.Label(self.page, text="Set up your camera", bg=BG, fg=INK, font=("Segoe UI", 25, "bold")).pack(anchor="w", pady=5)
        item = self._exercise()
        joints = ", ".join(item["required_joints"])
        tk.Label(self.page, text=item["camera_instructions"], bg=BG, fg=MUTED, wraplength=800, justify="left").pack(anchor="w")
        tk.Label(self.page, text=f"This exercise uses {len(item['required_joints'])} required joints: {joints}.", bg=BG, fg=GREEN, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(5, 8))
        self.tracking_side = tk.StringVar(value="Auto (choose once)")
        self._field(self.page, "Which leg should the camera track?", ttk.Combobox(self.page, textvariable=self.tracking_side, values=["Auto (choose once)", "My left leg", "My right leg"], state="readonly"))
        tk.Label(self.page, text="Choose the leg closest to the camera. Left and right mean your own sides. The selection stays fixed during the session.", bg=BG, fg=MUTED, wraplength=800, justify="left").pack(anchor="w", pady=(2, 10))
        tk.Label(self.page, text="Pose model: MediaPipe Heavy", bg=BG, fg=INK, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(2, 4))
        tk.Label(
            self.page,
            text="The live camera estimate uses clinical flexion convention (0° = straight, larger = more bend). It is still a 2D estimate and has not been clinically validated.",
            bg=BG, fg=MUTED, wraplength=800, justify="left",
        ).pack(anchor="w", pady=(0, 8))
        range_text = f"Form cues: move smoothly through a comfortable range; pause near the endpoint for {self.prescribed_hold_seconds:g} second(s). Stop if pain increases."
        self.range_status_label = tk.Label(self.page, text=range_text, bg=BG, fg=MUTED, wraplength=800, justify="left")
        self.range_status_label.pack(anchor="w", pady=(2, 4))
        if self.exercise_type == "knee_extension":
            tk.Label(self.page, text="Camera extension warning: estimated near-full extension only. This is not a medical safety limit.", bg=BG, fg=MUTED, wraplength=800).pack(anchor="w")
        ttk.Button(self.page, text="Set prescribed hold time…", command=self._configure_hold).pack(anchor="w", pady=(2, 6))
        ttk.Button(self.page, text="Set minimum controlled cycle time…", command=self._configure_cycle_time).pack(anchor="w", pady=(2, 6))
        self._speech_model_selector(self.page)
        voice_controls = tk.Frame(self.page, bg=BG)
        voice_controls.pack(anchor="w", pady=(4, 8))
        ttk.Button(voice_controls, text="Enable hands-free voice", command=self._start_voice).pack(side="left")
        self.voice_status = tk.Label(voice_controls, text="Voice off", bg=BG, fg=MUTED, font=("Segoe UI", 9))
        self.voice_status.pack(side="left", padx=10)

        if self.pre["pain"] == "Severe" or self.pre["symptoms"]:
            tk.Label(self.page, text="Please check with your physiotherapist before exercising if you have severe pain or new symptoms.", bg="#faeee9", fg="#995940", padx=12, pady=10, wraplength=800, justify="left").pack(fill="x", pady=14)
        else:
            tk.Label(self.page, text="Camera frames stay on this computer. They are not saved or uploaded.", bg=BG, fg=MUTED).pack(anchor="w", pady=12)
        ttk.Button(self.page, text="Start camera and exercise", command=self._start).pack(anchor="w", pady=8)

    def _configure_hold(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Prescribed endpoint hold")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(padx=16, pady=14)
        tk.Label(dialog, text="Enter the hold time from the clinician’s exercise plan. The patient should move only as far as comfortably possible. The camera cannot determine pain or a safe limit.", wraplength=440, justify="left").pack(anchor="w", pady=(0, 10))
        value = tk.StringVar(value=str(self.prescribed_hold_seconds))
        self._field(dialog, "Hold near the comfortable endpoint (seconds)", ttk.Entry(dialog, textvariable=value))
        def save():
            try:
                seconds = float(value.get())
                if not 0 <= seconds <= 30:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid hold time", "Enter a hold time from 0 to 30 seconds.", parent=dialog)
                return
            self.prescribed_hold_seconds = seconds
            dialog.destroy()
            self._show_setup()
        ttk.Button(dialog, text="Save hold time", command=save).pack(anchor="e", pady=(10, 0))

    def _configure_cycle_time(self):
        dialog=tk.Toplevel(self.root); dialog.title("Cycle control timing"); dialog.transient(self.root); dialog.grab_set(); dialog.configure(padx=16,pady=14)
        tk.Label(dialog,text="A minimum cycle duration checks pace only; it cannot verify smoothness, alignment, pain, or safe movement.",wraplength=440,justify="left").pack(anchor="w",pady=(0,10))
        value=tk.StringVar(value=str(self.minimum_cycle_seconds)); self._field(dialog,"Minimum seconds for one bend–extend–return cycle",ttk.Entry(dialog,textvariable=value))
        def save():
            try:
                seconds=float(value.get())
                if not 0<=seconds<=60: raise ValueError
            except ValueError:
                messagebox.showerror("Invalid duration","Enter a value from 0 to 60 seconds.",parent=dialog); return
            self.minimum_cycle_seconds=seconds; dialog.destroy(); self._show_setup()
        ttk.Button(dialog,text="Save timing",command=save).pack(anchor="e",pady=(10,0))

    def _start(self):
        try:
            side_choice = self.tracking_side.get()
            preferred_side = {"My left leg": "left", "My right leg": "right"}.get(side_choice, "auto")
            self.pose_model_complexity = 2
            self.exercise_type = self._selected_exercise()
            self.target_repetitions = self._exercise()["target_repetitions"]
            self.analyzer = SquatAnalyzer() if self.exercise_type == "squat" else KneeExtensionAnalyzer()
            self.tracker = PoseTracker(preferred_side=preferred_side, model_complexity=2, exercise_type=self.exercise_type)
            self.tracker.start()
        except Exception as error:
            self.tracker = None
            messagebox.showerror("Could not start camera", str(error), parent=self.root)
            return
        self.analyzer.reset()
        self.chat.clear()
        self.events.clear()
        self.symptom_reports.clear()
        self.voice_session_feedback.clear()
        self.angles.clear()
        self.angle_trace.clear()
        self.last_trace_bucket = -1
        self.camera_started_at = time.monotonic()
        self.started_at = None
        self.tracking_stable_since = None
        self.tracking_acquired = False
        self.voice_paused = False
        self.live_activity = {
            "movement_phase": "acquiring_tracking",
            "tracking_status": "acquiring",
            "tracking_guidance": "Hold still while the app finds the required joints.",
            "repetitions": 0,
        }
        self._show_session()
        if self.pre.get("pain") == "Severe" or self.pre.get("symptoms"):
            self._add_message("instructor", "Please check with your physiotherapist before exercising if you have severe pain or new symptoms. I cannot assess symptoms or decide if it is safe to continue.")
        else:
            joints = ", ".join(self._exercise()["required_joints"])
            self._add_message("instructor", f"Camera is on. Keep {joints} in view and hold still while I find them. Repetition counting starts when tracking is stable.")
        self._camera_tick()

    def _show_session(self):
        self._page()
        top = tk.Frame(self.page, bg=BG)
        top.pack(fill="x", pady=(0, 10))
        self.stage_title = tk.Label(top, text="Finding your leg…", bg=BG, fg=INK, font=("Segoe UI", 21, "bold"))
        self.stage_title.pack(side="left")
        self.timer_label = tk.Label(top, text="00:00", bg=BG, fg=MUTED, font=("Segoe UI", 12, "bold"))
        self.timer_label.pack(side="right")
        ttk.Button(top, text="Finish exercise", command=self._finish).pack(side="right", padx=(0, 12))
        self.columns = tk.Frame(self.page, bg=BG)
        self.columns.pack(fill="both", expand=True)
        left = tk.Frame(self.columns, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.camera_view = tk.Label(left, text="Starting camera…", bg="#293630", fg="white", font=("Segoe UI", 13))
        self.camera_view.pack(fill="both", expand=True)
        metrics = tk.Frame(left, bg="white", padx=12, pady=10)
        metrics.pack(fill="x", pady=8)
        self.angle_label = tk.Label(metrics, text="KNEE FLEXION ESTIMATE\n—°", bg="white", fg=INK, font=("Segoe UI", 15, "bold"))
        self.angle_label.pack(side="left", padx=12)
        self.rep_label = tk.Label(metrics, text="MOVEMENT CYCLES\n0 / 8", bg="white", fg=INK, font=("Segoe UI", 15, "bold"))
        self.rep_label.pack(side="left", padx=14)
        self.quality_label = tk.Label(metrics, text="TRACKING\nWaiting", bg="white", fg=INK, font=("Segoe UI", 13, "bold"))
        self.quality_label.pack(side="left", padx=14)
        form_status = f"Cues: comfortable range · controlled movement · hold {self.prescribed_hold_seconds:g}s. Camera timing is descriptive, not a clinical form score."
        self.form_status_label = tk.Label(left, text=form_status, bg=BG, fg=MUTED, font=("Segoe UI", 9))
        self.form_status_label.pack(anchor="w", pady=(0, 3))
        self.phase_label = tk.Label(left, text="Ready", bg=BG, fg=INK, font=("Segoe UI", 10, "bold"))
        self.phase_label.pack(anchor="w")
        initial_hint = "Stand comfortably, then begin a comfortable squat." if self.exercise_type == "squat" else "Bend your knee to begin a repetition"
        self.hint_label = tk.Label(left, text=initial_hint, bg=BG, fg=MUTED, font=("Segoe UI", 9), wraplength=720, justify="left")
        self.hint_label.pack(anchor="w", pady=2)
        self._speech_model_selector(left)
        controls = tk.Frame(left, bg=BG)
        controls.pack(fill="x", pady=7)
        ttk.Checkbutton(controls, text="Form cue after each rep", variable=self.guided).pack(side="left")
        ttk.Button(controls, text="Enable hands-free voice", command=self._start_voice).pack(side="right")
        self.voice_status = tk.Label(left, text="Voice off", bg=BG, fg=MUTED, font=("Segoe UI", 9))
        self.voice_status.pack(anchor="w")

        right = tk.LabelFrame(self.columns, text="AI instructor", bg="white", fg=INK, padx=10, pady=8, font=("Segoe UI", 10, "bold"), width=300)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self.chat_display = tk.Text(right, wrap="word", bg="#fbfcf9", fg=INK, relief="flat", font=("Segoe UI", 9), state="disabled")
        self.chat_display.pack(fill="both", expand=True, pady=(0, 7))
        self.chat_entry = ttk.Entry(right)
        self.chat_entry.pack(fill="x", pady=4)
        self.chat_entry.bind("<Return>", lambda _event: self._send_chat())
        ttk.Button(right, text="Ask", command=self._send_chat).pack(anchor="e")
        ttk.Button(right, text="LLM settings…", command=self._settings).pack(anchor="w", pady=7)

    def _speech_model_selector(self, parent):
        row = tk.Frame(parent, bg=parent.cget("bg"))
        row.pack(anchor="w", fill="x", pady=(5, 3))
        tk.Label(row, text="Speech recognition", bg=parent.cget("bg"), fg=INK, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 8))
        ttk.Combobox(
            row, textvariable=self.speech_model_choice, state="readonly", width=43,
            values=["Gemini Live · English (streaming)"],
        ).pack(side="left")
        tk.Label(parent, text="Live English transcription streams microphone audio to Google. If it cannot connect, the app falls back to local Vosk.", bg=parent.cget("bg"), fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")

    def _camera_tick(self):
        if self.tracker is None:
            return
        frame, reading = self.tracker.read()
        if frame is not None:
            now = time.monotonic()
            if not self.tracking_acquired:
                if reading.ready:
                    if self.tracking_stable_since is None:
                        self.tracking_stable_since = now
                    elif now - self.tracking_stable_since >= .75:
                        self.tracking_acquired = True
                        self.started_at = now
                        self.stage_title.configure(text=f"{self._exercise()['name']} · move comfortably")
                        self._add_message("instructor", f"Tracking is ready. Begin your comfortable {self._exercise()['short_name'].lower()} repetitions.")
                else:
                    self.tracking_stable_since = None
            update = self.analyzer.update(reading.angle, reading.reliable, reading.confidence, tracking_ready=reading.ready) if self.tracking_acquired and not self.voice_paused else None
            self.angle = camera_angle_to_flexion_degrees(reading.angle) if reading.reliable else None
            self.confidence = reading.confidence
            elapsed_seconds = now - self.camera_started_at
            trace_bucket = int(elapsed_seconds * 10)
            if trace_bucket > self.last_trace_bucket:
                self.angle_trace.append({
                    "time_seconds": round(elapsed_seconds, 1),
                    "camera_included_angle_degrees": round(reading.angle, 1) if reading.reliable else None,
                    "camera_flexion_estimate_degrees": round(self.angle, 1) if self.angle is not None else None,
                    "tracking_reliable": reading.reliable,
                    "tracking_ready": reading.ready,
                    "confidence": round(reading.confidence, 2),
                    "landmark_visibility": [round(score, 2) for score in reading.landmark_scores],
                    "side": reading.side or None,
                    "phase": "paused" if self.voice_paused else update.phase if update else "acquiring_tracking",
                    "repetitions": update.repetitions if update else self.analyzer.repetitions,
                })
                self.last_trace_bucket = trace_bucket
            if self.angle is not None:
                self.angles.append(self.angle)
                cv2.putText(frame, f"Camera knee-flexion estimate: {round(self.angle)} deg", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .7, (235, 248, 237), 2, cv2.LINE_AA)
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.camera_view.configure(image=photo, text="")
            self.camera_view.image = photo
            self.angle_label.configure(text=f"KNEE FLEXION ESTIMATE\n{round(self.angle)}°" if self.angle is not None else "KNEE FLEXION ESTIMATE\n—°")
            joint_scores = dict(zip(reading.joint_names, reading.landmark_scores))
            score_text = " · ".join(f"{name[:1].upper()} {score:.2f}" for name, score in joint_scores.items())
            if not self.tracking_acquired:
                quality = "Acquiring"
            elif reading.ready:
                quality = "Ready"
            elif reading.side is None:
                quality = "Pose lost"
            else:
                weak_joints = [name for name, score in joint_scores.items() if score < .50]
                quality = "Adjust " + " + ".join(weak_joints) + " view" if weak_joints else "Adjust camera view"
            status_color = GREEN if self.tracking_acquired and reading.ready else "#a56b00" if reading.reliable or not self.tracking_acquired else "#a5523d"
            self.quality_label.configure(text=f"TRACKING\n{quality}\n{score_text}", fg=status_color)
            self.rep_label.configure(text=f"MOVEMENT CYCLES\n{self.analyzer.repetitions} / {self.target_repetitions}")
            if not self.tracking_acquired:
                self.phase_label.configure(text="Finding your leg…")
                self.hint_label.configure(text=reading.guidance or f"Hold still and keep {', '.join(reading.joint_names)} visible.", fg=status_color)
                self.quality_label.configure(text=f"TRACKING\nAcquiring\n{score_text}", fg=status_color)
            elif self.voice_paused:
                self.phase_label.configure(text="Paused")
                self.hint_label.configure(text="Rep counting is paused. Say ‘resume’ to continue.", fg=MUTED)
            elif update is None:
                self.phase_label.configure(text="Waiting for movement data…")
                self.hint_label.configure(text=reading.guidance or reading.reason or "Keep your whole leg visible.", fg=status_color)
            else:
                self.phase_label.configure(text=update.phase.title())
                self.hint_label.configure(text=update.hint if reading.ready else (reading.guidance or reading.reason), fg=MUTED if reading.ready else status_color)
            phase = "paused" if self.voice_paused else update.phase if update else "acquiring_tracking"
            tracking_status = "acquiring" if not self.tracking_acquired else "ready" if reading.ready else "unreliable"
            self.live_activity = {
                "movement_phase": phase,
                "phase_description": {
                    "acquiring_tracking": "waiting for the app to find the leg",
                    "waiting": "waiting for the next movement",
                    "flexed": "knee is bent; the user can begin straightening",
                    "extending": "leg is being straightened",
                    "lowering": "leg is being lowered after extension",
                    "moving": "the user is lowering into a squat",
                    "returning": "the user is returning to standing",
                    "paused": "rep measurement is paused",
                }.get(phase, phase),
                "tracking_status": tracking_status,
                "tracking_confidence": round(float(reading.confidence), 2),
                "joint_confidence": {name: round(float(score), 2) for name, score in joint_scores.items()},
                "exercise": self.exercise_type,
                "required_joints": list(reading.joint_names),
                "tracking_guidance": reading.guidance or reading.reason or (update.hint if update else "Waiting for tracking."),
                "camera_knee_flexion_estimate_degrees": round(self.angle) if self.angle is not None else None,
                "repetitions": self.analyzer.repetitions,
                "tracking_side": reading.side or None,
            }
            if update and update.event:
                update.event["form_checks"] = assess_rep_checks(update.event, minimum_hold_seconds=self.prescribed_hold_seconds, minimum_cycle_seconds=self.minimum_cycle_seconds, exercise_type=self.exercise_type)
                self._handle_event(update.event)
        elapsed = int(time.monotonic() - self.started_at) if self.started_at is not None else 0
        self.timer_label.configure(text=f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        self.elapsed = elapsed
        self.root.after(30, self._camera_tick)

    def _context(self):
        recent_events = [
            {key: event.get(key) for key in (
                "rep_number", "movement_cycle_status", "peak_angle_degrees", "excursion_degrees",
                "extension_hold_seconds", "endpoint_hold_seconds", "rep_duration_seconds", "form_checks", "instructor_feedback",
            ) if key in event}
            for event in self.events[-3:]
        ]
        return {
            "exercise": self._exercise()["name"],
            "camera_knee_flexion_estimate_degrees": round(self.angle) if self.angle is not None else None,
            "tracking_reliable": self.angle is not None,
            "repetitions": self.analyzer.repetitions,
            "count_basis": "detected completed movement cycles",
            "prescribed_hold_seconds": self.prescribed_hold_seconds,
            "patient_pre_session_feedback": self.pre,
            "recent_events": recent_events,
            "live_activity": dict(self.live_activity),
            "session_summary": self._session_summary(),
            "current_plan": {
                "exercise": self._exercise()["name"],
                "target_repetitions": self.analyzer.repetition_target,
                "endpoint_hold_seconds": self.prescribed_hold_seconds,
                "minimum_cycle_seconds": self.minimum_cycle_seconds,
                "movement_cues": [
                    "Move smoothly through the comfortable range.",
                    "Pause near the comfortable endpoint for the configured hold.",
                    "Stop if pain increases.",
                ],
                "limits": "The camera cannot determine a safe range or pain. Camera values are movement estimates, not medical limits.",
                "required_joints": list(self._exercise()["required_joints"]),
            },
            "patient_reported_symptom_count": len(self.symptom_reports),
        }

    def _session_summary(self):
        completed = [event for event in self.events if event.get("movement_cycle_status") == "completed"]
        def count(check_name, status):
            return sum((event.get("form_checks") or {}).get(check_name, {}).get("status") == status for event in completed)
        summary = {
            "completed_repetitions": len(completed),
            "hold": {status: count("hold", status) for status in ("met", "not_met", "unable_to_assess")},
            "cycle_pace": {status: count("cycle_control", status) for status in ("met", "not_met", "unable_to_assess")},
            "symptom_reports_recorded": len(self.symptom_reports),
            "patient_reported_feedback_fields": sorted(self.voice_session_feedback),
            "note": "Counts describe configured prototype checks, not clinical form or safety assessments.",
        }
        if self.exercise_type == "knee_extension":
            summary["extension_warning"] = {"reached": count("extension_ceiling", "past_default_ceiling"), "unable_to_assess": count("extension_ceiling", "unable_to_assess")}
        return summary

    def _handle_event(self, event):
        event.update({"tracking_confidence": round(self.confidence, 2), "exercise": self.exercise_type})
        checks = event.get("form_checks", {})
        hold = checks.get("hold", {})
        control = checks.get("cycle_control", {})
        limit = checks.get("extension_ceiling", {})
        rep = self.analyzer.repetitions
        event["rep_number"] = rep
        # Keep the live instructor to one useful cue; full measurements and each
        # rule result remain in event['form_checks'] for the report.
        active_checks = (hold, control, limit) if self.exercise_type == "knee_extension" else (hold, control)
        if any(check.get("status") == "unable_to_assess" for check in active_checks):
            joints = ", ".join(self._exercise()["required_joints"])
            message = f"Rep {rep} counted. I couldn't check one part because tracking dropped; keep {joints} visible."
        elif self.exercise_type == "knee_extension" and limit.get("status") == "past_default_ceiling":
            message = f"Rep {rep} counted. The camera reached its extension warning; ease back a little if comfortable."
        elif hold.get("status") == "not_met":
            observed = hold.get("observed_seconds", 0)
            required = hold.get("required_seconds", self.prescribed_hold_seconds)
            message = f"Rep {rep} counted. Hold was {observed:.1f} of {required:g} seconds; try a little longer if comfortable."
        elif control.get("status") == "not_met":
            message = f"Rep {rep} counted. That cycle was quick; slow the next one down if comfortable."
        elif hold.get("status") == "met" and control.get("status") == "met":
            praise = ("Nice steady cycle.", "Good pace on that one.", "That met the timing checks.")[(rep - 1) % 3]
            message = f"Rep {rep}. {praise}"
        else:
            message = f"Rep {rep} counted."
        event["instructor_feedback"] = message
        self.events.append(event)
        if self.guided.get():
            needs_llm_cue = (
                (self.exercise_type == "knee_extension" and limit.get("status") == "past_default_ceiling")
                or hold.get("status") == "not_met"
                or control.get("status") == "not_met"
            )
            if needs_llm_cue and self.instructor.api_key:
                self._queue_rep_cue(event, message)
            else:
                self._add_message("instructor", message)
                self._speak(message)

    def _queue_rep_cue(self, event, fallback):
        with self.rep_cue_lock:
            try:
                self.rep_cue_queue.put_nowait((event, fallback))
            except queue.Full:
                try:
                    self.rep_cue_queue.get_nowait()
                except queue.Empty:
                    pass
                self.rep_cue_queue.put_nowait((event, fallback))
            if self.rep_cue_worker is None or not self.rep_cue_worker.is_alive():
                self.rep_cue_worker = threading.Thread(target=self._rep_cue_worker, daemon=True)
                self.rep_cue_worker.start()

    def _rep_cue_worker(self):
        while True:
            with self.rep_cue_lock:
                try:
                    event, fallback = self.rep_cue_queue.get_nowait()
                except queue.Empty:
                    self.rep_cue_worker = None
                    return
            try:
                message = self.instructor.rep_feedback(event).strip() or fallback
            except Exception:
                message = fallback
            self.root.after(0, lambda event=event, message=message: self._deliver_rep_cue(event, message))

    def _deliver_rep_cue(self, event, message):
        event["instructor_feedback"] = message
        if self.events and event is self.events[-1] and self.guided.get():
            self._add_message("instructor", message)
            self._speak(message)

    def _start_voice(self):
        backend = "gemini_live" if self.instructor.api_key else "vosk_en"
        if self.voice is not None:
            if self.voice_backend_active == backend:
                return
            self.voice.stop()
            self.voice = None
            self._set_voice_status("switching speech model…")
            self.root.after(500, self._start_voice)
            return
        try:
            self.voice = HandsFreeVoice(
                on_text=lambda text: self.root.after(0, lambda: self._on_voice_text(text)),
                on_status=lambda status: self.root.after(0, lambda: self._set_voice_status(status)),
                asr_backend=backend,
                api_key=self.instructor.api_key,
            )
            self.voice_backend_active = backend
            self.voice_ready_announced = False
            self.voice_language = "en"
            self.voice.start()
            self.voice_input_status = "microphone starting"
            self.voice_output_status = "speaker starting"
            self.voice_ai_status = "Gemini configured" if self.instructor.api_key else "offline replies"
            self._render_voice_status()
        except Exception as error:
            self.voice = None
            self._set_voice_status(str(error))

    def _set_voice_status(self, status):
        if status.startswith("ai:"):
            self.voice_ai_status = status.partition(":")[2].strip()
        elif status.startswith("output:"):
            self.voice_output_status = status.partition(":")[2].strip()
        elif status.startswith("input:"):
            self.voice_input_status = status.partition(":")[2].strip()
        else:
            self.voice_input_status = status
        self._render_voice_status()
        if status.startswith("input:Listening with") and self.voice and not self.voice_ready_announced:
            self.voice_ready_announced = True
            self.voice.speak("Voice is ready. You can ask a question or say a command.")

    def _render_voice_status(self):
        if hasattr(self, "voice_status") and self.voice_status.winfo_exists():
            if self.voice is None:
                shown = "Voice off"
            else:
                shown = f"Mic: {self.voice_input_status} · Speaker: {self.voice_output_status} · AI: {self.voice_ai_status}"
            self.voice_status.configure(text=shown)

    def _speak(self, text):
        if self.voice:
            self.voice.speak(text)

    def _on_voice_text(self, text):
        self.voice_language = "en"
        self._add_message("user", text)
        self._ask_instructor(text, history=self.chat[:-1])

    def _send_chat(self):
        prompt = self.chat_entry.get().strip()
        if not prompt:
            return
        history = list(self.chat)
        self.chat_entry.delete(0, "end")
        self._add_message("user", prompt)
        self._ask_instructor(prompt, history=history)

    def _ask_instructor(self, prompt, event_type=None, history=None):
        history = list(self.chat if history is None else history)
        context = self._context()
        self._set_voice_status("ai:thinking")
        def work():
            offline = False
            try:
                if not self.instructor.api_key:
                    offline = True
                    answer = self.instructor.fallback(prompt, context, event_type)
                    actions = []
                    status = "ai:no Gemini key; offline reply"
                else:
                    result = self.instructor.reply(prompt, context, history, event_type)
                    if isinstance(result, dict):
                        answer = result.get("text", "")
                        actions = result.get("actions") or ([{"name": result["action"], "arguments": result.get("arguments", {})}] if result.get("action") else [])
                    else:
                        answer, actions = result, []
                    status = "ai:Gemini replied"
            except Exception as error:
                offline = True
                answer = self.instructor.fallback(prompt, context, event_type)
                actions = []
                if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
                    status = "ai:Gemini timed out; offline reply"
                elif self.instructor.api_key:
                    status = f"ai:Gemini request failed ({type(error).__name__})"
                else:
                    status = "ai:no Gemini key; offline reply"
            label = "Instructor · offline" if offline else "Instructor"
            self.root.after(0, lambda status=status, answer=answer, offline=offline, label=label, actions=actions: self._deliver_instructor_reply(status, answer, offline, label, actions))
        threading.Thread(target=work, daemon=True).start()

    def _deliver_instructor_reply(self, status, answer, offline, label, actions):
        self._set_voice_status(status)
        finish_requested = False
        for item in actions if isinstance(actions, list) else []:
            if not isinstance(item, dict):
                continue
            action = item.get("name")
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            if action not in self.instructor.ALLOWED_TOOLS:
                continue
            if action == "pause_exercise":
                self.voice_paused = True
            elif action == "resume_exercise":
                self.voice_paused = False
            elif action == "repeat_last_cue":
                answer = self.events[-1].get("instructor_feedback", "Move smoothly and only as far as comfortable.") if self.events else "Move smoothly and only as far as comfortable."
            elif action == "finish_exercise":
                finish_requested = True
            elif action == "record_symptom_report":
                self._record_symptom_report(arguments)
            elif action == "record_session_feedback":
                self._record_session_feedback(arguments)
        if finish_requested:
            self.root.after(2500 if self.voice else 300, self._finish)
        if answer:
            self._add_message("instructor", answer, save_to_history=not offline, display_name=label)
            self._speak(answer)

    def _record_symptom_report(self, arguments):
        report = str(arguments.get("report", "")).strip()[:500]
        if not report or self.started_at is None:
            return
        item = {
            "reported_at_elapsed_seconds": max(0, int(time.monotonic() - self.started_at)),
            "rep_count": int(self.analyzer.repetitions),
            "report": report,
        }
        severity = arguments.get("severity_0_to_10")
        if isinstance(severity, int) and not isinstance(severity, bool) and 0 <= severity <= 10:
            item["severity_0_to_10"] = severity
        for key in ("body_area", "timing"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                item[key] = value.strip()[:120]
        self.symptom_reports.append(item)

    def _record_session_feedback(self, arguments):
        allowed = {"feeling", "pain", "difficulty", "notes", "report"}
        for key in allowed:
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                self.voice_session_feedback[key] = value.strip()[:500]
        actual = arguments.get("actual_repetitions")
        if isinstance(actual, int) and not isinstance(actual, bool) and 0 <= actual <= 1000:
            self.voice_session_feedback["actual_repetitions"] = actual

    def _add_message(self, role, text, save_to_history=True, display_name=None):
        if save_to_history:
            self.chat.append({"role": "user" if role == "user" else "assistant", "content": text})
        if not hasattr(self, "chat_display") or not self.chat_display.winfo_exists():
            return
        self.chat_display.configure(state="normal")
        name = display_name or ("You" if role == "user" else "Instructor")
        self.chat_display.insert("end", f"{name}: {text}\n\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def _settings(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Optional LLM connection")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(padx=16, pady=14)
        tk.Label(dialog, text="The API key stays in memory only for this app launch. Camera tracking and endpoint feedback work without an LLM key.", wraplength=420, justify="left").pack(anchor="w", pady=(0, 10))
        base = tk.StringVar(value=self.instructor.base_url)
        model = tk.StringVar(value=self.instructor.model)
        key = tk.StringVar(value=self.instructor.api_key)
        self._field(dialog, "API base URL", ttk.Entry(dialog, textvariable=base, width=48))
        self._field(dialog, "Model", ttk.Entry(dialog, textvariable=model, width=48))
        self._field(dialog, "API key", ttk.Entry(dialog, textvariable=key, show="•", width=48))
        def save():
            self.instructor.api_key = key.get().strip()
            self.instructor.base_url = base.get().strip() or "https://api.openai.com/v1"
            self.instructor.model = model.get().strip() or "gpt-4o-mini"
            dialog.destroy()
        ttk.Button(dialog, text="Save", command=save).pack(anchor="e", pady=(10, 0))

    def _finish(self):
        self.voice_paused = True
        if self.voice:
            self.voice.stop()
            self.voice = None
        if self.started_at:
            self.elapsed = int(time.monotonic() - self.started_at)
        if self.tracker:
            self.tracker.stop()
            self.tracker = None
        self._page()
        self._eyebrow(self.page, "POST-SESSION CHECK-IN")
        tk.Label(self.page, text="How did that feel?", bg=BG, fg=INK, font=("Segoe UI", 25, "bold")).pack(anchor="w", pady=8)
        card = tk.LabelFrame(self.page, text="Your feedback", bg="white", fg=INK, padx=14, pady=12)
        card.pack(fill="x")
        self.post_feeling = tk.StringVar(value="Good")
        self.post_pain = tk.StringVar(value="No")
        self.post_repetitions = tk.StringVar()
        self.post_difficulty = tk.StringVar()
        self.post_notes = tk.StringVar()
        self._field(card, "How are you feeling now?", ttk.Combobox(card, textvariable=self.post_feeling, values=["Good", "Okay", "Not great"], state="readonly"))
        self._field(card, "Pain during a repetition?", ttk.Combobox(card, textvariable=self.post_pain, values=["No", "Yes, mild", "Yes, moderate", "Yes, severe"], state="readonly"))
        self._field(card, "How many repetitions did you actually complete? (optional)", ttk.Entry(card, textvariable=self.post_repetitions))
        self._field(card, "What felt difficult?", ttk.Entry(card, textvariable=self.post_difficulty))
        self._field(card, "Anything else you noticed?", ttk.Entry(card, textvariable=self.post_notes))
        ttk.Button(self.page, text="Create session report", command=self._report).pack(anchor="w", pady=14)

    def _report(self):
        actual_repetitions = None
        try:
            actual_repetitions = int(self.post_repetitions.get().strip()) if self.post_repetitions.get().strip() else None
        except ValueError:
            messagebox.showerror("Check repetition count", "Enter a whole number for the repetitions you actually completed, or leave it blank.", parent=self.root)
            return
        post = {"feeling": self.post_feeling.get(), "pain": self.post_pain.get(), "actual_repetitions": actual_repetitions, "difficulty": self.post_difficulty.get().strip(), "notes": self.post_notes.get().strip(), "in_session_symptom_reports": list(self.symptom_reports), "voice_session_feedback": dict(self.voice_session_feedback)}
        try:
            movement_events = [event for event in self.events if event.get("movement_cycle_status") == "completed"]
            report = build_report(self.analyzer.repetitions, self.elapsed, self.angles, self.analyzer.errors, self.pre, post,
                                  angle_trace=self.angle_trace, patient_reported_repetitions=actual_repetitions,
                                  movement_events=movement_events, prescribed_hold_seconds=self.prescribed_hold_seconds,
                                  exercise_type=self.exercise_type, target_repetitions=self.target_repetitions,
                                  tracked_joints=self._exercise()["required_joints"])
            path = save_report(report)
        except Exception as error:
            messagebox.showerror("Could not create session report", f"The report could not be saved.\n\n{error}", parent=self.root)
            return
        self._page()
        footer = tk.Frame(self.page, bg=BG)
        footer.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Button(footer, text="Start another demo →", command=self._reset).pack(anchor="w", pady=(6, 0))
        self._eyebrow(self.page, "SESSION SUMMARY")
        tk.Label(self.page, text="Your report", bg=BG, fg=INK, font=("Segoe UI", 25, "bold")).pack(anchor="w", pady=8)
        measured = report["measured_by_camera"]
        self._report_card(f"{report['exercise_name'].upper()} CYCLES DETECTED", f"Required joints: {', '.join(report['required_joints'])}\nPose model: {measured['pose_model']}\nCompleted cycles: {measured['repetitions']} / {measured['target_repetitions']}\n{measured['angle_summary']}\nReliable samples: {measured['reliable_samples']}")
        performance = measured["performance"]
        hold_pct = performance["hold_met_percent_of_assessable"]
        performance_text = (
            f"Prescribed hold: {performance['hold_goal_seconds']:g}s\n"
            f"Hold check: {performance['hold_met']} met · {performance['hold_not_met']} not met · {performance['hold_unable_to_assess']} unable to assess\n"
            f"Cycle pace: {performance['cycle_pace_met']} met · {performance['cycle_pace_not_met']} too fast · {performance['cycle_pace_unable_to_assess']} unable to assess\n"
            f"Hold met among assessable reps: {hold_pct:g}%\n" if hold_pct is not None else
            f"Prescribed hold: {performance['hold_goal_seconds']:g}s\n"
            f"Hold check: {performance['hold_met']} met · {performance['hold_not_met']} not met · {performance['hold_unable_to_assess']} unable to assess\n"
            f"Cycle pace: {performance['cycle_pace_met']} met · {performance['cycle_pace_not_met']} too fast · {performance['cycle_pace_unable_to_assess']} unable to assess\n"
            "Hold percentage: unavailable\n"
        )
        performance_text += (f"Average estimated movement range: {performance['mean_camera_estimated_excursion_degrees']}°\n"
                             f"Average cycle duration: {performance['mean_cycle_duration_seconds']} s\n"
                             f"Average endpoint hold: {performance['mean_endpoint_hold_seconds']} s\n")
        for item in performance["per_rep"]:
            performance_text += (f"Rep {item['rep']}: range {item['estimated_range_degrees']}° · duration {item['cycle_duration_seconds']}s · "
                                 f"hold {item['endpoint_hold_seconds']}s · pace {item['cycle_pace_check'].replace('_',' ')}")
            if self.exercise_type == "knee_extension":
                performance_text += f" · extension warning {item['extension_ceiling_check'].replace('_',' ')}"
            performance_text += "\n"
        performance_text += performance["note"]
        self._report_card("SESSION PERFORMANCE · CAMERA ESTIMATES", performance_text)
        if actual_repetitions is not None:
            self._report_card("COUNT COMPARISON", f"Camera detected {self.analyzer.repetitions}; you reported {actual_repetitions}.")
        self._report_card("REPORTED BY YOU", f"Before: {self.pre}\nAfter: {post}")
        self._report_card("SUMMARY", report["summary"])
        tk.Label(self.page, text=f"Saved locally to {path}", bg=BG, fg=MUTED).pack(anchor="w", pady=8)
        tk.Label(self.page, text="Camera measurements are estimates and are not a clinical assessment.", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")

    def _report_card(self, title, body):
        card = tk.LabelFrame(self.page, text=title, bg="white", fg=GREEN, padx=12, pady=9)
        card.pack(fill="x", pady=6)
        tk.Label(card, text=body, bg="white", fg=INK, justify="left", wraplength=850).pack(anchor="w")

    def _reset(self):
        self.analyzer.reset()
        self.chat.clear()
        self.events.clear()
        self.angles.clear()
        self.angle_trace.clear()
        self.last_trace_bucket = -1
        self.started_at = None
        self.camera_started_at = None
        self.tracking_stable_since = None
        self.tracking_acquired = False
        self.elapsed = 0
        self._show_checkin()

    def close(self):
        if self.voice:
            self.voice.stop()
        if self.tracker:
            self.tracker.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TButton", padding=(12, 7), font=("Segoe UI", 9, "bold"))
    style.configure("TCombobox", padding=5)
    PhysioPoseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

