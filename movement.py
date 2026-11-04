"""Rep counting for the seated knee-extension demo."""
from dataclasses import dataclass
import time


@dataclass
class MovementUpdate:
    phase: str
    hint: str
    repetitions: int
    event: dict | None = None


class KneeExtensionAnalyzer:
    repetition_target = 8
    flexed_position_limit = 140
    start_excursion_degrees = 15
    count_excursion_degrees = 25
    tracking_dropout_grace_seconds = 1.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.repetitions = 0
        self.phase = "waiting"
        self.started_at = 0.0
        self.flexed_angle = None
        self.peak_angle = None
        self.extension_hold_seconds = 0.0
        self.hold_last_sample_at = None
        self.lowering_candidate_count = 0
        self.minimum_tracking_confidence = None
        self.peak_tracking_confidence = None
        self.trajectory_samples = []
        self.trajectory_start_at = None
        self.trajectory_last_sample_at = None
        self.tracking_lost_at = None
        self.last_error_at = 0.0
        self.errors = []

    def update(self, angle, reliable, confidence=1.0, timestamp=None, tracking_ready=False):
        now = time.monotonic() if timestamp is None else float(timestamp)
        if not reliable or angle is None:
            if self.phase in ("flexed", "extending", "lowering") and self.trajectory_start_at is not None:
                if self.trajectory_last_sample_at is None or now - self.trajectory_last_sample_at >= 0.05:
                    self.trajectory_samples.append({
                        "elapsed_seconds": round(now - self.trajectory_start_at, 3),
                        "angle_degrees": None,
                        "confidence": 0.0,
                        "tracking_ready": False,
                    })
                    self.trajectory_last_sample_at = now
            if self.tracking_lost_at is None:
                self.tracking_lost_at = now
            if now - self.tracking_lost_at > self.tracking_dropout_grace_seconds:
                self.phase = "paused"
                hint = "Keep hip, knee and ankle visible"
            else:
                hint = "Reacquiring landmarks; hold your position briefly"
            return MovementUpdate(self.phase, hint, self.repetitions)
        if self.tracking_lost_at is not None:
            if self.phase == "paused":
                self.phase = "waiting"
                self.flexed_angle = None
                self.peak_angle = None
                self.extension_hold_seconds = 0.0
                self.hold_last_sample_at = None
                self.trajectory_samples = []
                self.trajectory_start_at = None
                self.trajectory_last_sample_at = None
            self.tracking_lost_at = None
        if self.phase == "paused":
            self.phase = "waiting"
        event = None
        if self.phase == "waiting":
            if angle <= self.flexed_position_limit:
                self.phase, self.started_at = "flexed", now
                self.flexed_angle = self.peak_angle = angle
                self.extension_hold_seconds = 0.0
                self.hold_last_sample_at = None
                self.lowering_candidate_count = 0
                self.minimum_tracking_confidence = confidence
                self.peak_tracking_confidence = confidence
                self.trajectory_samples = []
                self.trajectory_start_at = None
                self.trajectory_last_sample_at = None
                self._record_trajectory_sample(now, angle, confidence, tracking_ready)
        else:
            self._record_trajectory_sample(now, angle, confidence, tracking_ready)
        if self.phase == "flexed":
            self.minimum_tracking_confidence = min(self.minimum_tracking_confidence or confidence, confidence)
            self.flexed_angle = min(self.flexed_angle if self.flexed_angle is not None else angle, angle)
            if self.peak_angle is None or angle > self.peak_angle:
                self.peak_angle = angle
                self.peak_tracking_confidence = confidence
            if self.peak_angle - self.flexed_angle >= self.start_excursion_degrees:
                self.phase = "extending"
            if now - self.started_at > 5 and angle <= self.flexed_position_limit:
                self.phase = "waiting"
                self.trajectory_samples = []
                self.trajectory_start_at = None
                self.trajectory_last_sample_at = None
        elif self.phase == "extending":
            self.minimum_tracking_confidence = min(self.minimum_tracking_confidence or confidence, confidence)
            if self.peak_angle is None or angle > self.peak_angle:
                self.peak_angle = angle
                self.peak_tracking_confidence = confidence
            excursion = self.peak_angle - self.flexed_angle
            if excursion >= self.count_excursion_degrees:
                # Reaching the minimum excursion does not mean extension is
                # finished. Keep updating the peak until there is a sustained
                # drop away from it, so later extension is not lost.
                if angle >= self.peak_angle - 7:
                    if self.hold_last_sample_at is not None:
                        sample_gap = now - self.hold_last_sample_at
                        if 0 < sample_gap <= .2:
                            self.extension_hold_seconds += sample_gap
                    self.hold_last_sample_at = now
                    self.lowering_candidate_count = 0
                else:
                    self.hold_last_sample_at = None
                    self.lowering_candidate_count += 1
                if self.lowering_candidate_count >= 3:
                    self.phase = "lowering"
                    self.hold_last_sample_at = None
                    self.lowering_candidate_count = 0
            elif angle <= self.flexed_angle + 10:
                self.phase, self.started_at = "flexed", now
                self.flexed_angle = self.peak_angle = angle
                self.extension_hold_seconds = 0.0
                self.hold_last_sample_at = None
                self.lowering_candidate_count = 0
                self.minimum_tracking_confidence = confidence
                self.peak_tracking_confidence = confidence
                self.trajectory_samples = []
                self.trajectory_start_at = None
                self.trajectory_last_sample_at = None
                self._record_trajectory_sample(now, angle, confidence, tracking_ready)
        elif self.phase == "lowering":
            self.minimum_tracking_confidence = min(self.minimum_tracking_confidence or confidence, confidence)
            # A short dip can look like lowering even while the person is still
            # extending. Re-open peak tracking if the signal rises clearly again.
            if angle >= self.peak_angle + 3:
                self.peak_angle = angle
                self.peak_tracking_confidence = confidence
                self.phase = "extending"
                self.lowering_candidate_count = 0
                self.extension_hold_seconds = 0.0
                self.hold_last_sample_at = now
                angle_is_near_peak = True
            else:
                angle_is_near_peak = angle >= self.peak_angle - 7
            if angle_is_near_peak:
                if self.hold_last_sample_at is not None:
                    sample_gap = now - self.hold_last_sample_at
                    if 0 < sample_gap <= .2:
                        self.extension_hold_seconds += sample_gap
                self.hold_last_sample_at = now
            else:
                self.hold_last_sample_at = None
            if angle <= self.flexed_angle + 10:
                self.repetitions += 1
                event = {
                    "type": "rep_completed",
                    "exercise_type": "knee_extension",
                    "movement_cycle_status": "completed",
                    "form_assessment": "not_assessed",
                    "angle_degrees": round(angle),
                    "peak_angle_degrees": round(self.peak_angle, 1),
                    "excursion_degrees": round(self.peak_angle - self.flexed_angle),
                    "extension_hold_seconds": round(self.extension_hold_seconds, 2),
                    "endpoint_hold_seconds": round(self.extension_hold_seconds, 2),
                    "rep_duration_seconds": round(max(0.0, now - self.started_at), 2),
                    "tracking_confidence_min": round(self.minimum_tracking_confidence or confidence, 2),
                    "peak_tracking_confidence": round(self.peak_tracking_confidence or confidence, 2),
                    "angle_trajectory": list(self.trajectory_samples),
                }
                self.phase = "waiting"
                self.flexed_angle = None
                self.peak_angle = None
                self.extension_hold_seconds = 0.0
                self.hold_last_sample_at = None
                self.lowering_candidate_count = 0
                self.minimum_tracking_confidence = None
                self.peak_tracking_confidence = None
                self.trajectory_samples = []
                self.trajectory_start_at = None
                self.trajectory_last_sample_at = None
        hint = {
            "waiting": "Bend your knee to begin a repetition",
            "flexed": "Extend slowly when comfortable",
            "extending": "Slow and comfortable is just right",
            "lowering": "Lower your leg with control",
            "paused": "Keep your whole leg in view",
        }[self.phase]
        return MovementUpdate(self.phase, hint, self.repetitions, event)

    def _record_trajectory_sample(self, now, angle, confidence, tracking_ready):
        if self.trajectory_last_sample_at is not None and now - self.trajectory_last_sample_at < 0.05:
            return
        if self.trajectory_start_at is None:
            self.trajectory_start_at = now
        self.trajectory_samples.append({
            "elapsed_seconds": round(now - self.trajectory_start_at, 3),
            "angle_degrees": round(float(angle), 1),
            "confidence": round(float(confidence), 2),
            "tracking_ready": bool(tracking_ready),
        })
        self.trajectory_last_sample_at = now


class SquatAnalyzer:
    """Count a squat from a user's standing baseline, deepest point, and return.

    The knee angle is a 2D camera estimate. The counter adapts to the user's
    standing angle and does not require a prescribed squat depth.
    """

    repetition_target = 8
    minimum_excursion_degrees = 14.0
    tracking_dropout_grace_seconds = 1.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.repetitions = 0
        self.phase = "waiting"
        self.baseline = None
        self.start_angle = None
        self.extreme_angle = None
        self.started_at = None
        self.turning = False
        self.endpoint_hold_seconds = 0.0
        self.hold_last_sample_at = None
        self.minimum_tracking_confidence = None
        self.peak_tracking_confidence = None
        self.trajectory_samples = []
        self.trajectory_start_at = None
        self.trajectory_last_sample_at = None
        self.tracking_lost_at = None
        self.errors = []

    def update(self, angle, reliable, confidence=1.0, timestamp=None, tracking_ready=False):
        now = time.monotonic() if timestamp is None else float(timestamp)
        if not reliable or angle is None:
            if self.tracking_lost_at is None:
                self.tracking_lost_at = now
            if self.started_at is not None and (self.trajectory_last_sample_at is None or now - self.trajectory_last_sample_at >= .05):
                self._record_sample(None, now, 0.0, False)
            if now - self.tracking_lost_at > self.tracking_dropout_grace_seconds:
                self.phase = "paused"
                hint = "Reacquiring shoulder, hip, knee and ankle tracking."
            else:
                hint = "Tracking dipped; hold your position briefly."
            return MovementUpdate(self.phase, hint, self.repetitions)

        angle = float(angle)
        if self.tracking_lost_at is not None:
            if self.phase == "paused":
                if self.started_at is not None:
                    self.phase = "returning" if self.turning else "moving"
                else:
                    self.phase = "waiting"
            self.tracking_lost_at = None

        if self.phase in ("waiting", "paused"):
            self.phase = "waiting"
            if self.baseline is None or angle > self.baseline + 3.0:
                # Standing straighter refreshes the start position, preventing
                # an end-of-session stand-up from being counted as a rep.
                self.baseline = angle
            elif abs(angle - self.baseline) <= 3.0:
                self.baseline = .85 * self.baseline + .15 * angle

            if self.baseline - angle >= 8.0:
                self.start_angle = self.baseline
                self.extreme_angle = angle
                self.started_at = now
                self.phase = "moving"
                self.turning = False
                self.endpoint_hold_seconds = 0.0
                self.hold_last_sample_at = now
                self.minimum_tracking_confidence = confidence
                self.peak_tracking_confidence = confidence
                self.trajectory_samples = []
                self.trajectory_start_at = None
                self.trajectory_last_sample_at = None
                self._record_sample(angle, now, confidence, tracking_ready)
            return MovementUpdate(self.phase, self._hint(), self.repetitions)

        self._record_sample(angle, now, confidence, tracking_ready)
        self.minimum_tracking_confidence = min(self.minimum_tracking_confidence or confidence, confidence)
        if self.extreme_angle is None or angle < self.extreme_angle:
            self.extreme_angle = angle
            self.peak_tracking_confidence = confidence

        excursion = max(0.0, self.start_angle - self.extreme_angle)
        if self.phase == "moving":
            if excursion >= 7.0 and angle - self.extreme_angle >= 3.0:
                self.turning = True
                self.phase = "returning"
        elif self.phase == "returning":
            # If the patient bends farther after starting to rise, follow the
            # new bottom instead of ending the repetition early.
            if angle < self.extreme_angle - 2.0:
                self.extreme_angle = angle
                excursion = max(0.0, self.start_angle - self.extreme_angle)
                self.phase = "moving"
                self.turning = False
                self.hold_last_sample_at = now
            else:
                near_bottom = angle <= self.extreme_angle + 7.0
                if near_bottom:
                    if self.hold_last_sample_at is not None:
                        gap = now - self.hold_last_sample_at
                        if 0 < gap <= .2:
                            self.endpoint_hold_seconds += gap
                    self.hold_last_sample_at = now
                else:
                    self.hold_last_sample_at = None

                return_margin = max(5.0, excursion * .28)
                if angle >= self.start_angle - return_margin:
                    if excursion >= self.minimum_excursion_degrees:
                        self.repetitions += 1
                        event = {
                            "type": "rep_completed",
                            "exercise_type": "squat",
                            "movement_cycle_status": "completed",
                            "form_assessment": "not_assessed",
                            "rep_number": self.repetitions,
                            "start_angle_degrees": round(self.start_angle, 1),
                            "peak_angle_degrees": round(self.extreme_angle, 1),
                            "excursion_degrees": round(excursion, 1),
                            "endpoint_hold_seconds": round(self.endpoint_hold_seconds, 2),
                            "rep_duration_seconds": round(max(0.0, now - self.started_at), 2),
                            "tracking_confidence_min": round(self.minimum_tracking_confidence or confidence, 2),
                            "peak_tracking_confidence": round(self.peak_tracking_confidence or confidence, 2),
                            "angle_trajectory": list(self.trajectory_samples),
                        }
                        self.baseline = angle
                        self._clear_cycle()
                        return MovementUpdate("waiting", self._hint(), self.repetitions, event)
                    self.baseline = angle
                    self._clear_cycle()
                    return MovementUpdate("waiting", self._hint(), self.repetitions)

        return MovementUpdate(self.phase, self._hint(), self.repetitions)

    def _hint(self):
        return {
            "waiting": "Stand comfortably, then begin a comfortable squat.",
            "moving": "Lower at a comfortable pace; use support if needed.",
            "returning": "Return to standing at a comfortable pace.",
            "paused": "Reacquiring shoulder, hip, knee and ankle tracking.",
        }.get(self.phase, "Move within your comfortable range.")

    def _clear_cycle(self):
        self.phase = "waiting"
        self.start_angle = None
        self.extreme_angle = None
        self.started_at = None
        self.turning = False
        self.endpoint_hold_seconds = 0.0
        self.hold_last_sample_at = None
        self.minimum_tracking_confidence = None
        self.peak_tracking_confidence = None
        self.trajectory_samples = []
        self.trajectory_start_at = None
        self.trajectory_last_sample_at = None

    def _record_sample(self, angle, now, confidence, tracking_ready):
        if self.trajectory_last_sample_at is not None and now - self.trajectory_last_sample_at < .05:
            return
        if self.trajectory_start_at is None:
            self.trajectory_start_at = now
        self.trajectory_samples.append({
            "elapsed_seconds": round(now - self.trajectory_start_at, 3),
            "angle_degrees": round(float(angle), 1) if angle is not None else None,
            "confidence": round(float(confidence), 2),
            "tracking_ready": bool(tracking_ready),
        })
        self.trajectory_last_sample_at = now
