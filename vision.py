"""Local webcam and MediaPipe pose tracking. Frames stay on this computer."""
from dataclasses import dataclass
import time

import cv2
import mediapipe as mp
import numpy as np
from exercise_catalog import get_exercise


@dataclass
class Reading:
    reliable: bool
    angle: float | None = None
    confidence: float = 0.0
    reason: str = ""
    side: str = ""
    landmark_scores: tuple[float, ...] = (0.0, 0.0, 0.0)
    ready: bool = False
    guidance: str = ""
    joint_names: tuple[str, ...] = ("hip", "knee", "ankle")


def knee_angle(hip, knee, ankle):
    return knee_angle_xy((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))


def knee_angle_xy(hip, knee, ankle):
    a = np.array([hip[0] - knee[0], hip[1] - knee[1]], dtype=float)
    b = np.array([ankle[0] - knee[0], ankle[1] - knee[1]], dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return None
    cosine = np.clip(np.dot(a, b) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class PoseTracker:
    def __init__(self, camera_index=0, preferred_side="auto", model_complexity=2, exercise_type="knee_extension"):
        self.camera_index = camera_index
        self.camera = None
        self.pose = None
        self.preferred_side = preferred_side
        self.exercise_type = exercise_type if exercise_type in ("knee_extension", "squat") else "knee_extension"
        self.exercise = get_exercise(self.exercise_type)
        if model_complexity not in (0, 1, 2):
            raise ValueError("MediaPipe model complexity must be 0, 1 or 2.")
        self.model_complexity = model_complexity
        self.side = None if preferred_side == "auto" else preferred_side
        self.samples = []
        self.smoothed = None
        self.last_reliable_at = None

    def start(self):
        self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            self.camera = None
            raise RuntimeError("Could not open a camera. Check that it is connected and not in use by another app.")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.pose = mp.solutions.pose.Pose(model_complexity=self.model_complexity, smooth_landmarks=True, min_detection_confidence=.35, min_tracking_confidence=.35)

    def read(self):
        if self.camera is None:
            return None, Reading(False, reason="Camera is off")
        ok, frame = self.camera.read()
        if not ok:
            return None, Reading(False, reason="Waiting for camera frame")
        # Run pose inference on the camera's original orientation so MediaPipe's
        # anatomical LEFT/RIGHT names match the user. Mirror only the preview.
        result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = cv2.flip(frame, 1)
        if not result.pose_landmarks:
            if self.last_reliable_at is None or time.monotonic() - self.last_reliable_at > .4:
                self.samples.clear()
                self.smoothed = None
            required = ", ".join(self.exercise["required_joints"])
            cv2.putText(frame, f"No pose: show {required}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, .58, (60, 190, 240), 2, cv2.LINE_AA)
            guidance = f"No pose found. {self.exercise['camera_instructions']}"
            return frame, Reading(False, reason=guidance, guidance=guidance, joint_names=self.exercise["required_joints"])
        landmarks = result.pose_landmarks.landmark
        pose_landmark = mp.solutions.pose.PoseLandmark
        point_ids = {
            "left": {"shoulder": pose_landmark.LEFT_SHOULDER.value, "hip": pose_landmark.LEFT_HIP.value, "knee": pose_landmark.LEFT_KNEE.value, "ankle": pose_landmark.LEFT_ANKLE.value},
            "right": {"shoulder": pose_landmark.RIGHT_SHOULDER.value, "hip": pose_landmark.RIGHT_HIP.value, "knee": pose_landmark.RIGHT_KNEE.value, "ankle": pose_landmark.RIGHT_ANKLE.value},
        }
        required_names = self.exercise["required_joints"]
        sides = {side: {name: landmarks[ids[name]] for name in required_names} for side, ids in point_ids.items()}
        scores = {side: sum(float(point.visibility) for point in points.values()) / len(points) for side, points in sides.items()}
        # Pick a side once, then keep it for the whole session. Switching to
        # whichever leg happens to score higher frame-by-frame causes jumps.
        if self.side is None:
            self.side = max(scores, key=scores.get)
        selected = sides[self.side]
        hip, knee, ankle = selected["hip"], selected["knee"], selected["ankle"]
        raw_points = [selected[name] for name in required_names]
        point_scores = tuple(float(point.visibility) for point in raw_points)
        confidence = min(point_scores)
        angle = knee_angle(hip, knee, ankle)

        # Draw the selected chain even when confidence is too low to measure.
        # This lets us see whether the model missed a point or the view is poor.
        height, width = frame.shape[:2]
        segment_lengths = [float(np.linalg.norm(np.array([a.x - b.x, a.y - b.y]))) for a, b in zip(raw_points, raw_points[1:])]
        geometry_ok = all(length >= .035 for length in segment_lengths)
        points = [(int(np.clip(1 - point.x, 0, 1) * (width - 1)), int(np.clip(point.y, 0, 1) * (height - 1))) for point in raw_points]
        reliable = all(score >= .35 for score in point_scores) and angle is not None and geometry_ok
        color = (115, 210, 145) if reliable else (40, 170, 245)
        for first, second in zip(points, points[1:]):
            cv2.line(frame, first, second, color, 4, cv2.LINE_AA)
        names = tuple("HIP/WAIST" if name == "hip" else name.upper() for name in required_names)
        for name, point, score in zip(names, points, point_scores):
            cv2.circle(frame, point, 8, color, -1, cv2.LINE_AA)
            cv2.circle(frame, point, 8, (245, 250, 245), 2, cv2.LINE_AA)
            label_y = max(18, point[1] - 12)
            cv2.putText(frame, f"{name} {score:.2f}", (point[0] + 8, label_y), cv2.FONT_HERSHEY_SIMPLEX, .48, (250, 250, 250), 2, cv2.LINE_AA)

        weak = [f"{name} {score:.2f}" for name, score in zip(required_names, point_scores) if score < .35]
        outside = [name for name, point in zip(required_names, raw_points) if not (0 <= point.x <= 1 and 0 <= point.y <= 1)]
        ready = all(score >= .50 for score in point_scores) and not outside and geometry_ok and angle is not None
        if outside:
            guidance = f"{', '.join(outside).title()} out of frame. {self.exercise['camera_instructions']}"
        elif not geometry_ok:
            guidance = f"The required joints are hard to measure from this view. {self.exercise['camera_instructions']}"
        else:
            joint_scores = tuple(zip(("hip/waist" if name == "hip" else name for name in required_names), point_scores))
            weak_for_guidance = [(name, score) for name, score in joint_scores if score < .50]
            if weak_for_guidance:
                tips = {
                    "shoulder": "move the camera back or up until your shoulder is visible",
                    "hip/waist": "reframe slightly upward/back so your hip and upper thigh are visible",
                    "knee": "clear anything covering the knee and keep the camera directly beside you",
                    "ankle": "move the camera back and tilt it slightly down so the ankle and foot are in frame",
                }
                details = "; ".join(f"{name} {score:.2f}: {tips[name]}" for name, score in weak_for_guidance)
                guidance = f"Tracking confidence is low ({details}). Adjust the view; better light can also help."
            else:
                guidance = "Tracking is ready. Keep the camera in the recommended position."
        side_label = f"{self.side.upper()} SIDE"
        cv2.putText(frame, side_label, (14, height - 15), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)

        if not reliable or outside:
            if self.last_reliable_at is None or time.monotonic() - self.last_reliable_at > .4:
                self.samples.clear()
                self.smoothed = None
            details = ", ".join(weak) if weak else ""
            if outside:
                details = (details + "; " if details else "") + "out of frame: " + ", ".join(outside)
            reason = "Low landmark tracking" + (f": {details}" if details else f". Keep {', '.join(required_names)} in view.")
            return frame, Reading(False, confidence=confidence, reason=reason, side=self.side, landmark_scores=point_scores, ready=ready, guidance=guidance, joint_names=required_names)
        self.last_reliable_at = time.monotonic()
        self.samples = (self.samples + [angle])[-5:]
        angle = float(np.median(self.samples))
        self.smoothed = angle if self.smoothed is None else self.smoothed * .65 + angle * .35
        return frame, Reading(True, self.smoothed, confidence, side=self.side, landmark_scores=point_scores, ready=ready, guidance=guidance, joint_names=required_names)

    def stop(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.pose is not None:
            self.pose.close()
            self.pose = None
        self.side = None if self.preferred_side == "auto" else self.preferred_side
        self.samples.clear()
        self.smoothed = None
        self.last_reliable_at = None
