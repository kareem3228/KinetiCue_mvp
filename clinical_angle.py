"""Angle convention and per-repetition hold check."""
import math

DEFAULT_EXTENSION_CEILING_INCLUDED_ANGLE = 170.0


def camera_angle_to_flexion_degrees(included_angle):
    if included_angle is None:
        return None
    value=float(included_angle)
    return 180.0-value if math.isfinite(value) and 0<=value<=180 else None


def assess_rep_checks(event, minimum_hold_seconds=1.0, minimum_cycle_seconds=2.0, minimum_confidence=.50):
    trajectory=event.get("angle_trajectory") or []
    ready=bool(trajectory) and all(s.get("tracking_ready") is True and s.get("confidence",0)>=minimum_confidence for s in trajectory)
    status="unable_to_assess" if not ready else ("met" if event.get("extension_hold_seconds",0)>=minimum_hold_seconds else "not_met")
    angle = event.get("peak_angle_degrees")
    endpoint_status = "unable_to_assess" if not ready else ("past_default_ceiling" if angle > DEFAULT_EXTENSION_CEILING_INCLUDED_ANGLE else "within_default_ceiling")
    duration_status="unable_to_assess" if not ready else ("met" if event.get("rep_duration_seconds",0)>=minimum_cycle_seconds else "not_met")
    return {"hold":{"status":status,"observed_seconds":event.get("extension_hold_seconds"),"required_seconds":minimum_hold_seconds},"cycle_control":{"status":duration_status,"observed_seconds":event.get("rep_duration_seconds"),"minimum_seconds":minimum_cycle_seconds},"extension_ceiling":{"status":endpoint_status,"default_included_angle_degrees":DEFAULT_EXTENSION_CEILING_INCLUDED_ANGLE,"observed_included_angle_degrees":angle},"tracking_ready_throughout":ready}
