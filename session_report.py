"""Build and save local session reports for the knee-extension demo."""
from datetime import datetime
import json
from pathlib import Path
from statistics import mean


def build_report(repetitions, elapsed_seconds, angles, errors, pre_checkin, post_checkin, angle_trace=(), patient_reported_repetitions=None, movement_events=(), prescribed_hold_seconds=1.0, exercise_type="knee_extension", target_repetitions=8, tracked_joints=None):
    summary = "No reliable camera angle estimates were captured."
    if angles:
        summary = f"Camera-estimated knee flexion varied from {min(angles):.0f}° to {max(angles):.0f}° during the session; this is not a clinical measurement."
    holds = [event.get("endpoint_hold_seconds", event.get("extension_hold_seconds", 0)) for event in movement_events]
    durations = [event.get("rep_duration_seconds", 0) for event in movement_events]
    ranges = [event.get("excursion_degrees", 0) for event in movement_events]
    met = sum((e.get("form_checks") or {}).get("hold", {}).get("status") == "met" for e in movement_events)
    missed = sum((e.get("form_checks") or {}).get("hold", {}).get("status") == "not_met" for e in movement_events)
    unable = sum((e.get("form_checks") or {}).get("hold", {}).get("status") == "unable_to_assess" for e in movement_events)
    limit_count = sum((e.get("form_checks") or {}).get("extension_ceiling", {}).get("status") == "past_default_ceiling" for e in movement_events)
    limit_unable = sum((e.get("form_checks") or {}).get("extension_ceiling", {}).get("status") == "unable_to_assess" for e in movement_events)
    control_met=sum((e.get("form_checks") or {}).get("cycle_control",{}).get("status")=="met" for e in movement_events)
    control_missed=sum((e.get("form_checks") or {}).get("cycle_control",{}).get("status")=="not_met" for e in movement_events)
    control_unable=sum((e.get("form_checks") or {}).get("cycle_control",{}).get("status")=="unable_to_assess" for e in movement_events)
    performance = {
        "hold_goal_seconds": prescribed_hold_seconds,
        "hold_met": met, "hold_not_met": missed, "hold_unable_to_assess": unable,
        "cycle_pace_met":control_met,"cycle_pace_not_met":control_missed,"cycle_pace_unable_to_assess":control_unable,
        "hold_met_percent_of_assessable": round(100*met/(met+missed),1) if met+missed else None,
        "mean_camera_estimated_excursion_degrees": round(mean(ranges),1) if ranges else None,
        "mean_cycle_duration_seconds": round(mean(durations),2) if durations else None,
        "mean_endpoint_hold_seconds": round(mean(holds),2) if holds else None,
        "per_rep": [{"rep":i,"estimated_range_degrees":e.get("excursion_degrees"),"cycle_duration_seconds":e.get("rep_duration_seconds"),"endpoint_hold_seconds":e.get("endpoint_hold_seconds",e.get("extension_hold_seconds")),"hold_check":(e.get("form_checks") or {}).get("hold",{}).get("status","unknown"),"cycle_pace_check":(e.get("form_checks") or {}).get("cycle_control",{}).get("status","unknown"),"extension_ceiling_check":(e.get("form_checks") or {}).get("extension_ceiling",{}).get("status","not_configured")} for i,e in enumerate(movement_events,1)],
        "note":"Camera estimates are descriptive and not a clinical form score. Cycle duration does not establish movement control. Tracking gaps are unable to assess, not failures. Camera cannot determine pain or prevent over-stretching."
    }
    if exercise_type == "knee_extension":
        performance["past_default_ceiling"] = limit_count
        performance["extension_ceiling_unable_to_assess"] = limit_unable
    exercise_name = "Squat" if exercise_type == "squat" else "Seated knee extension"
    result = {"created_at":datetime.now().isoformat(timespec="seconds"),"exercise":exercise_type,"exercise_name":exercise_name,"required_joints":list(tracked_joints or (("shoulder","hip","knee","ankle") if exercise_type == "squat" else ("hip","knee","ankle"))),"measured_by_camera":{"pose_model":"MediaPipe Heavy","angle_convention":"2D knee included-angle movement estimate; not clinically validated","repetitions":repetitions,"target_repetitions":target_repetitions,"angle_summary":summary,"reliable_samples":len(angles),"angle_trace_interval_seconds":0.1,"angle_trace":list(angle_trace),"performance":performance,"movement_events":list(movement_events),"movement_errors":errors},"reported_by_patient":{"pre_session":pre_checkin,"post_session":post_checkin},"summary":f"Detected {repetitions} {exercise_name.lower()} movement cycles in {elapsed_seconds//60}m {elapsed_seconds%60}s. Patient feedback is self-reported; camera values are estimates."}
    return result


def save_report(report,directory=None):
    folder=Path(directory) if directory is not None else Path(__file__).resolve().parent/"session_reports"
    if not folder.is_absolute(): folder=Path(__file__).resolve().parent/folder
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"session_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return path
