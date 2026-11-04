"""Exercise-specific camera requirements and patient-facing setup text."""

EXERCISES = {
    "knee_extension": {
        "name": "Seated knee extension",
        "short_name": "Knee extension",
        "target_repetitions": 8,
        "required_joints": ("hip", "knee", "ankle"),
        "joint_labels": ("HIP", "KNEE", "ANKLE"),
        "checkin_instructions": "Sit tall in a stable chair. Slowly straighten your leg, then lower it with control.",
        "camera_instructions": "Sit side-on to the camera. Place it beside the exercising leg at knee height and move it back until the hip, knee and ankle are visible.",
    },
    "squat": {
        "name": "Squat",
        "short_name": "Squat",
        "target_repetitions": 8,
        "required_joints": ("shoulder", "hip", "knee", "ankle"),
        "joint_labels": ("SHOULDER", "HIP", "KNEE", "ANKLE"),
        "checkin_instructions": "Stand near a stable chair or counter for support. Bend and return within a comfortable range, using support if needed.",
        "camera_instructions": "Stand side-on to the camera. Place it near hip height and move it back until the shoulder, hip, knee and ankle on the camera-facing side are visible.",
    },
}


def get_exercise(exercise_type):
    """Return a known exercise definition, defaulting to the stable extension flow."""
    return EXERCISES.get(exercise_type, EXERCISES["knee_extension"])
