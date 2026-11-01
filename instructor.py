"""Conversational instructor over an OpenAI-compatible chat endpoint."""
import json
import os
from pathlib import Path
import requests


class Instructor:
    TOOLS = [
        {"type": "function", "function": {"name": "pause_exercise", "description": "Pause repetition measurement when the user clearly asks to pause or stop, or reports pain during the exercise. Do not finish the session.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "resume_exercise", "description": "Resume repetition measurement only when the user clearly asks to continue or resume.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "finish_exercise", "description": "Finish the current exercise session only when the user clearly says they are done or asks to finish.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "repeat_last_cue", "description": "Repeat the most recent movement cue when the user asks to repeat it.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "get_current_activity", "description": "Read the latest pose-tracking snapshot when the user asks what their leg is doing, what movement phase is detected, or whether tracking is working.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "get_last_rep_result", "description": "Read the latest completed repetition and its measured hold, pace, extension-warning, and tracking results when the user asks how the last rep went or why a check missed.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "coach_from_latest_check", "description": "When the user asks how to improve or correct the latest repetition, read its configured check results and give one specific cue. Do not invent a clinical form fault.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "get_session_summary", "description": "Read aggregate hold, pace, extension-warning, and tracking-check counts across all completed repetitions when the user asks about overall progress or session performance.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "get_current_plan", "description": "Read the exercise targets and instructions currently configured in this demo when the user asks what the plan says. Do not change the plan.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "record_symptom_report", "description": "Record a symptom the user explicitly reports during the session. Capture only what they said; do not infer severity or diagnose. If they report current or worsening pain, also call pause_exercise.", "parameters": {"type": "object", "properties": {"report": {"type": "string", "description": "Brief faithful summary of the user's self-reported symptom."}, "severity_0_to_10": {"type": "integer", "minimum": 0, "maximum": 10, "description": "Only include if the user explicitly gives a number from 0 to 10."}, "body_area": {"type": "string", "description": "Only include the body area the user explicitly identifies."}, "timing": {"type": "string", "description": "When the user says it happened, if stated."}}, "required": ["report"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "record_session_feedback", "description": "Save self-reported session feedback when the user explicitly asks to record it or clearly provides post-exercise feedback. Do not invent missing values.", "parameters": {"type": "object", "properties": {"feeling": {"type": "string"}, "pain": {"type": "string"}, "actual_repetitions": {"type": "integer", "minimum": 0}, "difficulty": {"type": "string"}, "notes": {"type": "string"}, "report": {"type": "string", "description": "Faithful summary of feedback that does not fit the structured fields."}}, "additionalProperties": False}}},
    ]
    ALLOWED_TOOLS = {"pause_exercise", "resume_exercise", "finish_exercise", "repeat_last_cue", "get_current_activity", "get_last_rep_result", "coach_from_latest_check", "get_session_summary", "get_current_plan", "record_symptom_report", "record_session_feedback"}
    READ_ONLY_TOOLS = {"repeat_last_cue", "get_current_activity", "get_last_rep_result", "coach_from_latest_check", "get_session_summary", "get_current_plan"}
    WRITE_TOOLS = {"record_symptom_report", "record_session_feedback"}

    def __init__(self):
        self._load_env_file()
        self.api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.model = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

    @staticmethod
    def _load_env_file():
        """Load local .env values without requiring another dependency or printing secrets."""
        env_path = Path(__file__).resolve().parent / ".env"
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name, value = name.strip(), value.strip().strip("\"'")
                if name and value:
                    os.environ.setdefault(name, value)
        except OSError:
            pass

    def reply(self, prompt, context, history=(), event_type=None):
        if not self.api_key:
            return {"text": self.fallback(prompt, context, event_type), "action": None}
        messages = [{"role": "system", "content": (
            "You are a warm, conversational voice instructor during a physiotherapist-prescribed seated knee-extension demo. "
            "Answer the person's actual question directly, naturally, and briefly (usually 1-3 sentences). "
            "Always reply in clear, natural English suitable for spoken conversation. "
            "Vary wording and avoid repeating generic exercise reminders unless relevant. Ask a short follow-up only when needed. "
            "Use recent conversation so follow-up questions make sense. Do not diagnose, prescribe, or change the clinician's plan. "
            "If they report increased pain or a new concerning symptom, advise pausing/stopping and contacting their physiotherapist. "
            "Camera values are rough estimates, not clinical truth; don't claim to judge form or safety. "
            "Use the matching tool for clear pause/stop, resume/continue, finish/done, or repeat requests. "
            "Use get_current_activity for questions about the live camera or what movement phase is happening. Use get_last_rep_result for questions about the most recent completed rep or a missed check. "
            "Use coach_from_latest_check when the user asks what to change or how to correct the most recent rep. Only give a correction for a configured check that actually missed; if tracking was unable to assess, explain the tracking issue, and if all checks were met, say the app has no measured correction for that rep. "
            "For missed holds, cue a pause near the comfortable endpoint before lowering; never tell the user to fully straighten or move farther. For an extension warning, suggest easing back slightly. For a fast cycle, cue slowing the whole movement within the comfortable range. "
            "Use get_session_summary for session-wide trends and get_current_plan for the currently configured exercise targets. Never infer an all-session trend from the recent-events sample alone. "
            "Use record_symptom_report only for symptoms the user actually reports. Do not ask for a pain score after every repetition. If worsening pain is reported during exercise, request both pause_exercise and record_symptom_report. Use record_session_feedback for explicit session feedback; treat all such details as patient-reported. "
            "Do not call a tool when the person is asking hypothetically or discussing the app. For pain during exercise, pause first, then briefly advise them to stop if pain increases and contact their physiotherapist. "
            "Do not claim a rep or form check passed unless the session data says so. "
            "Session context (for answering, not for making clinical decisions): " + json.dumps(context, ensure_ascii=True, separators=(",", ":"))
        )}]
        messages.extend(list(history)[-6:])
        messages.append({"role": "user", "content": prompt or f"A reliable movement event occurred: {event_type}. Give one short cue."})
        response = self._chat_completion(messages, tools=self.TOOLS)
        assistant = response["choices"][0]["message"]
        calls = assistant.get("tool_calls") or []
        if not calls:
            return {"text": (assistant.get("content") or "").strip(), "action": None}

        selected_calls = []
        for call in calls:
            function = call.get("function") or {}
            action = function.get("name")
            if action not in self.ALLOWED_TOOLS:
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            selected_calls.append({"id": call.get("id"), "name": action, "arguments": arguments, "raw": call})
        if not selected_calls:
            return {"text": (assistant.get("content") or "").strip(), "action": None, "actions": []}

        messages.append({"role": "assistant", "content": assistant.get("content"), "tool_calls": [item["raw"] for item in selected_calls]})
        for item in selected_calls:
            action, arguments, call_id = item["name"], item["arguments"], item["id"]
            if action in self.READ_ONLY_TOOLS:
                tool_result = self._read_tool(action, context)
            elif action in self.WRITE_TOOLS:
                tool_result = {"accepted": True, "action": action, "recording": "the app will store only the supplied self-report fields locally in this session report"}
            else:
                tool_result = {"accepted": True, "action": action}
            if call_id:
                messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(tool_result, ensure_ascii=True)})

        try:
            final = self._chat_completion(messages)
            text = (final["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            text = "Okay. I’ve noted that for this session." if any(item["name"] in self.WRITE_TOOLS for item in selected_calls) else "Okay, I’ve updated the session as requested."
        actions = [{"name": item["name"], "arguments": item["arguments"]} for item in selected_calls]
        return {"text": text, "action": actions[0]["name"], "arguments": actions[0]["arguments"], "actions": actions}

    @staticmethod
    def _read_tool(action, context):
        if action == "get_current_activity":
            return context.get("live_activity") or {"available": False, "reason": "No live camera snapshot is available."}
        if action == "get_session_summary":
            return context.get("session_summary") or {"available": False, "reason": "No session summary is available yet."}
        if action == "get_current_plan":
            return context.get("current_plan") or {"available": False, "reason": "No plan is configured."}
        if action in {"repeat_last_cue", "get_last_rep_result", "coach_from_latest_check"}:
            recent = context.get("recent_events") or []
            if not recent:
                return {"available": False, "reason": "No completed repetitions yet.", "repetitions": context.get("repetitions", 0)}
            latest = recent[-1]
            if action == "repeat_last_cue":
                return {"rep_number": latest.get("rep_number"), "cue": latest.get("instructor_feedback")}
            if action == "coach_from_latest_check":
                checks = latest.get("form_checks", {})
                hold = checks.get("hold", {})
                pace = checks.get("cycle_control", {})
                extension = checks.get("extension_ceiling", {})
                if any(check.get("status") == "unable_to_assess" for check in (hold, pace, extension)):
                    suggestion = "Tracking was insufficient for one or more checks; explain the camera visibility guidance instead of guessing about form."
                elif extension.get("status") == "past_default_ceiling":
                    suggestion = "The camera extension warning was reached; suggest easing back slightly if comfortable. Do not cue more extension."
                elif hold.get("status") == "not_met":
                    suggestion = "The measured endpoint hold was short; cue a pause near the comfortable endpoint before lowering. Do not cue more extension."
                elif pace.get("status") == "not_met":
                    suggestion = "The measured full cycle was faster than its configured duration; cue slowing the bend, straighten, and return within comfort."
                else:
                    suggestion = "The configured timing and camera-warning checks did not miss; say the app has no measured correction and did not assess clinical form."
                return {"rep_number": latest.get("rep_number"), "checks": checks, "recorded_result": latest.get("instructor_feedback"), "cue_constraint": suggestion}
            return latest
        return {"available": False}

    def rep_feedback(self, event):
        """Phrase one cue from deterministic check results; the model does not judge form."""
        checks = event.get("form_checks", {})
        hold = checks.get("hold", {})
        control = checks.get("cycle_control", {})
        limit = checks.get("extension_ceiling", {})
        if limit.get("status") == "past_default_ceiling":
            reason = {"check": "camera extension warning", "instruction": "Suggest easing back slightly if comfortable."}
        elif hold.get("status") == "not_met":
            reason = {"check": "endpoint hold not met", "observed_seconds": hold.get("observed_seconds"), "required_seconds": hold.get("required_seconds"), "instruction": "Cue a brief pause at the comfortable endpoint on the next repetition, only if comfortable."}
        elif control.get("status") == "not_met":
            reason = {"check": "cycle duration below configured minimum", "observed_seconds": control.get("observed_seconds"), "minimum_seconds": control.get("minimum_seconds"), "instruction": "Cue a slower bend, straighten, and return on the next repetition."}
        else:
            return ""

        messages = [
            {"role": "system", "content": (
                "Write one natural, concise spoken coaching cue (one sentence, under 22 words) for the stated timing result. "
                "The check status and numbers are computed by the app; do not reinterpret them or claim to assess form. "
                "Do not call it a bad rep, diagnose, mention pain unless the event does, or change the clinician's plan. "
                "Be specific, supportive, and avoid generic filler."
            )},
            {"role": "user", "content": json.dumps({"rep": event.get("rep_number"), **reason}, ensure_ascii=True)},
        ]
        response = self._chat_completion(messages, max_tokens=45, temperature=.25, timeout=(2, 5))
        return (response["choices"][0]["message"].get("content") or "").strip()

    def _chat_completion(self, messages, tools=None, max_tokens=100, temperature=.5, timeout=(3, 7)):
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            payload.update({"tools": tools, "tool_choice": "auto"})
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def fallback(prompt, context, event_type=None):
        text = (prompt or "").lower()
        if event_type == "rep_completed":
            return f"Movement cycle detected: {context.get('repetitions', 0)} of 8. Follow your prescribed comfortable range."
        if any(word in text for word in ("angle", "measurement", "degrees")):
            return "The camera estimates the angle from your hip, knee and ankle. It can shift with camera position, so use it as a rough movement trace rather than a clinical measurement."
        if any(word in text for word in ("pain", "hurt", "symptom")):
            return "Stop if pain increases or you notice a new symptom, and contact your physiotherapist. I can’t assess symptoms or change your plan."
        if any(word in text for word in ("how", "exercise", "do this", "what should", "what do")):
            return "Sit tall in a stable chair. Slowly straighten your knee within the range your physiotherapist prescribed, then lower it with control. Stop if pain increases."
        if any(word in text for word in ("rep", "count", "repetition")):
            return f"I have counted {context.get('repetitions', 0)} movement cycles so far. I count a bend, straighten, and return sequence; camera tracking can miss cycles if the leg leaves view."
        return "The live chat model is unavailable right now, so I can’t answer that conversationally. I can still hear pause, resume, finish, and repeat."
