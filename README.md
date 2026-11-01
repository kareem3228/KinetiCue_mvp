# PhysioPose AI — Python MVP

Local Python desktop demo for a seated knee extension. MediaPipe Heavy tracks hip, knee and ankle; one movement analyzer counts bend–extend–return cycles.

Run with `python -m pip install -r requirements.txt` then `python main.py`.

General cues: move smoothly, go only as far as comfortable, pause at the endpoint for the prescribed hold, and stop if pain increases. The clinician can set the hold duration. The camera cannot judge pain or keep a patient from over-stretching, so the comfortable limit remains the patient's own symptom-guided limit.

The session report shows repetitions, estimated movement range, cycle duration, endpoint-hold duration, and per-rep hold status (met, not met, or unable to assess). Cycle duration is descriptive; it does not prove control. A default camera-angle ceiling warns when the estimated knee approaches full straightening. This threshold is a prototype heuristic, not a validated safety limit, and the camera cannot sense pain. A tracking gap makes a check unable to assess. These camera statistics are not a clinical form score. Video is processed locally and is not saved or uploaded.

## Hands-free voice (experimental)

Voice recognition and speech output are optional. Install the app packages with `python -m pip install -r requirements.txt`, then install the isolated Gemini voice SDK with `python -m pip install --target voice_packages "google-genai>=1.0" "typing_extensions>=4.14.0"`. With `GEMINI_API_KEY` configured, the app streams microphone audio to **Gemini 3.5 Transcribe Live** for English recognition and falls back to local Vosk if the cloud connection fails. Place `vosk-model-small-en-us-0.15` in the `models` folder beside `main.py` for that fallback. English replies use the built-in Windows System.Speech voice. If voice dependencies are already present, refresh the bundled typing-extensions with `python -m pip install --target voice_packages --upgrade "typing_extensions>=4.14.0"`.

The instructor exposes 11 allowlisted tools: pause, resume, finish, repeat the last cue, read current activity, read the latest repetition, explain the latest configured check, summarize all completed repetitions, read the configured demo plan, record an explicitly reported symptom, and record explicit session feedback. The model selects one or more tools from what the user says; Python validates the tool names and accepted values, applies session controls, and stores self-reports in the local session JSON. For example, a report of worsening pain during exercise can trigger both pause and symptom-recording tools. The app never infers pain severity, diagnoses, or lets the model decide whether a camera check passed. Ask questions in English.

For conversational answers, put `GEMINI_API_KEY=your_key_here` in the ignored `.env` file in the same folder as `main.py`. Gemini uses the OpenAI-compatible chat endpoint. Without a working key/network, the app gives a small offline fallback and labels that status in the voice indicator.

Speech recognition is a prototype: noise and accents can cause transcription errors. Voice pause/stop phrases are recognized locally after an utterance endpoint; do not rely on voice as an emergency control. Use the on-screen Finish button too. Audio is not recorded to disk. Speech is not sent to the LLM unless converted into a recognized question; the transcript and session context go to the configured chat API.
