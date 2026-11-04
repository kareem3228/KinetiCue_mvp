# KinetiCue MVP

A Windows Python demo that tracks seated knee extensions or squats, counts reps, and saves session reports. The camera checks only the joints needed for the selected exercise: hip, knee, and ankle for extensions; shoulder, hip, knee, and ankle for squats.

## Setup and run (Windows)

```powershell
git clone https://github.com/kareem3228/KinetiCue_mvp.git
cd KinetiCue_mvp
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the repository is private, ask the owner for access. To enable Gemini instructor and hands-free voice, copy `.env.example` to `.env` and add your own Gemini key after `GEMINI_API_KEY=`. Install voice support:

```powershell
.\.venv\Scripts\python.exe -m pip install --target voice_packages "google-genai>=1.0" "typing_extensions>=4.14.0"
.\.venv\Scripts\python.exe main.py
```

Run `main.py`, choose the exercise, and follow its camera setup instructions. Keep the required joints visible. Never commit `.env` or share your API key.

Reports are saved in `session_reports`. Camera measurements are experimental and are not a clinical assessment.
