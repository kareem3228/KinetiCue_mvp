# KinetiCue MVP

A Windows Python demo that tracks seated knee extensions, counts reps, and saves session reports.

## Run the app

```powershell
git clone https://github.com/kareem3228/KinetiCue_mvp.git
cd KinetiCue_mvp
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

If the repository is private, ask the owner for access. Allow camera access and sit sideways with your whole leg visible.

## Optional voice

Copy `.env.example` to `.env` and add your own Gemini key after `GEMINI_API_KEY=`. Then install voice support:

```powershell
.\.venv\Scripts\python.exe -m pip install --target voice_packages "google-genai>=1.0" "typing_extensions>=4.14.0"
```

Restart the app. Never commit `.env` or share your API key.

Reports are saved in `session_reports`. Camera measurements are experimental and are not a clinical assessment.
