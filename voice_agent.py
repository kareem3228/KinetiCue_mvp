"""Hands-free speech recognition with Gemini Live and a local Vosk fallback."""
import asyncio
import importlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


class HandsFreeVoice:
    def __init__(self, on_text, on_status, model_dir=None, asr_backend="gemini_live", api_key=None):
        self.on_text = on_text
        self.on_status = on_status
        self.models_dir = Path(__file__).resolve().parent / "models"
        self.asr_backend = asr_backend
        self.api_key = api_key
        default_dir = {
            "vosk_ar": self.models_dir / "vosk-model-ar-mgb2-0.4",
            "vosk_en": self.models_dir / "vosk-model-small-en-us-0.15",
            "whisper": self.models_dir / "vosk-model-small-en-us-0.15",
        }.get(asr_backend, self.models_dir / "vosk-model-small-en-us-0.15")
        self.model_dir = Path(model_dir) if model_dir else default_dir
        self.stop_event = threading.Event()
        self.speech_queue = queue.Queue()
        self.listener_thread = None
        self.speaker_thread = None
        self.speaking = threading.Event()
        self.transcribing = threading.Event()
        self.ignore_audio_until = 0.0

    def start(self):
        if self.asr_backend == "gemini_live" and not self.api_key:
            raise RuntimeError("Gemini API key is missing; add GEMINI_API_KEY to .env or choose local English recognition.")
        if self.asr_backend != "gemini_live" and not self.model_dir.is_dir():
            model_name = self.model_dir.name
            raise FileNotFoundError(f"Speech model is missing: extract {model_name} into {self.models_dir}")
        self.stop_event.clear()
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)
        self.speaker_thread = threading.Thread(target=self._speak, daemon=True)
        self.listener_thread.start()
        self.speaker_thread.start()

    def speak(self, text):
        if text:
            self.speech_queue.put(str(text))

    def stop(self):
        self.stop_event.set()
        self.speech_queue.put(None)

    def _listen(self):
        if self.asr_backend == "gemini_live":
            try:
                asyncio.run(self._listen_gemini_live())
                return
            except Exception as error:
                if self.stop_event.is_set():
                    return
                self.on_status(f"input:Gemini Live unavailable ({error}); switching to local English recognition.")
                self.asr_backend = "vosk_en"
        self._listen_offline()

    async def _listen_gemini_live(self):
        import sounddevice as sd
        sdk_path = self.models_dir.parent / "voice_packages"
        if sdk_path.is_dir() and str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))
        # Other app dependencies may have imported an older system copy before
        # the isolated Gemini SDK path was added. anyio in voice_packages needs
        # typing_extensions.sentinel, so replace only a cached copy that lacks it.
        cached_typing_extensions = sys.modules.get("typing_extensions")
        bundled_typing_extensions = sdk_path / "typing_extensions.py"
        if (
            cached_typing_extensions is not None
            and not hasattr(cached_typing_extensions, "sentinel")
            and bundled_typing_extensions.is_file()
        ):
            sys.modules.pop("typing_extensions", None)
            importlib.invalidate_caches()
        import typing_extensions
        if not hasattr(typing_extensions, "sentinel"):
            raise ImportError(
                "Gemini Live needs a newer typing_extensions in voice_packages. "
                "Run: python -m pip install --target voice_packages --upgrade 'typing_extensions>=4.14.0'"
            )
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        audio_queue = queue.Queue(maxsize=40)

        def callback(indata, frames, timing, status):
            if self.stop_event.is_set() or self.speaking.is_set() or time.monotonic() < self.ignore_audio_until:
                return
            try:
                audio_queue.put_nowait(bytes(indata))
            except queue.Full:
                # Keep the stream current rather than transcribing stale queued audio.
                try:
                    audio_queue.get_nowait()
                    audio_queue.put_nowait(bytes(indata))
                except queue.Empty:
                    pass

        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=["en-US"],
                custom_vocabulary=["knee extension", "repetitions", "range of motion", "tracking", "hold time"],
            ),
        )

        with sd.RawInputStream(samplerate=16000, blocksize=1600, dtype="int16", channels=1, callback=callback):
            failures = 0
            while not self.stop_event.is_set():
                try:
                    async with client.aio.live.connect(model="gemini-3.5-transcribe-live", config=config) as session:
                        failures = 0
                        self.on_status("input:Listening with Gemini Live Transcribe (English).")

                        async def send_audio():
                            while not self.stop_event.is_set():
                                if self.speaking.is_set() or time.monotonic() < self.ignore_audio_until:
                                    while True:
                                        try:
                                            audio_queue.get_nowait()
                                        except queue.Empty:
                                            break
                                    await asyncio.sleep(.05)
                                    continue
                                try:
                                    chunk = audio_queue.get_nowait()
                                except queue.Empty:
                                    await asyncio.sleep(.01)
                                    continue
                                await session.send_realtime_input(
                                    audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                                )

                        async def receive_transcripts():
                            async for response in session.receive():
                                content = response.server_content
                                if content and content.input_transcription:
                                    transcript = content.input_transcription.text.strip()
                                    if transcript:
                                        self.on_text(transcript)

                        sender = asyncio.create_task(send_audio())
                        receiver = asyncio.create_task(receive_transcripts())
                        try:
                            while not self.stop_event.is_set() and not sender.done() and not receiver.done():
                                await asyncio.sleep(.05)
                            for task in (sender, receiver):
                                if not task.done():
                                    task.cancel()
                            results = await asyncio.gather(sender, receiver, return_exceptions=True)
                            failure = next((result for result in results if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError)), None)
                            if failure:
                                raise failure
                        finally:
                            for task in (sender, receiver):
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(sender, receiver, return_exceptions=True)
                except Exception as error:
                    if self.stop_event.is_set():
                        break
                    failures += 1
                    if failures >= 2:
                        raise RuntimeError(f"Live transcription connection failed: {error}") from error
                    self.on_status(f"input:Gemini Live disconnected; reconnecting ({error})…")
                    await asyncio.sleep(1)

    def _listen_offline(self):
        try:
            import sounddevice as sd
            from vosk import Model, KaldiRecognizer
            self.on_status(f"input:Loading {self.asr_backend} speech model…")
            # Vosk also provides the low-latency end-of-utterance signal for
            # Whisper; only the selected recognizer's words reach the app.
            endpoint_model = Model(str(self.model_dir))
            endpoint_recognizer = KaldiRecognizer(endpoint_model, 16000)
            whisper_model = None
            if self.asr_backend == "whisper":
                from faster_whisper import WhisperModel
                whisper_dir = self.models_dir / "faster-whisper"
                whisper_dir.mkdir(parents=True, exist_ok=True)
                self.on_status("input:Loading multilingual Whisper small (first run downloads the model)…")
                whisper_model = WhisperModel(
                    "small", device="cpu", compute_type="int8",
                    cpu_threads=max(1, min(6, (os.cpu_count() or 2) - 1)),
                    download_root=str(whisper_dir),
                )

            if self.stop_event.is_set():
                return
            audio_queue = queue.Queue(maxsize=80)
            def callback(indata, frames, timing, status):
                # Never feed the instructor's own synthesized voice back to ASR.
                if not self.stop_event.is_set() and not self.speaking.is_set() and not self.transcribing.is_set() and time.monotonic() >= self.ignore_audio_until:
                    try:
                        audio_queue.put_nowait(bytes(indata))
                    except queue.Full:
                        pass
            with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype="int16", channels=1, callback=callback):
                labels = {"whisper": "multilingual Whisper small", "vosk_en": "Vosk English", "vosk_ar": "Vosk Arabic"}
                self.on_status(f"input:Listening with {labels.get(self.asr_backend, 'speech recognition')}.")
                utterance_audio = []
                while not self.stop_event.is_set():
                    try:
                        data = audio_queue.get(timeout=.2)
                    except queue.Empty:
                        continue
                    if self.speaking.is_set() or self.transcribing.is_set() or time.monotonic() < self.ignore_audio_until:
                        while True:
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break
                        utterance_audio.clear()
                        endpoint_recognizer = KaldiRecognizer(endpoint_model, 16000)
                        continue
                    if self.asr_backend in {"vosk_en", "vosk_ar"}:
                        if endpoint_recognizer.AcceptWaveform(data):
                            text = json.loads(endpoint_recognizer.Result()).get("text", "").strip()
                            if text:
                                self.on_text(text)
                    else:
                        utterance_audio.append(data)
                        if endpoint_recognizer.AcceptWaveform(data):
                            payload = b"".join(utterance_audio)
                            utterance_audio.clear()
                            # Vosk marks the pause boundary; multilingual
                            # Whisper performs the actual Arabic/English ASR.
                            endpoint_recognizer = KaldiRecognizer(endpoint_model, 16000)
                            self.transcribing.set()
                            try:
                                import numpy as np
                                samples = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
                                segments, info = whisper_model.transcribe(
                                    samples, task="transcribe", language=None, beam_size=3,
                                    condition_on_previous_text=False, vad_filter=False,
                                )
                                text = " ".join(segment.text.strip() for segment in segments).strip()
                                if text:
                                    self.on_status(f"input:Whisper detected {info.language}; transcribed utterance.")
                                    self.on_text(text)
                            except Exception as error:
                                self.on_status(f"input:Whisper transcription failed: {error}")
                            finally:
                                self.transcribing.clear()
                                self.ignore_audio_until = time.monotonic() + .25
        except Exception as error:
            self.on_status(f"input:Speech recognition stopped: {error}")

    def _speak(self):
        speaker = None
        arabic_voice = None
        try:
            arabic_model = self.models_dir / "piper-ar_JO-kareem-medium.onnx"
            if arabic_model.is_file():
                try:
                    from piper import PiperVoice
                    arabic_voice = PiperVoice.load(arabic_model, download_dir=self.models_dir / "piper")
                    # Avoid downloading an extra diacritization model at runtime.
                    arabic_voice.use_tashkeel = False
                except Exception as error:
                    self.on_status(f"output:Arabic voice unavailable: {error}")
            powershell = shutil.which("powershell.exe") or shutil.which("powershell")
            if not powershell:
                if arabic_voice is None:
                    raise RuntimeError("Windows PowerShell was not found")
                powershell = None
            if powershell:
                script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$speaker.Volume = 100; "
                    "while ($null -ne ($line = [Console]::ReadLine())) { "
                    "if ($line -eq '__PHYSIOPOSE_EXIT__') { break }; "
                    "try { $text = ConvertFrom-Json -InputObject $line; "
                    "$speaker.Speak([string]$text); [Console]::Out.WriteLine('__PHYSIOPOSE_DONE__') } "
                    "catch { [Console]::Error.WriteLine($_.Exception.Message); "
                    "[Console]::Out.WriteLine('__PHYSIOPOSE_DONE__') } }"
                )
                speaker = subprocess.Popen(
                    [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, encoding="utf-8", bufsize=1,
                )
                if speaker.poll() is not None:
                    raise RuntimeError("Windows speech engine could not start")
            self.on_status("output:Arabic Piper voice ready; English Windows voice ready." if arabic_voice else "output:English Windows voice ready; Arabic voice unavailable.")
            while not self.stop_event.is_set():
                try:
                    text = self.speech_queue.get(timeout=.2)
                except queue.Empty:
                    continue
                if text is None:
                    break
                self.speaking.set()
                self.on_status("output:Speaking…")
                contains_arabic = any("\u0600" <= char <= "\u06ff" for char in text)
                if contains_arabic and arabic_voice is not None:
                    import numpy as np
                    import sounddevice as sd
                    chunks = list(arabic_voice.synthesize(text))
                    if chunks:
                        audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
                        sd.play(audio, samplerate=arabic_voice.config.sample_rate, blocking=True)
                elif speaker is not None:
                    # Escaping Unicode keeps redirected Windows PowerShell pipes
                    # independent of the machine's legacy console code page.
                    speaker.stdin.write(json.dumps(text, ensure_ascii=True) + "\n")
                    speaker.stdin.flush()
                    marker = speaker.stdout.readline().strip()
                    if marker != "__PHYSIOPOSE_DONE__":
                        raise RuntimeError("Windows speech engine stopped unexpectedly")
                else:
                    self.on_status("output:Arabic reply shown as text; no Arabic voice model loaded.")
                self.speaking.clear()
                self.ignore_audio_until = time.monotonic() + 0.65
                self.on_status("output:Speech output ready.")
        except Exception as error:
            self.speaking.clear()
            self.on_status(f"output:Speech output unavailable: {error}")
        finally:
            if speaker is not None and speaker.poll() is None:
                try:
                    speaker.stdin.write("__PHYSIOPOSE_EXIT__\n")
                    speaker.stdin.flush()
                    speaker.wait(timeout=1)
                except Exception:
                    speaker.kill()
