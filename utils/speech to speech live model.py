"""
CrisisLens AI — Live Voice Commander
=====================================

Voice interface for CrisisLens AI. A field responder, NGO worker, or call-center
operator speaks a disaster report out loud (optionally showing the scene on
camera), and CrisisLens AI Commander responds back in real-time speech with a
structured emergency briefing — situation summary, threat level, immediate
actions, and resource needs.

This replaces generic small-talk behavior with the CrisisLens AI Commander
persona and routes every spoken report through your existing ML pipeline:

    classify_disaster()  -> disaster type
    extract_location()   -> location
    predict_severity()   -> severity level
    detect_fake_news()   -> authenticity
    generate_relief_plan() -> full briefing (spoken back to the user)

PHOTO SUPPORT
--------------
This version adds a photo library you can build up during a session:

    - Upload a photo from disk:        type its file path into the terminal,
                                        or say "upload photo <path>"
    - Capture a photo from the camera: say "take a photo" / "capture photo"
    - Ask about a specific photo:      say "what do you see in photo 2" or
                                        "ask about the last photo" — the
                                        Commander is shown that exact image
                                        and answers using Gemini's vision.

Every captured/uploaded photo is numbered and kept in memory for the rest
of the session so you can refer back to "photo 1", "photo 2", "the last
photo", etc.

Usage:
    python crisislens_voice_commander.py
    (press Q in the camera window, or Ctrl+C in the terminal, to stop)
"""

import os
import sys
import cv2
import asyncio
import threading
import traceback
import time
import queue
import re
import types as _types
from dotenv import load_dotenv

load_dotenv()

# Import your existing CrisisLens AI pipeline.
# These must be importable from your project's utils/ package.
try:
    from utils.disaster_classifier import classify_disaster
    from utils.location_extractor import extract_location
    from utils.severity_prediction import predict_severity
    from utils.fake_news_detection import detect_fake_news
    from utils.gemini_relief_agent import generate_relief_plan
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("WARNING: utils/ pipeline not found on path. Voice transcripts will be "
          "logged but NOT run through classify_disaster / generate_relief_plan. "
          "Run this file from your project root so 'utils' is importable.")


# =====================================================
# PERSISTENT STORE
# =====================================================
_KEY = "__crisislens_voice_store__"
if _KEY not in sys.modules:
    _m              = _types.ModuleType(_KEY)
    _m.frame        = None
    _m.lock         = threading.Lock()
    _m.running      = False
    _m.gemini_q     = queue.Queue(maxsize=2)
    _m.pya          = None
    _m.client       = None
    _m.transcript   = []     # rolling buffer of spoken report text
    _m.last_report_time = 0.0
    _m.photos       = []     # list of dicts: {id, path, source, bytes, caption}
    _m.photo_lock    = threading.Lock()
    sys.modules[_KEY] = _m

_store = sys.modules[_KEY]


def set_frame(f):
    with _store.lock:
        _store.frame = f


def get_frame():
    with _store.lock:
        return _store.frame


# =====================================================
# PHOTO LIBRARY
# =====================================================
PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "field_photos")
os.makedirs(PHOTO_DIR, exist_ok=True)


def _next_photo_id() -> int:
    with _store.photo_lock:
        return len(_store.photos) + 1


def add_photo_from_path(path: str) -> dict:
    """
    Adds an uploaded photo (from disk) to the in-session photo library.
    Returns the photo record dict, or raises FileNotFoundError / ValueError.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        raise ValueError(f"Unsupported image type: {ext}")

    with open(path, "rb") as f:
        raw = f.read()

    photo_id = _next_photo_id()
    dest_name = f"photo_{photo_id}{ext}"
    dest_path = os.path.join(PHOTO_DIR, dest_name)
    with open(dest_path, "wb") as out:
        out.write(raw)

    record = {
        "id": photo_id,
        "path": dest_path,
        "source": "upload",
        "bytes": raw,
        "mime": "image/png" if ext == ".png" else "image/jpeg",
        "caption": None,
    }
    with _store.photo_lock:
        _store.photos.append(record)

    print(f"PHOTO {photo_id}: uploaded from '{path}' -> saved as '{dest_path}'")
    return record


def add_photo_from_camera() -> dict:
    """
    Captures the current live camera frame and adds it to the photo library.
    Returns the photo record dict, or raises RuntimeError if no frame is
    available yet.
    """
    frame = get_frame()
    if frame is None:
        raise RuntimeError("No camera frame available yet — is the camera running?")

    photo_id = _next_photo_id()
    dest_name = f"photo_{photo_id}.jpg"
    dest_path = os.path.join(PHOTO_DIR, dest_name)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Failed to encode camera frame as JPEG.")

    raw = bytes(buf)
    with open(dest_path, "wb") as out:
        out.write(raw)

    record = {
        "id": photo_id,
        "path": dest_path,
        "source": "camera",
        "bytes": raw,
        "mime": "image/jpeg",
        "caption": None,
    }
    with _store.photo_lock:
        _store.photos.append(record)

    print(f"PHOTO {photo_id}: captured from camera -> saved as '{dest_path}'")
    return record


def get_photo(photo_id: int) -> "dict | None":
    with _store.photo_lock:
        for p in _store.photos:
            if p["id"] == photo_id:
                return p
    return None


def get_last_photo() -> "dict | None":
    with _store.photo_lock:
        return _store.photos[-1] if _store.photos else None


def list_photos() -> list:
    with _store.photo_lock:
        return list(_store.photos)


def resolve_photo_reference(text: str) -> "dict | None":
    """
    Parses spoken/typed phrases like 'photo 2', 'the last photo', 'photo
    number three', 'this photo' and returns the matching photo record.
    Falls back to the most recent photo if no number is found but the
    text clearly refers to a photo (e.g. 'this picture', 'that image').
    """
    text_l = text.lower()

    match = re.search(r"photo\s*(?:number\s*)?(\d+)", text_l)
    if match:
        return get_photo(int(match.group(1)))

    if any(kw in text_l for kw in ["last photo", "latest photo", "most recent photo",
                                     "this photo", "that photo", "the picture",
                                     "this picture", "that picture", "this image",
                                     "that image"]):
        return get_last_photo()

    return None


# =====================================================
# AUDIO / GEMINI CONSTANTS
# =====================================================
import pyaudio as _pyaudio

FORMAT              = _pyaudio.paInt16
CHANNELS             = 1
SEND_SAMPLE_RATE     = 16000
RECEIVE_SAMPLE_RATE  = 24000
CHUNK_SIZE           = 1024
MODEL                = "models/gemini-2.5-flash-native-audio-latest"

# How long to wait after the user stops speaking before treating the
# accumulated transcript as a complete disaster report worth analyzing.
REPORT_SILENCE_GAP_SECONDS = 2.5


# =====================================================
# CRISISLENS AI COMMANDER — SYSTEM INSTRUCTION
# =====================================================
COMMANDER_SYSTEM_INSTRUCTION = """
You are CrisisLens AI Commander — a voice-based Disaster Intelligence and
Emergency Response assistant for Pakistan, speaking with field responders,
NGO workers, and emergency call-center operators in real time.

Your job during this voice session:
- Listen for disaster reports being spoken aloud (flood, earthquake, and
  related emergencies in Pakistan).
- If you can see a live camera feed, briefly note anything visually relevant
  to the disaster (flooded streets, structural damage, crowds, debris) when
  asked or when it is clearly useful context.
- The user can upload photos or capture photos with the camera during this
  session. When the user asks about "photo 1", "photo 2", "the last photo",
  "this picture", etc., you will be shown that exact image directly in the
  conversation. Describe what you see in disaster-relevant terms: damage
  severity, flooding extent, structural risk, visible hazards, number of
  people, and anything relevant to an emergency response decision.
- Ask short, targeted clarifying questions ONLY if the location or disaster
  type is unclear and ONLY one question at a time. Do not interrogate.
- Once a report has enough detail (disaster type + location, at minimum),
  say: "Understood. Analyzing the report now." and stop talking — the
  system will run the full analysis pipeline and read back a structured
  emergency briefing to the user automatically.
- Never use uncertain phrases like "I think", "maybe", "possibly".
- Keep spoken responses brief (1-3 sentences) since this is a voice
  channel — detailed analysis comes from the structured briefing, not from
  your live chatter.
- You are not a general assistant. Politely decline unrelated small talk
  and steer back to disaster reporting.
"""


# =====================================================
# CAMERA WORKER — 30 fps, OpenCV window
# =====================================================
def camera_worker():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    if not cap.isOpened():
        print("CAMERA: could not be opened (continuing in audio-only mode)")
        return

    print("CAMERA: started — press Q in the video window to quit")
    last_sent = 0.0

    while _store.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        set_frame(frame.copy())

        cv2.imshow("CrisisLens AI — Live Field Camera  |  press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            _store.running = False
            break

        now = time.time()
        if now - last_sent >= 1.0:
            small = cv2.resize(frame, (320, 240))
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 40])
            try:
                _store.gemini_q.put_nowait(bytes(buf))
            except queue.Full:
                pass
            last_sent = now

    cap.release()
    cv2.destroyAllWindows()
    print("CAMERA: stopped")


# =====================================================
# DISASTER REPORT PIPELINE RUNNER
# =====================================================
def run_pipeline_and_build_briefing(report_text: str) -> str:
    """
    Runs the spoken transcript through the full CrisisLens AI pipeline and
    returns a short spoken-friendly summary of the emergency briefing.

    This is intentionally short for text-to-speech playback — the full
    written briefing should still be generated and saved/displayed in your
    main Streamlit app; here we only need a spoken digest.
    """
    if not PIPELINE_AVAILABLE:
        return ("Pipeline modules are not available. "
                "Please run this from the CrisisLens AI project root.")

    if not report_text or not report_text.strip():
        return "No report content was captured to analyze."

    try:
        disaster     = classify_disaster(report_text)
        location     = extract_location(report_text)
        severity     = predict_severity(report_text)
        authenticity = detect_fake_news(report_text)

        is_fake = "fake" in authenticity.lower() or "suspicious" in authenticity.lower()

        if is_fake:
            spoken = (
                f"Caution. This report has been flagged as {authenticity}. "
                f"I am not activating full emergency response. "
                f"Please verify with N D M A or P D M A before taking action."
            )
            print("\n" + "=" * 60)
            print("VOICE REPORT — FLAGGED AS SUSPICIOUS / FAKE")
            print(f"Disaster: {disaster} | Location: {location} | "
                  f"Severity: {severity} | Authenticity: {authenticity}")
            print("=" * 60 + "\n")
            return spoken

        briefing = generate_relief_plan(
            disaster_type=disaster,
            location=location,
            severity=severity,
            authenticity=authenticity,
            original_report=report_text,
        )

        print("\n" + "=" * 60)
        print("VOICE REPORT — FULL BRIEFING GENERATED")
        print(f"Disaster: {disaster} | Location: {location} | Severity: {severity}")
        print("-" * 60)
        print(briefing)
        print("=" * 60 + "\n")

        spoken = (
            f"Briefing ready. {disaster} reported in {location}. "
            f"Severity is {severity}. "
            f"The full written emergency briefing has been generated and "
            f"saved. Key priority: check your screen for the complete "
            f"response plan."
        )
        return spoken

    except Exception as e:
        traceback.print_exc()
        return f"Pipeline error while analyzing the report: {e}"


# =====================================================
# GEMINI ASSISTANT
# =====================================================
class CrisisLensVoiceCommander:

    def __init__(self):
        self.audio_in_q  = asyncio.Queue()
        self.audio_out_q = asyncio.Queue()
        self.session     = None
        self._pending_transcript_parts = []
        self._last_text_time = 0.0
        self._analysis_lock  = asyncio.Lock()
        self._analyzing      = False

    # ---------------- Microphone capture ----------------

    async def listen_mic(self):
        pya = _store.pya
        stream = None
        try:
            mic = pya.get_default_input_device_info()
            stream = pya.open(
                format=FORMAT, channels=CHANNELS,
                rate=SEND_SAMPLE_RATE, input=True,
                input_device_index=mic["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
            print("MIC: started — speak your disaster report now")
            while _store.running:
                data = await asyncio.to_thread(
                    stream.read, CHUNK_SIZE, exception_on_overflow=False
                )
                if data:
                    await self.audio_out_q.put(data)
        except Exception as e:
            print(f"MIC ERROR: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

    async def send_mic_audio(self):
        from google.genai import types
        while _store.running:
            try:
                data = await asyncio.wait_for(self.audio_out_q.get(), timeout=1.0)
                if self.session and _store.running:
                    await self.session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm")
                    )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"AUDIO SEND ERROR: {e}")

    # ---------------- Camera frame streaming ----------------

    async def send_camera_frames(self):
        from google.genai import types
        while _store.running:
            try:
                raw = _store.gemini_q.get_nowait()
                if self.session and _store.running:
                    await self.session.send_realtime_input(
                        video=types.Blob(data=raw, mime_type="image/jpeg")
                    )
            except queue.Empty:
                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"CAMERA FRAME SEND ERROR: {e}")

    # ---------------- Response receiving ----------------

    async def receive_responses(self):
        while _store.running:
            try:
                async for response in self.session.receive():
                    if not _store.running:
                        break

                    # Capture spoken transcript text (for pipeline analysis)
                    if hasattr(response, "text") and response.text:
                        print(f"COMMANDER: {response.text}")
                        self._pending_transcript_parts.append(response.text)
                        self._last_text_time = time.time()

                    audio_data = None

                    if hasattr(response, "server_content") and response.server_content:
                        sc = response.server_content
                        if hasattr(sc, "model_turn") and sc.model_turn:
                            for part in sc.model_turn.parts:
                                if hasattr(part, "inline_data") and part.inline_data:
                                    audio_data = part.inline_data.data
                                    break

                    if audio_data is None and hasattr(response, "data") and response.data:
                        audio_data = response.data

                    if audio_data:
                        await self.audio_in_q.put(audio_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"RESPONSE RECEIVING ERROR: {e}")
                if not _store.running:
                    break
                await asyncio.sleep(0.1)

    # ---------------- Report completion watcher ----------------

    async def watch_for_complete_report(self):
        """
        Watches the live transcript. Once the user has stopped speaking for
        REPORT_SILENCE_GAP_SECONDS and we have transcript text accumulated,
        treat it as a complete report, run the CrisisLens pipeline, and
        speak back the briefing summary as a new turn.
        """
        while _store.running:
            await asyncio.sleep(0.5)

            if not self._pending_transcript_parts:
                continue

            idle_for = time.time() - self._last_text_time
            if idle_for < REPORT_SILENCE_GAP_SECONDS:
                continue

            if self._analyzing:
                continue

            async with self._analysis_lock:
                self._analyzing = True
                report_text = " ".join(self._pending_transcript_parts).strip()
                self._pending_transcript_parts = []

                print("\nANALYZING SPOKEN REPORT...\n")
                spoken_summary = await asyncio.to_thread(
                    run_pipeline_and_build_briefing, report_text
                )

                # Send the spoken summary back into the live session so the
                # model speaks it out loud to the user.
                if self.session and _store.running:
                    try:
                        await self.session.send_client_content(
                            turns={"role": "user", "parts": [{
                                "text": (
                                    "[SYSTEM: Speak this emergency briefing "
                                    f"summary aloud verbatim, calmly and clearly: "
                                    f"\"{spoken_summary}\"]"
                                )
                            }]},
                            turn_complete=True,
                        )
                    except Exception as e:
                        print(f"FAILED TO SEND BRIEFING TO SESSION: {e}")

                self._analyzing = False

    # ---------------- Photo Q&A ----------------

    async def send_photo_question(self, photo: dict, question: str):
        """
        Sends a specific photo plus a question about it into the live
        session as a new user turn, so the model answers using that
        exact image (vision) rather than just the live camera stream.
        """
        from google.genai import types

        if not self.session or not _store.running:
            return

        prompt_text = (
            f"[SYSTEM: The user is asking about photo {photo['id']} "
            f"(source: {photo['source']}). Question: \"{question}\". "
            f"Look at the attached image and answer in disaster-response "
            f"terms — damage, flooding, hazards, people, structural risk.]"
        )

        try:
            await self.session.send_client_content(
                turns={
                    "role": "user",
                    "parts": [
                        {"text": prompt_text},
                        {"inline_data": {"mime_type": photo["mime"], "data": photo["bytes"]}},
                    ],
                },
                turn_complete=True,
            )
            print(f"PHOTO Q&A: sent photo {photo['id']} with question -> '{question}'")
        except Exception as e:
            print(f"PHOTO Q&A ERROR: {e}")

    async def handle_text_command(self, raw_text: str):
        """
        Handles a typed (or transcribed) command from the operator console:
          - 'upload photo <path>'
          - 'take photo' / 'capture photo'
          - 'photo <n> <question>' / 'ask about photo <n>: <question>'
          - 'what do you see in the last photo'
          - anything else -> ignored here (spoken reports go through the
            normal mic pipeline instead)
        """
        text = raw_text.strip()
        text_l = text.lower()

        # --- Upload photo from path ---
        if text_l.startswith("upload photo"):
            path = text[len("upload photo"):].strip().strip('"').strip("'")
            try:
                photo = await asyncio.to_thread(add_photo_from_path, path)
                print(f"-> Photo {photo['id']} ready. Ask about it any time, "
                      f"e.g. 'ask about photo {photo['id']}: how bad is the flooding?'")
            except Exception as e:
                print(f"UPLOAD FAILED: {e}")
            return

        # --- Capture photo from camera ---
        if text_l in ("take photo", "take a photo", "capture photo", "capture a photo"):
            try:
                photo = await asyncio.to_thread(add_photo_from_camera)
                print(f"-> Photo {photo['id']} captured. Ask about it any time, "
                      f"e.g. 'ask about photo {photo['id']}: what hazards do you see?'")
            except Exception as e:
                print(f"CAPTURE FAILED: {e}")
            return

        # --- List photos ---
        if text_l in ("list photos", "show photos"):
            photos = list_photos()
            if not photos:
                print("No photos in this session yet.")
            else:
                for p in photos:
                    print(f"  Photo {p['id']} — source: {p['source']} — path: {p['path']}")
            return

        # --- Ask about a specific / the last photo ---
        # Patterns: "ask about photo 2: <question>" or "photo 2 <question>"
        m = re.match(r"(?:ask about\s+)?photo\s*(?:number\s*)?(\d+)\s*[:\-]?\s*(.*)", text_l)
        if m:
            photo_id = int(m.group(1))
            question = m.group(2).strip() or "Describe what you see in this photo."
            photo = get_photo(photo_id)
            if photo is None:
                print(f"No photo numbered {photo_id} found. Use 'list photos' to see what's available.")
                return
            await self.send_photo_question(photo, question)
            return

        # Generic reference: "what do you see in the last photo", "describe this picture"
        photo = resolve_photo_reference(text_l)
        if photo is not None:
            await self.send_photo_question(photo, text)
            return

    async def listen_console_commands(self):
        """
        Reads typed commands from stdin in a background thread so the
        operator can type 'upload photo path/to/file.jpg', 'take photo',
        or 'ask about photo 1: ...' alongside the live voice session.
        """
        loop = asyncio.get_event_loop()
        print("\nTYPE COMMANDS HERE (voice keeps working at the same time):")
        print("  upload photo <path>")
        print("  take photo")
        print("  list photos")
        print("  ask about photo <n>: <question>\n")

        while _store.running:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except Exception:
                break
            if not line:
                await asyncio.sleep(0.2)
                continue
            line = line.strip()
            if not line:
                continue
            await self.handle_text_command(line)

    # ---------------- Audio playback ----------------

    async def play_audio(self):
        pya = _store.pya
        stream = None
        try:
            stream = pya.open(
                format=FORMAT, channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE, output=True,
            )
            print("SPEAKER: ready")
            while _store.running:
                data = await self.audio_in_q.get()
                if data:
                    await asyncio.to_thread(stream.write, data)
        except Exception as e:
            print(f"SPEAKER ERROR: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

    # ---------------- Session lifecycle ----------------

    async def run(self):
        from google.genai import types

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part(text=COMMANDER_SYSTEM_INSTRUCTION)]
            ),
        )

        async with _store.client.aio.live.connect(
            model=MODEL, config=config
        ) as session:
            self.session = session
            print("\nCrisisLens AI Commander connected — describe the disaster report now.\n")
            await asyncio.gather(
                self.listen_mic(),
                self.send_mic_audio(),
                self.send_camera_frames(),
                self.receive_responses(),
                self.watch_for_complete_report(),
                self.listen_console_commands(),
                self.play_audio(),
            )


# =====================================================
# MAIN
# =====================================================
def main():
    try:
        import pyaudio as _pa
        from google import genai

        print("Initializing PyAudio...")
        _store.pya = _pa.PyAudio()
        print("PyAudio OK")

        print("Initializing Gemini client...")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("ERROR: GEMINI_API_KEY environment variable not set!")
            return

        _store.client = genai.Client(
            api_key=gemini_api_key,
            http_options={"api_version": "v1beta"},
        )
        print("Gemini client OK")

        _store.running = True

        threading.Thread(target=camera_worker, daemon=True).start()
        asyncio.run(CrisisLensVoiceCommander().run())

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        err = str(e)
        if "getaddrinfo failed" in err or "11001" in err:
            print("\nNetwork Error: Cannot reach Google servers.")
            print("   -> Use a VPN if Google APIs are blocked in your region")
            print("   -> Check GEMINI_API_KEY in your .env file")
        elif "Could not construct Bearer token" in err:
            print("\nAuthentication Error: Invalid GEMINI_API_KEY")
            print("   -> Verify your GEMINI_API_KEY is correct in .env")
        else:
            print(f"\nError: {e}")
            traceback.print_exc()
    finally:
        _store.running = False
        if _store.pya:
            try:
                _store.pya.terminate()
            except Exception:
                pass
        print("Cleanup complete")


if __name__ == "__main__":
    print("=" * 60)
    print("  CrisisLens AI — Live Voice Commander")
    print("  Speak a disaster report. Press Q in the camera window")
    print("  or Ctrl+C in this terminal to stop.")
    print("=" * 60)
    main()