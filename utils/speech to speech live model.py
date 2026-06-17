"""
CrisisLens AI — Live Voice Commander
=====================================

Voice + console interface for CrisisLens AI. A field responder, NGO worker,
or call-center operator speaks (or types) a disaster report — optionally
showing the scene on camera — and CrisisLens AI Commander responds back
with a structured emergency briefing: situation summary, threat level,
immediate actions, and resource needs.

This routes every report through your existing ML pipeline:

    classify_disaster()    -> disaster type
    extract_location()     -> location
    predict_severity()     -> severity level
    detect_fake_news()     -> authenticity
    generate_relief_plan() -> full briefing (spoken / emailed / texted back)

CONSOLE FEATURES
-----------------
While the live mic + camera session is running, type any of the following
into the terminal at the same time (press Enter after each):

    📝 Report Text        report: <your disaster report text>
                           submit report <your disaster report text>

    📷 Upload Image        upload photo <path/to/file.jpg>

    📸 Capture Image       take photo
                           capture photo

    🎤 Voice Report        (automatic — just speak into the mic, no command
                            needed; CrisisLens AI listens continuously)

    📍 Current Location    location
                           current location
                           where am i

    🚨 Analyze Incident    analyze
                           analyze incident
                           (forces immediate pipeline analysis of whatever
                            report text has been spoken/typed so far,
                            without waiting for the normal silence timeout)

    🤖 Ask AI Assistant    ask <any question>
                           ask about photo <n>: <question>   (vision Q&A)

    🔊 Read Response       read
                           read response
                           (reads the last AI answer / briefing aloud using
                            a local, offline TTS engine — independent of the
                            live Gemini audio channel)

    📧 Send Alert          send alert
                           send alert to <email>
                           (emails the latest briefing — requires SMTP_USER /
                            SMTP_PASSWORD / ALERT_EMAIL_TO in .env)

    📱 Send SMS            send sms
                           send sms to <phone>
                           (texts the latest briefing via Twilio — requires
                            TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN /
                            TWILIO_FROM_NUMBER / ALERT_SMS_TO in .env)

    help                   shows this command list again

OPTIONAL DEPENDENCIES
-----------------------
Core:        pip install pyaudio opencv-python google-genai python-dotenv
Location:    pip install requests
Read aloud:  pip install pyttsx3
SMS alerts:  pip install twilio
(Email alerts use Python's built-in smtplib — no extra install needed.)

ENVIRONMENT VARIABLES (.env)
------------------------------
    GEMINI_API_KEY          (required)
    SMTP_HOST               (default: smtp.gmail.com)
    SMTP_PORT               (default: 587)
    SMTP_USER               your sending email address
    SMTP_PASSWORD           your email app password
    ALERT_EMAIL_FROM        (default: SMTP_USER)
    ALERT_EMAIL_TO          default recipient for "send alert"
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_FROM_NUMBER      your Twilio sending number, e.g. +1...
    ALERT_SMS_TO            default recipient for "send sms", e.g. +92...

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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import types as _types
from dotenv import load_dotenv
from utils.vision_report_generator import generate_incident_report
from utils.pipeline import analyze_report


load_dotenv()

if __name__ == "__main__":
    image_description = "A severe flood situation in urban area with submerged buildings."
    report = generate_incident_report(image_description)
    result = analyze_report(report)
    print(result)


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
    print("WARNING: utils/ pipeline not found on path. Reports will be "
          "logged but NOT run through classify_disaster / generate_relief_plan. "
          "Run this file from your project root so 'utils' is importable.")

# Optional third-party helpers — each feature degrades gracefully if its
# package isn't installed, instead of crashing the whole app.
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False


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
    _m.last_report_text = ""    # 📝 most recent report (typed or spoken)
    _m.last_briefing     = ""   # 🚨 full written briefing from the pipeline
    _m.last_ai_response  = ""   # 🤖 most recent "ask" answer (for 🔊 read)
    _m.last_location      = None  # 📍 most recent resolved location dict
    _m.tts_engine          = None  # 🔊 lazily-initialized offline TTS engine
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
# ALERTING CONFIGURATION (Email + SMS) — set these in .env
# =====================================================
SMTP_HOST              = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT              = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER              = os.getenv("SMTP_USER")
SMTP_PASSWORD          = os.getenv("SMTP_PASSWORD")
ALERT_EMAIL_FROM       = os.getenv("ALERT_EMAIL_FROM", SMTP_USER)
ALERT_EMAIL_DEFAULT_TO = os.getenv("ALERT_EMAIL_TO")

TWILIO_ACCOUNT_SID   = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN    = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER   = os.getenv("TWILIO_FROM_NUMBER")
ALERT_SMS_DEFAULT_TO = os.getenv("ALERT_SMS_TO")

SMS_MAX_CHARS = 300  # keep SMS short / affordable; truncate longer briefings


# =====================================================
# LOCATION, ALERTING & TEXT-TO-SPEECH HELPERS
# =====================================================
def get_current_location() -> dict:
    """
    📍 Resolves an approximate current location using IP-based geolocation
    (no GPS hardware required). Returns a dict with lat, lon, city, region,
    country, ip, and a human-readable 'address' string. Raises RuntimeError
    if the lookup is unavailable or fails.

    Note: IP-based location is approximate (typically city-level) and
    reflects the network's location, not necessarily the device's exact
    GPS position. For precise field coordinates, wire this up to a phone's
    GPS / browser geolocation API instead.
    """
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("The 'requests' package is not installed. Run: pip install requests")

    try:
        resp = requests.get("http://ip-api.com/json/", timeout=5)
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Location lookup failed: {e}")

    if data.get("status") != "success":
        raise RuntimeError(f"Location lookup failed: {data.get('message', 'unknown error')}")

    location = {
        "lat": data.get("lat"),
        "lon": data.get("lon"),
        "city": data.get("city"),
        "region": data.get("regionName"),
        "country": data.get("country"),
        "ip": data.get("query"),
    }
    location["address"] = ", ".join(
        part for part in [location["city"], location["region"], location["country"]] if part
    ) or "Unknown"

    with _store.lock:
        _store.last_location = location
    return location


def send_email_alert(subject: str, body: str, to_email: str = None) -> str:
    """
    📧 Sends the given subject/body as an emergency alert email via SMTP.
    Falls back to ALERT_EMAIL_DEFAULT_TO if to_email isn't provided.
    Raises ValueError on missing configuration, RuntimeError on send failure.
    """
    to_email = to_email or ALERT_EMAIL_DEFAULT_TO
    if not to_email:
        raise ValueError("No recipient email given and ALERT_EMAIL_TO is not set in .env")
    if not SMTP_USER or not SMTP_PASSWORD:
        raise ValueError("SMTP_USER / SMTP_PASSWORD are not configured in .env")

    msg = MIMEMultipart()
    msg["From"] = ALERT_EMAIL_FROM or SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(msg["From"], [to_email], msg.as_string())
    except Exception as e:
        raise RuntimeError(f"Failed to send alert email: {e}")

    return f"Alert email sent to {to_email}"


def send_sms_alert(body: str, to_phone: str = None) -> str:
    """
    📱 Sends the given body as an SMS emergency alert via Twilio.
    Falls back to ALERT_SMS_DEFAULT_TO if to_phone isn't provided.
    Raises ValueError/RuntimeError on missing configuration or failure.
    """
    to_phone = to_phone or ALERT_SMS_DEFAULT_TO
    if not to_phone:
        raise ValueError("No recipient phone number given and ALERT_SMS_TO is not set in .env")
    if not TWILIO_AVAILABLE:
        raise RuntimeError("The 'twilio' package is not installed. Run: pip install twilio")
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        raise ValueError(
            "TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM_NUMBER are not configured in .env"
        )

    sms_body = (body or "").strip()
    if len(sms_body) > SMS_MAX_CHARS:
        sms_body = sms_body[: SMS_MAX_CHARS - 3] + "..."

    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(body=sms_body, from_=TWILIO_FROM_NUMBER, to=to_phone)
    except Exception as e:
        raise RuntimeError(f"Failed to send SMS: {e}")

    return f"SMS sent to {to_phone} (sid: {message.sid})"


def speak_text_locally(text: str):
    """
    🔊 Reads text aloud using a local, OFFLINE TTS engine (pyttsx3),
    independent of the live Gemini audio session. This powers the explicit
    "read" / "read response" command so typed reports and AI answers can be
    read back even when no live voice turn is in progress.
    """
    if not text or not text.strip():
        print("Nothing to read yet.")
        return

    if not PYTTSX3_AVAILABLE:
        print("pyttsx3 not installed — cannot read aloud locally. Run: pip install pyttsx3")
        print(f"[TEXT] {text}")
        return

    try:
        with _store.lock:
            if _store.tts_engine is None:
                _store.tts_engine = pyttsx3.init()
            engine = _store.tts_engine
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS ERROR: {e}")
        print(f"[TEXT] {text}")


# =====================================================
# AUDIO / GEMINI CONSTANTS
# =====================================================
try:
    import pyaudio as _pyaudio
    FORMAT = _pyaudio.paInt16
except ImportError:
    _pyaudio = None
    FORMAT = None

CHANNELS             = 1
SEND_SAMPLE_RATE     = 16000
RECEIVE_SAMPLE_RATE  = 24000
CHUNK_SIZE           = 1024
MODEL                = "models/gemini-2.5-flash-native-audio-latest"

# How long to wait after the user stops speaking before treating the
# accumulated transcript as a complete disaster report worth analyzing.
REPORT_SILENCE_GAP_SECONDS = 2.5

# How long after an "ask" / photo question we keep treating the model's
# spoken reply as a Q&A answer rather than a fresh disaster-report transcript.
QA_REPLY_WINDOW_SECONDS = 15.0


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
- The user may also ask you direct questions unrelated to a specific photo
  (general "ask" queries). Answer briefly and directly using whatever
  report/briefing context is available.
- The user may share an approximate current location with you as context;
  treat it as authoritative for the report's location unless contradicted
  by the spoken report itself.
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
def _auto_scan_worker():
    """Background thread: periodically grabs the latest camera frame and runs
    vision analysis + ML pipeline WITHOUT blocking the live camera feed."""
    import tempfile, os
    from utils.vision_report_generator import generate_incident_report
    from utils.pipeline import analyze_report
    from utils.storage import save_incident

    AUTO_SCAN_INTERVAL = 30
    os.makedirs("data/reports", exist_ok=True)

    while _store.running:
        time.sleep(AUTO_SCAN_INTERVAL)
        if not _store.running:
            break

        frame = get_frame()
        if frame is None:
            continue

        try:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            raw = bytes(jpeg)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.write(raw)
            tmp_path = tmp.name
            tmp.close()

            print("\n🔍 AUTO-SCAN: Analyzing camera frame for disasters...")
            analysis = generate_incident_report(tmp_path)
            os.unlink(tmp_path)

            if "DISASTER TYPE:" in analysis or "FLOOD" in analysis.upper() or "EARTHQUAKE" in analysis.upper() or "FIRE" in analysis.upper() or "LANDSLIDE" in analysis.upper():
                print("🚨 AUTO-SCAN: Disaster detected! Running ML pipeline...")
                result = analyze_report(analysis)
                disaster = result.get("disaster", "Unknown")
                location = result.get("location", "Unknown")
                severity = result.get("severity", "Unknown")

                with _store.lock:
                    _store.last_briefing = result.get("briefing", "")
                    _store.last_report_text = analysis

                report_text = f"""
🚨 AUTO-DETECTED DISASTER FROM LIVE CAMERA
===========================================
Disaster: {disaster}
Location: {location}
Severity: {severity}
Authenticity: {result.get('authenticity', 'Unknown')}

ALERTS:
Citizen: {result.get('citizen_alert', 'N/A')}
NGO: {result.get('ngo_alert', 'N/A')}
Government: {result.get('government_alert', 'N/A')}

AI BRIEFING:
{result.get('briefing', 'N/A')}

VISION ANALYSIS:
{analysis}
                """.strip()

                rpath = os.path.join("data/reports", f"auto_cam_scan_{int(time.time())}.txt")
                with open(rpath, "w", encoding="utf-8") as f:
                    f.write(report_text)
                print(f"📁 Auto-scan report saved: {rpath}")
                print("✅ AUTO-SCAN: Pipeline complete — alerts generated & report saved\n")
            else:
                print("🔍 AUTO-SCAN: No clear disaster detected in this frame.\n")
        except Exception as scan_err:
            print(f"⚠️ AUTO-SCAN error: {scan_err}\n")


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

    # Start auto-scan in a separate background thread
    scan_thread = threading.Thread(target=_auto_scan_worker, daemon=True)
    scan_thread.start()

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
    🚨 Runs a disaster report (spoken or typed) through the full CrisisLens
    AI pipeline and returns a short spoken-friendly summary of the emergency
    briefing.

    Side effect: stores the report text and the full written briefing in
    the session store (_store.last_report_text / _store.last_briefing) so
    other features — 🔊 Read Response, 📧 Send Alert, 📱 Send SMS — can
    reuse the latest analysis without re-running the pipeline.
    """
    with _store.lock:
        _store.last_report_text = report_text

    if not PIPELINE_AVAILABLE:
        msg = ("Pipeline modules are not available. "
               "Please run this from the CrisisLens AI project root.")
        with _store.lock:
            _store.last_briefing = msg
        return msg

    if not report_text or not report_text.strip():
        msg = "No report content was captured to analyze."
        with _store.lock:
            _store.last_briefing = msg
        return msg

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
            print("REPORT — FLAGGED AS SUSPICIOUS / FAKE")
            print(f"Disaster: {disaster} | Location: {location} | "
                  f"Severity: {severity} | Authenticity: {authenticity}")
            print("=" * 60 + "\n")
            with _store.lock:
                _store.last_briefing = spoken
            try:
                from utils.storage import save_incident
                save_incident({
                    "disaster": disaster,
                    "location": location,
                    "severity": severity,
                    "authenticity": authenticity,
                    "report": report_text,
                    "briefing": "FAKE/SUSPICIOUS — no briefing generated",
                    "source": "voice",
                    "timestamp": time.time(),
                })
            except Exception:
                pass
            return spoken

        briefing = generate_relief_plan(
            disaster_type=disaster,
            location=location,
            severity=severity,
            authenticity=authenticity,
            original_report=report_text,
        )

        print("\n" + "=" * 60)
        print("REPORT — FULL BRIEFING GENERATED")
        print(f"Disaster: {disaster} | Location: {location} | Severity: {severity}")
        print("-" * 60)
        print(briefing)
        print("=" * 60 + "\n")

        with _store.lock:
            _store.last_briefing = briefing

        # ── Save to incidents.json + data/reports/ ──
        try:
            from utils.storage import save_incident
            save_incident({
                "disaster": disaster,
                "location": location,
                "severity": severity,
                "authenticity": authenticity,
                "report": report_text,
                "briefing": briefing,
                "source": "voice",
                "timestamp": time.time(),
            })
            import os as _os2
            _os2.makedirs("data/reports", exist_ok=True)
            report_path = _os2.path.join("data/reports", f"voice_report_{int(time.time())}.txt")
            full_report = f"""
🚨 VOICE REPORT — AUTO GENERATED
================================
Disaster: {disaster}
Location: {location}
Severity: {severity}
Authenticity: {authenticity}

ALERTS (generated by alert engine):
(Open the Streamlit app to view full alerts)

AI BRIEFING:
{briefing}

ORIGINAL REPORT:
{report_text}
            """.strip()
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(full_report)
            print(f"📁 Voice report saved: {report_path}")
            print(f"💾 Incident saved to incidents.json")
        except Exception as save_err:
            print(f"⚠️ Could not save report: {save_err}")

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
        msg = f"Pipeline error while analyzing the report: {e}"
        with _store.lock:
            _store.last_briefing = msg
        return msg


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
        # When set in the future, model replies are treated as Q&A answers
        # (🤖 ask / photo Q&A) rather than disaster-report transcript text.
        self._qa_mode_until  = 0.0

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

                    # Capture spoken transcript text. While a 🤖 ask / photo
                    # Q&A is in flight, treat this as a Q&A answer instead of
                    # disaster-report transcript so it doesn't get fed into
                    # the pipeline by mistake.
                    if hasattr(response, "text") and response.text:
                        if time.time() < self._qa_mode_until:
                            with _store.lock:
                                _store.last_ai_response = (
                                    (_store.last_ai_response or "") + response.text
                                )
                            print(f"AI ASSISTANT: {response.text}")
                        else:
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

    # ---------------- Speak text back through the live session ----------

    async def _speak_via_session(self, spoken_summary: str):
        """
        Pushes a short text into the live Gemini session so it gets spoken
        aloud verbatim through the voice channel. Shared by the automatic
        silence-triggered analysis and the manual 🚨 'analyze' / 📝 'report:'
        commands.
        """
        if self.session and _store.running:
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{
                        "text": (
                            "[SYSTEM: Speak this emergency briefing summary "
                            f"aloud verbatim, calmly and clearly: \"{spoken_summary}\"]"
                        )
                    }]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"FAILED TO SEND BRIEFING TO SESSION: {e}")

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
                await self._speak_via_session(spoken_summary)

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
            print("No active session to ask.")
            return

        with _store.lock:
            _store.last_ai_response = ""
        self._qa_mode_until = time.time() + QA_REPLY_WINDOW_SECONDS

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

    # ---------------- 🤖 Generic AI assistant Q&A ----------------

    async def send_ai_assistant_question(self, question: str):
        """
        Sends a direct question to the live session that is NOT tied to a
        specific photo — e.g. "ask what should we prioritize first". Uses
        whatever report/briefing context is already available.
        """
        if not self.session or not _store.running:
            print("No active session to ask.")
            return

        with _store.lock:
            _store.last_ai_response = ""
            context_bits = []
            if _store.last_report_text:
                context_bits.append(f"Last report: {_store.last_report_text}")
            if _store.last_briefing:
                context_bits.append(f"Last briefing: {_store.last_briefing}")
        context = " ".join(context_bits) if context_bits else "none yet"

        self._qa_mode_until = time.time() + QA_REPLY_WINDOW_SECONDS

        prompt_text = (
            f"[SYSTEM: The operator is asking the AI assistant a direct "
            f"question, not tied to a specific photo. Context so far: "
            f"{context}. Question: \"{question}\". Answer briefly and "
            f"directly in 1-3 sentences.]"
        )

        try:
            await self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": prompt_text}]},
                turn_complete=True,
            )
            print(f"AI ASSISTANT: asked -> '{question}'")
        except Exception as e:
            print(f"ASK AI ERROR: {e}")

    # ---------------- Console command handling ----------------

    def _print_help(self):
        print("""
Available console commands:
  📝 report: <text>              — submit a typed disaster report for analysis
     submit report <text>
  📷 upload photo <path>         — add a photo from disk
  📸 take photo / capture photo  — capture a photo from the live camera
     list photos                — list all photos in this session
     ask about photo <n>: <q>   — ask the AI about a specific photo (vision)
  📍 location                   — show approximate current location
     current location / where am i
  🚨 analyze / analyze incident — force-analyze the report right now
  🤖 ask <question>             — ask the AI assistant a direct question
  🔊 read / read response       — read the last answer/briefing aloud (offline TTS)
  📧 send alert [to <email>]    — email the latest briefing
  📱 send sms [to <phone>]      — text the latest briefing
     help                       — show this list again
(🎤 voice reports work automatically any time you speak into the mic)
""")

    async def handle_text_command(self, raw_text: str):
        """
        Handles a typed (or transcribed) command from the operator console.
        See _print_help() / module docstring for the full command list.
        """
        text = raw_text.strip()
        text_l = text.lower()

        if not text:
            return

        # --- 📷 Upload photo from path ---
        if text_l.startswith("upload photo"):
            path = text[len("upload photo"):].strip().strip('"').strip("'")
            try:
                photo = await asyncio.to_thread(add_photo_from_path, path)
                print(f"-> Photo {photo['id']} ready. Ask about it any time, "
                      f"e.g. 'ask about photo {photo['id']}: how bad is the flooding?'")
            except Exception as e:
                print(f"UPLOAD FAILED: {e}")
            return

        # --- 📸 Capture photo from camera ---
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

        # --- Ask about a specific / numbered photo (vision Q&A) ---
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

        # --- 📧 Send Alert (email) ---
        m = re.match(r"send alert(?:\s+to\s+(\S+))?", text, re.IGNORECASE)
        if m:
            to_email = m.group(1)
            body = _store.last_briefing or _store.last_report_text
            if not body:
                print("Nothing to alert on yet. Submit a report ('report: ...') or 'analyze' first.")
                return
            try:
                result = await asyncio.to_thread(
                    send_email_alert, "CrisisLens AI — Emergency Alert", body, to_email
                )
                print(f"-> {result}")
            except Exception as e:
                print(f"SEND ALERT ERROR: {e}")
            return

        # --- 📱 Send SMS ---
        m = re.match(r"send sms(?:\s+to\s+(\S+))?", text, re.IGNORECASE)
        if m:
            to_phone = m.group(1)
            body = _store.last_briefing or _store.last_report_text
            if not body:
                print("Nothing to text out yet. Submit a report ('report: ...') or 'analyze' first.")
                return
            try:
                result = await asyncio.to_thread(send_sms_alert, body, to_phone)
                print(f"-> {result}")
            except Exception as e:
                print(f"SEND SMS ERROR: {e}")
            return

        # --- 📍 Current location ---
        if text_l in ("location", "current location", "where am i", "get location"):
            try:
                loc = await asyncio.to_thread(get_current_location)
                print(f"-> Approximate location: {loc['address']} "
                      f"(lat {loc['lat']}, lon {loc['lon']})")
            except Exception as e:
                print(f"LOCATION ERROR: {e}")
            return

        # --- 🚨 Analyze incident (manual trigger, doesn't wait for silence) ---
        if text_l in ("analyze", "analyze incident", "analyse", "analyse incident"):
            if self._pending_transcript_parts:
                report_text = " ".join(self._pending_transcript_parts).strip()
                self._pending_transcript_parts = []
            else:
                report_text = _store.last_report_text
            if not report_text:
                print("No report text available yet. Speak a report, type 'report: ...', or wait for transcript.")
                return
            print("\nANALYZING INCIDENT (manual trigger)...\n")
            spoken_summary = await asyncio.to_thread(run_pipeline_and_build_briefing, report_text)
            print(f"BRIEFING SUMMARY: {spoken_summary}")
            await self._speak_via_session(spoken_summary)
            return

        # --- 🔊 Read last response aloud (offline TTS) ---
        if text_l in ("read", "read response", "read last response", "read aloud"):
            text_to_read = _store.last_ai_response or _store.last_briefing or _store.last_report_text
            if not text_to_read:
                print("Nothing to read yet — ask a question, submit a report, or wait for a briefing first.")
                return
            print("\n🔊 Reading aloud...\n")
            await asyncio.to_thread(speak_text_locally, text_to_read)
            return

        # --- 📝 Typed text report ---
        m = re.match(r"(?:submit report|report)\b\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
        if m and len(m.group(1).strip()) > 3:
            report_text = m.group(1).strip()
            print("\nANALYZING TYPED REPORT...\n")
            spoken_summary = await asyncio.to_thread(run_pipeline_and_build_briefing, report_text)
            print(f"BRIEFING SUMMARY: {spoken_summary}")
            await self._speak_via_session(spoken_summary)
            return

        # --- 🤖 Ask AI assistant (generic, non-photo question) ---
        if text_l.startswith("ask "):
            question = text[4:].strip()
            if question:
                await self.send_ai_assistant_question(question)
            return

        # --- Help ---
        if text_l in ("help", "commands", "?"):
            self._print_help()
            return

        # --- Generic photo reference fallback: "the last photo", "this picture", etc. ---
        photo = resolve_photo_reference(text_l)
        if photo is not None:
            await self.send_photo_question(photo, text)
            return

    async def listen_console_commands(self):
        """
        Reads typed commands from stdin in a background thread so the
        operator can use any console feature alongside the live voice
        session.
        """
        loop = asyncio.get_event_loop()
        self._print_help()

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

        print("\nFeature availability:")
        print(f"  Pipeline (classify/severity/relief plan): "
              f"{'OK' if PIPELINE_AVAILABLE else 'MISSING - utils/ not importable'}")
        print(f"  📍 Location lookup (requests):             "
              f"{'OK' if REQUESTS_AVAILABLE else 'MISSING - pip install requests'}")
        print(f"  🔊 Offline read-aloud (pyttsx3):            "
              f"{'OK' if PYTTSX3_AVAILABLE else 'MISSING - pip install pyttsx3'}")
        print(f"  📱 SMS alerts (twilio):                     "
              f"{'OK' if TWILIO_AVAILABLE else 'MISSING - pip install twilio'}")
        print(f"  📧 Email alerts (SMTP):                     "
              f"{'OK' if (SMTP_USER and SMTP_PASSWORD) else 'NOT CONFIGURED - set SMTP_USER/SMTP_PASSWORD in .env'}")
        print()

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
    print("  Speak or type a disaster report. Type 'help' for all")
    print("  console commands. Press Q in the camera window or")
    print("  Ctrl+C in this terminal to stop.")
    print("=" * 60)
    main()