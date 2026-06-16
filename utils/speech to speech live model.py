import os
import sys
import cv2
import asyncio
import threading
import traceback
import time
import queue
import types as _types
from dotenv import load_dotenv

load_dotenv()

# =====================================================
# PERSISTENT STORE
# =====================================================
_KEY = "__gemini_store__"
if _KEY not in sys.modules:
    _m          = _types.ModuleType(_KEY)
    _m.frame    = None
    _m.lock     = threading.Lock()
    _m.running  = False
    _m.gemini_q = queue.Queue(maxsize=2)
    _m.pya      = None
    _m.client   = None
    sys.modules[_KEY] = _m

_store = sys.modules[_KEY]

def set_frame(f):
    with _store.lock:
        _store.frame = f

def get_frame():
    with _store.lock:
        return _store.frame

# =====================================================
# AUDIO / GEMINI CONSTANTS
# =====================================================
import pyaudio as _pyaudio
FORMAT              = _pyaudio.paInt16
CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024
MODEL               = "models/gemini-2.5-flash-native-audio-latest"

# =====================================================
# CAMERA WORKER — 30 fps, OpenCV window
# =====================================================
def camera_worker():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    if not cap.isOpened():
        print("❌ Camera could not be opened")
        return

    print("📷 Camera started — press Q to quit")
    last_sent = 0.0

    while _store.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        set_frame(frame.copy())

        cv2.imshow("Gemini Live Camera  |  press Q to quit", frame)
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
    print("📷 Camera stopped")

# =====================================================
# GEMINI ASSISTANT
# =====================================================
class GeminiAssistant:

    def __init__(self):
        # FIX 1: unlimited queue so no audio chunk is ever dropped
        self.audio_in_q  = asyncio.Queue()
        self.audio_out_q = asyncio.Queue()
        self.session     = None

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
            print("🎤 Microphone started — speak now")
            while _store.running:
                data = await asyncio.to_thread(
                    stream.read, CHUNK_SIZE, exception_on_overflow=False
                )
                if data:
                    await self.audio_out_q.put(data)
        except Exception as e:
            print(f"❌ Microphone error: {e}")
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
                print(f"❌ Audio send error: {e}")

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
                print(f"❌ Camera frame send error: {e}")

    async def receive_responses(self):
        # FIX 2: correctly extract audio data from all response types
        while _store.running:
            try:
                async for response in self.session.receive():
                    if not _store.running:
                        break
                        
                    # Print any text (transcripts / debug)
                    if hasattr(response, "text") and response.text:
                        print(f"🤖 Gemini: {response.text}")

                    # Extract audio bytes — handle both old and new SDK formats
                    audio_data = None

                    # New SDK: server_content → model_turn → parts
                    if hasattr(response, "server_content") and response.server_content:
                        sc = response.server_content
                        if hasattr(sc, "model_turn") and sc.model_turn:
                            for part in sc.model_turn.parts:
                                if hasattr(part, "inline_data") and part.inline_data:
                                    audio_data = part.inline_data.data
                                    break

                    # Old SDK / fallback: response.data directly
                    if audio_data is None and hasattr(response, "data") and response.data:
                        audio_data = response.data

                    if audio_data:
                        # FIX 3: use put (blocking) not put_nowait so chunks are NEVER dropped
                        await self.audio_in_q.put(audio_data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Response receiving error: {e}")
                if not _store.running:
                    break
                await asyncio.sleep(0.1)

    async def play_audio(self):
        pya = _store.pya
        stream = None
        try:
            stream = pya.open(
                format=FORMAT, channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE, output=True,
            )
            print("🔊 Speaker ready")
            while _store.running:
                data = await self.audio_in_q.get()
                if data:
                    await asyncio.to_thread(stream.write, data)
        except Exception as e:
            print(f"❌ Speaker error: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

    async def run(self):
        from google.genai import types
        CONFIG = types.LiveConnectConfig(response_modalities=["AUDIO"])
        async with _store.client.aio.live.connect(
            model=MODEL, config=CONFIG
        ) as session:
            self.session = session
            print("✅ Gemini connected — ready!\n")
            await asyncio.gather(
                self.listen_mic(),
                self.send_mic_audio(),
                self.send_camera_frames(),
                self.receive_responses(),
                self.play_audio(),
            )

# =====================================================
# MAIN
# =====================================================
def main():
    try:
        import pyaudio as _pa
        from google import genai

        print("🔧 Initializing PyAudio...")
        _store.pya = _pa.PyAudio()
        print("✅ PyAudio OK")

        print("🔧 Initializing Gemini client...")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("❌ ERROR: GEMINI_API_KEY environment variable not set!")
            return
        
        _store.client = genai.Client(
            api_key=gemini_api_key,
            http_options={"api_version": "v1beta"},
        )
        print("✅ Gemini client OK")

        _store.running = True

        threading.Thread(target=camera_worker, daemon=True).start()
        asyncio.run(GeminiAssistant().run())

    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        err = str(e)
        if "getaddrinfo failed" in err or "11001" in err:
            print("\n❌ Network Error: Cannot reach Google servers.")
            print("   → Use a VPN (Google APIs may be blocked in your region)")
            print("   → Check GEMINI_API_KEY in your .env file")
        elif "Could not construct Bearer token" in err:
            print("\n❌ Authentication Error: Invalid GEMINI_API_KEY")
            print("   → Verify your GEMINI_API_KEY is correct in .env")
        else:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
    finally:
        _store.running = False
        if _store.pya:
            try:
                _store.pya.terminate()
            except Exception:
                pass
        print("✅ Cleanup complete")


if __name__ == "__main__":
    print("=" * 50)
    print("  🎤 Gemini Speech-to-Speech + Camera Assistant")
    print("=" * 50)
    main()