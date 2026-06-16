# CrisisLensAI: Speech-to-Speech Live Model Guide

**Disaster Response AI Assistant** — Real-time voice interaction with Gemini for crisis briefings, emergency response planning, and disaster analysis in Pakistan.

---

## Prerequisites

### 1. Install Required Packages
```bash
pip install -r requirements.txt
pip install pyaudio opencv-python
```

**Troubleshooting PyAudio on Windows:**
- Download pre-built wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
- Install: `pip install PyAudio-0.2.13-cp311-cp311-win_amd64.whl`

### 2. Set Environment Variables

Update `.env` file with:
```env
GEMINI_API_KEY=your_api_key_here
GROQ_API_KEY=your_groq_key_here
```

Get API keys from:
- Gemini: https://ai.google.dev/
- Groq: https://console.groq.com/

### 3. Hardware Requirements (for CrisisLensAI Operations Center)
- **Microphone**: For real-time voice commands and situation reporting
- **Speaker**: For receiving AI briefings and alert notifications
- **Camera**: For live disaster zone monitoring (optional but recommended)

## Running the Live Model

### From CrisisLensAI Dashboard (Recommended)
1. Start the dashboard: `streamlit run app.py`
2. Navigate to **"🎤 Speech-to-Speech Live Model"** section
3. Click **"▶️ Start live model"** button
4. Wait for initialization (30-60 seconds)
5. Check **terminal for logs** confirming readiness:
   ```
   ✅ PyAudio OK
   ✅ Gemini client OK
   ✅ Gemini connected — ready!
   ```
6. Once ready, interact with the AI for:
   - **Live disaster briefing requests** during incidents
   - **Emergency response planning** guidance
   - **Flood/earthquake impact analysis** from voice descriptions
   - **Relief resource allocation** recommendations
   - **Real-time situation updates** to crisis command center

### Example Crisis Scenarios
```
"Analyze flood situation in Swat Valley based on latest reports"
"Generate emergency relief plan for Quetta earthquake"
"What evacuation routes should we use for the Sindh floods?"
"Provide damage assessment briefing from the Balochistan incident"
```

### From Command Line (Standalone Testing)
```bash
cd utils
python "speech to speech live model.py"
```

## Troubleshooting

### "GEMINI_API_KEY not set"
- ✅ Add API key to `.env` file in project root
- ✅ Restart Streamlit: `streamlit run app.py`
- ✅ Verify with PowerShell: `$env:GEMINI_API_KEY`

### "Network Error: Cannot reach Google servers"
- ✅ Common in Pakistan — use a VPN to access Google APIs
- ✅ Check internet connectivity: `ping 8.8.8.8`
- ✅ Verify Gemini API status
- ⚠️ Model will fallback to Groq API if available

### "Authentication Error: Invalid GEMINI_API_KEY"
- ✅ Regenerate key from https://ai.google.dev/
- ✅ Copy the full key (no spaces)
- ✅ Verify key format in `.env` (should start with `AIza...`)

### "Camera could not be opened"
- ✅ Check camera permissions in Windows settings
- ✅ Try: Settings → Privacy & Security → Camera → App permissions
- ✅ Model still works for **disaster audio briefings** without camera
- ℹ️ Camera is optional; voice-only mode is fully functional

### "Microphone not detected"
- ✅ Check system audio settings: Settings → Sound → Input devices
- ✅ Test microphone works: Settings → Sound → Volume mixer
- ✅ Reinstall PyAudio: `pip install --force-reinstall pyaudio`
- ⚠️ Critical for crisis response — test before deployment

### "Model stuck at initialization"
- ✅ Restart Streamlit: Ctrl+C and `streamlit run app.py`
- ✅ Check terminal for specific error messages
- ✅ Try VPN if in Pakistan (network connectivity issue)
- ✅ Increase timeout: Wait 2-3 minutes before declaring failure

### "Response audio not playing"
- ✅ Check speaker volume: Settings → Sound → Volume mixer
- ✅ Test speaker: Play system sound in Settings → Sound
- ✅ Verify output device selected in Sound settings

### Status shows "Idle" but button says "Stop"
- ✅ Thread may still be starting (wait 15-20 seconds)
- ✅ Check terminal for specific error messages
- ✅ Try clicking "⏹️ Stop" then "▶️ Start" again
- ✅ Restart Streamlit if persistent

## How It Works

1. **Audio Input Loop**
   - Captures microphone audio (16kHz, PCM)
   - Sends to Gemini Live API in real-time

2. **Video Input Loop**  
   - Captures camera frames at 30 FPS
   - Compresses to JPEG (~40% quality)
   - Sends to Gemini Live API (~1 frame per second)

3. **Response Processing**
   - Receives audio responses from Gemini
   - Plays through speakers (24kHz)
   - Prints transcripts to terminal

4. **Graceful Shutdown**
   - Click "⏹️ Stop live model" button
   - Or press 'Q' in OpenCV window
   - Or press Ctrl+C in terminal

## Performance Tips

- Close unnecessary applications to free CPU/RAM
- Use a quieter room to improve speech recognition
- Position camera for good lighting
- Test microphone levels before starting
- Use USB audio devices for better quality

## Advanced Configuration

Edit `utils/speech to speech live model.py`:

```python
# Change model (line ~42):
MODEL = "models/gemini-2.5-flash-native-audio-latest"

# Change camera resolution (line ~63):
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Was 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Was 480

# Change sample rates (line ~40):
SEND_SAMPLE_RATE = 16000      # Microphone sample rate
RECEIVE_SAMPLE_RATE = 24000   # Speaker sample rate
```

## Support

Check terminal output for detailed error messages. The model prints:
- ✅ Initialization success
- 🎤 Microphone status
- 🔊 Speaker status
- 📷 Camera status
- 🤖 Gemini responses
- ❌ Any errors encountered

---

**Last Updated**: 2026-06-16  
**Model Version**: Gemini 2.5 Flash Native Audio  
**Status**: Production Ready
