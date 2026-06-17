# CrisisLensAI 🚨

**Pakistan Disaster Intelligence Platform** — Real-time flood & earthquake monitoring, AI-powered image analysis, voice-enabled crisis commander, and automated alerting.

---

## Features ✨

- **🧠 Dual ML Model Classification** — MobileNetV2-based deep learning models (`earthquake_model.h5` / `flood_model.h5`) classify disaster scenes from uploaded or live-captured images with confidence scores
- **📸 Photo Analysis Pipeline** — Upload/take a photo → DL model classifies (Earthquake/Flood) → Gemini Vision generates detailed incident report → text pipeline classifies, locates, scores severity, detects fake news → generates alerts + AI Commander briefing
- **🎤 Live Voice Commander** — Real-time speech-to-speech interaction with Gemini Live API for on-the-ground crisis reporting and relief planning (microphone + speaker required)
- **📹 Live Camera Auto-Scan** — Background thread captures frames every 30s → Gemini Vision checks for disaster → if detected, runs full pipeline and saves report automatically
- **🗺️ Interactive Maps** — Folium-based inline maps with location markers and severity coloring
- **🚨 Multi-Tier Alerts** — Citizen (👤), NGO (🏛️), and Government (🏢) alerts with English + Urdu text
- **🤖 AI Commander** — Groq-powered relief planning, evacuation routes, resource allocation, and situational briefings
- **📄 Automated Reports** — Full crisis reports saved to `data/reports/` with download buttons (`.txt`)
- **📧 Email Alerts** — SMTP-based alert dispatch (configurable via `.env`)
- **📊 Analytics Dashboard** — Historical incident metrics, disaster distribution charts, trend analysis
- **🔍 Fake News Detection** — ML-based authenticity scoring for incoming reports
- **🌐 FastAPI Backend** — REST API endpoints for analytics and data access

---

## Project Structure 📁

```
CrisisLensAI/
├── app.py                      # Main Streamlit entry point
├── api/
│   └── main.py                 # FastAPI backend
├── pages/
│   ├── dashboard.py            # Analytics dashboard
│   ├── analytics.py            # Detailed analytics
│   ├── prediction.py           # Prediction workflows
│   ├── live_map.py             # Live disaster map
│   └── ai_assistant.py         # AI Commander chat
├── utils/
│   ├── dl_classifier.py        # .h5 model loader & image inference (Earthquake/Flood)
│   ├── disaster_classifier.py  # Keyword + TF-IDF disaster classifier
│   ├── pipeline.py             # Full analysis pipeline (text + image)
│   ├── severity_prediction.py  # Severity scoring engine
│   ├── fake_news_detection.py  # Authenticity checker
│   ├── location_extractor.py   # Location extraction from text
│   ├── alert_engine.py         # Multi-tier alert generator
│   ├── gemini_relief_agent.py  # Groq AI Commander + chat
│   ├── vision_report_generator.py  # Gemini Vision image analysis
│   ├── maps.py                 # Folium map manager
│   ├── storage.py              # Incident JSON persistence
│   ├── email_alert.py          # SMTP email sender
│   ├── pdf_report_generator.py # PDF report generation
│   └── speech to speech live model.py  # Live voice commander
├── models/
│   ├── earthquake_model.h5     # MobileNetV2 DL model (Earthquake)
│   ├── flood_model.h5          # MobileNetV2 DL model (Flood)
│   ├── earthquake_model.pkl    # sklearn earthquake classifier
│   ├── xgboost_flood_model.pkl # XGBoost flood classifier
│   ├── tfidf.pkl               # TF-IDF vectorizer
│   ├── scaler.pkl              # Feature scaler
│   ├── label_encoder.pkl       # Label encoder
│   ├── imputer.pkl             # Missing value imputer
│   ├── fake_news_model.pkl     # Fake news detection model
│   └── categorical_encoders.pkl # Category encoders
├── data/
│   ├── incidents.json          # Saved incidents
│   ├── reports/                # Auto-generated crisis reports
│   └── *.csv                   # Training datasets
├── tests/
│   ├── test_model_loader.py
│   ├── test_runtime_fallbacks.py
│   └── test_groq_agent_fallback.py
├── notebooks/                  # Training & experimentation scripts
├── .env                        # API keys & config (not committed)
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
└── SPEECH_MODEL_GUIDE.md       # Voice model setup guide
```

---

## Requirements 📋

- Python 3.10+
- pip
- Internet connection for AI API access
- **API Keys** (see [Environment Variables](#environment-variables-))

---

## Installation 🛠️

```bash
# 1. Clone the repository
git clone <repo-url>
cd CrisisLensAI

# 2. Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create environment file
copy .env.example .env   # Windows
cp .env.example .env      # Linux/Mac

# 5. Edit .env with your API keys (see below)
```

---

## Environment Variables 🔑

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key (Vision analysis, Live model) — [Get one](https://ai.google.dev/) |
| `GROQ_API_KEY` | ✅ Yes | Groq API key (AI Commander, chat assistant) — [Get one](https://console.groq.com/keys) |
| `EMAIL_USER` | ❌ No | Gmail address for SMTP alerts |
| `EMAIL_PASS` | ❌ No | Gmail app password for SMTP alerts |

`.env` format:
```env
# === GEMINI ===
GEMINI_API_KEY=AIzaXXXXXXXXXXXX

# === GROQ ===
GROQ_API_KEY=gsk_XXXXXXXXXXXX

# === SMTP (optional) ===
EMAIL_USER=your.email@gmail.com
EMAIL_PASS=your-app-password
```

---

## Running the App 🚀

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### Quick Test with Sample Data
1. Open the **Upload Photo** tab
2. Drop in a disaster scene image (flood, earthquake damage, etc.)
3. Watch the pipeline auto-run:
   - **Step 1** 🧠 DL Model classifies earthquake/flood probability
   - **Step 2** 🔍 Gemini Vision analyzes the scene
   - **Step 3** ⚙️ Full pipeline runs (classify → locate → severity → fake news → alerts → briefing)
   - **Step 4** 🗺️ Inline map + 🚨 Alerts + 📄 Downloadable report

### Live Voice Commander 🎤
1. Click **▶️ Start Live Model** in the app
2. Check terminal for initialization logs
3. Speak crisis reports directly — the AI responds with voice briefings

### FastAPI Backend 🌐
```bash
uvicorn api.main:app --reload
```
API docs at `http://localhost:8000/docs`

---

## ML Pipeline Flow 🔄

```
Image Upload/Capture
        │
        ▼
┌─────────────────────────────┐
│   🧠 DL Classifier (.h5)    │
│   Earthquake Model (75%)    │
│   Flood Model (85%)         │
│   → Prediction + Confidence │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   🔍 Gemini Vision           │
│   Image analysis + report   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   ⚙️ Crisis Pipeline         │
│   ├── Classify disaster      │
│   ├── Extract location       │
│   ├── Predict severity       │
│   ├── Detect fake news       │
│   ├── Generate 3-tier alerts │
│   ├── AI Commander briefing  │
│   └── Save incident          │
└─────────────────────────────┘
        │
        ▼
   📊 Results Display
   ├── DL probabilities
   ├── Disaster type / Location / Severity / Authenticity
   ├── Inline Folium map
   ├── Citizen / NGO / Govt alerts
   ├── AI Commander briefing
   ├── Report saved to data/reports/
   └── Download button
```

---

## Deep Learning Models 🧠

Both models are **MobileNetV2 (224×224)** binary classifiers fine-tuned for disaster imagery:

| Model | Architecture | Input | Output | Classes |
|---|---|---|---|---|
| `earthquake_model.h5` | MobileNetV2 → Pool → Dense(128) → Dropout → Dense(1) | 224×224×3 | Sigmoid probability | Earthquake vs Not |
| `flood_model.h5` | MobileNetV2 → Pool → Dense(128) → Dropout → Dense(1) | 224×224×3 | Sigmoid probability | Flood vs Not |

Keras 3.13.2 / TensorFlow 2.21.0 backend.

---

## Voice Commander Setup 🎤

See [SPEECH_MODEL_GUIDE.md](./SPEECH_MODEL_GUIDE.md) for detailed setup of:
- Microphone/speaker requirements
- PyAudio installation (Windows wheels)
- Camera configuration
- Voice command examples

---

## Testing ✅

```bash
# Install pytest
pip install pytest

# Run tests
python -m pytest tests/
```

---

## API Endpoints 🌐

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/analytics` | Incident analytics summary |
| `GET` | `/api/reports` | List all saved reports |
| `GET` | `/api/health` | Health check |

---

## Troubleshooting 🔧

### DL Model fails to load
- Ensure `tensorflow` is installed: `pip install tensorflow`
- Verify model files exist in `models/`
- RAM usage: models require ~500MB combined

### Gemini API errors (503 / 429)
- Automatic retry (3 attempts, 5s backoff) built in
- Check API key validity in `.env`
- Try a VPN if in Pakistan (Google API restrictions)

### Groq API errors (401)
- Regenerate key at https://console.groq.com/keys
- Update `GROQ_API_KEY` in `.env`
- Restart the app

### Camera not working
- Check permissions: Settings → Privacy → Camera
- Camera auto-scan runs in a background daemon thread

### "Cannot read clipboard" in chat
- This is from the chat interface, not CrisisLensAI — ignore

---

## Deployment ☁️

For production deployment:
- Set environment variables in hosting platform
- Install `requirements.txt` on target environment
- Ensure model files (`models/`) are included in deployment
- Use `st.secrets` instead of `.env` for Streamlit Cloud

---

## Security 🔒

- `.env` is git-ignored — never commit real API keys
- SMTP credentials stored only in `.env`
- All AI API calls use HTTPS

---

## License 📄

MIT
