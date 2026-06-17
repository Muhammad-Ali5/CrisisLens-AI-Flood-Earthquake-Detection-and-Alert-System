import importlib.util
import threading
import tempfile
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap, MarkerCluster
import time
from PIL import Image

from utils.disaster_classifier import classify_disaster
from utils.location_extractor import extract_location
from utils.severity_prediction import predict_severity
from utils.fake_news_detection import detect_fake_news
from utils.alert_engine import alert_engine
from utils.gemini_relief_agent import generate_relief_plan, chat_with_assistant
from utils.maps import map_manager
from utils.vision_report_generator import generate_incident_report
from utils.pipeline import analyze_report



# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CrisisLens AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS — Full Advanced UI
# =========================================================

st.markdown("""
<style>

/* ── Global Reset & Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #0a0e1a;
}

/* ── Hide Streamlit Default UI ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1120 !important;
    border-right: 1px solid #1e2d4a;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #8899bb !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

/* ── Top Navigation Bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.5rem;
    background: #0d1120;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}

.topbar-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.topbar-logo-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #1a6fc4, #0f4a8a);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}

.topbar-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8f0fe;
    letter-spacing: -0.02em;
}

.topbar-subtitle {
    font-size: 0.7rem;
    color: #5577aa;
    margin-top: 1px;
}

.topbar-right {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.live-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: #0a2010;
    border: 1px solid #1a4a2a;
    border-radius: 20px;
    padding: 0.3rem 0.75rem;
    font-size: 0.7rem;
    color: #4dbb7a;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.live-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #4dbb7a;
    animation: livepulse 1.5s ease-in-out infinite;
}

@keyframes livepulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Section Header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.5rem 0 0.75rem 0;
}

.section-icon {
    width: 30px;
    height: 30px;
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    flex-shrink: 0;
}

.si-blue  { background: #0d2545; color: #4d9ef5; }
.si-red   { background: #2a0a0a; color: #f56565; }
.si-green { background: #0a2a15; color: #4dbb7a; }
.si-amber { background: #2a1a00; color: #f5a623; }

.section-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #c8d8f0;
    letter-spacing: -0.01em;
}

.section-divider {
    height: 1px;
    background: #1e2d4a;
    margin: 1.5rem 0;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: #0d1120;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.kpi-card:hover {
    border-color: #2a4a7a;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}

.kpi-card.blue::before  { background: #1a6fc4; }
.kpi-card.green::before { background: #27a85a; }
.kpi-card.amber::before { background: #d97706; }
.kpi-card.red::before   { background: #dc2626; }

.kpi-label {
    font-size: 0.68rem;
    color: #5577aa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.kpi-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e8f0fe;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.kpi-sub {
    font-size: 0.7rem;
    color: #3d5a8a;
    margin-top: 0.3rem;
}

.kpi-icon {
    position: absolute;
    top: 1rem;
    right: 1rem;
    font-size: 1.3rem;
    opacity: 0.3;
}

/* ── Analysis Input Card ── */
.input-card {
    background: #0d1120;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* ── Text Area Override ── */
.stTextArea textarea {
    background: #060911 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 8px !important;
    color: #c8d8f0 !important;
    font-family: 'Inter', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
}

.stTextArea textarea:focus {
    border-color: #1a6fc4 !important;
    box-shadow: 0 0 0 2px rgba(26, 111, 196, 0.2) !important;
}

.stTextArea label {
    color: #8899bb !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ── Analyze Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1a6fc4, #0f4a8a) !important;
    color: #e8f0fe !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1.5rem !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

/* ── Severity Badge ── */
.sev-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.sev-critical { background: #2a0505; color: #f87171; border: 1px solid #7a1515; }
.sev-high     { background: #2a1200; color: #fb923c; border: 1px solid #7a3500; }
.sev-medium   { background: #2a1f00; color: #fbbf24; border: 1px solid #7a5a00; }
.sev-low      { background: #0a2010; color: #4ade80; border: 1px solid #1a5a2a; }
.sev-fake     { background: #1a1a2a; color: #a78bfa; border: 1px solid #4a3a8a; }
.sev-real     { background: #0a2010; color: #4dbb7a; border: 1px solid #1a4a2a; }
.sev-suspect  { background: #2a1a00; color: #f5a623; border: 1px solid #7a4a00; }

/* ── Result Metric Boxes ── */
.result-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}

.result-metric {
    background: #060911;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 0.875rem 1rem;
    text-align: center;
}

.result-metric-label {
    font-size: 0.65rem;
    color: #5577aa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.result-metric-value {
    font-size: 0.95rem;
    font-weight: 700;
    color: #e8f0fe;
}

/* ── Alert Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #060911 !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #1e2d4a !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #5577aa !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 0.4rem 1rem !important;
}

.stTabs [aria-selected="true"] {
    background: #1a3a6a !important;
    color: #e8f0fe !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1rem !important;
}

/* ── Alert Text Box ── */
.alert-box {
    background: #060911;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-size: 0.82rem;
    color: #9fb3d0;
    line-height: 1.7;
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
    min-height: 180px;
}

/* ── AI Briefing ── */
.briefing-container {
    background: #060911;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 0.5rem;
}

.briefing-section {
    margin-bottom: 1.25rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #1e2d4a;
}

.briefing-section:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}

.briefing-section-head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
}

.briefing-section-icon {
    width: 22px;
    height: 22px;
    border-radius: 5px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    flex-shrink: 0;
}

.briefing-section-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #c8d8f0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.briefing-section-body {
    font-size: 0.82rem;
    color: #8899bb;
    line-height: 1.75;
    white-space: pre-wrap;
    padding-left: 1.7rem;
}

/* ── Fake Warning Banner ── */
.fake-banner {
    background: #1a0a00;
    border: 1px solid #7a3500;
    border-radius: 10px;
    padding: 0.875rem 1.25rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #fb923c;
    line-height: 1.5;
}

.fake-banner-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }

/* ── Pipeline Steps ── */
.pipeline-steps {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.75rem 0 1.25rem 0;
    flex-wrap: wrap;
}

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: #0d1120;
    border: 1px solid #1e2d4a;
    border-radius: 20px;
    padding: 0.3rem 0.75rem;
    font-size: 0.7rem;
    color: #5577aa;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.pipeline-step.done {
    border-color: #1a4a2a;
    color: #4dbb7a;
    background: #0a1a10;
}

.pipeline-step.active {
    border-color: #1a5a9a;
    color: #4d9ef5;
    background: #0a1a2a;
    animation: stepglow 1s ease-in-out infinite;
}

@keyframes stepglow {
    0%, 100% { border-color: #1a5a9a; }
    50% { border-color: #4d9ef5; }
}

.step-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: currentColor;
}

.step-arrow { color: #2a3a5a; font-size: 0.7rem; }

/* ── Sidebar Stats ── */
.sidebar-stat {
    background: #0a1020;
    border: 1px solid #1a2a3a;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}

.sidebar-stat-label {
    font-size: 0.65rem;
    color: #445577;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.sidebar-stat-val {
    font-size: 1rem;
    font-weight: 700;
    color: #c8d8f0;
}

/* ── Spinner Override ── */
.stSpinner > div {
    border-top-color: #1a6fc4 !important;
}

/* ── Select / Input labels ── */
.stSelectbox label, .stTextInput label {
    color: #8899bb !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

.stSelectbox > div > div {
    background: #060911 !important;
    border-color: #1e2d4a !important;
    color: #c8d8f0 !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060911; }
::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 3px; }

/* ── Map container ── */
.map-wrap {
    background: #0d1120;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    overflow: hidden;
    margin-top: 0.5rem;
}

.stMarkdown h3 {
    color: #c8d8f0 !important;
    font-size: 0.95rem !important;
}

div[data-testid="metric-container"] {
    background: #0d1120;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 0.875rem 1rem;
}

div[data-testid="metric-container"] label {
    color: #5577aa !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8f0fe !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "results" not in st.session_state:
    st.session_state.results = {}
if "total_reports" not in st.session_state:
    st.session_state.total_reports = 1248
if "active_alerts" not in st.session_state:
    st.session_state.active_alerts = 7
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []
if "incident_context" not in st.session_state:
    st.session_state.incident_context = None
if "speech_live_thread" not in st.session_state:
    st.session_state.speech_live_thread = None
if "speech_live_module" not in st.session_state:
    st.session_state.speech_live_module = None
if "photo_capture_bytes" not in st.session_state:
    st.session_state.photo_capture_bytes = None
if "photo_capture_name" not in st.session_state:
    st.session_state.photo_capture_name = None
if "photo_analysis" not in st.session_state:
    st.session_state.photo_analysis = None
if "photo_report" not in st.session_state:
    st.session_state.photo_report = None
if "cam_key" not in st.session_state:
    st.session_state.cam_key = 0
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0


def load_speech_live_model():
    """Load the standalone speech-to-speech module from its file path."""
    module_path = Path(__file__).resolve().parent / "utils" / "speech to speech live model.py"
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("speech_live_model", module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;padding:0.5rem 0 1rem 0;border-bottom:1px solid #1e2d4a;margin-bottom:1rem;">
        <div style="width:32px;height:32px;background:#1a3a6a;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:1rem;">🚨</div>
        <div>
            <div style="font-size:0.9rem;font-weight:700;color:#e8f0fe;">CrisisLens AI</div>
            <div style="font-size:0.65rem;color:#445577;margin-top:1px;">v2.0 · Pakistan</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p>System status</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Total reports processed</div>
        <div class="sidebar-stat-val">1,248</div>
    </div>
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Active alerts</div>
        <div class="sidebar-stat-val" style="color:#f87171;">7 critical</div>
    </div>
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Flood reports</div>
        <div class="sidebar-stat-val" style="color:#4d9ef5;">842</div>
    </div>
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Earthquake reports</div>
        <div class="sidebar-stat-val" style="color:#fbbf24;">406</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br><p>Model performance</p>', unsafe_allow_html=True)
    st.progress(0.932, text="Disaster classifier: 93.2%")
    st.progress(0.891, text="Fake news detector: 89.1%")

    st.markdown('<br><p>Monitored provinces</p>', unsafe_allow_html=True)
    provinces = {
        "KPK": 78,
        "Punjab": 62,
        "Balochistan": 45,
        "Sindh": 38,
        "AJK": 21,
    }
    for prov, val in provinces.items():
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
            <span style="font-size:0.72rem;color:#8899bb;">{prov}</span>
            <span style="font-size:0.72rem;font-weight:600;color:#c8d8f0;">{val}%</span>
        </div>
        <div style="height:3px;background:#0d1120;border-radius:2px;margin-bottom:0.5rem;">
            <div style="width:{val}%;height:100%;background:#1a6fc4;border-radius:2px;"></div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TOP BAR
# =========================================================

st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        <div class="topbar-logo-icon">🛰️</div>
        <div>
            <div class="topbar-title">CrisisLens AI Commander</div>
            <div class="topbar-subtitle">AI-Powered Flood & Earthquake Emergency Intelligence · Pakistan</div>
        </div>
    </div>
    <div class="topbar-right">
        <div class="live-badge">
            <div class="live-dot"></div>
            LIVE MONITORING
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD KPIs
# =========================================================

st.markdown("""
<div class="kpi-grid">
    <div class="kpi-card blue">
        <div class="kpi-icon">📊</div>
        <div class="kpi-label">Total reports</div>
        <div class="kpi-value">1,248</div>
        <div class="kpi-sub">↑ 12% from yesterday</div>
    </div>
    <div class="kpi-card red">
        <div class="kpi-icon">🚨</div>
        <div class="kpi-label">High priority</div>
        <div class="kpi-value">126</div>
        <div class="kpi-sub">↑ 20% from yesterday</div>
    </div>
    <div class="kpi-card green">
        <div class="kpi-icon">✅</div>
        <div class="kpi-label">Verified real</div>
        <div class="kpi-value">1,089</div>
        <div class="kpi-sub">87.3% authenticity rate</div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-icon">⚠️</div>
        <div class="kpi-label">Fake / suspicious</div>
        <div class="kpi-value">159</div>
        <div class="kpi-sub">61 fake · 98 suspicious</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("""
<div class="section-header">
    <div class="section-icon si-blue">📋</div>
    <div class="section-title">Disaster report input</div>
</div>
""", unsafe_allow_html=True)

col_input, col_options = st.columns([3, 1])

with col_input:
    report = st.text_area(
        "REPORT TEXT",
        placeholder="Paste or type the raw disaster report here. The AI pipeline will classify, extract location, predict severity, detect fake news, and generate a full emergency briefing...",
        height=160,
        key="report_input"
    )

with col_options:
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.7rem;color:#5577aa;text-transform:uppercase;letter-spacing:0.07em;font-weight:600;margin-bottom:0.4rem;">Quick examples</p>', unsafe_allow_html=True)

    if st.button("🌊 Flood — Swat", key="ex1"):
        st.session_state["report_input"] = (
            "Heavy flooding reported in Swat district, Khyber Pakhtunkhwa. "
            "River Swat overflowed its banks. 500+ families displaced. "
            "Roads blocked, bridge collapsed near Mingora. Hospital overwhelmed. "
            "Urgent need for food, water, and medical supplies."
        )
        st.rerun()

    if st.button("🏔️ Earthquake — Quetta", key="ex2"):
        st.session_state["report_input"] = (
            "Magnitude 5.1 earthquake struck 25km north of Quetta, Balochistan. "
            "Multiple buildings collapsed in old city area. "
            "At least 12 casualties reported, 40+ injured. "
            "Gas lines ruptured. Emergency services deployed."
        )
        st.rerun()

    if st.button("⚠️ Suspicious report", key="ex3"):
        st.session_state["report_input"] = (
            "BREAKING: 100,000 people homeless in Lahore due to earthquake. "
            "All hospitals destroyed. Government hiding the truth. "
            "Share immediately before they delete this."
        )
        st.rerun()


# =========================================================
# ANALYZE BUTTON
# =========================================================

st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

analyze_btn = st.button("🔍  Run AI Analysis Pipeline", key="analyze", use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <div class="section-icon si-red">🎤</div>
    <div class="section-title">📸 Photo Analysis & 🎙️ Voice Model</div>
</div>
""", unsafe_allow_html=True)

# ── Voice model start/stop ──
voice_col, status_col = st.columns([2, 1])

with voice_col:
    st.caption("🎙️ Voice commander — بول کر آفات کی رپورٹ کریں | 📸 Photo tools — تصویر سے رپورٹ تیار کریں")

with status_col:
    if st.session_state.speech_live_module is None:
        try:
            st.session_state.speech_live_module = load_speech_live_model()
            if st.session_state.speech_live_module is None:
                st.error("❌ Live model file could not be loaded — ماڈیل لوڈ نہیں ہو سکا")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")

    if st.session_state.speech_live_module is not None:
        live_thread = st.session_state.get("speech_live_thread")
        is_running = live_thread is not None and live_thread.is_alive()

        if not is_running:
            if st.button("▶️ Start Live Model  |  ماڈیل شروع کریں", key="start_live_voice_model", use_container_width=True):
                try:
                    st.session_state.speech_live_thread = threading.Thread(
                        target=st.session_state.speech_live_module.main,
                        name="speech-live-model",
                        daemon=True,
                    )
                    st.session_state.speech_live_thread.start()
                    st.success("✅ Live model started — ماڈیل چل رہا ہے۔ ٹرمینل میں لاگ دیکھیں۔")
                except Exception as exc:
                    st.error(f"❌ Failed to start: {str(exc)[:100]}")
        else:
            if st.button("⏹️ Stop Live Model  |  ماڈیل بند کریں", key="stop_live_voice_model", use_container_width=True):
                try:
                    import sys
                    store_key = "__crisislens_voice_store__"
                    if store_key in sys.modules:
                        store = sys.modules[store_key]
                        store.running = False
                    st.success("✅ Stopping live model — ماڈیل بند ہو رہا ہے۔")
                except Exception as exc:
                    st.error(f"❌ Error stopping: {str(exc)[:100]}")

        live_thread = st.session_state.get("speech_live_thread")
        is_running = live_thread is not None and live_thread.is_alive()

        if is_running:
            st.info("🟢 Status: Running  |  حالت: چل رہا ہے")
        else:
            st.caption("⚪ Status: Idle  |  حالت: غیر فعال")

# ── Divider ──
st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

# ── Photo Tools ──
photo_tabs = st.tabs([
    "📸 Take Photo  |  تصویر لیں",
    "📁 Upload Photo  |  تصویر اپ لوڈ کریں",
    "🔍 Analyze  |  تجزیہ کریں",
    "📄 Report  |  رپورٹ"
])

with photo_tabs[0]:
    cam = st.camera_input("📷 Capture a disaster scene photo  |  آفات کی تصویر لیں", key=f"cam_{st.session_state.cam_key}")

    if cam is not None:
        st.session_state.photo_capture_bytes = cam.getvalue()
        st.session_state.photo_capture_name = "camera_capture.jpg"

    saved = st.session_state.photo_capture_bytes is not None and st.session_state.photo_capture_name == "camera_capture.jpg"
    if saved:
        st.image(st.session_state.photo_capture_bytes, caption="📸 Saved photo  |  محفوظ کردہ تصویر", use_container_width=True)
        st.info("✅ Ready for analysis — اب Analyze ٹیب پر جائیں")
        if st.button("🗑️ Clear  |  ہٹائیں", key="clear_cam_photo", use_container_width=True):
            st.session_state.photo_capture_bytes = None
            st.session_state.photo_capture_name = None
            st.session_state.cam_key += 1
            st.rerun()

with photo_tabs[1]:
    uploaded = st.file_uploader(
        "📤 Upload a disaster scene image  |  آفات کی تصویر اپ لوڈ کریں",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key=f"upload_{st.session_state.upload_key}",
    )
    if uploaded is not None:
        st.session_state.photo_capture_bytes = uploaded.getvalue()
        st.session_state.photo_capture_name = uploaded.name
        st.image(uploaded, caption="📁 Uploaded image  |  اپ لوڈ کردہ تصویر", use_container_width=True)
        st.success(f"✅ Photo uploaded: {uploaded.name} — تصویر اپ لوڈ ہو گئی!")

        # ── Auto-analyze + pipeline + alerts ──
        try:
            ext = os.path.splitext(uploaded.name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(st.session_state.photo_capture_bytes)
                tmp_path = tmp.name

            with st.spinner("🔍 Analyzing with Gemini Vision...  |  تجزیہ ہو رہا ہے۔۔۔"):
                analysis = generate_incident_report(tmp_path)
            os.unlink(tmp_path)

            st.session_state.photo_analysis = analysis
            st.success("✅ Vision analysis done!  |  بصری تجزیہ مکمل!")

            with st.spinner("⚙️ Running ML pipeline + alerts...  |  پائپ لائن چل رہی ہے۔۔۔"):
                result = analyze_report(analysis)

            st.session_state.photo_report = result

            # ── Show Results ──
            disaster = result.get("disaster", "Unknown")
            location = result.get("location", "Unknown")
            severity = result.get("severity", "Unknown")
            authenticity = result.get("authenticity", "Unknown")

            st.markdown("""<div style="border-top:2px solid #1e2d4a;margin:1rem 0 0.5rem 0"></div>""", unsafe_allow_html=True)
            st.markdown("### 📊 Auto-Generated Report  |  خودکار رپورٹ")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**🆘 Disaster  |  آفت:** {disaster}")
                st.markdown(f"**📍 Location  |  مقام:** {location}")
                st.markdown(f"**⚠️ Severity  |  شدت:** {severity}")
                st.markdown(f"**🔍 Authenticity  |  مستندیت:** {authenticity}")
            with col_b:
                st.markdown("**🚨 Alerts  |  الرٹس:**")
                citizen_alert = str(result.get('citizen_alert', ''))
                ngo_alert = str(result.get('ngo_alert', ''))
                gov_alert = str(result.get('government_alert', ''))
                if citizen_alert:
                    with st.expander("👤 Citizen Alert  |  شہری الرٹ", expanded=False):
                        st.markdown(citizen_alert.replace("\n", "  \n"))
                if ngo_alert:
                    with st.expander("🏛️ NGO Alert  |  این جی او الرٹ", expanded=False):
                        st.markdown(ngo_alert.replace("\n", "  \n"))
                if gov_alert:
                    with st.expander("🏢 Govt Alert  |  حکومتی الرٹ", expanded=False):
                        st.markdown(gov_alert.replace("\n", "  \n"))

            with st.expander("📋 Full AI Briefing  |  مکمل بریفنگ"):
                st.markdown(result.get("briefing", "N/A"))

            # ── Inline map ──
            try:
                disaster_map = map_manager.create_map(location, disaster, severity)
                st_folium(disaster_map, width=None, height=350, returned_objects=[])
            except Exception as map_err:
                st.caption(f"📍 Map preview: {location} ({map_err})")

            # ── Save report to disk (record copy) + ML pipeline ──
            report_text = f"""
╔══════════════════════════════════════════════════════════╗
║          🚨 CRISIS DISASTER REPORT — AUTO-GENERATED      ║
║          🚨 آفات کی خودکار رپورٹ                         ║
╚══════════════════════════════════════════════════════════╝

🆘 Disaster Type  |  آفت کی قسم: {disaster}
📍 Location  |  مقام: {location}
⚠️ Severity  |  شدت: {severity}
🔍 Authenticity  |  مستندیت: {authenticity}

════════════════════════════════════════════════════════════
🚨 ALERTS  |  الرٹس
════════════════════════════════════════════════════════════

👤 CITIZEN ALERT  |  شہری الرٹ:
{result.get('citizen_alert', 'N/A')}

🏛️ NGO ALERT  |  این جی او الرٹ:
{result.get('ngo_alert', 'N/A')}

🏢 GOVERNMENT ALERT  |  حکومتی الرٹ:
{result.get('government_alert', 'N/A')}

════════════════════════════════════════════════════════════
🤖 AI COMMANDER BRIEFING  |  AI کمانڈر بریفنگ
════════════════════════════════════════════════════════════

{result.get('briefing', 'N/A')}

════════════════════════════════════════════════════════════
🔍 VISION ANALYSIS  |  بصری تجزیہ
════════════════════════════════════════════════════════════

{analysis}

════════════════════════════════════════════════════════════
✅ Report auto-generated by CrisisLens AI  |  CrisisLens AI
   کے ذریعے خودکار رپورٹ تیار کی گئی
════════════════════════════════════════════════════════════
            """.strip()

            os.makedirs("data/reports", exist_ok=True)
            report_filename = f"crisis_report_{uploaded.name.rsplit('.',1)[0]}_{int(time.time())}.txt"
            report_path = os.path.join("data/reports", report_filename)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            st.success(f"📁 Report auto-saved to `{report_path}`  |  رپورٹ خودکار محفوظ: `{report_path}`")

            st.download_button(
                "📥 Download Report (.txt)  |  رپورٹ ڈاؤن لوڈ کریں",
                data=report_text,
                file_name=f"crisis_report_{uploaded.name.rsplit('.',1)[0]}.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.info("✅ ML pipeline processed + report saved to records  |  ML پروسیسنگ مکمل + رپورٹ محفوظ!")

            if st.button("🗑️ Clear & Upload New  |  نئی تصویر اپ لوڈ کریں", key="clear_after_upload", use_container_width=True):
                st.session_state.photo_capture_bytes = None
                st.session_state.photo_capture_name = None
                st.session_state.upload_key += 1
                st.rerun()
        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                st.error("❌ Gemini API is overloaded (503). Try again in a few seconds  |  Gemini API مصروف ہے، کچھ سیکنڈ بعد دوبارہ کوشش کریں")
            elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                st.error("❌ API rate limit reached. Wait a moment and retry  |  API کی حد ختم، تھوڑی دیر بعد کوشش کریں")
            elif "API_KEY" in err_str.upper() or "unauthorized" in err_str.lower():
                st.error("❌ Invalid Gemini API key. Check your .env file  |  Gemini API کلید درست نہیں، .env چیک کریں")
            else:
                st.error(f"❌ Auto pipeline failed: {e}  |  خودکار پائپ لائن ناکام")
    elif st.session_state.photo_capture_bytes is not None and st.session_state.photo_capture_name != "camera_capture.jpg":
        st.image(st.session_state.photo_capture_bytes, caption="📁 Stored upload  |  محفوظ کردہ تصویر", use_container_width=True)
        st.info("✅ Photo saved — اب نیا اپ لوڈ کرنے کے لیے Clear دبائیں")
        if st.button("🗑️ Clear Photo  |  تصویر ہٹائیں", key="clear_upload_photo", use_container_width=True):
            st.session_state.photo_capture_bytes = None
            st.session_state.photo_capture_name = None
            st.session_state.upload_key += 1
            st.rerun()

with photo_tabs[2]:
    st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)
    if st.button("🔍 Analyze Photo  |  تصویر کا تجزیہ کریں", key="analyze_photo_btn", use_container_width=True):
        if st.session_state.photo_capture_bytes is None:
            st.warning("⚠️ Please take or upload a photo first  |  پہلے تصویر لیں یا اپ لوڈ کریں")
        else:
            try:
                ext = os.path.splitext(st.session_state.photo_capture_name)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(st.session_state.photo_capture_bytes)
                    tmp_path = tmp.name

                with st.spinner("🔍 Analyzing image with Gemini Vision...  |  تصویر کا تجزیہ ہو رہا ہے۔۔۔"):
                    analysis = generate_incident_report(tmp_path)
                st.session_state.photo_analysis = analysis
                os.unlink(tmp_path)

                st.success("✅ Analysis complete!  |  تجزیہ مکمل!")
                st.markdown("### 🔍 Vision Analysis Result  |  بصری تجزیہ کا نتیجہ")
                st.markdown(analysis)
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}  |  تجزیہ ناکام")
    if st.session_state.photo_analysis:
        with st.expander("📋 Last Analysis  |  پچھلا تجزیہ", expanded=True):
            st.markdown(st.session_state.photo_analysis)
        if st.button("🔄 Clear Analysis  |  تجزیہ ہٹائیں", key="clear_analysis"):
            st.session_state.photo_analysis = None
            st.rerun()

with photo_tabs[3]:
    st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)
    if st.button("📄 Generate Complete Report  |  مکمل رپورٹ تیار کریں", key="gen_report_btn", use_container_width=True):
        analysis_text = st.session_state.photo_analysis
        if not analysis_text:
            st.warning("⚠️ Please analyze a photo first (Analyze tab)  |  پہلے تصویر کا تجزیہ کریں (تجزیہ والے ٹیب میں)")
        else:
            with st.spinner("⚙️ Running full crisis pipeline...  |  مکمل پائپ لائن چل رہی ہے۔۔۔"):
                result = analyze_report(analysis_text)
            st.session_state.photo_report = result

            st.success("✅ Report generated!  |  رپورٹ تیار!")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**🆘 Disaster  |  آفت:** {result.get('disaster', 'N/A')}")
                st.markdown(f"**📍 Location  |  مقام:** {result.get('location', 'N/A')}")
                st.markdown(f"**⚠️ Severity  |  شدت:** {result.get('severity', 'N/A')}")
                st.markdown(f"**🔍 Authenticity  |  مستندیت:** {result.get('authenticity', 'N/A')}")
            with col2:
                st.markdown("**🚨 Alerts  |  الرٹس:**")
                st.markdown(f"- 👤 Citizen  |  شہری: {str(result.get('citizen_alert', ''))[:120]}...")
                st.markdown(f"- 🏛️ NGO  |  این جی او: {str(result.get('ngo_alert', ''))[:120]}...")
                st.markdown(f"- 🏢 Govt  |  حکومت: {str(result.get('government_alert', ''))[:120]}...")
            with st.expander("📋 Full AI Briefing  |  مکمل بریفنگ"):
                st.markdown(result.get("briefing", "No briefing generated.  |  کوئی بریفنگ تیار نہیں ہوئی۔"))
    if st.session_state.photo_report:
        if st.button("🔄 Clear Report  |  رپورٹ ہٹائیں", key="clear_report"):
            st.session_state.photo_report = None
            st.rerun()


# =========================================================
# ANALYSIS PIPELINE
# =========================================================

def get_severity_class(severity: str) -> str:
    s = severity.upper()
    if "CRITICAL" in s: return "sev-critical"
    if "HIGH" in s:     return "sev-high"
    if "MEDIUM" in s:   return "sev-medium"
    if "LOW" in s:      return "sev-low"
    return "sev-medium"

def get_auth_class(auth: str) -> str:
    a = auth.upper()
    if "FAKE" in a:      return "sev-fake"
    if "SUSPICIOUS" in a: return "sev-suspect"
    return "sev-real"

def get_briefing_sections(text: str) -> list:
    """Parse the AI briefing into labeled sections."""
    section_map = [
        ("SITUATION SUMMARY",            "🚨", "si-red"),
        ("THREAT ASSESSMENT",            "⚠️",  "si-amber"),
        ("EMERGENCY RESPONSE PLAN",      "🚑", "si-red"),
        ("RESOURCE ALLOCATION",          "📦", "si-blue"),
        ("PUBLIC SAFETY RECOMMENDATIONS","📢", "si-amber"),
        ("GOVERNMENT ACTION PLAN",       "🏛️", "si-blue"),
        ("RESCUE PRIORITIES",            "🚁", "si-red"),
        ("RECOVERY STRATEGY",            "🔄", "si-green"),
    ]
    sections = []
    current_key = None
    current_body = []

    for line in text.split("\n"):
        clean = line.strip().upper()
        clean_alpha = "".join(c for c in clean if c.isalpha() or c == " ").strip()
        matched = False
        for key, icon, cls in section_map:
            if key in clean_alpha:
                if current_key:
                    sections.append((current_key[0], current_key[1], current_key[2], "\n".join(current_body).strip()))
                current_key = (key.title(), icon, cls)
                current_body = []
                matched = True
                break
        if not matched and current_key:
            current_body.append(line)

    if current_key:
        sections.append((current_key[0], current_key[1], current_key[2], "\n".join(current_body).strip()))

    return sections


if analyze_btn:
    if not report.strip():
        st.warning("⚠️ Please enter a disaster report before running the analysis.")
    else:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Pipeline progress display ──
        step_placeholder = st.empty()

        steps = [
            ("Classifying disaster", "done"),
            ("Extracting location", "active"),
            ("Predicting severity", ""),
            ("Fake news detection", ""),
            ("AI Commander briefing", ""),
        ]

        def render_steps(completed: int):
            html = '<div class="pipeline-steps">'
            for i, (label, _) in enumerate(steps):
                if i < completed:
                    cls = "done"
                    dot = "✓"
                elif i == completed:
                    cls = "active"
                    dot = "●"
                else:
                    cls = ""
                    dot = "○"
                html += f'<div class="pipeline-step {cls}"><span class="step-dot">{dot}</span>{label}</div>'
                if i < len(steps) - 1:
                    html += '<span class="step-arrow">→</span>'
            html += "</div>"
            return html

        # Step 1 — Classify
        step_placeholder.markdown(render_steps(0), unsafe_allow_html=True)
        with st.spinner("Classifying disaster type..."):
            disaster = classify_disaster(report)
        time.sleep(0.3)

        # Step 2 — Location
        step_placeholder.markdown(render_steps(1), unsafe_allow_html=True)
        with st.spinner("Extracting location..."):
            location = extract_location(report)
        time.sleep(0.3)

        # Step 3 — Severity
        step_placeholder.markdown(render_steps(2), unsafe_allow_html=True)
        with st.spinner("Predicting severity..."):
            severity = predict_severity(report)
        time.sleep(0.3)

        # Step 4 — Fake news
        step_placeholder.markdown(render_steps(3), unsafe_allow_html=True)
        with st.spinner("Detecting fake news..."):
            authenticity = detect_fake_news(report)
        time.sleep(0.3)

        step_placeholder.markdown(render_steps(4), unsafe_allow_html=True)

        # Store results
        st.session_state.analyzed = True
        st.session_state.results = {
            "disaster": disaster,
            "location": location,
            "severity": severity,
            "authenticity": authenticity,
            "report": report,
        }
        st.session_state.total_reports += 1


# =========================================================
# RESULTS — shown if analysis ran
# =========================================================

if st.session_state.analyzed and st.session_state.results:

    r = st.session_state.results
    disaster     = r["disaster"]
    location     = r["location"]
    severity     = r["severity"]
    authenticity = r["authenticity"]
    report_text  = r["report"]

    is_fake = "fake" in authenticity.lower() or "suspicious" in authenticity.lower()

    # ── Result KPIs ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <div class="section-icon si-green">✅</div>
        <div class="section-title">Analysis results</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Disaster type", disaster)
    with c2:
        st.metric("Location", location)
    with c3:
        st.metric("Severity", severity)
    with c4:
        st.metric("Authenticity", authenticity)

    # ── Fake warning ──
    if is_fake:
        st.markdown("""
        <div class="fake-banner">
            <div class="fake-banner-icon">🚫</div>
            <div>
                <strong>Misinformation alert detected.</strong>
                This report has been flagged as suspicious or fake by the CrisisLens AI detection model.
                Do not activate full emergency deployment. Initiate cross-verification with NDMA, PDMA,
                and local authorities before taking action.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── MAP ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <div class="section-icon si-blue">📍</div>
        <div class="section-title">Disaster location map</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
    try:
        disaster_map = map_manager.create_map(location, disaster, severity)
        st_folium(disaster_map, width=None, height=440, returned_objects=[])
    except Exception as e:
        st.error(f"Map error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── ALERTS ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <div class="section-icon si-red">🚨</div>
        <div class="section-title">Generated alerts</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Generating targeted alerts..."):
        alerts = alert_engine.generate_alerts(disaster, location, severity, authenticity)

    citizen_tab, ngo_tab, gov_tab = st.tabs(["👥 Citizens", "🤝 NGOs", "🏛️ Government"])

    with citizen_tab:
        st.markdown(f'<div class="alert-box">{alerts.get("citizen", "No alert generated.")}</div>', unsafe_allow_html=True)

    with ngo_tab:
        st.markdown(f'<div class="alert-box">{alerts.get("ngo", "No alert generated.")}</div>', unsafe_allow_html=True)

    with gov_tab:
        st.markdown(f'<div class="alert-box">{alerts.get("government", "No alert generated.")}</div>', unsafe_allow_html=True)

    # ── AI COMMANDER BRIEFING ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <div class="section-icon si-red">🤖</div>
        <div class="section-title">CrisisLens AI Commander — Emergency briefing</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("CrisisLens AI Commander generating emergency briefing..."):
        briefing = generate_relief_plan(
            disaster_type=disaster,
            location=location,
            severity=severity,
            authenticity=authenticity,
            original_report=report_text
        )

    sections = get_briefing_sections(briefing)

    if sections:
        st.markdown('<div class="briefing-container">', unsafe_allow_html=True)
        for title, icon, cls, body in sections:
            if body.strip():
                st.markdown(f"""
                <div class="briefing-section">
                    <div class="briefing-section-head">
                        <div class="briefing-section-icon {cls}">{icon}</div>
                        <div class="briefing-section-title">{title}</div>
                    </div>
                    <div class="briefing-section-body">{body}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="briefing-container"><div class="briefing-section-body">{briefing}</div></div>', unsafe_allow_html=True)

    # ── AI Assistant Chat (synced with dedicated AI Assistant page) ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <div class="section-icon si-blue">💬</div>
        <div class="section-title">CrisisLens AI Assistant</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Ask follow-up questions below, or open the **AI Assistant** page for the full experience.")

    # Sync incident context to the shared state so the AI Assistant page picks it up
    st.session_state.incident_context = {
        "disaster":     r["disaster"],
        "location":     r["location"],
        "severity":     r["severity"],
        "authenticity": r["authenticity"],
        "report":       report_text,
    }

    for message in st.session_state.assistant_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask the assistant about this incident...", key="assistant_chat_postanalysis"):
        st.session_state.assistant_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    context = st.session_state.incident_context
                    reply = chat_with_assistant(
                        user_message=prompt,
                        chat_history=st.session_state.assistant_messages,
                        context=context,
                    )
                except Exception as exc:
                    reply = f"I could not generate a response right now. Error: {exc}"
            st.markdown(reply)

        st.session_state.assistant_messages.append({"role": "assistant", "content": reply})

    # ── Download briefing ──
    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    st.download_button(
        label="⬇️  Download full briefing (.txt)",
        data=briefing,
        file_name=f"crisislens_briefing_{disaster.lower().replace(' ', '_')}_{location.split(',')[0].lower().replace(' ', '_')}.txt",
        mime="text/plain",
        use_container_width=True
    )