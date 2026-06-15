import streamlit as st
import requests
import time
from datetime import datetime, timezone

from utils.disaster_classifier import classify_disaster
from utils.location_extractor    import extract_location
from utils.severity_prediction   import predict_severity
from utils.fake_news_detection   import detect_fake_news

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Live Model & Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif !important; }
.stApp { background:#0a0e1a; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:1.5rem 2rem !important; max-width:100% !important; }

/* ── Header bar ── */
.page-header {
    display:flex; align-items:center; justify-content:space-between;
    background:#0d1120; border:1px solid #1e2d4a; border-radius:12px;
    padding:.75rem 1.25rem; margin-bottom:1.5rem;
}
.header-left { display:flex; align-items:center; gap:.75rem; }
.header-icon {
    width:36px; height:36px;
    background:linear-gradient(135deg,#1a6fc4,#0f4a8a);
    border-radius:8px; display:flex; align-items:center;
    justify-content:center; font-size:1.1rem;
}
.header-title { font-size:1.05rem; font-weight:700; color:#e8f0fe; }
.header-sub   { font-size:.68rem; color:#5577aa; margin-top:2px; }

/* ── Live badge ── */
.live-badge {
    display:flex; align-items:center; gap:.4rem;
    background:#0a2010; border:1px solid #1a4a2a;
    border-radius:20px; padding:.3rem .75rem;
    font-size:.68rem; color:#4dbb7a; font-weight:700; letter-spacing:.05em;
}
.live-dot {
    width:7px; height:7px; border-radius:50%;
    background:#4dbb7a; animation:livepulse 1.4s ease-in-out infinite;
}
@keyframes livepulse { 0%,100%{opacity:1} 50%{opacity:.2} }

.offline-badge {
    display:flex; align-items:center; gap:.4rem;
    background:#1a1a2a; border:1px solid #2a2a4a;
    border-radius:20px; padding:.3rem .75rem;
    font-size:.68rem; color:#5577aa; font-weight:700; letter-spacing:.05em;
}

/* ── Section header ── */
.sec-hdr {
    display:flex; align-items:center; gap:.5rem; margin:1.25rem 0 .65rem 0;
}
.sec-icon {
    width:28px; height:28px; border-radius:6px;
    display:flex; align-items:center; justify-content:center; font-size:.85rem;
}
.si-blue  { background:#0d2545; color:#4d9ef5; }
.si-red   { background:#2a0a0a; color:#f56565; }
.si-green { background:#0a2a15; color:#4dbb7a; }
.si-amber { background:#2a1a00; color:#f5a623; }

.sec-title { font-size:.9rem; font-weight:600; color:#c8d8f0; }

/* ── KPI strip ── */
.kpi-row {
    display:grid; grid-template-columns:repeat(4,1fr); gap:.75rem;
    margin-bottom:1.25rem;
}
.kpi-card {
    background:#0d1120; border:1px solid #1e2d4a; border-radius:10px;
    padding:.85rem 1rem; position:relative; overflow:hidden;
}
.kpi-card::before {
    content:''; position:absolute; top:0;left:0;right:0; height:3px; border-radius:10px 10px 0 0;
}
.kpi-card.blue::before  { background:#1a6fc4; }
.kpi-card.green::before { background:#27a85a; }
.kpi-card.amber::before { background:#d97706; }
.kpi-card.red::before   { background:#dc2626; }
.kpi-label { font-size:.62rem; color:#5577aa; text-transform:uppercase; letter-spacing:.08em; font-weight:600; margin-bottom:.35rem; }
.kpi-val   { font-size:1.3rem; font-weight:700; color:#e8f0fe; line-height:1.1; }
.kpi-sub   { font-size:.65rem; color:#3d5a8a; margin-top:.25rem; }

/* ── Signal log card ── */
.signal-card {
    background:#0d1120; border:1px solid #1e2d4a; border-radius:10px;
    padding:.9rem 1.1rem; margin-bottom:.65rem; transition:border-color .2s;
}
.signal-card:hover { border-color:#2a4a7a; }
.signal-ts   { font-size:.65rem; color:#5577aa; font-weight:500; margin-bottom:.4rem; }
.signal-text { font-size:.8rem; color:#c8d8f0; line-height:1.55; margin-bottom:.6rem; }
.badges { display:flex; gap:.4rem; flex-wrap:wrap; }

/* ── Badges ── */
.badge {
    display:inline-flex; align-items:center; gap:.25rem;
    padding:.18rem .55rem; border-radius:20px;
    font-size:.62rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase;
}
.b-type     { background:#0d2545; color:#4d9ef5; border:1px solid #1a4a9a; }
.b-loc      { background:#1a1a2a; color:#c8d8f0; border:1px solid #2a3a5a; }
.b-critical { background:#2a0505; color:#f87171; border:1px solid #7a1515; }
.b-high     { background:#2a1200; color:#fb923c; border:1px solid #7a3500; }
.b-medium   { background:#2a1f00; color:#fbbf24; border:1px solid #7a5a00; }
.b-low      { background:#0a2010; color:#4ade80; border:1px solid #1a5a2a; }
.b-fake     { background:#1a1a2a; color:#a78bfa; border:1px solid #4a3a8a; }
.b-real     { background:#0a2010; color:#4dbb7a; border:1px solid #1a4a2a; }
.b-suspect  { background:#2a1a00; color:#f5a623; border:1px solid #7a4a00; }
.b-source   { background:#0d1825; color:#60a5fa; border:1px solid #1a3a5a; }

/* ── Divider ── */
.divider { height:1px; background:#1e2d4a; margin:1.5rem 0; }

/* Streamlit overrides */
.stTextArea textarea {
    background:#060911 !important; border:1px solid #1e2d4a !important;
    border-radius:8px !important; color:#c8d8f0 !important; font-size:.85rem !important;
}
.stButton > button {
    background:linear-gradient(135deg,#1a6fc4,#0f4a8a) !important;
    color:#e8f0fe !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important;
}
.stSelectbox > div > div {
    background:#060911 !important; border-color:#1e2d4a !important;
    color:#c8d8f0 !important; border-radius:8px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "live_signals"  not in st.session_state: st.session_state.live_signals  = []
if "live_active"   not in st.session_state: st.session_state.live_active   = False
if "last_refresh"  not in st.session_state: st.session_state.last_refresh  = 0
if "eq_seen"       not in st.session_state: st.session_state.eq_seen       = set()

# =========================================================
# USGS LIVE EARTHQUAKE FETCHER
# =========================================================
USGS_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
    "?format=geojson"
    "&limit=20"
    "&minlatitude=20&maxlatitude=38"
    "&minlongitude=60&maxlongitude=80"
    "&minmagnitude=2.5"
    "&orderby=time"
)

GDACS_URL = (
    "https://www.gdacs.org/gdacsapi/api/events/geteventlist/MAP"
    "?eventtypes=FL,EQ,TC&country=PAK&limit=10"
)

def _sev_badge_cls(s: str) -> str:
    s = s.lower()
    if "critical" in s: return "b-critical"
    if "high"     in s: return "b-high"
    if "medium"   in s: return "b-medium"
    return "b-low"

def _auth_badge_cls(a: str) -> str:
    a = a.lower()
    if "fake" in a:       return "b-fake"
    if "suspicious" in a: return "b-suspect"
    return "b-real"

def fetch_usgs_signals():
    """Pull live earthquake events from USGS for Pakistan/Afghanistan region."""
    try:
        r = requests.get(USGS_URL, timeout=8)
        r.raise_for_status()
        features = r.json().get("features", [])
        signals = []
        for f in features:
            eid = f.get("id", "")
            if eid in st.session_state.eq_seen:
                continue
            p = f["properties"]
            mag   = p.get("mag", 0) or 0
            place = p.get("place", "Unknown location")
            ts_ms = p.get("time", 0)
            ts    = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            sig   = p.get("sig", 0) or 0
            alert = p.get("alert") or ""

            text = f"Magnitude {mag:.1f} earthquake — {place}."
            if sig > 600:
                text += " Significant seismic event. Casualties possible."
            elif sig > 300:
                text += " Moderate impact expected. Structural checks advised."

            disaster     = classify_disaster(text)
            location     = extract_location(text) if extract_location(text) != "Pakistan" else (
                               extract_location(place) if extract_location(place) != "Pakistan" else place.split(",")[-1].strip()
                           )
            severity     = predict_severity(text)
            authenticity = "Likely Real"  # USGS is authoritative

            signals.append({
                "id":          eid,
                "ts":          ts,
                "text":        text,
                "disaster":    disaster,
                "location":    location,
                "severity":    severity,
                "authenticity":authenticity,
                "source":      "USGS Live",
                "mag":         mag,
                "sig":         sig,
            })
            st.session_state.eq_seen.add(eid)
        return signals
    except Exception as e:
        return []


# ── Pakistan-specific simulated disaster feeds (flood / landslide / other)
_PAK_FEEDS = [
    {
        "text": "Pakistan Meteorological Dept issues flood watch for Indus River catchment area in Sindh. Water level at Sukkur Barrage at warning stage.",
        "source": "PMD Alert"
    },
    {
        "text": "NDMA confirms flash flooding in Chitral district, KPK after 120mm rainfall in 24 hours. Bridge washed away near Garam Chashma, 200 families displaced.",
        "source": "NDMA Report"
    },
    {
        "text": "River Swat overflowing banks near Mingora. PDMA has evacuated 500 households from low-lying areas. Relief camps established.",
        "source": "PDMA KPK"
    },
    {
        "text": "Landslide blocks Karakoram Highway near Gilgit at Km 120. Heavy machinery deployed. Traffic suspended on both sides.",
        "source": "NHA Alert"
    },
    {
        "text": "Urban flooding reported in Karachi after 85mm rainfall in 3 hours. Lyari and Malir rivers overflowing. Rescue 1122 teams deployed.",
        "source": "Rescue 1122"
    },
    {
        "text": "Glacial lake outburst flood (GLOF) detected in Hunza Valley. Attabad Lake water levels rising. DC Hunza issues evacuation advisory.",
        "source": "GLOF Warning"
    },
    {
        "text": "Magnitude 5.2 earthquake felt strongly in Quetta, Balochistan. Multiple buildings cracked. No casualties reported so far. Aftershocks expected.",
        "source": "PMD Seismic"
    },
    {
        "text": "Heavy rains lash Dera Ismail Khan, KPK. 15 houses collapsed, 3 people injured. District administration releases emergency relief funds.",
        "source": "DC DI Khan"
    },
]
_feed_idx = 0

def fetch_pak_simulation():
    """Return one Pakistan-specific simulated signal."""
    global _feed_idx
    feed = _PAK_FEEDS[_feed_idx % len(_PAK_FEEDS)]
    _feed_idx += 1
    text        = feed["text"]
    disaster    = classify_disaster(text)
    location    = extract_location(text)
    severity    = predict_severity(text)
    authenticity = detect_fake_news(text)
    return {
        "id":           f"pak-{_feed_idx}",
        "ts":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text":         text,
        "disaster":     disaster,
        "location":     location,
        "severity":     severity,
        "authenticity": authenticity,
        "source":       feed["source"],
        "mag":          None,
        "sig":          None,
    }

# =========================================================
# PAGE HEADER
# =========================================================
st.markdown(f"""
<div class="page-header">
    <div class="header-left">
        <div class="header-icon">🔮</div>
        <div>
            <div class="header-title">Live Model & Prediction Center</div>
            <div class="header-sub">Real-time USGS earthquake API + Pakistan disaster feed — fully classified by CrisisLens AI</div>
        </div>
    </div>
    <div class="{'live-badge' if st.session_state.live_active else 'offline-badge'}">
        {'<div class="live-dot"></div> LIVE STREAMING ACTIVE' if st.session_state.live_active else '⚪ MONITORING OFF'}
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LIVE TOGGLE + CONTROLS
# =========================================================
st.markdown('<div class="sec-hdr"><div class="sec-icon si-blue">📡</div><div class="sec-title">Streaming Controller</div></div>', unsafe_allow_html=True)

col_tog, col_int, col_src, col_fetch, col_clr = st.columns([2, 2, 2, 2, 2])

with col_tog:
    live_on = st.toggle("ACTIVATE LIVE MONITORING", value=st.session_state.live_active, key="_tog")
    st.session_state.live_active = live_on

with col_int:
    refresh_interval = st.selectbox(
        "Auto-Refresh Interval",
        ["Manual only", "30 seconds", "60 seconds", "2 minutes"],
        index=0,
        label_visibility="collapsed"
    )

with col_src:
    source_filter = st.selectbox(
        "Signal Source",
        ["All Sources", "USGS Earthquakes", "Pakistan Disaster Feed"],
        index=0,
        label_visibility="collapsed"
    )

with col_fetch:
    fetch_now = st.button("⚡ Fetch Now", use_container_width=True, disabled=not live_on)

with col_clr:
    clear_btn = st.button("🧹 Clear Log", use_container_width=True)

if clear_btn:
    st.session_state.live_signals = []
    st.session_state.eq_seen = set()
    st.rerun()

# ── Auto-refresh logic ──
interval_map = {
    "Manual only": 0,
    "30 seconds": 30,
    "60 seconds": 60,
    "2 minutes": 120,
}
interval_sec = interval_map[refresh_interval]
now = time.time()
auto_fetch = (
    live_on and
    interval_sec > 0 and
    (now - st.session_state.last_refresh) >= interval_sec
)

# =========================================================
# FETCH & PROCESS
# =========================================================
if live_on and (fetch_now or auto_fetch):
    with st.spinner("Fetching live signals and running AI pipeline..."):
        new_signals = []
        if source_filter in ("All Sources", "USGS Earthquakes"):
            new_signals += fetch_usgs_signals()
        if source_filter in ("All Sources", "Pakistan Disaster Feed"):
            new_signals.append(fetch_pak_simulation())

        # Prepend newest signals
        st.session_state.live_signals = new_signals + st.session_state.live_signals
        # Keep last 50 only
        st.session_state.live_signals = st.session_state.live_signals[:50]
        st.session_state.last_refresh = time.time()

    if new_signals:
        st.success(f"✅ {len(new_signals)} new signal(s) ingested and classified.")
    else:
        st.info("No new signals since last fetch.")
    st.rerun()

# =========================================================
# LIVE FEED DISPLAY
# =========================================================
if live_on or st.session_state.live_signals:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    signals = st.session_state.live_signals

    # ── KPIs ──
    total  = len(signals)
    crit   = sum(1 for s in signals if s["severity"].lower() in ("critical","high"))
    fake   = sum(1 for s in signals if s["authenticity"].lower() in ("fake","suspicious"))
    real   = total - fake
    eqs    = sum(1 for s in signals if "earthquake" in s["disaster"].lower())

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card blue">
            <div class="kpi-label">Total Signals</div>
            <div class="kpi-val">{total}</div>
            <div class="kpi-sub">since monitoring started</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">High/Critical Alerts</div>
            <div class="kpi-val">{crit}</div>
            <div class="kpi-sub">require immediate action</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Verified Real</div>
            <div class="kpi-val">{real}</div>
            <div class="kpi-sub">{f'{real/total*100:.0f}% authenticity' if total else '—'}</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">Misinformation</div>
            <div class="kpi-val">{fake}</div>
            <div class="kpi-sub">flagged &amp; suppressed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Signal type sub-header ──
    st.markdown('<div class="sec-hdr"><div class="sec-icon si-red">📶</div><div class="sec-title">Live Classified Signals Feed</div></div>', unsafe_allow_html=True)

    if not signals:
        st.info("Pipeline active. Click **⚡ Fetch Now** to pull live data.")
    else:
        for sig in signals:
            sc   = _sev_badge_cls(sig["severity"])
            ac   = _auth_badge_cls(sig["authenticity"])
            mag_badge = (
                f'<span class="badge b-source">📊 M {sig["mag"]:.1f}</span>'
                if sig.get("mag") else ""
            )
            st.markdown(f"""
            <div class="signal-card">
                <div class="signal-ts">⏱️ {sig['ts']} &nbsp;|&nbsp; 📡 {sig['source']}</div>
                <div class="signal-text">{sig['text']}</div>
                <div class="badges">
                    <span class="badge b-type">📁 {sig['disaster']}</span>
                    <span class="badge b-loc">📍 {sig['location']}</span>
                    <span class="badge {sc}">⚠️ {sig['severity']}</span>
                    <span class="badge {ac}">🛡️ {sig['authenticity']}</span>
                    {mag_badge}
                    <span class="badge b-source">🔗 {sig['source']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Auto-refresh countdown hint ──
if live_on and interval_sec > 0:
    elapsed   = int(time.time() - st.session_state.last_refresh)
    remaining = max(0, interval_sec - elapsed)
    st.caption(f"⏱️ Next auto-fetch in **{remaining}s**  •  Last refresh: "
               f"{datetime.fromtimestamp(st.session_state.last_refresh).strftime('%H:%M:%S') if st.session_state.last_refresh else 'never'}")
    # trigger a rerun after the interval
    time.sleep(min(remaining + 1, 10))
    st.rerun()

# =========================================================
# MANUAL PREDICTION SECTION
# =========================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-hdr"><div class="sec-icon si-blue">📋</div><div class="sec-title">Manual Report Predictor</div></div>', unsafe_allow_html=True)

col_in, col_out = st.columns([3, 2])

with col_in:
    manual_text = st.text_area(
        "DISASTER REPORT TEXT",
        placeholder="Paste any disaster report text here. The AI will classify it, extract location, predict severity, and check authenticity...",
        height=140,
        key="manual_pred_input"
    )
    run_btn = st.button("🔮 Analyze Report", use_container_width=True)

with col_out:
    if run_btn and manual_text.strip():
        with st.spinner("Running AI pipeline..."):
            d  = classify_disaster(manual_text)
            lo = extract_location(manual_text)
            sv = predict_severity(manual_text)
            au = detect_fake_news(manual_text)

        sc = _sev_badge_cls(sv)
        ac = _auth_badge_cls(au)

        st.markdown(f"""
        <div style="background:#0d1120;border:1px solid #1e2d4a;border-radius:10px;padding:1.1rem;">
            <p style="font-size:.68rem;color:#5577aa;text-transform:uppercase;letter-spacing:.07em;font-weight:600;margin-bottom:.9rem;">AI Model Outputs</p>
            <div style="margin-bottom:.75rem;">
                <div style="font-size:.65rem;color:#5577aa;margin-bottom:.2rem;">DISASTER TYPE</div>
                <strong style="color:#e8f0fe;font-size:.95rem;">📁 {d}</strong>
            </div>
            <div style="margin-bottom:.75rem;">
                <div style="font-size:.65rem;color:#5577aa;margin-bottom:.2rem;">DETECTED LOCATION</div>
                <strong style="color:#e8f0fe;font-size:.95rem;">📍 {lo}</strong>
            </div>
            <div style="margin-bottom:.75rem;">
                <div style="font-size:.65rem;color:#5577aa;margin-bottom:.2rem;">PRIORITY / SEVERITY</div>
                <span class="badge {sc}">⚠️ {sv}</span>
            </div>
            <div>
                <div style="font-size:.65rem;color:#5577aa;margin-bottom:.2rem;">AUTHENTICITY</div>
                <span class="badge {ac}">🛡️ {au}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif run_btn:
        st.warning("Please enter some report text first.")
    else:
        st.markdown("""
        <div style="background:#0d1120;border:1px dashed #1e2d4a;border-radius:10px;
                    padding:2rem 1rem;text-align:center;">
            <div style="font-size:1.5rem;margin-bottom:.5rem;">🔮</div>
            <div style="color:#5577aa;font-size:.78rem;">Enter a report and click<br><strong style="color:#4d9ef5;">Analyze Report</strong></div>
        </div>
        """, unsafe_allow_html=True)
