import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Live Emergency Map",
    page_icon="📍",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #0a0e1a;
}

.map-container {
    background: #0d1120;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    overflow: hidden;
    margin-top: 0.5rem;
}

/* ── Live Badge ── */
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

.incident-card {
    background: #0d1120;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}

.incident-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #e8f0fe;
}

.incident-subtitle {
    font-size: 0.7rem;
    color: #5577aa;
}

.sev-tag {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 12px;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
}
.sev-high { background: #2a1200; color: #fb923c; border: 1px solid #7a3500; }
.sev-critical { background: #2a0505; color: #f87171; border: 1px solid #7a1515; }
.sev-medium { background: #2a1f00; color: #fbbf24; border: 1px solid #7a5a00; }
.sev-low { background: #0a2010; color: #4ade80; border: 1px solid #1a5a2a; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;">
    <div style="display:flex;align-items:center;gap:0.6rem;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg, #1a6fc4, #0f4a8a);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;">📍</div>
        <div>
            <div style="font-size:1.1rem;font-weight:700;color:#e8f0fe;letter-spacing:-0.02em;">Live Emergency Map Tracker</div>
            <div style="font-size:0.7rem;color:#5577aa;margin-top:1px;">Geographical visualization of ongoing disaster events across Pakistan.</div>
        </div>
    </div>
    <div class="live-badge">
        <div class="live-dot"></div>
        LIVE TRACKING ACTIVE
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# MOCK EMERGENCY INCIDENTS
# =========================================================
INCIDENTS = [
    {"location": "Swat", "coords": (34.8065, 72.3602), "type": "Flood", "severity": "High", "desc": "River Swat overflowed. 500+ families displaced. Road blocks near Mingora."},
    {"location": "Quetta", "coords": (30.1798, 66.9750), "type": "Earthquake", "severity": "Critical", "desc": "Magnitude 5.1 earthquake. Multiple buildings collapsed in old city area."},
    {"location": "Karachi", "coords": (24.8607, 67.0011), "type": "Flood", "severity": "Medium", "desc": "Urban flooding from monsoon storms. Severe waterlogging on major roads."},
    {"location": "Murree", "coords": (33.9070, 73.3943), "type": "Landslide", "severity": "Medium", "desc": "Rainfall triggered landslide. Road block on Expressway, tourists stranded."},
    {"location": "Sukkur", "coords": (27.7244, 68.8228), "type": "Flood", "severity": "High", "desc": "River Indus rising warning. Low-lying riverine villages evacuated."},
    {"location": "Peshawar", "coords": (34.0151, 71.5249), "type": "Flood", "severity": "Low", "desc": "Canal overflow in suburban villages. Minor crop damage."}
]

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.markdown('<p style="font-size:0.85rem;font-weight:700;color:#e8f0fe;margin-bottom:0.75rem;">MAP FILTERS</p>', unsafe_allow_html=True)

disaster_filter = st.sidebar.multiselect(
    "DISASTER TYPE",
    options=["Flood", "Earthquake", "Landslide"],
    default=["Flood", "Earthquake", "Landslide"]
)

severity_filter = st.sidebar.multiselect(
    "SEVERITY LEVEL",
    options=["Low", "Medium", "High", "Critical"],
    default=["Low", "Medium", "High", "Critical"]
)

# Filter data
filtered_incidents = [
    inc for inc in INCIDENTS 
    if inc["type"] in disaster_filter and inc["severity"] in severity_filter
]

# =========================================================
# MAIN LAYOUT
# =========================================================
col_map, col_details = st.columns([3, 1])

with col_map:
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    
    # Initialize Folium Map centered on Pakistan
    m = folium.Map(location=[30.3753, 69.3451], zoom_start=6, tiles="OpenStreetMap")
    
    for inc in filtered_incidents:
        # Determine Color
        s = inc["severity"].lower()
        if "critical" in s: color = "darkred"
        elif "high" in s: color = "red"
        elif "medium" in s: color = "orange"
        else: color = "green"
        
        folium.Marker(
            location=inc["coords"],
            popup=f"<b>{inc['type']}</b><br>{inc['location']}<br>Severity: {inc['severity']}",
            tooltip=f"{inc['type']} - {inc['location']}",
            icon=folium.Icon(color=color, icon="exclamation-sign")
        ).add_to(m)
        
        folium.Circle(
            location=inc["coords"],
            radius=15000,
            color=color,
            fill=True,
            fill_opacity=0.15
        ).add_to(m)
        
    st_folium(m, width=None, height=520, returned_objects=[])
    st.markdown('</div>', unsafe_allow_html=True)

with col_details:
    st.markdown('<p style="font-size:0.75rem;color:#5577aa;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;margin-bottom:0.75rem;">Active Incident Log</p>', unsafe_allow_html=True)
    
    if not filtered_incidents:
        st.caption("No active incidents match current filters.")
    else:
        for inc in filtered_incidents:
            s_lower = inc["severity"].lower()
            if "critical" in s_lower: tag_cls = "sev-critical"
            elif "high" in s_lower: tag_cls = "sev-high"
            elif "medium" in s_lower: tag_cls = "sev-medium"
            else: tag_cls = "sev-low"
            
            st.markdown(f"""
            <div class="incident-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
                    <span class="incident-title">{inc['type']} — {inc['location']}</span>
                    <span class="sev-tag {tag_cls}">{inc['severity']}</span>
                </div>
                <div class="incident-subtitle">{inc['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
