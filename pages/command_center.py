import streamlit as st

st.set_page_config(
    page_title="CrisisLens AI Command Center",
    layout="wide"
)

st.title("🚨 CrisisLens AI Command Center")

# ===================================
# REPORT INPUT
# ===================================

report = st.text_area(
    "📝 Report Text",
    height=150
)

# ===================================
# IMAGE INPUT
# ===================================

uploaded_file = st.file_uploader(
    "📷 Upload Disaster Image",
    type=["jpg", "jpeg", "png"]
)

captured_image = st.camera_input(
    "📸 Capture Disaster Image"
)

# ===================================
# LOCATION
# ===================================

location = st.text_input(
    "📍 Current Location"
)

# ===================================
# ACTION BUTTONS
# ===================================

col1, col2, col3 = st.columns(3)

with col1:

    analyze_btn = st.button(
        "🚨 Analyze Incident"
    )

with col2:

    ai_btn = st.button(
        "🤖 Ask AI Assistant"
    )

with col3:

    voice_btn = st.button(
        "🎤 Voice Report"
    )

# ===================================
# SECOND ROW
# ===================================

col4, col5, col6 = st.columns(3)

with col4:

    read_btn = st.button(
        "🔊 Read Response"
    )

with col5:

    email_btn = st.button(
        "📧 Send Alert"
    )

with col6:

    sms_btn = st.button(
        "📱 Send SMS"
    )

# ===================================
# RESULTS
# ===================================

if analyze_btn:

    st.success(
        "Incident Analysis Started"
    )

if ai_btn:

    st.success(
        "AI Assistant Activated"
    )

if voice_btn:

    st.success(
        "Voice Commander Started"
    )

if read_btn:

    st.success(
        "Reading Response"
    )

if email_btn:

    st.success(
        "Email Alert Sent"
    )

if sms_btn:

    st.success(
        "SMS Alert Sent"
    )