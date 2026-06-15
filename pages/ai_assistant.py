import streamlit as st
from dotenv import load_dotenv
from utils.gemini_relief_agent import (
    chat_with_assistant,
    generate_relief_plan,
    CHAT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)

load_dotenv()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CrisisLens AI Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
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
.block-container { padding:1.5rem 2rem 1rem 2rem !important; max-width:100% !important; }

/* ── Page header ── */
.page-hdr {
    display:flex; align-items:center; justify-content:space-between;
    background:#0d1120; border:1px solid #1e2d4a; border-radius:12px;
    padding:.75rem 1.25rem; margin-bottom:1.25rem;
}
.phdr-left  { display:flex; align-items:center; gap:.75rem; }
.phdr-icon  {
    width:38px; height:38px;
    background:linear-gradient(135deg,#1a6fc4,#0f4a8a);
    border-radius:9px; display:flex; align-items:center;
    justify-content:center; font-size:1.2rem;
}
.phdr-title { font-size:1.05rem; font-weight:700; color:#e8f0fe; letter-spacing:-.02em; }
.phdr-sub   { font-size:.68rem; color:#5577aa; margin-top:2px; }

/* ── Powered-by badge ── */
.powered-badge {
    display:flex; align-items:center; gap:.4rem;
    background:#0a1828; border:1px solid #1a3a5a;
    border-radius:20px; padding:.3rem .85rem;
    font-size:.68rem; color:#4d9ef5; font-weight:700; letter-spacing:.04em;
}

/* ── Two-column layout ── */
.chat-col {
    background:#0d1120; border:1px solid #1e2d4a;
    border-radius:12px; padding:1rem 1.25rem;
    min-height:520px;
}
.ctx-col {
    background:#0d1120; border:1px solid #1e2d4a;
    border-radius:12px; padding:1rem 1.25rem;
}

/* ── Context panel ── */
.ctx-title {
    font-size:.72rem; color:#5577aa; font-weight:700;
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:.75rem;
}
.ctx-row {
    display:flex; align-items:flex-start; gap:.5rem;
    background:#060911; border:1px solid #1e2d4a;
    border-radius:8px; padding:.6rem .75rem; margin-bottom:.5rem;
}
.ctx-label {
    font-size:.62rem; color:#5577aa; text-transform:uppercase;
    letter-spacing:.06em; font-weight:600; width:70px; flex-shrink:0; padding-top:2px;
}
.ctx-val  { font-size:.8rem; color:#c8d8f0; font-weight:600; line-height:1.4; }

/* ── Quick prompts ── */
.qp-title {
    font-size:.72rem; color:#5577aa; font-weight:700;
    text-transform:uppercase; letter-spacing:.08em; margin-top:1rem; margin-bottom:.5rem;
}

/* ── Session stats ── */
.stat-row {
    display:grid; grid-template-columns:1fr 1fr 1fr; gap:.5rem; margin-top:.75rem;
}
.stat-card {
    background:#060911; border:1px solid #1e2d4a;
    border-radius:8px; padding:.5rem .75rem; text-align:center;
}
.stat-val   { font-size:1rem; font-weight:700; color:#e8f0fe; }
.stat-label { font-size:.6rem; color:#5577aa; text-transform:uppercase; letter-spacing:.06em; margin-top:2px; }

/* ── Streamlit overrides ── */
.stTextArea textarea {
    background:#060911 !important; border:1px solid #1e2d4a !important;
    border-radius:8px !important; color:#c8d8f0 !important; font-size:.85rem !important;
}
.stButton > button {
    background:linear-gradient(135deg,#1a6fc4,#0f4a8a) !important;
    color:#e8f0fe !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important; font-size:.82rem !important;
}
.stButton > button:hover { opacity:.87 !important; }

/* secondary button style */
.stButton.secondary > button {
    background:#0d1120 !important;
    border:1px solid #1e2d4a !important; color:#8899bb !important;
}

div[data-testid="stChatMessage"] {
    background:transparent !important;
}

.stChatInputContainer {
    background:#0a0e1a !important;
    border-top:1px solid #1e2d4a !important;
    padding:.75rem 2rem !important;
}

/* Chat input box */
.stChatInputContainer textarea {
    background:#060911 !important; border:1px solid #1e2d4a !important;
    border-radius:10px !important; color:#c8d8f0 !important; font-size:.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Hey there! I'm the CrisisLens AI Assistant** — powered by the Groq LLM engine.\n\n"
                "I'm here to help your emergency response team with:\n"
                "* 🌊 **Flood safety & evacuation guidance**\n"
                "* 🏔️ **Earthquake preparedness & response protocols**\n"
                "* 🩹 **First aid basics & survivor care**\n"
                "* 📋 **Relief planning & resource allocation advice**\n"
                "* 📊 **Explaining CrisisLens AI reports in plain language**\n\n"
                "You can also load any **analyzed report** from the main dashboard and I'll "
                "give context-aware answers based on the actual incident data.\n\n"
                "What can I help you with today?"
            ),
        }
    ]
if "incident_context" not in st.session_state:
    st.session_state.incident_context = None
if "briefing_cache"   not in st.session_state:
    st.session_state.briefing_cache = None

# ── Carry over context from main dashboard if available ──
if st.session_state.get("results"):
    r = st.session_state.results
    st.session_state.incident_context = {
        "disaster":      r.get("disaster", "Unknown"),
        "location":      r.get("location", "Unknown"),
        "severity":      r.get("severity",  "Unknown"),
        "authenticity":  r.get("authenticity", "Unknown"),
        "report":        r.get("report", ""),
    }

# =========================================================
# PAGE HEADER
# =========================================================
st.markdown("""
<div class="page-hdr">
    <div class="phdr-left">
        <div class="phdr-icon">💬</div>
        <div>
            <div class="phdr-title">CrisisLens AI Assistant</div>
            <div class="phdr-sub">Conversational emergency response intelligence powered by Groq LLM</div>
        </div>
    </div>
    <div class="powered-badge">⚡ Groq LLM · llama-3.3-70b</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LAYOUT: Chat (left 70%) | Context Panel (right 30%)
# =========================================================
col_chat, col_ctx = st.columns([7, 3])

# ─────────────────────────────────────────────────────────
# RIGHT PANEL — Incident Context & Quick Prompts
# ─────────────────────────────────────────────────────────
with col_ctx:
    # ── Incident context ──────────────────────────────────
    ctx = st.session_state.incident_context

    st.markdown('<div class="ctx-title">📋 Active Incident Context</div>', unsafe_allow_html=True)

    if ctx:
        def _sev_color(s):
            s = s.lower()
            if "critical" in s: return "#f87171"
            if "high"     in s: return "#fb923c"
            if "medium"   in s: return "#fbbf24"
            return "#4ade80"

        def _auth_color(a):
            a = a.lower()
            if "fake"       in a: return "#a78bfa"
            if "suspicious" in a: return "#f5a623"
            return "#4dbb7a"

        st.markdown(f"""
        <div class="ctx-row">
            <div class="ctx-label">Type</div>
            <div class="ctx-val">📁 {ctx['disaster']}</div>
        </div>
        <div class="ctx-row">
            <div class="ctx-label">Location</div>
            <div class="ctx-val">📍 {ctx['location']}</div>
        </div>
        <div class="ctx-row">
            <div class="ctx-label">Severity</div>
            <div class="ctx-val" style="color:{_sev_color(ctx['severity'])};">
                ⚠️ {ctx['severity']}
            </div>
        </div>
        <div class="ctx-row">
            <div class="ctx-label">Auth.</div>
            <div class="ctx-val" style="color:{_auth_color(ctx['authenticity'])};">
                🛡️ {ctx['authenticity']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🧹 Clear incident context", use_container_width=True):
            st.session_state.incident_context = None
            st.session_state.briefing_cache   = None
            st.rerun()

        # ── Generate full briefing ──
        st.markdown('<div class="qp-title">🤖 AI Commander Briefing</div>', unsafe_allow_html=True)
        if st.button("📄 Generate Full Emergency Briefing", use_container_width=True):
            with st.spinner("CrisisLens AI Commander generating briefing..."):
                try:
                    briefing = generate_relief_plan(
                        disaster_type=ctx["disaster"],
                        location=ctx["location"],
                        severity=ctx["severity"],
                        authenticity=ctx["authenticity"],
                        original_report=ctx["report"],
                    )
                    st.session_state.briefing_cache = briefing
                    # Push briefing into chat as assistant message
                    st.session_state.assistant_messages.append({
                        "role": "assistant",
                        "content": f"📋 **Full Emergency Briefing for {ctx['disaster']} — {ctx['location']}**\n\n{briefing}"
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Briefing error: {e}")

        if st.session_state.briefing_cache:
            st.download_button(
                label="⬇️ Download Briefing (.txt)",
                data=st.session_state.briefing_cache,
                file_name=f"crisislens_briefing_{ctx['disaster'].lower().replace(' ','_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    else:
        st.markdown("""
        <div style="background:#060911;border:1px dashed #1e2d4a;border-radius:8px;
                    padding:1.25rem;text-align:center;color:#445577;font-size:.78rem;line-height:1.7;">
            No incident loaded.<br>
            Run the <strong style="color:#4d9ef5;">AI Analysis Pipeline</strong>
            on the main dashboard to load incident context here automatically.
        </div>
        """, unsafe_allow_html=True)

    # ── Quick prompt suggestions ──────────────────────────
    st.markdown('<div class="qp-title">⚡ Quick Prompts</div>', unsafe_allow_html=True)

    quick_prompts = [
        ("🌊 Flood evacuation steps", "What are the standard evacuation steps for a major flood event?"),
        ("🏔️ Earthquake safety", "What should people do immediately after a strong earthquake?"),
        ("🩹 Mass casualty protocol", "Explain the triage protocol for a mass casualty incident."),
        ("📦 Resource allocation", "How should relief resources be prioritized after a major disaster?"),
        ("🚁 Rescue prioritization", "How do rescue teams prioritize trapped victims in a collapsed building?"),
        ("📢 Public comms", "What key messages should authorities broadcast to affected populations?"),
    ]

    for label, qprompt in quick_prompts:
        if st.button(label, use_container_width=True):
            st.session_state.assistant_messages.append({"role": "user", "content": qprompt})
            with st.spinner("Thinking..."):
                try:
                    reply = chat_with_assistant(
                        user_message=qprompt,
                        chat_history=st.session_state.assistant_messages,
                        context=st.session_state.incident_context,
                    )
                except Exception as e:
                    reply = f"⚠️ Error: {e}"
            st.session_state.assistant_messages.append({"role": "assistant", "content": reply})
            st.rerun()

    # ── Session stats ──────────────────────────────────────
    msgs = st.session_state.assistant_messages
    user_count  = sum(1 for m in msgs if m["role"] == "user")
    asst_count  = sum(1 for m in msgs if m["role"] == "assistant")
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-val">{user_count}</div>
            <div class="stat-label">Questions</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{asst_count}</div>
            <div class="stat-label">Answers</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{'✅' if ctx else '—'}</div>
            <div class="stat-label">Context</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear entire conversation", use_container_width=True):
        st.session_state.assistant_messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Conversation cleared. I'm ready to help again!**\n\n"
                    "Ask me anything about disaster preparedness, emergency protocols, "
                    "or load an incident report from the main dashboard."
                ),
            }
        ]
        st.rerun()

# ─────────────────────────────────────────────────────────
# LEFT PANEL — Chat Messages + Input
# ─────────────────────────────────────────────────────────
with col_chat:
    # Render all chat messages
    for msg in st.session_state.assistant_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Spacer so the last message isn't hidden by the fixed input
    st.markdown('<div style="height:80px;"></div>', unsafe_allow_html=True)

    # ── Chat input — at root level of col_chat for footer pinning ──
    if user_input := st.chat_input(
        "Ask anything about disaster response, safety, or the current incident...",
        key="main_assistant_input",
    ):
        st.session_state.assistant_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("CrisisLens AI thinking..."):
                try:
                    reply = chat_with_assistant(
                        user_message=user_input,
                        chat_history=st.session_state.assistant_messages,
                        context=st.session_state.incident_context,
                    )
                except Exception as exc:
                    reply = f"⚠️ I couldn't generate a response right now. Error: {exc}"
            st.markdown(reply)

        st.session_state.assistant_messages.append({"role": "assistant", "content": reply})
        st.rerun()
