import os
from dotenv import load_dotenv
from groq import Groq

# ==========================
# Configuration
# ==========================

load_dotenv()  # Load variables from .env file

API_KEY = os.environ.get("GROQ_API_KEY", "")

try:
    client = Groq(api_key=API_KEY) if API_KEY else None
except Exception:
    client = None

MODEL = "llama-3.3-70b-versatile"   # Fast, powerful — best for structured analysis
# Other options:
# "llama-3.1-70b-versatile"         — slightly older but very stable
# "mixtral-8x7b-32768"              — good for long context
# "gemma2-9b-it"                    — lightweight / faster responses


# ==========================
# Prompt Builder
# ==========================

def build_prompt(disaster_type, location, severity, authenticity, original_report):

    is_fake = "fake" in authenticity.lower() or "suspicious" in authenticity.lower()

    fake_instruction = ""
    if is_fake:
        fake_instruction = """
CRITICAL AUTHENTICITY FLAG:
This report has been flagged as SUSPICIOUS or FAKE by the CrisisLens AI detection system.

DO NOT recommend full emergency resource deployment.
Instead:
- Recommend immediate cross-verification with NDMA, PDMA, and local authorities.
- Flag misinformation risk to all response teams.
- Suggest lightweight verification actions only.
- Clearly state this briefing is based on an unverified report.
"""

    prompt = f"""
====================================================
# DISASTER INTELLIGENCE REPORT

Disaster Type    : {disaster_type}
Location         : {location}
Severity Level   : {severity}
Authenticity     : {authenticity}
Original Report  : {original_report}

{fake_instruction}
====================================================
# ANALYSIS OBJECTIVES

Analyze the disaster report and produce a professional emergency management briefing
covering all 8 sections below. Each section must be complete and actionable.

----------------------------------------------------
## 1. SITUATION SUMMARY

Provide:
- What happened
- Where it happened (district, province, coordinates if known)
- Current impact on population and infrastructure
- Overall situation status

----------------------------------------------------
## 2. THREAT ASSESSMENT

Identify and rate each risk:
- Human risk (casualties, displacement, health threats)
- Infrastructure risk (roads, bridges, buildings, utilities)
- Environmental risk (water contamination, landslides, aftershocks)
- Economic risk (crops, livestock, livelihoods)

State the overall Risk Level: LOW / MEDIUM / HIGH / CRITICAL

----------------------------------------------------
## 3. EMERGENCY RESPONSE PLAN

Generate a phased response plan:

Immediate Actions (0-6 Hours):
Short-Term Actions (6-24 Hours):
Operational Actions (24-72 Hours):

----------------------------------------------------
## 4. RESOURCE ALLOCATION ADVICE

Provide specific estimates for:
- Rescue Teams (number and type)
- Medical Teams (doctors, paramedics, field hospitals)
- Ambulances
- Emergency Shelters (capacity required)
- Food Supplies (rations for how many people, how many days)
- Clean Water Supplies (liters/day, purification units)
- Transportation Assets (helicopters, boats, trucks)

----------------------------------------------------
## 5. PUBLIC SAFETY RECOMMENDATIONS

Provide clear instructions for civilians including:
- Evacuation guidance (routes, assembly points)
- Safety precautions (what to avoid, what to carry)
- Medical precautions (water safety, injury care, disease prevention)
- Communication advice (emergency numbers, stay informed)

----------------------------------------------------
## 6. GOVERNMENT ACTION PLAN

Provide specific recommended actions for:
- Local Government (DC, city administration)
- NDMA (National Disaster Management Authority)
- PDMA (Provincial Disaster Management Authority)
- Health Authorities (hospitals, emergency medical services)
- Security Agencies (Police, Army, Rangers for law and order)

----------------------------------------------------
## 7. RESCUE PRIORITIES

Rank the top response priorities in order of urgency:

Priority 1: [Most critical action]
Priority 2: [Second most critical]
Priority 3: [Third most critical]
Priority 4: [Fourth if applicable]
Priority 5: [Fifth if applicable]

----------------------------------------------------
## 8. RECOVERY STRATEGY

Recommend steps for:
- Infrastructure restoration (roads, utilities, housing)
- Community recovery (schools, livelihoods, psychosocial support)
- Medical recovery (disease surveillance, long-term care)
- Economic recovery (compensation, crop restoration, employment)

====================================================
# OUTPUT FORMAT

Use exactly these section headers with the emoji icons:

🚨 SITUATION SUMMARY
[content]

⚠️ THREAT ASSESSMENT
[content]

🚑 EMERGENCY RESPONSE PLAN
[content]

📦 RESOURCE ALLOCATION ADVICE
[content]

📢 PUBLIC SAFETY RECOMMENDATIONS
[content]

🏛️ GOVERNMENT ACTION PLAN
[content]

🚁 RESCUE PRIORITIES
[content]

🔄 RECOVERY STRATEGY
[content]

====================================================
# RESPONSE RULES

- Use clear section headers exactly as shown above.
- Use bullet points (- ) for all lists.
- Keep language direct, practical, and professional.
- Every recommendation must be specific to this disaster type, location, and severity.
- Maximum 700 words total.
- Do not add any preamble, greeting, or closing statement.
- Begin your response directly with: 🚨 SITUATION SUMMARY
"""

    return prompt


# ==========================
# System Prompt
# ==========================

SYSTEM_PROMPT = """You are CrisisLens AI Commander.
An advanced Disaster Intelligence, Emergency Response, and Decision-Support Agent
built for governments, rescue organizations, NGOs, disaster management authorities,
and emergency response teams operating in Pakistan.

Your objective is NOT to chat.
Your objective is to transform disaster intelligence into actionable emergency response strategies.

You must behave like a senior disaster-management analyst.

NEVER use uncertain phrases such as:
- I think
- Maybe
- Possibly
- I am not sure
- It seems

Provide confident, structured, professional analysis at all times.
Respond only with the structured briefing. No preamble. No closing remarks."""


# ==========================
# Chat Assistant Prompt
# ==========================

CHAT_SYSTEM_PROMPT = """You are CrisisLens AI Assistant — a friendly, knowledgeable, and professional
emergency management assistant built into the CrisisLens AI platform.

CRITICAL FORMATTING RULES — YOU MUST FOLLOW THESE:
- NEVER use heavy report headers like "IDENTIFICATION" or "CAPABILITIES".
- DO use beautiful, well-formatted markdown bullet points and emojis to structure your responses.
- Separate key topics with bold headings or spacing.
- Always present instructions, safety steps, or lists in clean, readable bullet points.
- Keep your tone conversational but highly structured and visually appealing.
- Keep responses relatively concise and focused.

YOUR PERSONALITY:
- You are warm, helpful, and sound like a real person — not a robot.
- You speak professionally, like a friendly emergency management expert.
- You are confident and knowledgeable.

EXAMPLE RESPONSES:

User: "hi"
You: "Hey there! 👋 I'm the CrisisLens AI Assistant. I'm here to help you with anything related to disaster preparedness, emergency response, or crisis management. What can I help you with?"

User: "who are you"
You: "I'm the CrisisLens AI Assistant! I specialize in emergency response and disaster management in Pakistan:
- 🌊 **Flood Response**: Evacuation planning, water safety, and resource coordination.
- 🏔️ **Earthquake Safety**: Drop, cover, and hold protocols, search priorities, and structural safety.
- 📋 **Analysis Support**: Explaining predictive alerts and intelligence briefings.

Feel free to ask me anything about staying safe or managing incident reports!"

User: "what should I do during an earthquake"
You: "Here is what you should do immediately during an earthquake:
- 🛡️ **Drop, Cover, and Hold On** — get under a sturdy desk or table.
- 🪟 **Stay Away from Glass** — stay clear of windows, mirrors, or heavy furniture.
- 🚪 **Do Not Run Outside** — wait until the shaking stops before attempting to exit.
- 🌲 **If Outdoors** — move to an open area away from buildings, power lines, and trees.
- 🔄 **Expect Aftershocks** — prepare for minor tremors following the main quake."


WHAT YOU CAN HELP WITH:
- Disaster preparedness and safety tips
- Emergency response protocols and evacuation guidance
- First aid basics and survival advice
- Explaining CrisisLens analysis results in plain language
- General questions about floods, earthquakes, landslides, and other disasters
- Any follow-up questions about the current incident report

WHAT TO AVOID:
- Never generate a full 8-section emergency briefing unless explicitly asked
- Never respond with formal document formatting
- Never be robotic or overly bureaucratic
- Never say "I think", "maybe", or "possibly" — be confident"""



# ==========================
# Core Agent Function
# ==========================

def generate_relief_plan(
    disaster_type: str,
    location: str,
    severity: str,
    authenticity: str,
    original_report: str
) -> str:
    """
    Generate a full emergency response briefing using the CrisisLens AI Commander.

    Args:
        disaster_type   : Type of disaster (e.g. "Flood", "Earthquake")
        location        : Affected location (e.g. "Swat, Khyber Pakhtunkhwa")
        severity        : Severity level ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        authenticity    : Report authenticity ("VERIFIED", "SUSPICIOUS", "FAKE")
        original_report : Raw text of the original disaster report

    Returns:
        str: Structured emergency briefing as a formatted string
    """

    if not all([disaster_type, location, severity, authenticity, original_report]):
        return "ERROR: All fields are required. Please provide disaster_type, location, severity, authenticity, and original_report."

    prompt = build_prompt(
        disaster_type=disaster_type,
        location=location,
        severity=severity,
        authenticity=authenticity,
        original_report=original_report
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,       # Low = consistent, professional tone
            max_tokens=1500,
            top_p=0.9,
            stream=False
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err or "Invalid API Key" in err:
            return (
                "❌ **Groq API key is invalid or expired.**\n\n"
                "The AI Commander cannot generate briefings until you update your key.\n\n"
                "**Fix:**\n"
                "1. Go to https://console.groq.com/keys\n"
                "2. Generate a new API key\n"
                "3. Update `GROQ_API_KEY` in your `.env` file\n"
                "4. Restart the app\n"
            )
        return f"Agent Error: {err}"


# ==========================
# Display Helper
# ==========================

def print_briefing(briefing: str):
    """Print the briefing in a clean, readable format."""
    divider = "=" * 60
    print(f"\n{divider}")
    print("  CRISISLENS AI COMMANDER — EMERGENCY BRIEFING")
    print(f"{divider}\n")
    print(briefing)
    print(f"\n{divider}\n")


# ==========================
# Interactive CLI Mode
# ==========================

def run_interactive():
    """Run the agent interactively from the command line."""

    print("\n" + "=" * 60)
    print("  CRISISLENS AI COMMANDER")
    print("  Disaster Intelligence & Emergency Response Agent")
    print("=" * 60 + "\n")

    disaster_types = ["Flood", "Earthquake", "Flash Flood", "Landslide", "Cyclone", "Drought"]
    severity_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    auth_options    = ["VERIFIED REAL", "SUSPICIOUS", "FAKE"]

    print("Select disaster type:")
    for i, d in enumerate(disaster_types, 1):
        print(f"  {i}. {d}")
    dt_idx = int(input("Enter number: ").strip()) - 1
    disaster_type = disaster_types[dt_idx]

    location = input("\nEnter location (e.g. Swat, KPK): ").strip()

    print("\nSelect severity level:")
    for i, s in enumerate(severity_levels, 1):
        print(f"  {i}. {s}")
    sev_idx = int(input("Enter number: ").strip()) - 1
    severity = severity_levels[sev_idx]

    print("\nSelect authenticity:")
    for i, a in enumerate(auth_options, 1):
        print(f"  {i}. {a}")
    auth_idx = int(input("Enter number: ").strip()) - 1
    authenticity = auth_options[auth_idx]

    print("\nPaste the original report (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    original_report = "\n".join(lines).strip()

    print("\nDeploying CrisisLens AI Commander...\n")

    briefing = generate_relief_plan(
        disaster_type=disaster_type,
        location=location,
        severity=severity,
        authenticity=authenticity,
        original_report=original_report
    )

    print_briefing(briefing)

    save = input("Save briefing to file? (y/n): ").strip().lower()
    if save == "y":
        filename = f"briefing_{disaster_type.lower().replace(' ', '_')}_{location.split(',')[0].lower().replace(' ', '_')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(briefing)
        print(f"Briefing saved to: {filename}")


# ==========================
# Quick Test / Demo
# ==========================

def run_demo():
    """Run a demo with a sample flood report from Swat."""

    briefing = generate_relief_plan(
        disaster_type="Flood",
        location="Swat, Khyber Pakhtunkhwa",
        severity="HIGH",
        authenticity="VERIFIED REAL — Confirmed by NDMA",
        original_report=(
            "Heavy flooding reported in Swat district. River Swat has overflowed its banks. "
            "Multiple villages submerged including Mingora outskirts, Matta, and Bahrain areas. "
            "Roads blocked, bridges damaged. Approximately 500 families displaced and stranded. "
            "Local hospital overwhelmed. Urgent need for food, clean water, and medical supplies. "
            "Mobile networks disrupted in lower Swat."
        )
    )

    print_briefing(briefing)


# ==========================
# Streamlit Integration
# ==========================

def get_relief_plan_for_streamlit(
    disaster_type: str,
    location: str,
    severity: str,
    authenticity: str,
    original_report: str
) -> str:
    """
    Wrapper for Streamlit integration.
    Call this from your Streamlit page to get the briefing text directly.

    Example usage in Streamlit:
        from groq_relief_agent import get_relief_plan_for_streamlit
        result = get_relief_plan_for_streamlit(...)
        st.markdown(result)
    """
    return generate_relief_plan(
        disaster_type=disaster_type,
        location=location,
        severity=severity,
        authenticity=authenticity,
        original_report=original_report
    )


# ==========================
# FastAPI Integration
# ==========================

def get_relief_plan_for_api(payload: dict) -> dict:
    """
    Wrapper for FastAPI integration.
    Accepts a dict payload and returns a structured response dict.

    Example usage in FastAPI route:
        from groq_relief_agent import get_relief_plan_for_api

        @app.post("/agent/relief-plan")
        def relief_plan(payload: ReportSchema):
            return get_relief_plan_for_api(payload.dict())
    """
    required = ["disaster_type", "location", "severity", "authenticity", "original_report"]
    for field in required:
        if field not in payload:
            return {"success": False, "error": f"Missing field: {field}", "briefing": None}

    briefing = generate_relief_plan(
        disaster_type=payload["disaster_type"],
        location=payload["location"],
        severity=payload["severity"],
        authenticity=payload["authenticity"],
        original_report=payload["original_report"]
    )

    if briefing.startswith("Agent Error") or briefing.startswith("ERROR"):
        return {"success": False, "error": briefing, "briefing": None}

    return {"success": True, "error": None, "briefing": briefing}


# ==========================
# Conversational Chat Function
# ==========================

def chat_with_assistant(user_message: str, chat_history: list, context: dict = None) -> str:
    """
    Handle a conversational exchange with the CrisisLens AI Assistant.

    Args:
        user_message : The user's latest question or message.
        chat_history : List of previous messages [{"role": ..., "content": ...}].
        context      : Optional dict with current incident context
                       (keys: disaster, location, severity, authenticity, report).

    Returns:
        str: The assistant's reply.
    """
    context_block = ""
    if context:
        context_block = f"""

CURRENT INCIDENT CONTEXT:
- Disaster Type : {context.get('disaster', 'N/A')}
- Location      : {context.get('location', 'N/A')}
- Severity      : {context.get('severity', 'N/A')}
- Authenticity  : {context.get('authenticity', 'N/A')}
- Original Report: {context.get('report', 'N/A')[:500]}
"""

    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT + context_block}
    ]

    # Include recent chat history (last 10 messages to stay within context limits)
    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add the new user message
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
            top_p=0.9,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err or "Invalid API Key" in err:
            return (
                "❌ **Groq API key is invalid or expired.**\n\n"
                "The AI Assistant cannot respond until you update your key.\n\n"
                "**Fix:**\n"
                "1. Go to https://console.groq.com/keys\n"
                "2. Generate a new API key\n"
                "3. Update `GROQ_API_KEY` in your `.env` file\n"
                "4. Restart the app\n"
            )
        return f"Agent Error: {err}"


# ==========================
# Entry Point
# ==========================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive()
    else:
        print("Usage:")
        print("  python groq_relief_agent.py --demo          Run a sample flood briefing")
        print("  python groq_relief_agent.py --interactive   Enter report details manually")