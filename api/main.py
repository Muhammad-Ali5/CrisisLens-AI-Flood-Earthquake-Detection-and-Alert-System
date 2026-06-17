from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from utils.pipeline import analyze_report
from utils.storage import load_incidents

from utils.gemini_relief_agent import (
    generate_relief_plan
)

# =====================================================
# APP
# =====================================================

app = FastAPI(
    title="CrisisLens AI API",
    version="1.0.0",
    description="AI-Powered Disaster Intelligence Platform"
)

# =====================================================
# REQUEST MODELS
# =====================================================

class ReportRequest(BaseModel):
    report: str


class ChatRequest(BaseModel):
    question: str


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():

    return {
        "message": "CrisisLens AI Backend Running",
        "version": "1.0.0"
    }


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =====================================================
# ANALYZE REPORT
# =====================================================

@app.post("/analyze")
def analyze(data: ReportRequest):

    result = analyze_report(
        data.report
    )

    return result


# =====================================================
# AI CHAT
# =====================================================

@app.post("/chat")
def chat(data: ChatRequest):

    try:

        answer = generate_relief_plan(
            disaster_type="General Inquiry",
            location="Unknown",
            severity="Unknown",
            authenticity="Unknown",
            original_report=data.question
        )

        return {
            "question": data.question,
            "answer": answer
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# =====================================================
# INCIDENTS
# =====================================================

@app.get("/incidents")
def incidents():

    try:

        return load_incidents()

    except Exception as e:

        return {
            "error": str(e)
        }


# =====================================================
# ANALYTICS
# =====================================================

@app.get("/analytics")
def analytics():

    try:

        incidents = load_incidents()

        total_reports = len(incidents)

        flood_reports = sum(
            1
            for i in incidents
            if str(
                i.get(
                    "disaster",
                    ""
                )
            ).lower() == "flood"
        )

        earthquake_reports = sum(
            1
            for i in incidents
            if str(
                i.get(
                    "disaster",
                    ""
                )
            ).lower() == "earthquake"
        )

        high_severity = sum(
            1
            for i in incidents
            if str(
                i.get(
                    "severity",
                    ""
                )
            ).lower() == "high"
        )

        medium_severity = sum(
            1
            for i in incidents
            if str(
                i.get(
                    "severity",
                    ""
                )
            ).lower() == "medium"
        )

        low_severity = sum(
            1
            for i in incidents
            if str(
                i.get(
                    "severity",
                    ""
                )
            ).lower() == "low"
        )

        return {

            "total_reports": total_reports,

            "flood_reports": flood_reports,

            "earthquake_reports": earthquake_reports,

            "high_severity_reports": high_severity,

            "medium_severity_reports": medium_severity,

            "low_severity_reports": low_severity
        }

    except Exception as e:

        return {
            "error": str(e)
        }