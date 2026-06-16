from utils.alert_engine import alert_engine
from utils.disaster_classifier import classify_disaster
from utils.fake_news_detection import detect_fake_news
from utils.gemini_relief_agent import generate_relief_plan
from utils.location_extractor import extract_location
from utils.severity_prediction import predict_severity
from utils.storage import save_incident


def analyze_report(report: str):

    # VALIDATION
    if not report or not report.strip():
        return {
            "error": "Empty report provided"
        }

    # DISASTER CLASSIFICATION
    try:
        disaster = classify_disaster(report)
    except Exception as e:
        disaster = "Unknown"
        print(f"Disaster Classification Error: {e}")

    # LOCATION EXTRACTION
    try:
        location = extract_location(report)
    except Exception as e:
        location = "Unknown"
        print(f"Location Extraction Error: {e}")

    # SEVERITY PREDICTION
    try:
        severity = predict_severity(report)
    except Exception as e:
        severity = "Unknown"
        print(f"Severity Prediction Error: {e}")

    # FAKE NEWS DETECTION
    try:
        authenticity = detect_fake_news(report)
    except Exception as e:
        authenticity = "Unknown"
        print(f"Fake News Detection Error: {e}")

    # SAVE INCIDENT
    try:
        save_incident({
            "disaster": disaster,
            "location": location,
            "severity": severity,
            "authenticity": authenticity,
            "report": report
        })
    except Exception as e:
        print(f"Storage Error: {e}")

    # ALERT ENGINE
    try:
        alerts = alert_engine.generate_alerts(
            disaster,
            location,
            severity,
            authenticity
        )
    except Exception as e:
        print(f"Alert Error: {e}")

        alerts = {
            "citizen": "Alert generation failed.",
            "ngo": "Alert generation failed.",
            "government": "Alert generation failed."
        }

    # AI COMMANDER
    try:
        briefing = generate_relief_plan(
            disaster,
            location,
            severity,
            authenticity,
            report
        )
    except Exception as e:
        briefing = f"AI Agent Error: {e}"

    # RESPONSE
    return {
        "disaster": disaster,
        "location": location,
        "severity": severity,
        "authenticity": authenticity,
        "citizen_alert": alerts.get("citizen"),
        "ngo_alert": alerts.get("ngo"),
        "government_alert": alerts.get("government"),
        "briefing": briefing
    }