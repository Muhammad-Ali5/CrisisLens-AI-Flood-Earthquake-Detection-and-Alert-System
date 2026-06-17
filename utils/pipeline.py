from utils.alert_engine import alert_engine
from utils.disaster_classifier import classify_disaster
from utils.fake_news_detection import detect_fake_news
from utils.gemini_relief_agent import generate_relief_plan
from utils.location_extractor import extract_location
from utils.severity_prediction import predict_severity
from utils.storage import save_incident
from utils.dl_classifier import classify_image


def analyze_report(report: str = None, image_bytes: bytes = None):
    if not report and not image_bytes:
        return {"error": "No report text or image provided"}

    dl_result = None
    if image_bytes is not None:
        try:
            dl_result = classify_image(image_bytes)
        except Exception as e:
            print(f"DL Classifier Error: {e}")
            dl_result = {"error": str(e)}

    report_text = report or ""

    # VALIDATION
    if not report_text.strip() and dl_result and dl_result.get("dl_prediction"):
        report_text = f"Image classified as: {dl_result.get('dl_prediction', 'Unknown')}"
    elif not report_text.strip():
        return {"error": "Empty report provided", "dl_result": dl_result}

    # DISASTER CLASSIFICATION
    disaster = "Unknown"
    try:
        disaster = classify_disaster(report_text)
    except Exception as e:
        print(f"Disaster Classification Error: {e}")

    if dl_result and dl_result.get("dl_prediction") and "No disaster" not in dl_result["dl_prediction"]:
        dl_label = dl_result["dl_prediction"].split("|")[0].strip().lower()
        if disaster == "Unknown":
            disaster = dl_label.capitalize()
        elif disaster.lower() != dl_label.lower():
            if dl_result.get("dl_confidence", 0) and dl_result["dl_confidence"] > 0.8:
                disaster = dl_label.capitalize()

    # LOCATION EXTRACTION
    try:
        location = extract_location(report_text)
    except Exception as e:
        location = "Unknown"
        print(f"Location Extraction Error: {e}")

    # SEVERITY PREDICTION
    try:
        severity = predict_severity(report_text)
    except Exception as e:
        severity = "Unknown"
        print(f"Severity Prediction Error: {e}")

    # FAKE NEWS DETECTION
    try:
        authenticity = detect_fake_news(report_text)
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
            "report": report_text
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
            report_text
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
        "briefing": briefing,
        "dl_result": dl_result
    }