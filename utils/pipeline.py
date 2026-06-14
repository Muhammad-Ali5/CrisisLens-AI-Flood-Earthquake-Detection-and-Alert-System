from utils.alert_engine import alert_engine
from utils.disaster_classifier import classify_disaster
from utils.fake_news_detection import detect_fake_news
from utils.gemini_relief_agent import generate_relief_plan
from utils.location_extractor import extract_location
from utils.severity_prediction import predict_severity


def analyze_report(report):
    disaster = classify_disaster(report)
    location = extract_location(report)
    severity = predict_severity(report)
    authenticity = detect_fake_news(report)

    alerts = alert_engine.generate_alerts(
        disaster,
        location,
        severity,
        authenticity,
    )

    try:
        briefing = generate_relief_plan(
            disaster,
            location,
            severity,
            authenticity,
            report,
        )
    except Exception as exc:
        briefing = f"Agent Error: {exc}"

    return {
        "disaster": disaster,
        "location": location,
        "severity": severity,
        "authenticity": authenticity,
        "alerts": alerts,
        "briefing": briefing,
    }