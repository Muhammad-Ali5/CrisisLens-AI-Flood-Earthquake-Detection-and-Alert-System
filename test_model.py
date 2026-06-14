# tests/test_alert_engine.py

from utils.alert_engine import alert_engine

alerts = alert_engine.generate_alerts(
    disaster_type="Flood",
    location="Swat",
    severity="High",
    authenticity="Real"
)

print("\n=== CITIZEN ALERT ===")
print(alerts["citizen"])

print("\n=== NGO ALERT ===")
print(alerts["ngo"])

print("\n=== GOVERNMENT ALERT ===")
print(alerts["government"])