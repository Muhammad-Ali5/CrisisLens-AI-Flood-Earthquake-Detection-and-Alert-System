# tests/test_alert_engine.py
import os
import pickle

def load_artifact(path, fallback=None):
    """Safely load pickle files, returning fallback if empty, missing, or load fails."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return fallback
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return fallback


if __name__ == "__main__":
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