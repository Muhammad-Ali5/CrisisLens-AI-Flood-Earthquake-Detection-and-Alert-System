import pickle
import os


def load_artifact(path, fallback=None):
    try:
        if os.path.getsize(path) == 0:
            return fallback
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return fallback


if __name__ == "__main__":
    from utils.email_alert import send_email_alert

    send_email_alert(
        "maliuetm507@gmail.com",
        "Flood Alert",
        "Move to higher ground immediately."
    )
