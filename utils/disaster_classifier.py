# utils/disaster_classifier.py

import os
import pickle


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TFIDF_PATH = os.path.join(BASE_DIR, "models", "tfidf.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "earthquake_model.pkl")


def load_pickle_artifact(path):
    """Safely load pickle files"""

    if not os.path.exists(path):
        print(f"❌ Missing file: {path}")
        return None

    try:

        if os.path.getsize(path) == 0:
            print(f"❌ Empty file: {path}")
            return None

        with open(path, "rb") as f:
            return pickle.load(f)

    except Exception as e:

        print(f"❌ Load Error ({path})")
        print(e)

        return None


class DisasterClassifier:

    def __init__(self):

        self.vectorizer = load_pickle_artifact(TFIDF_PATH)
        self.model = load_pickle_artifact(MODEL_PATH)

        if self.vectorizer:
            print("✅ TFIDF Loaded")

        if self.model:
            print("✅ Model Loaded")

    # -------------------------------------------------
    # Fallback Keyword Classifier
    # -------------------------------------------------

    def keyword_classifier(self, text):

        text = text.lower()

        flood_keywords = [
            "flood",
            "flooding",
            "rain",
            "rainfall",
            "river",
            "water",
            "overflow",
            "storm",
            "dam"
        ]

        earthquake_keywords = [
            "earthquake",
            "quake",
            "richter",
            "tremor",
            "epicenter",
            "magnitude",
            "aftershock",
            "seismic"
        ]

        for word in flood_keywords:

            if word in text:
                return "Flood"

        for word in earthquake_keywords:

            if word in text:
                return "Earthquake"

        return "Unknown"

    # -------------------------------------------------
    # ML Prediction
    # -------------------------------------------------

    def ml_classifier(self, text):

        try:

            X = self.vectorizer.transform([text])

            prediction = self.model.predict(X)[0]

            label = str(prediction).lower()

            if label in ["0", "flood"]:
                return "Flood"

            elif label in ["1", "earthquake"]:
                return "Earthquake"

            return str(prediction)

        except Exception as e:

            print("⚠ ML Prediction Error:")
            print(e)

            return None

    # -------------------------------------------------
    # Main Classification Function
    # -------------------------------------------------

    def predict(self, text):

        if not text:
            return "Unknown"

        # Try ML first

        if self.vectorizer is not None and self.model is not None:

            result = self.ml_classifier(text)

            if result is not None:
                return result

        # Fallback

        return self.keyword_classifier(text)


classifier = DisasterClassifier()


def classify_disaster(text):
    return classifier.predict(text)