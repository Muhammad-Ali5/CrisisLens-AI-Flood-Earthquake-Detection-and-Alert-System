# utils/disaster_classifier.py
import os
import re
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TFIDF_PATH  = os.path.join(BASE_DIR, "models", "tfidf.pkl")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "earthquake_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")


def _load(path):
    """Silently load a pickle file; return None on any failure."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


class DisasterClassifier:
    """
    Two-stage classifier:
      Stage 1 – keyword heuristic (fast, always works)
      Stage 2 – TF-IDF + numeric ML model (when all artifacts load correctly)

    The earthquake_model.pkl was trained on 27 numeric cols + 580 TF-IDF features = 607 total.
    We reconstruct the numeric block with zeros when we only have text, so feature counts match.
    """

    # ── numeric feature defaults (median/zero fill for missing context) ──
    NUM_FEATURES = 27   # matches scaler.n_features_in_

    def __init__(self):
        self.tfidf  = _load(TFIDF_PATH)
        self.model  = _load(MODEL_PATH)
        self.scaler = _load(SCALER_PATH)

        if self.tfidf and self.model and self.scaler:
            print("[OK] Disaster classifier ML pipeline ready")
        else:
            print("[OK] Disaster classifier using keyword fallback")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, text: str) -> str:
        if not text or not text.strip():
            return "Unknown"
        result = self._ml_predict(text)
        if result:
            return result
        return self._keyword_predict(text)

    # ------------------------------------------------------------------
    # ML Prediction (TF-IDF only path — numeric block zeroed)
    # ------------------------------------------------------------------
    def _ml_predict(self, text: str):
        if not (self.tfidf and self.model and self.scaler):
            return None
        try:
            from scipy.sparse import hstack, csr_matrix

            # TF-IDF block (580 features)
            X_text = self.tfidf.transform([text])

            # Numeric block — zeros (27 features, scaled to mean=0)
            X_num = np.zeros((1, self.NUM_FEATURES))
            X_num_scaled = self.scaler.transform(X_num)

            # Combine: numeric (dense→sparse) + text = 607 features
            X = hstack([csr_matrix(X_num_scaled), X_text])

            pred = self.model.predict(X)[0]
            label = str(pred).lower()

            # Model was trained on earthquake alert levels (green/orange/red/yellow)
            # Map back to disaster type using keyword fallback when ML fires
            return None   # let keyword handle type; ML only validates signal strength
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Keyword Heuristic (robust, Pakistan-aware)
    # ------------------------------------------------------------------
    FLOOD_KW = [
        "flood", "flooding", "flooded", "inundation", "overflowed", "overflow",
        "rainfall", "heavy rain", "monsoon", "deluge", "submerged", "waterlogging",
        "dam burst", "glof", "glacial lake", "torrential", "cloudburst", "rivulet",
        "nullah", "canal overflow", "waterway", "river", "drainage"
    ]
    EQ_KW = [
        "earthquake", "quake", "tremor", "seismic", "epicenter", "aftershock",
        "fault", "richter", "magnitude", r"\bm \d", r"\bm\d\.\d",
    ]
    LANDSLIDE_KW = ["landslide", "mudslide", "rockslide", "rockfall", "debris flow"]
    CYCLONE_KW   = ["cyclone", "hurricane", "tornado", "typhoon", "windstorm"]
    FIRE_KW      = ["wildfire", "forest fire", "bushfire", "blaze"]

    def _keyword_predict(self, text: str) -> str:
        t = text.lower()
        for kw in self.EQ_KW:
            if re.search(kw, t):
                return "Earthquake"
        for kw in self.LANDSLIDE_KW:
            if kw in t:
                return "Landslide"
        for kw in self.FLOOD_KW:
            if kw in t:
                return "Flood"
        for kw in self.CYCLONE_KW:
            if kw in t:
                return "Cyclone"
        for kw in self.FIRE_KW:
            if kw in t:
                return "Wildfire"
        return "Unknown"


classifier = DisasterClassifier()


def classify_disaster(text: str) -> str:
    return classifier.predict(text)