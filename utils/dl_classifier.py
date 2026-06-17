import os
import numpy as np
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EQ_MODEL_PATH = os.path.join(BASE_DIR, "models", "earthquake_model.h5")
FLOOD_MODEL_PATH = os.path.join(BASE_DIR, "models", "flood_model.h5")

_models_loaded = False
_eq_model = None
_flood_model = None
_img_size = 224


def _load_models():
    global _eq_model, _flood_model, _models_loaded
    if _models_loaded:
        return True
    try:
        import tensorflow as tf
        if os.path.exists(EQ_MODEL_PATH):
            _eq_model = tf.keras.models.load_model(EQ_MODEL_PATH)
            print("[OK] Earthquake DL model loaded")
        else:
            print("[WARN] earthquake_model.h5 not found")
        if os.path.exists(FLOOD_MODEL_PATH):
            _flood_model = tf.keras.models.load_model(FLOOD_MODEL_PATH)
            print("[OK] Flood DL model loaded")
        else:
            print("[WARN] flood_model.h5 not found")
        _models_loaded = True
        return True
    except Exception as e:
        print(f"[WARN] DL models could not be loaded: {e}")
        return False


def preprocess_image(image_bytes):
    try:
        import tensorflow as tf
        img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [_img_size, _img_size])
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return tf.expand_dims(img, axis=0)
    except Exception as e:
        print(f"[WARN] Image preprocessing failed: {e}")
        return None


def classify_image(image_bytes):
    if not _load_models():
        return {
            "earthquake_probability": None,
            "flood_probability": None,
            "dl_prediction": "Model not loaded  |  ماڈیل لوڈ نہیں ہوا",
            "dl_confidence": None,
            "error": "Models could not be loaded"
        }

    processed = preprocess_image(image_bytes)
    if processed is None:
        return {
            "earthquake_probability": None,
            "flood_probability": None,
            "dl_prediction": "Preprocessing failed  |  پروسیسنگ ناکام",
            "dl_confidence": None,
            "error": "Image preprocessing failed"
        }

    eq_prob = None
    flood_prob = None

    if _eq_model is not None:
        try:
            pred = _eq_model.predict(processed, verbose=0)
            eq_prob = float(pred[0][0])
        except Exception as e:
            print(f"[WARN] EQ model inference failed: {e}")

    if _flood_model is not None:
        try:
            pred = _flood_model.predict(processed, verbose=0)
            flood_prob = float(pred[0][0])
        except Exception as e:
            print(f"[WARN] Flood model inference failed: {e}")

    EQ_THRESHOLD = 0.5
    FLOOD_THRESHOLD = 0.5

    if eq_prob is not None and flood_prob is not None:
        if eq_prob > EQ_THRESHOLD and eq_prob > flood_prob:
            pred_label = "Earthquake  |  زلزلہ"
            confidence = eq_prob
        elif flood_prob > FLOOD_THRESHOLD and flood_prob > eq_prob:
            pred_label = "Flood  |  سیلاب"
            confidence = flood_prob
        elif eq_prob > EQ_THRESHOLD:
            pred_label = "Earthquake  |  زلزلہ"
            confidence = eq_prob
        elif flood_prob > FLOOD_THRESHOLD:
            pred_label = "Flood  |  سیلاب"
            confidence = flood_prob
        else:
            pred_label = "No disaster detected  |  کوئی آفت نہیں ملی"
            confidence = max(eq_prob, flood_prob)
    elif eq_prob is not None:
        if eq_prob > EQ_THRESHOLD:
            pred_label = "Earthquake  |  زلزلہ"
            confidence = eq_prob
        else:
            pred_label = "No disaster detected (EQ model)  |  کوئی آفت نہیں ملی"
            confidence = eq_prob
    elif flood_prob is not None:
        if flood_prob > FLOOD_THRESHOLD:
            pred_label = "Flood  |  سیلاب"
            confidence = flood_prob
        else:
            pred_label = "No disaster detected (Flood model)  |  کوئی آفت نہیں ملی"
            confidence = flood_prob
    else:
        pred_label = "Inference failed  |  پیش گوئی ناکام"
        confidence = None

    return {
        "earthquake_probability": round(eq_prob, 4) if eq_prob is not None else None,
        "flood_probability": round(flood_prob, 4) if flood_prob is not None else None,
        "dl_prediction": pred_label,
        "dl_confidence": round(confidence, 4) if confidence is not None else None,
        "error": None
    }
