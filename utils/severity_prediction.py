# utils/severity_prediction.py
import re


class SeverityPredictor:
    """
    Multi-signal severity predictor.

    Signals checked (in order of priority):
      1. Explicit death / casualty count         → Critical or High
      2. Earthquake magnitude extracted from text → Critical / High / Medium
      3. High-impact keyword score               → Critical / High / Medium / Low
    """

    # ── keyword tables ──────────────────────────────────────────────
    CRITICAL_KW = [
        "hundreds dead", "thousands dead", "mass casualty", "catastrophic",
        "state of emergency", "massive destruction", "totally destroyed",
        "complete devastation", "dam collapse", "dam burst", "glof",
    ]

    HIGH_KW = [
        "collapsed", "destroyed", "fatalities", "deaths", "killed",
        "casualties", "dead", "critical", "emergency", "evacuation",
        "evacuated", "submerged", "swept away", "rescue operation",
        "stranded", "missing persons", "buried alive", "building collapse",
        "bridge collapse", "major flooding",
    ]

    MEDIUM_KW = [
        "damaged", "injured", "wounded", "affected", "flooding",
        "blocked", "disrupted", "displaced", "power outage", "waterlogging",
        "road blocked", "crops damaged", "livestock lost", "partial collapse",
        "tremors felt", "felt strongly", "cracks appeared",
    ]

    LOW_KW = [
        "warning issued", "alert", "heavy rain expected", "watch",
        "advisory", "minor tremor", "light earthquake", "slight damage",
        "no casualties reported", "precautionary", "monitoring",
    ]

    # death-count pattern: "12 dead", "40 killed", "3 fatalities"
    _DEATH_RE = re.compile(
        r'(\d+)\s+(?:dead|killed|deaths|fatalities|casualties)', re.I
    )
    # magnitude pattern: "magnitude 5.2", "M 5.2", "5.2 magnitude"
    _MAG_RE = re.compile(
        r'(?:magnitude|mag\.?|m)\s*(\d+(?:\.\d+)?)', re.I
    )

    # ────────────────────────────────────────────────────────────────
    def predict(self, text: str) -> str:
        if not text or not text.strip():
            return "Low"

        t = text.lower()

        # ── 1. Death / casualty count ────────────────────────────────
        death_match = self._DEATH_RE.search(t)
        if death_match:
            n = int(death_match.group(1))
            if n >= 50:
                return "Critical"
            if n >= 10:
                return "High"
            if n >= 1:
                return "Medium"

        # ── 2. Earthquake magnitude ──────────────────────────────────
        mag_match = self._MAG_RE.search(text)  # use original for M pattern
        if mag_match:
            mag = float(mag_match.group(1))
            if mag >= 6.5:
                return "Critical"
            if mag >= 5.5:
                return "High"
            if mag >= 4.5:
                return "Medium"

        # ── 3. Keyword score ─────────────────────────────────────────
        for kw in self.CRITICAL_KW:
            if kw in t:
                return "Critical"

        high_score = sum(1 for kw in self.HIGH_KW if kw in t)
        if high_score >= 2:
            return "High"
        if high_score == 1:
            # bump up to High if any medium keyword also present
            medium_hit = any(kw in t for kw in self.MEDIUM_KW)
            return "High" if medium_hit else "Medium"

        medium_score = sum(1 for kw in self.MEDIUM_KW if kw in t)
        if medium_score >= 1:
            return "Medium"

        return "Low"


predictor = SeverityPredictor()


def predict_severity(text: str) -> str:
    return predictor.predict(text)