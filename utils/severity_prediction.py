# utils/severity_prediction.py

class SeverityPredictor:

    def __init__(self):

        self.high_keywords = [
            "thousands",
            "collapsed",
            "destroyed",
            "fatalities",
            "deaths",
            "severe",
            "critical",
            "emergency",
            "evacuation",
            "submerged"
        ]

        self.medium_keywords = [
            "damaged",
            "injured",
            "affected",
            "flooding",
            "blocked",
            "displaced",
            "power outage"
        ]

    def predict(self, text):

        if not text:
            return "Low"

        text = text.lower()

        high_score = 0
        medium_score = 0

        for word in self.high_keywords:

            if word in text:
                high_score += 1

        for word in self.medium_keywords:

            if word in text:
                medium_score += 1

        if high_score >= 2:
            return "High"

        if medium_score >= 1:
            return "Medium"

        return "Low"


predictor = SeverityPredictor()


def predict_severity(text):
    return predictor.predict(text)