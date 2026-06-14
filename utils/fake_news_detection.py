# utils/fake_news_detection.py

class FakeNewsDetector:

    def __init__(self):

        self.fake_keywords = [

            "world ending",
            "entire pakistan destroyed",
            "100000 dead",
            "all cities destroyed",
            "government collapsed",
            "apocalypse",
            "fake news",
            "unverified"
        ]

        self.suspicious_keywords = [

            "reportedly",
            "rumor",
            "allegedly",
            "unconfirmed",
            "social media claim",
            "viral post"
        ]

    def detect(self, text):

        if not text:
            return "Unknown"

        text = text.lower()

        for word in self.fake_keywords:

            if word in text:
                return "Fake"

        for word in self.suspicious_keywords:

            if word in text:
                return "Suspicious"

        return "Likely Real"


detector = FakeNewsDetector()


def detect_fake_news(text):
    return detector.detect(text)