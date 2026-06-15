# utils/fake_news_detection.py
import re


class FakeNewsDetector:
    """
    Three-tier authenticity classifier:
      - Fake       : strong misinformation signals
      - Suspicious : exaggerated or unverified claims
      - Likely Real: no red flags detected
    """

    # ── FAKE signals ─────────────────────────────────────────────────
    FAKE_KW = [
        "world ending", "entire pakistan destroyed", "all hospitals destroyed",
        "government hiding", "hidden from public", "mainstream media not reporting",
        "share before they delete", "share immediately", "before it gets deleted",
        "fake news", "100000 dead", "1 million dead", "millions killed",
        "apocalypse", "government collapsed", "martial law declared",
        "nobody survived", "all cities flooded", "entire city wiped out",
    ]

    # ── SUSPICIOUS / exaggeration signals ───────────────────────────
    SUSPICIOUS_KW = [
        "allegedly", "reportedly", "rumor", "rumour", "unconfirmed",
        "social media claim", "viral post", "whatsapp forward",
        "unverified", "sources say", "some people claim",
        "breaking news unconfirmed", "not yet confirmed",
        "could be", "possibly", "might be", "spread awareness",
        "share this now", "please share",
    ]

    # ── CREDIBILITY boosters (reduce false positives) ────────────────
    CREDIBLE_KW = [
        "ndma", "pdma", "rescue 1122", "civil defence", "government of pakistan",
        "confirmed by", "official statement", "press release", "according to",
        "district administration", "deputy commissioner", "army", "pakistan army",
        "provincial government", "federal government", "ministry of",
        "usgs", "meteorological department", "pmd", "geo news", "dawn news",
        "ary news", "the express tribune", "bbc", "reuters",
    ]

    # ── extreme numbers that are implausible ─────────────────────────
    _EXAG_RE = re.compile(
        r'(\d{6,})\s*(?:dead|killed|homeless|displaced|affected)', re.I
    )

    def detect(self, text: str) -> str:
        if not text or not text.strip():
            return "Unknown"

        t = text.lower()

        # Hard FAKE check first
        for kw in self.FAKE_KW:
            if kw in t:
                return "Fake"

        # Implausibly large numbers
        m = self._EXAG_RE.search(t)
        if m and int(m.group(1)) > 500_000:
            return "Fake"

        # Count credibility signals
        credible_hits = sum(1 for kw in self.CREDIBLE_KW if kw in t)

        # Suspicious check (skip if enough credible sources cited)
        if credible_hits < 2:
            for kw in self.SUSPICIOUS_KW:
                if kw in t:
                    return "Suspicious"

        return "Likely Real"


detector = FakeNewsDetector()


def detect_fake_news(text: str) -> str:
    return detector.detect(text)