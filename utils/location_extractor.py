# utils/location_extractor.py
import re


class LocationExtractor:
    """
    Enhanced location extractor for Pakistan emergency reports.

    Priority order:
      1. Known specific city/district match (longest match first)
      2. Province / territory match
      3. "Unknown"
    """

    # ── City / District → canonical label ─────────────────────────────
    CITIES = {
        # KPK
        "swat": "Swat, KPK",
        "mingora": "Mingora, Swat, KPK",
        "peshawar": "Peshawar, KPK",
        "abbottabad": "Abbottabad, KPK",
        "mansehra": "Mansehra, KPK",
        "chitral": "Chitral, KPK",
        "mardan": "Mardan, KPK",
        "kohistan": "Kohistan, KPK",
        "dera ismail khan": "Dera Ismail Khan, KPK",
        "di khan": "Dera Ismail Khan, KPK",
        "bannu": "Bannu, KPK",
        "nowshera": "Nowshera, KPK",
        "charsadda": "Charsadda, KPK",
        "haripur": "Haripur, KPK",
        "battagram": "Battagram, KPK",
        "shangla": "Shangla, KPK",
        "tank": "Tank, KPK",
        "karak": "Karak, KPK",
        "hangu": "Hangu, KPK",
        "kohat": "Kohat, KPK",
        "waziristan": "Waziristan, KPK",

        # Punjab
        "lahore": "Lahore, Punjab",
        "rawalpindi": "Rawalpindi, Punjab",
        "islamabad": "Islamabad",
        "multan": "Multan, Punjab",
        "faisalabad": "Faisalabad, Punjab",
        "sialkot": "Sialkot, Punjab",
        "gujranwala": "Gujranwala, Punjab",
        "gujrat": "Gujrat, Punjab",
        "sargodha": "Sargodha, Punjab",
        "sheikhupura": "Sheikhupura, Punjab",
        "rahim yar khan": "Rahim Yar Khan, Punjab",
        "bahawalpur": "Bahawalpur, Punjab",
        "jhang": "Jhang, Punjab",
        "sahiwal": "Sahiwal, Punjab",
        "dera ghazi khan": "Dera Ghazi Khan, Punjab",
        "mianwali": "Mianwali, Punjab",
        "chakwal": "Chakwal, Punjab",
        "attock": "Attock, Punjab",
        "jhelum": "Jhelum, Punjab",
        "murree": "Murree, Punjab",

        # Sindh
        "karachi": "Karachi, Sindh",
        "hyderabad": "Hyderabad, Sindh",
        "sukkur": "Sukkur, Sindh",
        "larkana": "Larkana, Sindh",
        "jacobabad": "Jacobabad, Sindh",
        "badin": "Badin, Sindh",
        "thatta": "Thatta, Sindh",
        "dadu": "Dadu, Sindh",
        "nawabshah": "Nawabshah, Sindh",
        "mirpurkhas": "Mirpurkhas, Sindh",
        "sanghar": "Sanghar, Sindh",

        # Balochistan
        "quetta": "Quetta, Balochistan",
        "gwadar": "Gwadar, Balochistan",
        "turbat": "Turbat, Balochistan",
        "khuzdar": "Khuzdar, Balochistan",
        "ziarat": "Ziarat, Balochistan",
        "pishin": "Pishin, Balochistan",
        "chaman": "Chaman, Balochistan",
        "dera bugti": "Dera Bugti, Balochistan",
        "sibi": "Sibi, Balochistan",
        "mastung": "Mastung, Balochistan",
        "kalat": "Kalat, Balochistan",
        "panjgur": "Panjgur, Balochistan",

        # Gilgit-Baltistan
        "gilgit": "Gilgit, GB",
        "skardu": "Skardu, GB",
        "hunza": "Hunza, GB",
        "ghanche": "Ghanche, GB",
        "ghizer": "Ghizer, GB",
        "astore": "Astore, GB",
        "diamer": "Diamer, GB",

        # AJK
        "muzaffarabad": "Muzaffarabad, AJK",
        "mirpur": "Mirpur, AJK",
        "rawalakot": "Rawalakot, AJK",
        "neelum": "Neelum Valley, AJK",
        "bagh": "Bagh, AJK",
        "kotli": "Kotli, AJK",
    }

    # ── Province fallback ────────────────────────────────────────────
    PROVINCES = {
        r"\bkpk\b": "Khyber Pakhtunkhwa",
        r"\bkhyber pakhtunkhwa\b": "Khyber Pakhtunkhwa",
        r"\bsindh\b": "Sindh",
        r"\bpunjab\b": "Punjab",
        r"\bbalochistan\b": "Balochistan",
        r"\bgilgit[\s-]baltistan\b": "Gilgit-Baltistan",
        r"\b(?:ajk|azad kashmir)\b": "Azad Kashmir",
        r"\bislamabad\b": "Islamabad",
    }

    def __init__(self):
        # Pre-sort by key length (longest first → more specific match wins)
        self._sorted_cities = sorted(
            self.CITIES.items(), key=lambda x: len(x[0]), reverse=True
        )

    def extract(self, text: str) -> str:
        if not text:
            return "Pakistan"

        t = text.lower()

        # 1. City/district exact word-boundary match
        for key, label in self._sorted_cities:
            if re.search(r'\b' + re.escape(key) + r'\b', t):
                return label

        # 2. Province fallback
        for pattern, label in self.PROVINCES.items():
            if re.search(pattern, t, re.I):
                return label

        return "Pakistan"


extractor = LocationExtractor()


def extract_location(text: str) -> str:
    return extractor.extract(text)