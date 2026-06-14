# utils/location_extractor.py

import re


class LocationExtractor:

    def __init__(self):

        self.locations = [

            # Major Cities
            "Karachi",
            "Lahore",
            "Islamabad",
            "Rawalpindi",
            "Peshawar",
            "Quetta",
            "Multan",
            "Faisalabad",
            "Hyderabad",
            "Sialkot",

            # Disaster-Prone Areas
            "Swat",
            "Chitral",
            "Gilgit",
            "Skardu",
            "Muzaffarabad",
            "Abbottabad",
            "Mardan",
            "Mansehra",
            "Kohistan",
            "Dera Ismail Khan",
            "Thatta",
            "Badin",
            "Sukkur",
            "Jacobabad",
            "Gwadar"
        ]

    def extract(self, text):

        if not text:
            return "Unknown"

        text_lower = text.lower()

        # Exact location matching
        for location in self.locations:

            if location.lower() in text_lower:
                return location

        return "Unknown"


# Global Object
extractor = LocationExtractor()


def extract_location(text):
    return extractor.extract(text)