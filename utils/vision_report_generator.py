import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
RETRY_DELAY = 5


def generate_incident_report(image_path):

    prompt = """
You are an Emergency Disaster Intelligence Analyst for Pakistan.

Analyze the uploaded image carefully and extract every detail.

CRITICAL — LOCATION DETECTION:
Look for any text (signboards, vehicle plates, building names), landmarks,
terrain features, or cultural clues that reveal the SPECIFIC city, district,
or province in Pakistan. If no specific location is visible, infer from
vegetation, architecture, terrain, or any contextual clue. Be as precise
as possible — village, city, district, or at minimum the province.

Tasks:

1. Identify SPECIFIC location (city, district, province in Pakistan)
2. Identify disaster type
   (Flood, Earthquake, Fire, Landslide, Storm, Accident)
3. Estimate severity
   (Low, Medium, High, Critical)
4. Describe visible damage in detail
5. Mention possible risks to people and infrastructure
6. Generate a complete incident report.

Return format:

LOCATION:
[Specific city/district/province — not just "Pakistan"]

DISASTER TYPE:
...

SEVERITY:
...

VISIBLE DAMAGE:
...

RISKS:
...

INCIDENT REPORT:
...
"""

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            uploaded_file = client.files.upload(file=image_path)
            response = client.models.generate_content(
                model=MODEL,
                contents=[uploaded_file, prompt]
            )
            return response.text
        except Exception as e:
            last_exc = e
            err_str = str(e)
            if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
            raise

    raise last_exc