# utils/alert_engine.py

class AlertEngine:

    def generate_alerts(
        self,
        disaster_type,
        location,
        severity,
        authenticity
    ):

        if authenticity and authenticity.lower() in ["fake", "suspicious"]:

            return {
                "citizen": "⚠ Report requires verification.",
                "ngo": "⚠ Verify report before deployment.",
                "government": "⚠ Verification required before action."
            }

        citizen_alert = self._citizen_alert(
            disaster_type,
            location,
            severity
        )

        ngo_alert = self._ngo_alert(
            disaster_type,
            location,
            severity
        )

        gov_alert = self._government_alert(
            disaster_type,
            location,
            severity
        )

        return {
            "citizen": citizen_alert,
            "ngo": ngo_alert,
            "government": gov_alert
        }

    # ===================================
    # CITIZENS
    # ===================================

    def _citizen_alert(
        self,
        disaster_type,
        location,
        severity
    ):

        disaster_type = disaster_type.lower()

        if disaster_type == "flood":

            return f"""
🚨 FLOOD ALERT

Location: {location}
Severity: {severity}

ENGLISH:

• Move to higher ground immediately
• Avoid crossing flood water
• Keep drinking water stored
• Prepare emergency supplies
• Follow official instructions
• Call 1122 in emergencies

اردو:

• فوراً محفوظ اور اونچی جگہ منتقل ہوں
• سیلابی پانی میں داخل نہ ہوں
• پینے کا صاف پانی محفوظ رکھیں
• ہنگامی سامان تیار رکھیں
• سرکاری ہدایات پر عمل کریں
• ایمرجنسی میں 1122 پر کال کریں
"""

        elif disaster_type == "earthquake":

            return f"""
🚨 EARTHQUAKE ALERT

Location: {location}
Severity: {severity}

ENGLISH:

• Drop, Cover and Hold On
• Stay away from windows
• Move to open areas after shaking
• Avoid damaged buildings
• Keep emergency kit ready

اردو:

• جھک جائیں، پناہ لیں، مضبوطی سے پکڑیں
• کھڑکیوں سے دور رہیں
• زلزلہ رکنے کے بعد کھلی جگہ جائیں
• متاثرہ عمارتوں سے دور رہیں
• ہنگامی سامان تیار رکھیں
"""

        return "General Emergency Alert"

    # ===================================
    # NGO ALERT
    # ===================================

    def _ngo_alert(
        self,
        disaster_type,
        location,
        severity
    ):

        return f"""
🚨 NGO PRIORITY RESPONSE ALERT

Disaster: {disaster_type}
Location: {location}
Severity: {severity}

Recommended Actions:

1. Deploy assessment teams
2. Prioritize vulnerable populations
3. Establish temporary shelters
4. Arrange food distribution
5. Arrange clean drinking water
6. Deploy medical support
7. Coordinate with local authorities

Priority Area:
{location}

Response Level:
{severity}
"""

    # ===================================
    # GOVERNMENT ALERT
    # ===================================

    def _government_alert(
        self,
        disaster_type,
        location,
        severity
    ):

        return f"""
🚨 GOVERNMENT RESPONSE ALERT

Disaster: {disaster_type}
Location: {location}
Severity: {severity}

Immediate Actions:

• Activate emergency response center
• Coordinate NDMA and PDMA
• Deploy Rescue 1122 teams
• Monitor hospitals
• Ensure law and order
• Monitor transportation routes

Priority Deployment Zone:
{location}

Recommended Response Level:
{severity}
"""


alert_engine = AlertEngine()