# utils/maps.py

import folium
from geopy.geocoders import Nominatim


class DisasterMap:

    def __init__(self):
        self.geolocator = Nominatim(
            user_agent="crisislens_ai"
        )

    def get_coordinates(self, location):

        try:

            loc = self.geolocator.geocode(
                f"{location}, Pakistan"
            )

            if loc:

                return (
                    loc.latitude,
                    loc.longitude
                )

        except:
            pass

        return (
            33.6844,
            73.0479
        )

    def create_map(
        self,
        location,
        disaster_type,
        severity
    ):

        lat, lon = self.get_coordinates(location)

        m = folium.Map(
            location=[lat, lon],
            zoom_start=8
        )

        color = "green"

        if severity.lower() == "medium":
            color = "orange"

        elif severity.lower() == "high":
            color = "red"

        folium.Marker(
            [lat, lon],
            popup=f"""
            {disaster_type}
            <br>
            {location}
            <br>
            Severity: {severity}
            """,
            tooltip=location,
            icon=folium.Icon(
                color=color,
                icon="info-sign"
            )
        ).add_to(m)

        folium.Circle(
            location=[lat, lon],
            radius=5000,
            color=color,
            fill=True
        ).add_to(m)

        return m


map_manager = DisasterMap()