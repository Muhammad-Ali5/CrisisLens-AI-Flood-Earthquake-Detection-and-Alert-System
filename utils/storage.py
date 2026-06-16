import json
import os

DATA_DIR = "data"
DATA_FILE = os.path.join(
    DATA_DIR,
    "incidents.json"
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

if not os.path.exists(DATA_FILE):

    with open(
        DATA_FILE,
        "w"
    ) as f:

        json.dump([], f)


def load_incidents():

    with open(
        DATA_FILE,
        "r"
    ) as f:

        return json.load(f)


def save_incident(incident):

    incidents = load_incidents()

    incidents.append(
        incident
    )

    with open(
        DATA_FILE,
        "w"
    ) as f:

        json.dump(
            incidents,
            f,
            indent=4
        )