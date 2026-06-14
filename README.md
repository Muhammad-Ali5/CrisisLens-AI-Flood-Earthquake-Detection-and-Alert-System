# CrisisLensAI

CrisisLensAI is a Streamlit-based disaster intelligence dashboard for flood and earthquake monitoring, prediction, mapping, analytics, and AI-assisted relief planning.

## Features

- Live-style disaster dashboard and analytics pages
- Flood/earthquake prediction workflows
- AI-generated relief planning support
- Map and reporting utilities
- Local model artifacts and datasets for experimentation

## Project structure

- app.py — main Streamlit application entry point
- pages/ — dashboard, analytics, prediction, map, and assistant pages
- utils/ — prediction, alerting, map, and AI helper modules
- models/ — trained model files and encoders
- data/ — datasets, metadata, and training assets
- tests/ — basic verification tests
- notebooks/ — exploratory and experimental notebooks/scripts

## Requirements

- Python 3.10 or newer
- pip
- A Groq API key for AI-driven relief assistance

## Setup

1. Create and activate a virtual environment:
   python -m venv venv
   .\venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Create your local environment file:
   copy .env.example .env
   Then fill in your real secrets in .env.

4. Run the app:
   streamlit run app.py

## Environment variables

The project uses a local .env file for secret values. A ready-to-copy template is included in .env.example.

Recommended variables:
- GROQ_API_KEY — required for the relief planning AI assistant
- GOOGLE_API_KEY — optional, for Google GenAI related integrations
- GOOGLE_CLOUD_PROJECT — optional, for Google Cloud integrations

## Security notes

- Do not commit your real .env file.
- Keep API keys and credentials private.
- If you are preparing a public repository, add .env to .gitignore and remove any secret values from history.

## Troubleshooting

- If the app fails to start, confirm that all packages in requirements.txt are installed.
- If the AI assistant does not respond, verify that GROQ_API_KEY is set in your .env file.
- If you are using a fresh clone, recreate .env from .env.example before launching the app.
