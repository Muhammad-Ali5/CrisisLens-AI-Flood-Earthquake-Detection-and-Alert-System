# CrisisLensAI

CrisisLensAI is a Streamlit-based disaster intelligence application for flood and earthquake monitoring, predictions, analytics, mapping, and AI-assisted crisis response support.

## Overview

This project brings together:
- a multi-page Streamlit dashboard,
- disaster classification and severity prediction utilities,
- map and alerting components,
- AI-based relief planning helpers,
- local datasets and trained model artifacts.

It is designed to help you explore disaster events, review predictions, and build crisis-response insights from local data and AI services.

---

## Features

- Interactive dashboard and analytics overview
- Flood and earthquake-related prediction workflows
- AI assistant for relief planning and crisis briefing support
- Map-based visualization and alerting components
- Local data and model assets for experimentation and demos

---

## Project Structure

- app.py — main Streamlit entry point
- pages/ — dashboard, analytics, prediction, live map, and AI assistant interface
- utils/ — reusable logic for classification, severity prediction, alerts, maps, and AI integration
- models/ — pre-trained model files and encoders
- data/ — datasets, metadata, training assets, and reports
- tests/ — automated tests for model and fallback logic
- notebooks/ — exploratory scripts and pipeline experimentation
- requirements.txt — Python dependencies
- .env.example — example environment file for local secrets

---

## Requirements

- Python 3.10 or newer
- pip
- A working internet connection for package installation and any AI service access
- A Groq API key for AI-assisted relief planning features

---

## Installation

1. Clone the project repository.

2. Create and activate a virtual environment:

   python -m venv venv
   .\venv\Scripts\activate

3. Install dependencies:

   pip install -r requirements.txt

4. Create your local environment file:

   copy .env.example .env

   Then update the values in .env with your own real credentials.

---

## Running the App

Start the Streamlit application from the project root:

streamlit run app.py

The app will open in your browser and load the dashboard pages from the pages/ folder.

---

## Environment Variables

This project uses a local .env file for secrets and service credentials.

A safe template is provided in .env.example.

Recommended variables:
- GROQ_API_KEY — required for the AI relief planning assistant
- GOOGLE_API_KEY — optional, for Google GenAI-related integrations
- GOOGLE_CLOUD_PROJECT — optional, for Google Cloud integration setup

Important:
- Do not commit your real .env file to Git.
- Keep all API keys and credentials private.

---

## Data and Models

The following folders contain core project assets:
- data/ — CSV datasets, metadata, reports, and training data
- models/ — model binaries and encoders used by the prediction workflow

If you plan to train or replace models, keep the naming and expected format of the existing files consistent with the current code.

---

## Testing

You can run the project tests from the repository root:

pytest

This helps verify the model loader, fallback logic, and AI-related behavior.

---

## Troubleshooting

If the app does not start:
- confirm that all packages in requirements.txt are installed,
- verify that Python 3.10+ is being used,
- ensure that .env exists and includes the needed keys.

If the AI assistant does not work:
- confirm the Groq API key is present in .env,
- verify your network access,
- check whether the related module is importing the configured key correctly.

---

## Security Notes

- Use .env for local secrets only.
- Do not upload real API keys, credentials, SMTP settings, or personal data to GitHub.
- If you are pushing to a public repository, make sure .env is ignored and not present in the commit history.

---

## Deployment Notes

For local development, Streamlit is sufficient. If you want to deploy this project later:
- set environment variables in the hosting platform,
- install requirements.txt on the target environment,
- ensure model files and datasets are available in the deployed app.

---

## Summary

CrisisLensAI is a practical starter project for disaster intelligence and AI-assisted response workflows. It is suitable for local development, demos, experimentation, and future deployment with real APIs and models.

