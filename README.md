# Disaster Risk Dashboard - Backend

This repository contains the backend API and AI logic for the Disaster Risk Dashboard.

## Features
- **FastAPI Core**: High-performance asynchronous API.
- **AI News Classification**: Scrapes and classifies disaster news using a joblib-trained model.
- **GISTDA Integration**: Fetches historical flood recurrence data for OLT nodes.
- **Power Outage Tracking**: Integrates real-time PEA power outage notifications.
- **Impact Analysis**: Weighted risk scoring combining news scale, distance, and flood history.

## Tech Stack
- Python 3.9+
- FastAPI & Uvicorn
- BeautifulSoup4 & httpx (Scraping)
- PyThaiNLP (Named Entity Recognition)
- Scikit-learn (Classifier inference)
- APScheduler (Periodic news monitoring)

## Setup & Running
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
   ```

## Development
- `app/services`: Business logic (Disaster, PEA, Longdo services).
- `app/scrapers.py`: News scraping logic for Thai news sources.
- `news_classifier.pkl`: Pre-trained model for disaster news detection.
