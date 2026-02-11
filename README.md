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
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Spykurb/disaster-backend.git
   cd disaster-backend
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and fill in your database credentials and **Longdo Map API Key**.
4. **Start the server**:
   ```bash
   uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
   ```

## Pipeline Operations
- **Auto-Run**: The disaster monitoring pipeline triggers automatically upon server startup.
- **Scheduled**: By default, the pipeline runs every **10 minutes** to fetch the latest news and power outages.
- **Manual Trigger**: You can force an update by sending a POST request to `/api/system/refresh`.

## Development
- `app/services`: Business logic (Disaster, PEA, Longdo services).
- `app/scrapers.py`: News scraping logic for Thai news sources.
- `news_classifier.pkl`: Pre-trained model for disaster news detection.
