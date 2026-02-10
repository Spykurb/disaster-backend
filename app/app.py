from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.scrapers import scrape_all_sources, ScrapedNewsItem
import joblib
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware

# ------------------
# Scheduler Setup
# ------------------
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Scheduler...")
    scheduler.add_job(disaster_service.run_pipeline, 'interval', minutes=10)
    scheduler.start()
    
    # Run once on startup (in background to not block)
    asyncio.create_task(disaster_service.run_pipeline())
    
    yield
    
    # Shutdown
    print("Shutting down Scheduler...")
    scheduler.shutdown()

# ------------------
# load model
# ------------------
model = joblib.load("news_classifier.pkl")

app = FastAPI(
    title="Disaster News Classifier API",
    version="0.2",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------
# request schema
# ------------------
class NewsRequest(BaseModel):
    text: str

# ------------------
# response schema
# ------------------
class PredictionResponse(BaseModel):
    label: str
    probability: float

# ------------------
# endpoint
# ------------------

def predict_text(text: str) -> dict:
    """
    Helper function to classify text as disaster or non-disaster.
    Returns a dict with 'label' and 'probability'.
    """
    if not text:
        return {"label": "unknown", "probability": 0.0}
    
    # predict class (0 / 1)
    pred = model.predict([text])[0]
    # predict probability
    prob = model.predict_proba([text])[0]
    disaster_prob = float(prob[1])  # class = 1
    label = "disaster" if pred == 1 else "non-disaster"
    
    return {
        "label": label,
        "probability": disaster_prob
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_news(req: NewsRequest):
    result = predict_text(req.text)
    return result


# ------------------
# Location Extraction
# ------------------
from pythainlp.tag import NER

# Initialize NER engine (using 'thainer' - 1.4 version)
# This might download the model on first run
ner_engine = NER("thainer")

class LocationResponse(BaseModel):
    locations: list[str]

@app.post("/extract-location", response_model=LocationResponse)
def extract_location(req: NewsRequest):
    """
    Extract location entities (LOC) from the news text.
    """
    text = req.text
    # tag returns list of tuples. Length varies by engine/version.
    tags = ner_engine.tag(text)
    
    locations = []
    current_loc = []
    
    for item in tags:
        if len(item) == 3:
            word, pos, tag = item
        elif len(item) == 2:
            word, tag = item
        else:
            continue

        if "LOCATION" in tag:
            current_loc.append(word)
        else:
            if current_loc:
                locations.append("".join(current_loc))
                current_loc = []
    
    # Capture the last one if exists
    if current_loc:
        locations.append("".join(current_loc))
        
    # Deduplicate and clean
    unique_locations = sorted(list(set(locations)))
    
    return {"locations": unique_locations}

# ------------------
# OLT Management
# ------------------
from enum import Enum
from typing import List

class OLTStatus(str, Enum):
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

class OLT(BaseModel):
    id: str
    location: str
    status: OLTStatus = OLTStatus.ACTIVE

# Mock Database
olts_db: List[OLT] = [
    OLT(id="OLT-001", location="เชียงใหม่", status=OLTStatus.ACTIVE),
    OLT(id="OLT-002", location="กรุงเทพ", status=OLTStatus.ACTIVE),
    OLT(id="OLT-003", location="ภูเก็ต", status=OLTStatus.MAINTENANCE),
]

@app.get("/olts", response_model=List[OLT])
def get_olts():
    return olts_db

@app.post("/olts", response_model=OLT)
def create_olt(olt: OLT):
    olts_db.append(olt)
    return olt

# ------------------
# Impact Analysis
# ------------------
class ImpactResponse(BaseModel):
    is_disaster: bool
    confidence: float
    affected_locations: List[str]
    affected_olts: List[OLT]

@app.post("/analyze", response_model=ImpactResponse)
def analyze_impact(req: NewsRequest):
    text = req.text

    # 1. Check Disaster
    prob_res = model.predict_proba([text])[0]
    disaster_prob = float(prob_res[1])
    is_disaster = disaster_prob > 0.5

    affected_locations = []
    affected_olts = []

    if is_disaster:
        # 2. Extract Location
        tags = ner_engine.tag(text)
        current_loc = []
        locations = []
        for item in tags:
            if len(item) == 3:
                word, pos, tag = item
            elif len(item) == 2:
                word, tag = item
            else:
                continue

            if "LOCATION" in tag:
                current_loc.append(word)
            else:
                if current_loc:
                    locations.append("".join(current_loc))
                    current_loc = []
        if current_loc:
            locations.append("".join(current_loc))
            
        affected_locations = sorted(list(set(locations)))

        # 3. Match OLTs
        # Simple substring matching: if OLT location is in extracted locations (or vice versa)
        # For this MVP, we check if OLT location string appears in the unique locations list
        for olt in olts_db:
            # Check if any extracted location matches the OLT location
            # Using partial match might be better for "เชียงใหม่" vs "จังหวัดเชียงใหม่"
            # But here let's try direct check first.
            if any(olt.location in loc for loc in affected_locations) or \
               any(loc in olt.location for loc in affected_locations):
                affected_olts.append(olt)

    return {
        "is_disaster": is_disaster,
        "confidence": disaster_prob,
        "affected_locations": affected_locations,
        "affected_olts": affected_olts
    }


@app.post("/scrape-news", response_model=List[ScrapedNewsItem])
async def scrape_news_endpoint(limit: int = 5):
    """
    Scrape news from Dailynews, Thairath, and ThaiPBS.
    Returns a list of scraped articles enriched with prediction.
    """
    news_items = await scrape_all_sources(limit=limit)
    
    # Apply prediction
    for item in news_items:
        # Predict based on content, or title + content
        # Combining title + content usually gives better context
        text_to_predict = f"{item.title} {item.content}"
        pred_result = predict_text(text_to_predict)
        
        item.prediction_label = pred_result["label"]
        item.prediction_confidence = pred_result["probability"]
        
# ------------------
# Real-time Disaster Impact
# ------------------
from app.services.disaster_service import DisasterService, ImpactAlert, DisasterNews, AIExplanation, NearbyOLT, haversine_distance, NodeDownOLT

disaster_service = DisasterService()

@app.get("/api/olts/status/down", response_model=List[NodeDownOLT])
def get_node_down_olts_endpoint():
    """
    Get list of OLTs that are currently DOWN.
    Source: monitoring.node_down_detail (is_latest=True, batch_status='completed')
    """
    return disaster_service.get_node_down_olts()

@app.get("/api/disaster-impact", response_model=List[ImpactAlert])
def check_disaster_impact():
    """
    Get cached disaster alerts (instant).
    Background job updates this every 10 minutes.
    """
    return disaster_service.get_cached_pipeline_result()

@app.post("/api/system/refresh")
async def force_refresh():
    """Trigger immediate pipeline run."""
    await disaster_service.run_pipeline()
    return {"status": "refreshed", "count": len(disaster_service.get_cached_pipeline_result())}

@app.get("/api/monitor/node-down", response_model=List[NodeDownOLT])
def get_node_down_monitor():
    """
    Get list of OLTs that are currently DOWN.
    Source: monitoring.node_down_detail (joined with public.olts)
    """
    return disaster_service.get_node_down_olts()

@app.get("/api/flood-baseline")
async def get_flood_baseline():
    """
    Get historical flood frequency for all nodes 
    to create our own data-driven layer.
    """
    return await disaster_service.get_all_nodes_baseline()

# ------------------
# V1 API Endpoints
# ------------------

class NewsAnalysisRequest(BaseModel):
    title: str
    content: str

class ScaleInfo(BaseModel):
    scale: str
    reason: str
    keywords: List[str]

class PredictionInfo(BaseModel):
    label: str
    confidence: float

class NewsAnalysisResponse(BaseModel):
    prediction: PredictionInfo
    disaster_type: str
    extracted_locations: Dict[str, str]
    extracted_date: Optional[str]
    scale: ScaleInfo

@app.post("/api/v1/analyze/news", response_model=NewsAnalysisResponse)
def analyze_news_atomic(req: NewsAnalysisRequest):
    """
    Atomic analysis of a single news item. 
    Performs Classification, NER, Date Extraction, and Scale Analysis.
    """
    text = f"{req.title} {req.content}"
    
    # 1. Classification
    pred = predict_text(text)
    
    # 2. Extract Details
    locations = disaster_service.extract_location_details(text)
    date = disaster_service.extract_date(text)
    
    # 3. Scale Analysis
    # Construct temp object for compatibility
    temp_news = DisasterNews(
        news_id=0, title=req.title, content=req.content, 
        detail_url="", latitude=0, longitude=0,
        prediction_label=pred["label"],
        prediction_confidence=pred["probability"]
    )
    
    scale_res = disaster_service.analyze_scale(temp_news)
    
    # 4. Disaster Type (Simple mapping for now)
    # This logic matches what runs inside scrape pipeline
    disaster_type = "Unknown"
    all_types = ["ไฟไหม้ป่า", "ไฟป่า", "อัคคีภัย", "น้ำท่วม", "แผ่นดินไหว", "อุบัติเหตุ", "จราจร", "เรือชน", "ไฟไหม้"]
    for t in all_types:
        if t in text:
            disaster_type = t
            break
            
    return {
        "prediction": {"label": pred["label"], "confidence": pred["probability"]},
        "disaster_type": disaster_type,
        "extracted_locations": locations,
        "extracted_date": date,
        "scale": {
            "scale": scale_res["scale"],
            "reason": scale_res["reason"],
            "keywords": scale_res["keywords"]
        }
    }

class RiskAnalysisRequest(BaseModel):
    title: str
    content: str
    latitude: float
    longitude: float
    threshold_km: float = 50.0

@app.post("/api/v1/analyze/risk", response_model=ImpactAlert)
async def analyze_risk_e2e(req: RiskAnalysisRequest):
    """
    End-to-End Decision Endpoint.
    Calculates impact against OLTs using the Weighted Risk Model.
    """
    # 1. Construct News Object
    text = f"{req.title} {req.content}"
    pred = predict_text(text)
    
    news = DisasterNews(
        news_id=0, # Ephemeral
        title=req.title,
        content=req.content,
        detail_url="api-request",
        latitude=req.latitude,
        longitude=req.longitude,
        prediction_label=pred["label"],
        prediction_confidence=pred["probability"]
    )
    
    # 2. Extract Metadata (Type, Date, Scale)
    news.extracted_location = disaster_service.extract_location_details(text)
    news.date = disaster_service.extract_date(text)
    
    # Determine Type
    all_types = ["ไฟป่า", "อัคคีภัย", "น้ำท่วม", "แผ่นดินไหว", "อุบัติเหตุ", "จราจร", "เรือชน"]
    news.disaster_type = "Unknown"
    for t in all_types:
        if t in text:
            news.disaster_type = t
            break
            
    # Scale Analysis & RISK Calculation handled inside check_impact logic?
    # No, check_impact fetches FROM API. We need a method to check impact for a SINGLE news item.
    # We should expose `calculate_impact_for_news` in DisasterService or replicate logic.
    # Replicating logic here for transparency as "Pipeline" controller.
    
    scale_res = disaster_service.analyze_scale(news)
    
    # 3. AI Explanation Base
    ai_explain = AIExplanation(
        keywords=scale_res.get("keywords", []),
        location_confidence=0.8, # Placeholder
        scale_reason=scale_res.get("reason", "")
    )
    
    # 4. Fetch OLTs & Calculate
    # We need to fetch OLTs. DisasterService has fetch_olts but it is database bound.
    olts = disaster_service.fetch_olts()
    
    nearby_olts = []
    max_risk = 0.0
    dist_reason = ""
    
    for olt in olts:
        dist = haversine_distance(news.latitude, news.longitude, olt.latitude, olt.longitude)
        if dist <= req.threshold_km:
            risk = disaster_service.calculate_risk(
                scale=scale_res["scale"],
                distance=dist,
                disaster_type=news.disaster_type,
                probability=news.prediction_confidence
            )
            
            if risk["score"] > 0:
                if risk["score"] > max_risk:
                    max_risk = risk["score"]
                    dist_reason = risk["reason"]
                    
                nearby_olts.append(NearbyOLT(
                    olt_ip=olt.ip,
                    olt_address=olt.address,
                    distance_km=round(dist, 2),
                    risk_level=risk["level"],
                    risk_score=risk["score"]
                ))
    
    # Sort by risk
    nearby_olts.sort(key=lambda x: x.risk_score, reverse=True)
    
    ai_explain.distance_reason = dist_reason
    
    return ImpactAlert(
        disaster=news,
        nearby_olts=nearby_olts,
        ai_explanation=ai_explain
    )
