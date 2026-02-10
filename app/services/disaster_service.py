import httpx
from app.services.pea_service import PEAService
from app.services.longdo_service import LongdoService
from datetime import datetime
import json
import psycopg2
import math
import joblib
import asyncio
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from bs4 import BeautifulSoup
from pythainlp.tag import NER
import re
import os

# Configuration
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "postgres")

DISASTER_API = "https://dpmreporter.disaster.go.th/portal/services/news_map/get_list.php?type=0&province=0"
MODEL_PATH = "news_classifier.pkl"

# Load Model Global
try:
    CLASSIFIER = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    CLASSIFIER = None

class DisasterNews(BaseModel):
    news_id: Any
    title: str
    date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    latitude: float
    longitude: float
    detail_url: str
    target_type: Optional[str] = None # For specific targeting if needed
    # New fields
    content: Optional[str] = None
    extracted_location: Dict[str, str] = {}
    disaster_type: Optional[str] = "Unknown" 
    prediction_label: Optional[str] = None
    prediction_confidence: float = 0.0
    impact_radius: float = 0.0 # Scale-based radius for map
    source: str = "DPM" # "DPM" or "PEA" or "Longdo"

class OLTInfo(BaseModel):
    id: int
    ip: str
    address: str
    latitude: float
    longitude: float
    elevation: float = 0.0
    subdistrict: str = ""
    district: str = ""
    province: str = ""
    flood_frequency: int = 0

class NearbyOLT(BaseModel):
    olt_ip: str
    olt_address: str
    distance_km: float
    risk_level: str = "Low"
    risk_score: float = 0.0
    flood_frequency: int = 0
    latitude: float = 0.0
    longitude: float = 0.0

class AIExplanation(BaseModel):
    severity_level: int = 1
    keywords: List[str] = []
    location_confidence: float = 1.0 # Placeholder
    scale_reason: str = ""
    distance_reason: str = ""

class NodeDownOLT(BaseModel):
    id: int
    ip_address: str
    node_address: Optional[str] = None
    service_location: Optional[str] = None
    status: Optional[str] = None
    last_updated: Optional[datetime] = None
    latitude: float
    longitude: float
    province: Optional[str] = None
    rsl: Optional[str] = None

class ImpactAlert(BaseModel):
    disaster: DisasterNews
    nearby_olts: List[NearbyOLT]
    ai_explanation: Optional[AIExplanation] = None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class DisasterService:
    def __init__(self):
        self.client = httpx.AsyncClient(headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }, verify=False, timeout=10.0)
        try:
            self.ner = NER("thainer")
        except:
            self.ner = None
        
        self.pea_service = PEAService()
        self.longdo_service = LongdoService()
        self.baseline_cache = {} # Cache for flood recurrence: {ip: frequency}
        
        # Cache for Scheduler
        self.latest_alerts: List[ImpactAlert] = []

    async def fetch_disasters(self) -> List[DisasterNews]:
        """Fetch latest disasters from the internal API."""
        print("Fetching disaster data...")
        try:
            resp = await self.client.get(DISASTER_API)
            if resp.status_code != 200:
                print(f"API Error: {resp.status_code}")
                return []
            
            data = resp.json()
            items = data.get('data', {}).get('data', [])
            
            results = []
            for item in items:
                try:
                    # Initialize extracted_location with API data if available
                    province_from_api = item.get('province_name', '').strip()
                    initial_loc = {'tambon': '', 'amphoe': '', 'province': province_from_api}
                    
                    results.append(DisasterNews(
                        news_id=item.get('news_id'),
                        title=item.get('news_title'),
                        date=item.get('news_date'),
                        latitude=float(item.get('latitude', 0)),
                        longitude=float(item.get('longtitude', 0)),
                        detail_url=f"https://dpmreporter.disaster.go.th/portal/disaster-news/{item.get('news_id')}",
                        extracted_location=initial_loc
                    ))
                except (ValueError, TypeError) as e:
                    # indicate error but continue
                    continue
            print(f"Fetched {len(results)} disasters.")
            return results
        except Exception as e:
            print(f"Error fetching disasters: {e}")
            return []

    async def get_all_nodes_baseline(self) -> List[Dict[str, Any]]:
        """
        Fetch baseline flood frequency for EVERY known node concurrently.
        """
        olts = self.fetch_olts()
        
        async def fetch_one(olt):
            if olt.ip in self.baseline_cache:
                return {"ip": olt.ip, "lat": olt.latitude, "lon": olt.longitude, "freq": self.baseline_cache[olt.ip]}
            
            freq = await self.longdo_service.fetch_flood_recurrence(olt.latitude, olt.longitude)
            self.baseline_cache[olt.ip] = freq
            return {"ip": olt.ip, "lat": olt.latitude, "lon": olt.longitude, "freq": freq}

        # Concurrency limit to avoid hitting GISTDA API too hard
        semaphore = asyncio.Semaphore(10)
        async def fetch_with_sem(olt):
            async with semaphore:
                return await fetch_one(olt)

        tasks = [fetch_with_sem(olt) for olt in olts]
        raw_results = await asyncio.gather(*tasks)
        
        results = []
        for r in raw_results:
            if r["freq"] > 0:
                results.append({
                    "ip": r["ip"],
                    "lat": r["lat"],
                    "lon": r["lon"],
                    "freq": r["freq"],
                    "risk_level": "High" if r["freq"] > 10 else "Medium" if r["freq"] > 5 else "Low"
                })
        return results

    async def close(self):
        await self.client.aclose()

    async def scrape_detail(self, news: DisasterNews):
        """Scrape detail page to get disaster type and content."""
        try:
            resp = await self.client.get(news.detail_url)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                # Extract Description/Content
                desc_el = soup.select_one(".description textarea")
                if desc_el:
                    news.content = desc_el.text.strip()
                    # Extract address from content
                    new_loc = self.extract_location_details(news.content)
                    
                    # Merge with existing province from API if NER missed it
                    if not new_loc.get('province') and news.extracted_location and news.extracted_location.get('province'):
                         new_loc['province'] = news.extracted_location['province']
                    
                    news.extracted_location = new_loc
                    
                    # Extract Date if missing
                    if not news.date:
                        extracted_date = self.extract_date(news.content)
                        if extracted_date:
                            news.date = extracted_date

                # Selector: .detail-type > div:first-child
                type_el = soup.select_one(".detail-type > div")
                if type_el:
                     news.disaster_type = type_el.get_text(strip=True)
                else:
                     # Attempt fallback
                     icon_fire = soup.select_one(".icon-fire")
                     if icon_fire:
                         news.disaster_type = "อัคคีภัย (Inferred)"
        except Exception as e:
            print(f"Error scraping detail {news.news_id}: {e}")

    def classify(self, news: DisasterNews):
        """Run model on title and content."""
        if CLASSIFIER:
            try:
                # Use title + content (if available) for better recall as found in experiments
                text = f"{news.title} {news.content if news.content else ''}"
                pred = CLASSIFIER.predict([text])[0]
                proba = CLASSIFIER.predict_proba([text])[0]
                
                news.prediction_label = "Disaster" if pred == 1 else "Non-Disaster"
                news.prediction_confidence = float(proba[pred])
            except Exception as e:
                print(f"Classification error: {e}")

    def _extract_count(self, text: str, keywords: List[str]) -> int:
        """Helper to extract numbers associated with specific keywords (e.g., '10 ราย', 'บ้าน 5หลัง')."""
        count = 0
        for kw in keywords:
            # Pattern: [Number] [Keyword] (e.g., 5 ราย)
            pattern1 = rf"(\d+)\s*{kw}"
            matches1 = re.findall(pattern1, text)
            for m in matches1:
                count = max(count, int(m))
            # Pattern: [Keyword] [Number] (e.g., บ้าน 50)
            pattern2 = rf"{kw}\s*(\d+)"
            matches2 = re.findall(pattern2, text)
            for m in matches2:
                count = max(count, int(m))
        return count

    def analyze_scale(self, news: DisasterNews) -> Dict[str, Any]:
        """
        Analyze if the event is Level 1-4 scale based on Thai National Disaster Plan 
        and international standards (INFORM).
        """
        text = f"{news.title} {news.content if news.content else ''}"
        text_lower = text.lower()
        
        # 1. Keywords Groups
        large_keywords = ["น้ำท่วม", "พายุ", "แผ่นดินไหว", "น้ำป่า", "อุทกภัย", "วาตภัย", "ดินโคลนถล่ม", "ไฟป่า", "ไฟไหม้ป่า", "พื้นที่ป่า"]
        local_keywords = ["อัคคีภัย", "ไฟไหม้", "รถชน", "อุบัติเหตุ", "เฉี่ยวชน", "เพลิงไหม้", "เรือชน", "บาดเจ็บ", "ติดหน้าผา", "ตกผา", "กระโดดร่ม"]
        
        # 2. Extract and Filter Locations (Rescue Context Filtering)
        amphoe_matches = re.finditer(r"(?:อำเภอ|เขต)\s*([^\s\d,().]{2,})", text)
        unique_amphoes = set()
        rescue_context = ["ประจำ", "สังกัด", "ประสาน", "สนับสนุน", "เจ้าหน้าที่", "ปภ", "หน่วย", "ทีม"]
        
        for match in amphoe_matches:
            name = match.group(1).strip().replace("จังหวัด", "").strip()
            start_idx = max(0, match.start() - 30)
            context_before = text[start_idx:match.start()].lower()
            is_rescue = any(ctx in context_before for ctx in rescue_context)
            if not is_rescue and len(name) > 1:
                unique_amphoes.add(name)
        amphoe_count = len(unique_amphoes)

        # 3. Administrative Level Detection (Command/Authority)
        has_pm = any(k in text for k in ["นายกรัฐมนตรี", "นายกฯ", "ครม."])
        has_national = any(k in text for k in ["อธิบดี", "รัฐมนตรี"])
        has_provincial = any(k in text for k in ["ผู้ว่า", "ปภ.จังหวัด"])
        
        # Define active command (not just reporting)
        is_active_national = has_national and any(k in text for k in ["สั่งการ", "ลงพื้นที่", "ประธาน"])

        # 4. Impact Metrics Extraction
        deaths = self._extract_count(text, ["เสียชีวิต", "ศพ", "ผู้เสียชีวิต"])
        houses = self._extract_count(text, ["หลังคาเรือน", "บ้าน", "ครัวเรือน"])
        
        # 5. Determine Level (Thai Standard Levels 1-4)
        level = 1
        reason_parts = []
        
        # Level 4: Catastrophic (National Command / Extreme Impact)
        if has_pm or deaths >= 50 or "สเกลใหญ่" in text:
            level = 4
            reason_parts.append("Level 4: Catastrophic (National/PM Command)")
        # Level 3: Large (National DPM / Widespread Impact)
        elif is_active_national or deaths >= 10 or houses >= 100 or any(k in text_lower for k in ["วงกว้าง", "หลายพื้นที่", "หลายอำเภอ"]):
            level = 3
            reason_parts.append("Level 3: Large (National DPM Level)")
        # Level 2: Medium (Provincial Command / Multi-District)
        elif has_provincial or deaths >= 1 or houses >= 5 or amphoe_count > 1:
            level = 2
            reason_parts.append("Level 2: Medium (Provincial Level)")
        # Level 1: Small (Local/District level)
        else:
            level = 1
            reason_parts.append("Level 1: Small (Local/District Level)")

        # 6. Safety Capping for specific local incidents
        found_local = [k for k in local_keywords if k in text_lower]
        if found_local and level > 2 and deaths < 10:
             level = 2
             reason_parts.append("(Local event capped at Level 2)")

        # 7. Map to Display Scale (Backward compatibility for frontend)
        scale_map = {1: "SMALL", 2: "MEDIUM", 3: "LARGE", 4: "LARGE"}
        
        # Construct Detailed Reason
        stats = []
        if deaths > 0: stats.append(f"{deaths} dead")
        if houses > 0: stats.append(f"{houses} houses")
        if amphoe_count > 1: stats.append(f"{amphoe_count} districts")
        
        final_reason = reason_parts[0]
        if stats:
            final_reason += f" - Impact: {', '.join(stats)}"
        
        # Keywords for explanation
        found_keywords = [k for k in (large_keywords + local_keywords) if k in text_lower]

        return {
            "severity_level": level,
            "scale": scale_map[level],
            "keywords": found_keywords,
            "reason": final_reason
        }

    def calculate_risk(self, scale: str, distance: float, disaster_type: str, probability: float, severity_level: int = 1, elevation: float = 0.0) -> Dict[str, Any]:
        """
        Calculate Risk Score using Weighted Formula (Semi-AI).
        Modified to use severity_level for radius and weighting, and elevation for flood risk.
        """
        scale = scale.upper()
        
        # 1. Weights
        W_DIST = 0.40
        W_SCALE = 0.25
        W_TYPE = 0.25
        W_PROB = 0.10
        
        # 2. Score Factors (0.0 - 1.0)
        P = probability
        
        # Factor S: Scale (now influenced by level)
        S = severity_level / 4.0 
        
        # Factor T: Type
        high_risk_types = ["ไฟป่า", "อัคคีภัย", "ดินโคลนถล่ม", "ไฟไหม้", "เพลิงไหม้"]
        medium_risk_types = ["น้ำท่วม", "พายุ", "วาตภัย", "อุทกภัย"]
        
        T = 0.3
        dtype_lower = disaster_type.lower() if disaster_type else ""
        if any(t in dtype_lower for t in high_risk_types):
            T = 1.0
        elif any(t in dtype_lower for t in medium_risk_types):
            T = 0.7
            
        # Factor D: Distance (Dynamic Radius based on Level)
        radius_map = {1: 2.0, 2: 5.0, 3: 15.0, 4: 50.0}
        max_radius = radius_map.get(severity_level, 2.0)
        
        if distance > max_radius:
            D = 0.0
        else:
            D = 1.0 - (distance / max_radius)
        
        # 3. Terrain Factor (Elevation) - specifically for floods
        terrain_multiplier = 1.0
        if any(k in dtype_lower for k in ["น้ำท่วม", "พายุ", "อุทกภัย"]):
            if elevation < 2.0: # Very low land
                terrain_multiplier = 1.5
            elif elevation < 10.0:
                terrain_multiplier = 1.2
            elif elevation > 100.0: # Highland
                terrain_multiplier = 0.8
        
        # 4. Calculate Weighted Score
        base_impact = (0.4 * S) + (0.4 * T) + (0.2 * P)
        final_score = base_impact * D * terrain_multiplier * 100.0
        
        level_label = "Low"
        if final_score >= 75: level_label = "High"
        elif final_score >= 40: level_label = "Medium"
        
        return {
            "level": level_label, 
            "score": round(final_score, 2), 
            "reason": f"Level {severity_level} severity at {distance:.1f}km (Elev: {elevation}m)",
            "radius": max_radius
        }

    def extract_location_details(self, text: str) -> Dict[str, str]:
        """Use NER and refined Regex to extract Tambon, Amphoe, Province."""
        if not text:
            return {}
            
        location_info = {"tambon": "", "amphoe": "", "province": ""}
        blacklist = ["รับผิดชอบ", "พื้นที่", "ปฏิบัติการ", "ปภ", "หน่วย", "บริการ", "กู้ภัย", "เกิดเหตุ", "รายงาน"]
        
        from pythainlp.tokenize import word_tokenize

        def clean_val(val):
            if not val: return ""
            tokens = word_tokenize(val, engine="newmm")
            if not tokens: return ""
            
            stop_words = ["ได้รับ", "แจ้ง", "เกิด", "มี", "พบ", "รถ", "ถนน", "บริเวณ", "ที่", "ซึ่ง", "โดย", "ทำการ", "อาสากู้ภัย"]
            full_blacklist = set(blacklist + stop_words)
            
            kept_tokens = []
            for t in tokens:
                is_stop = False
                for sw in full_blacklist:
                    if sw in t:
                        is_stop = True
                        break
                if is_stop:
                    break
                kept_tokens.append(t)
            
            if not kept_tokens:
                return ""
            
            candidate = "".join(kept_tokens)
            candidate = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z0-9\s]", "", candidate).strip()
            if len(candidate) < 2:
                return ""
            return candidate

        try:
            m_tambon = re.search(r"(?:ตำบล|แขวง|ต\.)\s*((?:(?!อ\.|จ\.|อำเภอ|จังหวัด|เขต)[^\s\d,()./\-])+)", text)
            if m_tambon: location_info["tambon"] = clean_val(m_tambon.group(1))
            
            m_amphoe = re.search(r"(?:อำเภอ|เขต|อ\.)\s*((?:(?!จ\.|จังหวัด)[^\s\d,()./\-])+)", text)
            if m_amphoe: location_info["amphoe"] = clean_val(m_amphoe.group(1))
            
            m_province = re.search(r"(?:จังหวัด|จ\.)\s*([^\s\d,()./\-]+)", text)
            if m_province: location_info["province"] = clean_val(m_province.group(1))
            
            if not all(location_info.values()) and self.ner:
                tags = self.ner.tag(text)
                loc_parts = []
                for word, tag in tags:
                    if "LOCATION" in tag:
                        loc_parts.append(word)
                    else:
                        if loc_parts:
                            full_loc = "".join(loc_parts)
                            if "ตำบล" in full_loc and not location_info["tambon"]:
                                location_info["tambon"] = clean_val(full_loc.replace("ตำบล", ""))
                            elif "แขวง" in full_loc and not location_info["tambon"]:
                                location_info["tambon"] = clean_val(full_loc.replace("แขวง", ""))
                            elif "อำเภอ" in full_loc and not location_info["amphoe"]:
                                location_info["amphoe"] = clean_val(full_loc.replace("อำเภอ", ""))
                            elif "เขต" in full_loc and not location_info["amphoe"]:
                                location_info["amphoe"] = clean_val(full_loc.replace("เขต", ""))
                            elif "จังหวัด" in full_loc and not location_info["province"]:
                                location_info["province"] = clean_val(full_loc.replace("จังหวัด", ""))
                            elif "กรุงเทพ" in full_loc:
                                location_info["province"] = "กรุงเทพมหานคร"
                            loc_parts = []

            if any(k in text for k in ["กรุงเทพ", "กทม"]):
                if not location_info["province"]:
                    location_info["province"] = "กรุงเทพมหานคร"
        except Exception as e:
            print(f"NER Extraction Error: {e}")
        return location_info

    def extract_date(self, text: str) -> Optional[str]:
        """Extract date from text using patterns and NER."""
        if not text:
            return None
        thai_months = "มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม"
        thai_abbr_months = r"ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\."
        pattern1 = fr"(\d{{1,2}}\s+(?:{thai_months})\s*\d{{2,4}})"
        m1 = re.search(pattern1, text)
        if m1: return m1.group(1)
        pattern2 = fr"(\d{{1,2}}\s+(?:{thai_abbr_months})\s*\d{{2,4}})"
        m2 = re.search(pattern2, text)
        if m2: return m2.group(1)
        pattern3 = r"(\d{1,2}/\d{1,2}/\d{2,4})"
        m3 = re.search(pattern3, text)
        if m3: return m3.group(1)
        if self.ner:
            try:
                tags = self.ner.tag(text)
                for word, tag in tags:
                    if "DATE" in tag:
                        if any(c.isdigit() for c in word):
                            return word
            except:
                pass
        return None

    def fetch_olts(self) -> List[OLTInfo]:
        """Fetch OLTs with coordinates from PostgreSQL."""
        print("Fetching OLTs from DB...")
        olts = []
        conn = None
        try:
            conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
            cur = conn.cursor()
            query = "SELECT id, ip_address, service_location, latitude, longitude FROM public.olts WHERE latitude IS NOT NULL AND longitude IS NOT NULL;"
            cur.execute(query)
            rows = cur.fetchall()
            for row in rows:
                try:
                    olts.append(OLTInfo(id=row[0], ip=row[1] if row[1] else "Unknown IP", address=row[2] if row[2] else "Unknown Address", latitude=float(row[3]), longitude=float(row[4])))
                except ValueError: continue
            print(f"Loaded {len(olts)} OLTs.")
        except Exception as e: print(f"Database Error: {e}")
        finally:
            if conn: conn.close()
        return olts

    def get_node_down_olts(self) -> List[NodeDownOLT]:
        """Query OLTs that are currently DOWN based on monitoring table."""
        results = []
        conn = None
        try:
            conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
            cur = conn.cursor()
            query = """
                SELECT id, node_address, rsl, ip_address, latitude, longitude, location_point, service_location, province, olt_brand, is_active, status, last_updated, created_at, has_coordinates, coordinate_source, province_id
                FROM public.olts WHERE ip_address IN (
                    SELECT  ip_address
                    FROM monitoring.node_down_detail WHERE is_latest = TRUE AND batch_status = 'completed' 
                );
            """
            cur.execute(query)
            rows = cur.fetchall()
            for row in rows:
                try:
                    results.append(NodeDownOLT(id=row[0], ip_address=row[3], node_address=row[1], service_location=row[7], status=row[11], last_updated=row[12], latitude=float(row[4]) if row[4] else 0.0, longitude=float(row[5]) if row[5] else 0.0, province=row[8], rsl=row[2]))
                except Exception as e: continue
            print(f"Found {len(results)} Node Down OLTs.")
        except Exception as e: print(f"Database Error (Node Down): {e}")
        finally:
            if conn: conn.close()
        return results

    async def run_pipeline(self, distance_threshold_km: float = 50.0) -> List[ImpactAlert]:
        """Main flow updated for severity levels."""
        print("Pipeline running...")
        disasters = await self.fetch_disasters()
        try:
            pea_outages = await self.pea_service.run_pipeline()
            for po in pea_outages:
                disasters.append(DisasterNews(
                    news_id=po.id, title=po.title, date=po.start_date, 
                    start_time=po.start_date, end_time=po.end_date,
                    latitude=po.latitude, longitude=po.longitude,
                    detail_url=f"https://eservice.pea.co.th/PowerOutage/Home/Detail/{po.id}", content=po.location,
                    disaster_type="ไฟฟ้าดับ (PEA)", prediction_label="Disaster", prediction_confidence=1.0, impact_radius=2.0, source="PEA"
                ))
        except Exception as e: print(f"Failed to integrate PEA data: {e}")

        olts = self.fetch_olts()
        alerts = []
        
        # Concurrently fetch details and Longdo incidents
        tasks = [self.scrape_detail(d) for d in disasters if d.source == "DPM"]
        tasks.append(self.longdo_service.fetch_flood_incidents())
        
        results = await asyncio.gather(*tasks)
        longdo_incidents = results[-1] if results else []
        
        # Add Longdo Incidents to disasters
        for inc in longdo_incidents:
            try:
                # Longdo incidents often have 'title', 'lat', 'lon'
                lat = float(inc.get('lat', 0))
                lon = float(inc.get('lon', 0))
                if lat and lon:
                    disasters.append(DisasterNews(
                        news_id=f"longdo_{inc.get('id', 'unk')}",
                        title=inc.get('title', 'น้ำท่วมขัง (Longdo Live)'),
                        latitude=lat, longitude=lon,
                        detail_url="https://flood.longdo.com",
                        disaster_type="น้ำท่วมขัง (iTIC)",
                        prediction_label="Disaster",
                        prediction_confidence=0.9,
                        impact_radius=2.0,
                        source="Longdo"
                    ))
            except: continue

        for news in disasters:
            if news.source == "DPM": self.classify(news)
            
            # --- UPDATED SCALE ANALYSIS ---
            scale_info = self.analyze_scale(news)
            scale = scale_info["scale"]
            severity = scale_info.get("severity_level", 1)
            
            nearby = []
            max_risk_score = 0.0
            primary_risk_reason = ""
            
            for olt in olts:
                dist = haversine_distance(news.latitude, news.longitude, olt.latitude, olt.longitude)
                
                # Dynamic Radius
                radius_map = {1: 2.0, 2: 5.0, 3: 15.0, 4: 50.0}
                max_radius = radius_map.get(severity, 2.0)

                if dist <= max_radius:
                    # Enrich OLT with Longdo on-demand if not already enriched
                    if olt.ip in self.baseline_cache:
                        olt.flood_frequency = self.baseline_cache[olt.ip]
                        # Assume address/elevation also cached if frequency is
                    elif olt.elevation == 0.0:
                        extra = await self.longdo_service.reverse_geocode(olt.latitude, olt.longitude)
                        olt.elevation = extra.get("elevation", 0.0)
                        olt.subdistrict = extra.get("subdistrict", "")
                        olt.district = extra.get("district", "")
                        olt.province = extra.get("province", "")
                        
                        # Fetch and Cache Baseline
                        olt.flood_frequency = await self.longdo_service.fetch_flood_recurrence(olt.latitude, olt.longitude)
                        self.baseline_cache[olt.ip] = olt.flood_frequency

                    risk = self.calculate_risk(
                        scale=scale, distance=dist, disaster_type=news.disaster_type, 
                        probability=news.prediction_confidence, severity_level=severity,
                        elevation=olt.elevation
                    )
                    
                    # Manual adjustment for Baseline Risk (Repeat Flood Areas)
                    final_score = risk["score"]
                    if olt.flood_frequency >= 5: # High frequency area
                        final_score += 10.0 # Boost risk for chronic flood zones
                    elif olt.flood_frequency >= 1:
                        final_score += 5.0

                    risk["score"] = min(100.0, round(final_score, 2))

                    if risk["score"] > max_risk_score:
                        max_risk_score = risk["score"]
                        primary_risk_reason = risk["reason"]
                    
                    if risk["score"] > 0:
                        news.impact_radius = risk["radius"]
                        nearby.append(NearbyOLT(
                            olt_ip=olt.ip, 
                            olt_address=f"{olt.address} ({olt.subdistrict} {olt.district} Elev: {olt.elevation}m)", 
                            distance_km=round(dist, 2),
                            risk_level=risk["level"], 
                            risk_score=risk["score"],
                            flood_frequency=olt.flood_frequency,
                            latitude=olt.latitude, longitude=olt.longitude
                        ))
            
            if nearby:
                nearby.sort(key=lambda x: (-x.risk_score, x.distance_km))
                explanation = AIExplanation(
                    severity_level=severity,
                    keywords=scale_info["keywords"],
                    location_confidence=0.95 if news.extracted_location else 0.5,
                    scale_reason=scale_info["reason"],
                    distance_reason=primary_risk_reason
                )
                alerts.append(ImpactAlert(disaster=news, nearby_olts=nearby, ai_explanation=explanation))
        
        self.latest_alerts = alerts
        print(f"Pipeline finished. Cached {len(alerts)} alerts.")
        return alerts

    def get_cached_pipeline_result(self) -> List[ImpactAlert]:
        return self.latest_alerts
