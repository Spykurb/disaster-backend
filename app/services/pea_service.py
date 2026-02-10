import httpx
import re
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from bs4 import BeautifulSoup
from pydantic import BaseModel

class PEAOutage(BaseModel):
    id: str
    title: str
    location: str
    start_date: str
    end_date: str
    latitude: float
    longitude: float

class PEAService:
    def __init__(self):
        self.client = httpx.AsyncClient(headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest"
        }, verify=False, timeout=30.0)
        self.list_url = "https://eservice.pea.co.th/PowerOutage/Home/GetOutages"
        self.detail_url_base = "https://eservice.pea.co.th/PowerOutage/Home/Detail/"

    def get_be_date(self) -> str:
        """Get current date in DD/MM/YYYY (Buddhist Era) format."""
        now = datetime.now()
        year_be = now.year + 543
        return now.strftime(f"%d/%m/{year_be}")

    async def fetch_outage_list(self) -> List[Dict]:
        """Fetch list of outages from PEA API for today."""
        date_str = self.get_be_date()
        print(f"Fetching PEA outages for {date_str}...")
        
        data = {
            "province": "",
            "date": date_str,
            "detail": "",
            "draw": "1",
            "start": "0",
            "length": "100"
        }
        
        try:
            resp = await self.client.post(self.list_url, data=data)
            if resp.status_code == 200:
                data_json = resp.json()
                items = data_json.get("data", [])
                print(f"Found {len(items)} PEA outages.")
                return items
            else:
                print(f"PEA API Error: {resp.status_code}")
                return []
        except Exception as e:
            print(f"PEA Fetch Exception: {e}")
            return []

    async def get_coordinates(self, outage_id: str) -> Optional[Dict[str, float]]:
        """Extract LAT and LNG from PEA detail page."""
        url = f"{self.detail_url_base}{outage_id}"
        try:
            resp = await self.client.get(url)
            if resp.status_code == 200:
                html = resp.text
                # Try to extract LAT/LNG from hidden inputs
                lat_match = re.search(r'id="LAT"\s+name="LAT"\s+type="hidden"\s+value="([-0-9.]+)"', html)
                lng_match = re.search(r'id="LNG"\s+name="LNG"\s+type="hidden"\s+value="([-0-9.]+)"', html)
                
                if lat_match and lng_match:
                    return {
                        "lat": float(lat_match.group(1)),
                        "lng": float(lng_match.group(1))
                    }
                
                # Fallback: search for google maps URL in iframe
                map_match = re.search(r'src="https://www.google.com/maps\?output=embed&q=([-0-9.]+),([-0-9.]+)"', html)
                if map_match:
                    return {
                        "lat": float(map_match.group(1)),
                        "lng": float(map_match.group(2))
                    }
            return None
        except Exception as e:
            print(f"PEA Detail Exception for {outage_id}: {e}")
            return None

    async def run_pipeline(self) -> List[PEAOutage]:
        """Fetch list and coordinates for all outages."""
        raw_items = await self.fetch_outage_list()
        results = []
        
        # Limit concurrency to avoid overloading
        semaphore = asyncio.Semaphore(5)
        
        async def process_item(item):
            async with semaphore:
                outage_id = item.get("OUTAGE_ID")
                coords = await self.get_coordinates(outage_id)
                if coords:
                    return PEAOutage(
                        id=outage_id,
                        title=item.get("REASON", "Power Outage Notification"),
                        location=item.get("TEXT_LOCATION", ""),
                        start_date=item.get("START_DATE_DISPLAY", ""),
                        end_date=item.get("END_DATE_DISPLAY", ""),
                        latitude=coords["lat"],
                        longitude=coords["lng"]
                    )
                return None

        tasks = [process_item(item) for item in raw_items]
        processed = await asyncio.gather(*tasks)
        
        results = [p for p in processed if p]
        print(f"Successfully processed {len(results)} PEA outages with coordinates.")
        return results

    async def close(self):
        await self.client.aclose()
