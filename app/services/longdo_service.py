import httpx
from typing import Dict, Any, List, Optional
import re
import os

LONGDO_KEY = os.getenv("LONGDO_KEY", "your_api_key_here")
ADDRESS_API = "https://api.longdo.com/map/services/address"
INCIDENT_API = "https://search.longdo.com/mapsearch/json/search"

class LongdoService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)

    async def reverse_geocode(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get address details and elevation for a coordinate.
        Returns: {province, district, subdistrict, postcode, elevation, aoi}
        """
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "key": LONGDO_KEY
            }
            resp = await self.client.get(ADDRESS_API, params=params)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "province": data.get("province", ""),
                    "district": data.get("district", ""),
                    "subdistrict": data.get("subdistrict", ""),
                    "postcode": data.get("postcode", ""),
                    "elevation": float(data.get("elevation", 0)),
                    "aoi": data.get("aoi", "")
                }
        except Exception as e:
            print(f"Longdo Geocode Error: {e}")
        return {}

    async def fetch_flood_incidents(self) -> List[Dict[str, Any]]:
        """
        Search for real-time flood-related incidents/events.
        """
        try:
            params = {
                "keyword": "น้ำท่วม",
                "limit": 10,
                "key": LONGDO_KEY
            }
            resp = await self.client.get(INCIDENT_API, params=params)
            if resp.status_code == 200:
                data = resp.json()
                # If Search returns results in 'data' or directly as list
                items = data.get("data", []) if isinstance(data, dict) else []
                return items
        except Exception as e:
            print(f"Longdo Incident Error: {e}")
        return []

    async def fetch_flood_recurrence(self, lat: float, lon: float) -> int:
        """
        Get 13-year flood recurrence count from GISTDA Sphere.
        """
        try:
            url = f"https://api.sphere.gistda.or.th/services/info/disasters/flood-recurrence"
            params = {"key": LONGDO_KEY, "lat": lat, "lon": lon}
            resp = await self.client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    return int(data[0].get("total", 0))
        except Exception as e:
            print(f"GISTDA Recurrence Error: {e}")
        return 0

    async def close(self):
        await self.client.aclose()
