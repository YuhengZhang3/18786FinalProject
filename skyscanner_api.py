import requests
from config import SKYSCANNER_API_KEY

class SkyscannerAPIClient:
    """Wrapper for Skyscanner API endpoints"""
    
    def __init__(self, api_key=SKYSCANNER_API_KEY):
        self.api_key = api_key
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "skyscanner89.p.rapidapi.com"
        }
        # Cache for location IDs to avoid repeated API calls
        self.location_cache = {}
        
    def auto_complete(self, query, locale="en-US", market="US", currency="USD"):
        """Get location suggestions from Skyscanner"""
        # Create a cache key that includes all parameters
        cache_key = f"{query}_{locale}_{market}_{currency}"
        
        # Check cache first
        if cache_key in self.location_cache:
            return self.location_cache[cache_key]
            
        url = "https://skyscanner89.p.rapidapi.com/flights/auto-complete"
        querystring = {
            "query": query,
            "locale": locale,
            "market": market,
            "currency": currency
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=querystring)
            if response.status_code == 200:
                data = response.json()
                # Cache the result
                self.location_cache[cache_key] = data
                return data
            else:
                return {"error": f"API error: {response.status_code}", "message": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def get_location_details(self, query):
        """Extract location details from auto-complete results"""
        data = self.auto_complete(query)
        
        if "error" in data:
            return None
            
        results = data.get("inputSuggest", [])
        if not results:
            return None
        
        # 首先查找城市级别实体（更广泛的搜索）
        for result in results:
            flight_params = result.get("navigation", {}).get("relevantFlightParams", {})
            entity_type = result.get("navigation", {}).get("entityType", "")
            
            if entity_type == "CITY":
                return {
                    "entityId": flight_params.get("entityId", ""),
                    "skyId": flight_params.get("skyId", ""),
                    "name": flight_params.get("localizedName", ""),
                    "type": flight_params.get("flightPlaceType", "")
                }
        
        # 然后查找精确的机场代码匹配
        for result in results:
            flight_params = result.get("navigation", {}).get("relevantFlightParams", {})
            sky_id = flight_params.get("skyId", "")
            
            if sky_id and sky_id.upper() == query.upper():
                return {
                    "entityId": flight_params.get("entityId", ""),
                    "skyId": sky_id,
                    "name": flight_params.get("localizedName", ""),
                    "type": flight_params.get("flightPlaceType", "")
                }
        
        # 如果没有匹配，使用第一个结果
        if results:
            flight_params = results[0].get("navigation", {}).get("relevantFlightParams", {})
            return {
                "entityId": flight_params.get("entityId", ""),
                "skyId": flight_params.get("skyId", ""),
                "name": flight_params.get("localizedName", ""),
                "type": flight_params.get("flightPlaceType", "")
            }
        
        return None
        
    def search_one_way_flights(self, date, origin, origin_id, destination, destination_id, 
                            cabin_class="economy", adults=1, children=0, infants=0):
        """Search for one-way flights"""
        url = "https://skyscanner89.p.rapidapi.com/flights/one-way/list"
        
        # Validate cabin class
        valid_classes = ["economy", "premium_economy", "business"]
        if cabin_class.lower() not in valid_classes:
            cabin_class = "economy"
        
        # Prepare query parameters
        querystring = {
            "date": date,
            "origin": origin,
            "originId": origin_id,
            "destination": destination,
            "destinationId": destination_id,
            "cabinClass": cabin_class.lower(),
            "adults": str(adults),
        }
        
        # Add optional parameters if provided
        if children > 0:
            querystring["children"] = str(children)
        if infants > 0:
            querystring["infants"] = str(infants)
            
        try:
            response = requests.get(url, headers=self.headers, params=querystring)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"API error: {response.status_code}", "details": response.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
    def search_round_trip_flights(self, departure_date, return_date, origin, origin_id, destination, destination_id, 
                            cabin_class="economy", adults=1, children=0, infants=0):
        """Search for round-trip flights"""
        url = "https://skyscanner89.p.rapidapi.com/flights/roundtrip/list"
        
        # Validate cabin class
        valid_classes = ["economy", "premium_economy", "business"]
        if cabin_class.lower() not in valid_classes:
            cabin_class = "economy"
        
        # Prepare query parameters
        querystring = {
            "inDate": departure_date,
            "outDate": return_date,
            "origin": origin,
            "originId": origin_id,
            "destination": destination,
            "destinationId": destination_id,
            "cabinClass": cabin_class.lower(),
            "adults": str(adults),
            "locale": "en-US",
            "market": "US",
            "currency": "USD"
        }
        
        # Add optional parameters if provided
        if children > 0:
            querystring["children"] = str(children)
        if infants > 0:
            querystring["infants"] = str(infants)
            
        try:
            response = requests.get(url, headers=self.headers, params=querystring)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"API error: {response.status_code}", "details": response.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}