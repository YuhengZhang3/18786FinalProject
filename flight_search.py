import json
import datetime
from skyscanner_api import SkyscannerAPIClient
from config import SKYSCANNER_API_KEY

class SkyscannerFlightSearchTool:
    """Skyscanner Flight Search Tool"""
    name = "flight_search"
    description = "Search for flights between an origin and destination on a specific date using Skyscanner API. This tool can search for one-way flights with specified cabin class and number of passengers."
    args = {
        "origin": {
            "type": "string",
            "description": "Departure city or airport code"
        },
        "destination": {
            "type": "string",
            "description": "Arrival city or airport code"
        },
        "date": {
            "type": "string",
            "description": "Departure date in YYYY-MM-DD format"
        },
        "adults": {
            "type": "integer",
            "description": "Number of adult passengers (default: 1)"
        },
        "cabin_class": {
            "type": "string",
            "description": "Cabin class (Economy, Premium Economy, Business) (default: Economy)"
        },
        "children": {
            "type": "integer",
            "description": "Number of children (2-12 years) (default: 0)"
        },
        "infants": {
            "type": "integer",
            "description": "Number of infants (under 2 years) (default: 0)"
        }
    }

    def __init__(self):
        # Initialize Skyscanner API client
        self.api_client = SkyscannerAPIClient(SKYSCANNER_API_KEY)

    def invoke(self, input):
        """Call flight search tool"""
        try:
            # Parse input
            input_data = self._parse_input(input)
            
            # Extract parameters
            origin = input_data.get("origin", "")
            destination = input_data.get("destination", "")
            date = input_data.get("date", "")
            adults = input_data.get("adults", 1)
            cabin_class = input_data.get("cabin_class", "Economy")
            children = input_data.get("children", 0)
            infants = input_data.get("infants", 0)
            
            # Validate required fields
            if not origin or not destination or not date:
                return "Error: Required fields missing. Please provide origin, destination, and date."
            
            # Validate date format and ensure it's in the future
            try:
                date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
                today = datetime.datetime.now().date()
                if date_obj < today:
                    return f"Error: The requested departure date {date} is in the past. Please choose a future date."
            except ValueError:
                return f"Error: Invalid date format '{date}'. Please use YYYY-MM-DD format."
            
            # Get location details for origin
            origin_details = self.api_client.get_location_details(origin)
            if not origin_details:
                return f"Error: Could not find location information for origin '{origin}'."
            
            # Get location details for destination
            destination_details = self.api_client.get_location_details(destination)
            if not destination_details:
                return f"Error: Could not find location information for destination '{destination}'."
            
            # Map cabin class to API format
            cabin_map = {
                "Economy": "economy",
                "Premium Economy": "premium_economy",
                "Business": "business",
                "First": "business"  # API uses business for first class
            }
            skyscanner_cabin = cabin_map.get(cabin_class, "economy")
            
            # Perform flight search
            results = self.api_client.search_one_way_flights(
                date=date,
                origin=origin_details.get("skyId", origin),
                origin_id=origin_details.get("entityId", ""),
                destination=destination_details.get("skyId", destination),
                destination_id=destination_details.get("entityId", ""),
                cabin_class=skyscanner_cabin,
                adults=adults,
                children=children,
                infants=infants
            )
            
            # Check for API errors
            if results.get("status") == "error":
                return f"Error performing flight search: {results.get('message', 'Unknown error')}"
            
            # Format the results
            return self._format_flight_results(results, origin, destination, date, adults, cabin_class)
            
        except Exception as e:
            return f"Error performing flight search: {str(e)}"
    
    def _parse_input(self, input):
        """Parse tool input from various formats"""
        import re
        import json
        
        input_data = {}
        
        try:
            # Try parsing as JSON
            input_data = json.loads(input)
        except json.JSONDecodeError:
            # If JSON parsing fails, try parsing key=value pairs
            input = input.strip()
            if input.startswith('{') and input.endswith('}'):
                input = input[1:-1]
            
            pairs = re.split(r',\s*', input)
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes (if present)
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Convert to appropriate types
                    if key in ['adults', 'children', 'infants']:
                        try:
                            value = int(value)
                        except ValueError:
                            if key == 'adults':
                                value = 1
                            else:
                                value = 0
                    
                    input_data[key] = value
        
        return input_data
    
    def _format_flight_results(self, results, origin, destination, date, adults, cabin_class):
        """Format flight search results into a consistent format"""
        try:
            # Extract flight data from the structure
            flight_data = results.get("data", {}).get("itineraries", {})
            
            # Get items from buckets (prioritizing "Best" then "Cheapest")
            target_buckets = ["Best", "Cheapest"] 
            selected_bucket = None
            
            for bucket_id in target_buckets:
                for bucket in flight_data.get("buckets", []):
                    if bucket.get("id") == bucket_id and bucket.get("items"):
                        selected_bucket = bucket
                        break
                if selected_bucket:
                    break
            
            if not selected_bucket or not selected_bucket.get("items"):
                return "No flights found for your search criteria."
            
            # Format flight results
            flights = []
            for item in selected_bucket.get("items", [])[:5]:  # Limit to top 5 results
                # Get the leg information
                if not item.get("legs"):
                    continue
                
                leg = item["legs"][0]
                segments = leg.get("segments", [])
                if not segments:
                    continue
                
                # Get carrier information
                carrier_info = None
                if "carriers" in leg and "marketing" in leg["carriers"] and leg["carriers"]["marketing"]:
                    carrier_info = leg["carriers"]["marketing"][0]
                
                # Format the flight
                flight = {
                    "airline": carrier_info.get("name", "Unknown Airline") if carrier_info else "Unknown Airline",
                    "flight_number": segments[0].get("flightNumber", "") if segments else "",
                    "origin": leg.get("origin", {}).get("displayCode", origin),
                    "destination": leg.get("destination", {}).get("displayCode", destination),
                    "departure_date": date,
                    "departure_time": segments[0].get("departure", "").split("T")[1][:5] if segments and "T" in segments[0].get("departure", "") else "",
                    "arrival_time": segments[-1].get("arrival", "").split("T")[1][:5] if segments and "T" in segments[-1].get("arrival", "") else "",
                    "duration": f"{leg.get('durationInMinutes', 0) // 60}h {leg.get('durationInMinutes', 0) % 60}m",
                    "cabin_class": cabin_class,
                    "price": item.get("price", {}).get("raw", 0),
                    "formatted_price": item.get("price", {}).get("formatted", "$0"),
                    "stops": leg.get("stopCount", 0)
                }
                
                flights.append(flight)
            
            # Sort by price
            flights = sorted(flights, key=lambda x: x["price"])
            
            # Prepare the final response
            result = {
                "flights": flights,
                "search_parameters": {
                    "origin": origin,
                    "destination": destination,
                    "date": date,
                    "adults": adults,
                    "cabin_class": cabin_class
                }
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            return f"Error formatting flight results: {str(e)}"