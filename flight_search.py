import json
import datetime
from skyscanner_api import SkyscannerAPIClient
from config import SKYSCANNER_API_KEY

class SkyscannerFlightSearchTool:
    """Skyscanner Flight Search Tool"""
    name = "flight_search"
    description = "Search for flights between an origin and destination using Skyscanner API. This tool can search for one-way or round-trip flights with specified cabin class and number of passengers."
    args = {
        "origin": {
            "type": "string",
            "description": "Departure city or airport code"
        },
        "destination": {
            "type": "string",
            "description": "Arrival city or airport code"
        },
        "departure_date": {
            "type": "string",
            "description": "Departure date in YYYY-MM-DD format"
        },
        "return_date": {
            "type": "string",
            "description": "Return date in YYYY-MM-DD format (for round-trip only)"
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
            departure_date = input_data.get("departure_date", input_data.get("date", ""))  # Support both params
            return_date = input_data.get("return_date", "")
            adults = input_data.get("adults", 1)
            cabin_class = input_data.get("cabin_class", "Economy")
            children = input_data.get("children", 0)
            infants = input_data.get("infants", 0)
            
            # Validate required fields
            if not origin or not destination or not departure_date:
                return "Error: Required fields missing. Please provide origin, destination, and departure date."
            
            # Validate departure date format and ensure it's in the future
            try:
                date_obj = datetime.datetime.strptime(departure_date, "%Y-%m-%d").date()
                today = datetime.datetime.now().date()
                if date_obj < today:
                    return f"Error: The requested departure date {departure_date} is in the past. Please choose a future date."
            except ValueError:
                return f"Error: Invalid date format '{departure_date}'. Please use YYYY-MM-DD format."
            
            # Validate return date if provided
            if return_date:
                try:
                    return_date_obj = datetime.datetime.strptime(return_date, "%Y-%m-%d").date()
                    if return_date_obj < date_obj:
                        return f"Error: The return date {return_date} must be after the departure date {departure_date}."
                except ValueError:
                    return f"Error: Invalid date format '{return_date}'. Please use YYYY-MM-DD format."
            
            # Get location details
            origin_details = self.api_client.get_location_details(origin)
            if not origin_details:
                return f"Error: Could not find location information for origin '{origin}'."
            
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
            
            # Determine search type and perform search
            if return_date:
                # Round-trip search
                results = self.api_client.search_round_trip_flights(
                    departure_date=departure_date,  # This will become inDate
                    return_date=return_date,        # This will become outDate
                    origin=origin_details.get("skyId", origin),
                    origin_id=origin_details.get("entityId", ""),
                    destination=destination_details.get("skyId", destination),
                    destination_id=destination_details.get("entityId", ""),
                    cabin_class=skyscanner_cabin,
                    adults=adults,
                    children=children,
                    infants=infants
                )
            else:
                # One-way search
                results = self.api_client.search_one_way_flights(
                    date=departure_date,
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
            return self._format_flight_results(
                results, 
                origin, 
                destination, 
                departure_date, 
                adults, 
                cabin_class, 
                return_date=return_date
            )
            
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
    
    def _format_flight_results(self, results, origin, destination, departure_date, adults, cabin_class, return_date=None):
        """Format flight search results into a consistent format"""
        try:
            # Extract flight data
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
                if not item.get("legs"):
                    continue
                
                # For round-trip, we have two legs
                if return_date and len(item["legs"]) >= 2:
                    outbound_leg = item["legs"][0]
                    return_leg = item["legs"][1]
                    
                    # Process outbound leg
                    outbound_segments = outbound_leg.get("segments", [])
                    if not outbound_segments:
                        continue
                    
                    # Get carrier information for outbound
                    outbound_carrier_info = None
                    if "carriers" in outbound_leg and "marketing" in outbound_leg["carriers"] and outbound_leg["carriers"]["marketing"]:
                        outbound_carrier_info = outbound_leg["carriers"]["marketing"][0]
                    
                    # Process return leg
                    return_segments = return_leg.get("segments", [])
                    if not return_segments:
                        continue
                    
                    # Get carrier information for return
                    return_carrier_info = None
                    if "carriers" in return_leg and "marketing" in return_leg["carriers"] and return_leg["carriers"]["marketing"]:
                        return_carrier_info = return_leg["carriers"]["marketing"][0]
                    
                    # Format the round-trip flight
                    flight = {
                        "type": "round-trip",
                        "outbound": {
                            "airline": outbound_carrier_info.get("name", "Unknown Airline") if outbound_carrier_info else "Unknown Airline",
                            "flight_number": outbound_segments[0].get("flightNumber", "") if outbound_segments else "",
                            "origin": outbound_leg.get("origin", {}).get("displayCode", origin),
                            "destination": outbound_leg.get("destination", {}).get("displayCode", destination),
                            "departure_date": departure_date,
                            "departure_time": outbound_segments[0].get("departure", "").split("T")[1][:5] if outbound_segments and "T" in outbound_segments[0].get("departure", "") else "",
                            "arrival_time": outbound_segments[-1].get("arrival", "").split("T")[1][:5] if outbound_segments and "T" in outbound_segments[-1].get("arrival", "") else "",
                            "duration": f"{outbound_leg.get('durationInMinutes', 0) // 60}h {outbound_leg.get('durationInMinutes', 0) % 60}m",
                            "stops": outbound_leg.get("stopCount", 0)
                        },
                        "return": {
                            "airline": return_carrier_info.get("name", "Unknown Airline") if return_carrier_info else "Unknown Airline",
                            "flight_number": return_segments[0].get("flightNumber", "") if return_segments else "",
                            "origin": return_leg.get("origin", {}).get("displayCode", destination),
                            "destination": return_leg.get("destination", {}).get("displayCode", origin),
                            "departure_date": return_date,
                            "departure_time": return_segments[0].get("departure", "").split("T")[1][:5] if return_segments and "T" in return_segments[0].get("departure", "") else "",
                            "arrival_time": return_segments[-1].get("arrival", "").split("T")[1][:5] if return_segments and "T" in return_segments[-1].get("arrival", "") else "",
                            "duration": f"{return_leg.get('durationInMinutes', 0) // 60}h {return_leg.get('durationInMinutes', 0) % 60}m",
                            "stops": return_leg.get("stopCount", 0)
                        },
                        "cabin_class": cabin_class,
                        "price": item.get("price", {}).get("raw", 0),
                        "formatted_price": item.get("price", {}).get("formatted", "$0")
                    }
                else:
                    # One-way flight (existing logic)
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
                        "type": "one-way",
                        "airline": carrier_info.get("name", "Unknown Airline") if carrier_info else "Unknown Airline",
                        "flight_number": segments[0].get("flightNumber", "") if segments else "",
                        "origin": leg.get("origin", {}).get("displayCode", origin),
                        "destination": leg.get("destination", {}).get("displayCode", destination),
                        "departure_date": departure_date,
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
                    "departure_date": departure_date,
                    "adults": adults,
                    "cabin_class": cabin_class
                }
            }
            
            # Add return date to search parameters if it's a round-trip
            if return_date:
                result["search_parameters"]["return_date"] = return_date
                result["search_parameters"]["trip_type"] = "round-trip"
            else:
                result["search_parameters"]["trip_type"] = "one-way"
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            return f"Error formatting flight results: {str(e)}"