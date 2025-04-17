import json
import random
import datetime
import argparse
import time
from tqdm import tqdm

class BenchmarkDatasetGenerator:
    """Generator for flight search benchmark datasets"""
    
    def __init__(self, size=100, use_real_api=False):
        """Initialize dataset generator
        
        Args:
            size: Number of entries to generate
            use_real_api: Whether to use real Skyscanner API instead of mock data
        """
        self.dataset_size = size
        self.use_real_api = use_real_api
        # Add API results cache to avoid redundant calls
        self.api_results_cache = {}
        
        # Common US airports for dataset generation
        self.popular_airports = [
            {"code": "JFK", "name": "New York"},
            {"code": "LAX", "name": "Los Angeles"},
            {"code": "SFO", "name": "San Francisco"},
            {"code": "ORD", "name": "Chicago"},
            {"code": "MIA", "name": "Miami"}
        ]
        self.international_airports = [
            {"code": "LHR", "name": "London"},
            {"code": "CDG", "name": "Paris"},
            {"code": "NRT", "name": "Tokyo"},
            {"code": "YYZ", "name": "Toronto"},
            {"code": "MEX", "name": "Mexico City"}
        ]
        self.cabin_classes = ["Economy", "Premium Economy", "Business"]
        
        # Initialize API clients if using real API
        if self.use_real_api:
            try:
                from skyscanner_api import SkyscannerAPIClient
                from flight_search import SkyscannerFlightSearchTool
                from config import SKYSCANNER_API_KEY
                
                self.api_client = SkyscannerAPIClient(SKYSCANNER_API_KEY)
                self.flight_search = SkyscannerFlightSearchTool()
                print("Initialized Skyscanner API client for real data")
            except ImportError as e:
                print(f"Error initializing API clients: {e}")
                print("Falling back to mock data generation")
                self.use_real_api = False
    
    def _generate_date(self, min_days=7, max_days=60):
        """Generate a random future date"""
        today = datetime.datetime.now().date()
        random_days = random.randint(min_days, max_days)
        future_date = today + datetime.timedelta(days=random_days)
        return future_date.strftime("%Y-%m-%d")
    
    def _format_date_for_query(self, date_str):
        """Format date for natural language query"""
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        month_names = ["January", "February", "March", "April", "May", "June", 
                     "July", "August", "September", "October", "November", "December"]
        formatted_date = f"{month_names[date_obj.month-1]} {date_obj.day}"
        return formatted_date
    
    def _get_cache_key(self, params):
        """Generate a cache key for API results"""
        key_parts = [
            f"origin={params['origin']['code']}",
            f"dest={params['destination']['code']}",
            f"dep={params['departure_date']}",
        ]
        if params.get("is_round_trip") and params.get("return_date"):
            key_parts.append(f"ret={params['return_date']}")
        key_parts.append(f"cc={params['cabin_class']}")
        key_parts.append(f"adults={params['adults']}")
        return "|".join(key_parts)
    
    def _get_real_flight_results(self, params):
        """Get actual flight results from API with optimized calls"""
        # Check cache first
        cache_key = self._get_cache_key(params)
        if cache_key in self.api_results_cache:
            return self.api_results_cache[cache_key]
        
        # Try API call with retries
        for attempt in range(3):
            try:
                # Prepare input for flight search
                search_input = {
                    "origin": params["origin"]["code"],
                    "destination": params["destination"]["code"],
                    "departure_date": params["departure_date"],
                    "adults": params["adults"],
                    "cabin_class": params["cabin_class"]
                }
                
                # Add return date for round-trip
                if params["is_round_trip"]:
                    search_input["return_date"] = params["return_date"]
                
                # Convert to JSON
                input_json = json.dumps(search_input)
                
                # Search for flights
                results = self.flight_search.invoke(input_json)
                
                # Parse results
                if isinstance(results, str):
                    try:
                        results = json.loads(results)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON response (attempt {attempt+1}/3)")
                        if attempt < 2:
                            time.sleep(2)  # Delay 2 seconds before retry
                        continue
                
                # Validate results
                if results is None:
                    print(f"Null results (attempt {attempt+1}/3)")
                    if attempt < 2:
                        time.sleep(2)
                    continue
                
                # Check for flights
                if isinstance(results, dict):
                    if "flights" not in results or not results["flights"]:
                        print(f"No flights found (attempt {attempt+1}/3)")
                        if attempt < 2:
                            time.sleep(2)
                        continue
                
                # If we got here, we have valid results
                print(f"Found valid results on attempt {attempt+1}")
                self.api_results_cache[cache_key] = results
                return results
                
            except Exception as e:
                print(f"Error getting flight results: {e} (attempt {attempt+1}/3)")
                if attempt < 2:
                    time.sleep(2)
        
        # If we get here, all attempts failed
        return None
    
    def _generate_query(self, params):
        """Generate a natural language query from parameters"""
        # Select appropriate templates based on trip type
        if params["is_round_trip"]:
            templates = [
                "I need a round-trip flight from {origin} to {destination}, departing on {date} and returning on {return_date}",
                "Find me a round-trip from {origin} to {destination}, leaving {date}, coming back {return_date}",
                "Book a {cabin_class} class round-trip from {origin} to {destination}, outbound {date}, return {return_date}"
            ]
        else:
            templates = [
                "I need a flight from {origin} to {destination} on {date}",
                "Find me a flight from {origin} to {destination} on {date}",
                "Book a {cabin_class} class ticket from {origin} to {destination} for {date}"
            ]
        
        template = random.choice(templates)
        
        # Format dates for natural language
        departure_text = self._format_date_for_query(params["departure_date"])
        
        # Prepare format parameters
        format_params = {
            "origin": random.choice([params["origin"]["code"], params["origin"]["name"]]),
            "destination": random.choice([params["destination"]["code"], params["destination"]["name"]]),
            "date": departure_text,
            "cabin_class": params["cabin_class"],
            "adults": params["adults"]
        }
        
        # Add return date for round-trip
        if params["is_round_trip"]:
            format_params["return_date"] = self._format_date_for_query(params["return_date"])
        
        # Generate query
        query = template.format(**format_params)
        
        return query
    
    def _generate_mock_flight_results(self, params):
        """Generate mock flight search results"""
        # Create basic structure
        mock_results = {
            "flights": [],
            "search_parameters": {
                "origin": params["origin"]["code"],
                "destination": params["destination"]["code"],
                "departure_date": params["departure_date"],
                "adults": params["adults"],
                "cabin_class": params["cabin_class"],
                "trip_type": "round-trip" if params["is_round_trip"] else "one-way"
            }
        }
        
        # Add return date for round-trip
        if params["is_round_trip"]:
            mock_results["search_parameters"]["return_date"] = params["return_date"]
        
        # Generate 1-5 random flight options
        num_flights = random.randint(1, 5)
        
        for i in range(num_flights):
            airlines = ["American Airlines", "United Airlines", "Delta", "Southwest", "JetBlue"]
            airline = random.choice(airlines)
            
            # Random flight number
            flight_number = f"{airline[:2].upper()}{random.randint(100, 999)}"
            
            # Random departure and arrival times
            dep_hour = random.randint(6, 22)
            dep_minute = random.choice([0, 15, 30, 45])
            
            # Flight duration
            duration_hours = random.randint(1, 6)
            duration_minutes = random.choice([0, 15, 30, 45])
            
            # Calculate arrival time
            arr_hour = (dep_hour + duration_hours) % 24
            arr_minute = (dep_minute + duration_minutes) % 60
            
            # Random price
            base_price = 150 + random.randint(50, 300)
            if params["cabin_class"] == "Premium Economy":
                base_price *= 1.5
            elif params["cabin_class"] == "Business":
                base_price *= 3
                
            # Random stops
            stops = random.randint(0, 1)
            
            # Create flight data
            if params["is_round_trip"]:
                # Round-trip flight
                flight = {
                    "type": "round-trip",
                    "outbound": {
                        "airline": airline,
                        "flight_number": flight_number,
                        "origin": params["origin"]["code"],
                        "destination": params["destination"]["code"],
                        "departure_date": params["departure_date"],
                        "departure_time": f"{dep_hour:02d}:{dep_minute:02d}",
                        "arrival_time": f"{arr_hour:02d}:{arr_minute:02d}",
                        "duration": f"{duration_hours}h {duration_minutes}m",
                        "stops": stops
                    },
                    "return": {
                        "airline": airline,
                        "flight_number": flight_number.replace(flight_number[-1], str(int(flight_number[-1])+1)),
                        "origin": params["destination"]["code"],
                        "destination": params["origin"]["code"],
                        "departure_date": params["return_date"],
                        "departure_time": f"{(arr_hour+2)%24:02d}:{arr_minute:02d}",
                        "arrival_time": f"{(dep_hour+2)%24:02d}:{dep_minute:02d}",
                        "duration": f"{duration_hours}h {duration_minutes}m",
                        "stops": stops
                    },
                    "cabin_class": params["cabin_class"],
                    "price": base_price * 2,
                    "formatted_price": f"${base_price * 2}"
                }
            else:
                # One-way flight
                flight = {
                    "type": "one-way",
                    "airline": airline,
                    "flight_number": flight_number,
                    "origin": params["origin"]["code"],
                    "destination": params["destination"]["code"],
                    "departure_date": params["departure_date"],
                    "departure_time": f"{dep_hour:02d}:{dep_minute:02d}",
                    "arrival_time": f"{arr_hour:02d}:{arr_minute:02d}",
                    "duration": f"{duration_hours}h {duration_minutes}m",
                    "cabin_class": params["cabin_class"],
                    "price": base_price,
                    "formatted_price": f"${base_price}",
                    "stops": stops
                }
                
            mock_results["flights"].append(flight)
        
        return mock_results
    
    def _create_ground_truth(self, params):
        """Create ground truth parameters"""
        ground_truth = {
            "origin": params["origin"]["code"],
            "destination": params["destination"]["code"],
            "departure_date": params["departure_date"],
            "adults": params["adults"],
            "cabin_class": params["cabin_class"],
            "is_round_trip": params["is_round_trip"]
        }
        
        if params["is_round_trip"]:
            ground_truth["return_date"] = params["return_date"]
        
        return ground_truth
    
    def generate_dataset(self, output_file="benchmark_dataset.json"):
        """Generate benchmark dataset and save to file"""
        dataset = []
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        print(f"Generating {self.dataset_size} benchmark entries...")
        
        with tqdm(total=self.dataset_size) as pbar:
            while len(dataset) < self.dataset_size:
                # Break if too many consecutive failures to avoid getting stuck
                if consecutive_failures >= max_consecutive_failures:
                    print(f"Breaking after {consecutive_failures} consecutive failures")
                    break
                
                # For API queries, optimize for common routes that likely have data
                if self.use_real_api:
                    # Use common route pairs
                    popular_pairs = [
                        ("JFK", "LAX"), ("SFO", "JFK"), ("ORD", "MIA")
                    ]
                    origin_code, dest_code = random.choice(popular_pairs)
                    
                    origin = next(a for a in self.popular_airports if a["code"] == origin_code)
                    destination = next(a for a in self.popular_airports if a["code"] == dest_code)
                    
                    # Use near-term dates (API more likely to have data)
                    today = datetime.datetime.now().date()
                    future_days = random.randint(14, 30)
                    departure_date = (today + datetime.timedelta(days=future_days)).strftime("%Y-%m-%d")
                    
                    # Prefer Economy (more results)
                    cabin_class = "Economy"
                    
                    # Determine if round-trip
                    is_round_trip = random.choice([True, False])
                    
                    params = {
                        "origin": origin,
                        "destination": destination,
                        "departure_date": departure_date,
                        "is_round_trip": is_round_trip,
                        "cabin_class": cabin_class,
                        "adults": 1
                    }
                    
                    # Add return date for round-trip
                    if is_round_trip:
                        return_days = random.randint(3, 7)
                        dep_date_obj = datetime.datetime.strptime(departure_date, "%Y-%m-%d").date()
                        return_date = (dep_date_obj + datetime.timedelta(days=return_days)).strftime("%Y-%m-%d")
                        params["return_date"] = return_date
                    
                    # Get flight results
                    expected_results = self._get_real_flight_results(params)
                    
                    # Skip if no results
                    if expected_results is None:
                        consecutive_failures += 1
                        continue
                else:
                    # Mock data generation logic
                    origin, destination = random.sample(self.popular_airports, 2)
                    departure_date = self._generate_date(7, 60)
                    is_round_trip = random.choice([True, False])
                    
                    params = {
                        "origin": origin,
                        "destination": destination,
                        "departure_date": departure_date,
                        "is_round_trip": is_round_trip,
                        "cabin_class": random.choice(self.cabin_classes),
                        "adults": random.randint(1, 2)
                    }
                    
                    if is_round_trip:
                        return_days = random.randint(2, 14)
                        dep_date_obj = datetime.datetime.strptime(departure_date, "%Y-%m-%d").date()
                        return_date = (dep_date_obj + datetime.timedelta(days=return_days)).strftime("%Y-%m-%d")
                        params["return_date"] = return_date
                    
                    # Generate mock flight results
                    expected_results = self._generate_mock_flight_results(params)
                
                # Reset consecutive failures
                consecutive_failures = 0
                
                # Generate query text
                query = self._generate_query(params)
                
                # Create benchmark entry
                entry = {
                    "id": len(dataset) + 1,
                    "query": query,
                    "ground_truth": self._create_ground_truth(params),
                    "expected_results": expected_results
                }
                
                dataset.append(entry)
                pbar.update(1)
                
                # Save interim progress every 10 entries
                if len(dataset) % 10 == 0:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(dataset, f, ensure_ascii=False, indent=2)
                    print(f"\nInterim save: {len(dataset)}/{self.dataset_size} entries")
        
        # Save final dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Generated {len(dataset)} benchmark entries.")
        return dataset

def main():
    """Main function to generate dataset"""
    parser = argparse.ArgumentParser(description='Generate benchmark dataset for flight search')
    parser.add_argument('--size', type=int, default=100, 
                        help='Number of samples to generate (default: 100)')
    parser.add_argument('--use-real-api', action='store_true', 
                        help='Use real Skyscanner API instead of mock data')
    parser.add_argument('--output', type=str, default="benchmark_dataset.json", 
                        help='Output file path (default: benchmark_dataset.json)')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"- Generating {args.size} samples")
    print(f"- Using {'real API data' if args.use_real_api else 'mock data'}")
    print(f"- Output file: {args.output}")
    
    # Create generator and generate dataset
    generator = BenchmarkDatasetGenerator(size=args.size, use_real_api=args.use_real_api)
    generator.generate_dataset(output_file=args.output)
    
    print(f"Dataset generation complete. File saved to: {args.output}")

if __name__ == "__main__":
    main()