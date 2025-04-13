import json
import random
import datetime
import argparse
from tqdm import tqdm

class BenchmarkDatasetGenerator:
    """Generator for flight search benchmark datasets"""
    
    def __init__(self, size=200, use_real_api=False):
        """Initialize dataset generator
        
        Args:
            size: Number of entries to generate
            use_real_api: Whether to use real Skyscanner API instead of mock data
        """
        self.dataset_size = size
        self.use_real_api = use_real_api
        
        # Common US airports for dataset generation
        self.popular_airports = [
            {"code": "JFK", "name": "New York"},
            {"code": "LAX", "name": "Los Angeles"},
            {"code": "ORD", "name": "Chicago"},
            {"code": "DFW", "name": "Dallas"},
            {"code": "ATL", "name": "Atlanta"},
            {"code": "SFO", "name": "San Francisco"},
            {"code": "MIA", "name": "Miami"},
            {"code": "SEA", "name": "Seattle"},
            {"code": "BOS", "name": "Boston"},
            {"code": "LAS", "name": "Las Vegas"}
        ]
        
        # International airports (popular from US)
        self.international_airports = [
            {"code": "LHR", "name": "London"},
            {"code": "CDG", "name": "Paris"},
            {"code": "NRT", "name": "Tokyo"},
            {"code": "HKG", "name": "Hong Kong"},
            {"code": "FCO", "name": "Rome"},
            {"code": "YYZ", "name": "Toronto"},
            {"code": "MEX", "name": "Mexico City"},
            {"code": "CUN", "name": "Cancun"},
            {"code": "SYD", "name": "Sydney"},
            {"code": "AMS", "name": "Amsterdam"}
        ]
        
        # All airports combined
        self.all_airports = self.popular_airports + self.international_airports
        
        # Cabin classes
        self.cabin_classes = ["Economy", "Premium Economy", "Business"]
        
        # Templates for query generation - US English focused
        self.query_templates = [
            # One-way flight templates
            "I need a flight from {origin} to {destination} on {date}",
            "Find me a flight from {origin} to {destination} on {date}",
            "Book a {cabin_class} class ticket from {origin} to {destination} for {date}",
            "I want to fly from {origin} to {destination} on {date} with {adults} adults",
            "One-way flight from {origin} to {destination} on {date}",
            "What's the best {cabin_class} flight from {origin} to {destination} on {date}?",
            "Looking for a flight from {origin} to {destination} on {date}, {adults} passengers",
            
            # Round-trip flight templates
            "I need a round-trip flight from {origin} to {destination}, departing on {date} and returning on {return_date}",
            "Find me a round-trip from {origin} to {destination}, leaving {date}, coming back {return_date}",
            "Book a {cabin_class} class round-trip from {origin} to {destination}, outbound {date}, return {return_date}",
            "Round-trip from {origin} to {destination}: depart {date}, return {return_date}",
            "What's the cheapest round-trip from {origin} to {destination} on {date}, returning {return_date}?",
            "Need {cabin_class} tickets from {origin} to {destination}, leaving {date} and returning {return_date} for {adults} people"
        ]
        
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
    
    def _generate_query_params(self):
        """Generate random parameters for a flight query"""
        # Decide if domestic or international
        is_international = random.choice([True, False])
        
        if is_international:
            # Select one US and one international airport
            origin = random.choice(self.popular_airports)
            destination = random.choice(self.international_airports)
        else:
            # Select two different US airports
            origin, destination = random.sample(self.popular_airports, 2)
        
        # Generate departure date
        departure_date = self._generate_date(7, 60)
        
        # Decide if this is a one-way or round-trip
        is_round_trip = random.choice([True, False])
        
        params = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "is_round_trip": is_round_trip,
            "cabin_class": random.choice(self.cabin_classes),
            "adults": random.randint(1, 3),
            "children": random.randint(0, 2)
        }
        
        # Add return date for round-trip
        if is_round_trip:
            # Return between 2 and 14 days after departure
            dep_date_obj = datetime.datetime.strptime(departure_date, "%Y-%m-%d").date()
            return_days = random.randint(2, 14)
            return_date = (dep_date_obj + datetime.timedelta(days=return_days)).strftime("%Y-%m-%d")
            params["return_date"] = return_date
        
        return params
    
    def _generate_query(self, params):
        """Generate a natural language query from parameters"""
        # Select appropriate templates based on trip type
        if params["is_round_trip"]:
            templates = [t for t in self.query_templates if "return" in t.lower() or "round-trip" in t.lower()]
        else:
            templates = [t for t in self.query_templates if "return" not in t.lower() and "round-trip" not in t.lower()]
        
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
    
    def _get_real_flight_results(self, params):
        """Get actual flight results from API for ground truth and validate"""
        try:
            # Prepare input for flight search
            search_input = {
                "origin": params["origin"]["code"],
                "destination": params["destination"]["code"],
                "departure_date": params["departure_date"],
                "adults": params["adults"],
                "cabin_class": params["cabin_class"],
                "children": params["children"]
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
                    print(f"Invalid JSON response")
                    return None
            
            # Validate results
            if results is None:
                return None
                
            # Check for error messages
            if isinstance(results, str) and any(err in results.lower() for err in ["error", "not found", "invalid"]):
                print(f"API returned error: {results}")
                return None
                
            # Check for flights
            if isinstance(results, dict):
                # If API returned error in dictionary format
                if "status" in results and results["status"] == "error":
                    print(f"API error: {results.get('message', 'Unknown error')}")
                    return None
                    
                # Check if flights exist
                if "flights" in results and (not results["flights"] or len(results["flights"]) == 0):
                    print("No flights found")
                    return None
            
            return results
        except Exception as e:
            print(f"Error getting flight results: {e}")
            return None
    
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
            # Random airline
            us_airlines = ["American Airlines", "United Airlines", "Delta", "Southwest", "JetBlue", "Alaska Airlines"]
            international_airlines = ["British Airways", "Air France", "Lufthansa", "Emirates", "Japan Airlines", "Cathay Pacific"]
            
            if params["origin"]["code"] in [a["code"] for a in self.popular_airports] and params["destination"]["code"] in [a["code"] for a in self.popular_airports]:
                # Domestic flight
                airlines = us_airlines
            else:
                # International flight
                airlines = us_airlines + international_airlines
                
            airline = random.choice(airlines)
            
            # Random flight number
            flight_number = f"{airline[:2].upper()}{random.randint(100, 999)}"
            
            # Random departure and arrival times
            dep_hour = random.randint(6, 22)
            dep_minute = random.choice([0, 15, 30, 45])
            
            # Flight duration based on domestic/international
            if params["origin"]["code"] in [a["code"] for a in self.popular_airports] and params["destination"]["code"] in [a["code"] for a in self.popular_airports]:
                # Domestic flight (1-6 hours)
                duration_hours = random.randint(1, 6)
            else:
                # International flight (6-15 hours)
                duration_hours = random.randint(6, 15)
                
            duration_minutes = random.choice([0, 15, 30, 45])
            
            # Calculate arrival time
            arr_hour = (dep_hour + duration_hours) % 24
            arr_minute = (dep_minute + duration_minutes) % 60
            
            # Random price
            if params["origin"]["code"] in [a["code"] for a in self.popular_airports] and params["destination"]["code"] in [a["code"] for a in self.popular_airports]:
                # Domestic flight base price
                base_price = 150 + random.randint(50, 300)
            else:
                # International flight base price
                base_price = 500 + random.randint(100, 1000)
                
            if params["cabin_class"] == "Premium Economy":
                base_price *= 1.5
            elif params["cabin_class"] == "Business":
                base_price *= 3
                
            # Random stops
            if params["origin"]["code"] in [a["code"] for a in self.popular_airports] and params["destination"]["code"] in [a["code"] for a in self.popular_airports]:
                # Domestic flights have 0-1 stops
                stops = random.randint(0, 1)
            else:
                # International flights have 0-2 stops
                stops = random.randint(0, 2)
            
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
            "children": params["children"],
            "is_round_trip": params["is_round_trip"]
        }
        
        if params["is_round_trip"]:
            ground_truth["return_date"] = params["return_date"]
        
        return ground_truth
    
    def generate_dataset(self, output_file="benchmark_dataset.json"):
        """Generate benchmark dataset and save to file"""
        dataset = []
        
        print(f"Generating {self.dataset_size} benchmark entries...")
        
        # Keep track of retries
        retry_count = 0
        max_retries = self.dataset_size * 3  # Allow enough retries
        api_failures = 0
        
        with tqdm(total=self.dataset_size) as pbar:
            while len(dataset) < self.dataset_size and retry_count < max_retries:
                # Generate random parameters
                params = self._generate_query_params()
                
                # Generate natural language query
                query = self._generate_query(params)
                
                # Create ground truth parameters
                ground_truth = self._create_ground_truth(params)
                
                # Get expected flight results
                if self.use_real_api:
                    expected_results = self._get_real_flight_results(params)
                    # Skip if results are invalid
                    if expected_results is None:
                        retry_count += 1
                        api_failures += 1
                        continue
                else:
                    expected_results = self._generate_mock_flight_results(params)
                
                # Create benchmark entry
                entry = {
                    "id": len(dataset) + 1,
                    "query": query,
                    "ground_truth": ground_truth,
                    "expected_results": expected_results
                }
                
                dataset.append(entry)
                pbar.update(1)
        
        # Save dataset to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Generated {len(dataset)} benchmark entries.")
        if self.use_real_api:
            print(f"API stats: {retry_count} retries, {api_failures} API failures")
        
        return dataset

def main():
    """Main function to generate dataset"""
    parser = argparse.ArgumentParser(description='Generate benchmark dataset for flight search')
    parser.add_argument('--size', type=int, default=200, help='Number of samples to generate')
    parser.add_argument('--use-real-api', action='store_true', help='Use real Skyscanner API instead of mock data')
    parser.add_argument('--output', type=str, default="benchmark_dataset.json", help='Output file path')
    
    args = parser.parse_args()
    
    # Create generator and generate dataset
    generator = BenchmarkDatasetGenerator(size=args.size, use_real_api=args.use_real_api)
    generator.generate_dataset(output_file=args.output)

if __name__ == "__main__":
    main()