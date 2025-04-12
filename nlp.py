import re
import json
import datetime
from llm import llm

# Improved natural language date parsing
def parse_natural_date(date_str, current_date=None):
    """Parse natural language date expressions into YYYY-MM-DD format"""
    if current_date is None:
        current_date = datetime.datetime.now()
    
    if date_str is None or not isinstance(date_str, str):
        return None
    
    # Convert to lowercase for easier matching
    date_str = date_str.lower()
    
    # Handle explicit date format first
    if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
        return date_str
    
    # Special date keywords
    if "today" in date_str:
        return current_date.strftime("%Y-%m-%d")
    elif "tomorrow" in date_str:
        return (current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    elif "next week" in date_str:
        return (current_date + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Handle days of the week
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(days):
        if day in date_str:
            # Calculate days until the next occurrence of this day
            target_weekday = i
            current_weekday = current_date.weekday()
            days_ahead = (target_weekday - current_weekday) % 7
            
            # If it's the same day or "next" is specified, add 7 days
            if (days_ahead == 0 or "next" in date_str):
                days_ahead += 7
                
            target_date = current_date + datetime.timedelta(days=days_ahead)
            return target_date.strftime("%Y-%m-%d")
    
    # Handle specific date formats with month names
    try:
        # Try different date formats
        for fmt in ["%B %d, %Y", "%d %B %Y", "%B %d %Y", "%d %B, %Y"]:
            try:
                parsed_date = datetime.datetime.strptime(date_str, fmt)
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue
                
        # Handle formats like "April 12" (assume current year)
        months = ["january", "february", "march", "april", "may", "june", 
                "july", "august", "september", "october", "november", "december"]
        
        for i, month in enumerate(months, 1):
            if month in date_str:
                # Find the day number
                day_match = re.search(r'\d+', date_str)
                if day_match:
                    day = int(day_match.group())
                    # Assume current year
                    year = current_date.year
                    result_date = f"{year}-{i:02d}-{day:02d}"
                    
                    # Check if the date is in the past, if so, use next year
                    parsed_date = datetime.datetime.strptime(result_date, "%Y-%m-%d").date()
                    if parsed_date < current_date.date():
                        result_date = f"{year+1}-{i:02d}-{day:02d}"
                    
                    return result_date
    except Exception:
        pass
    
    # Return None if all parsing attempts fail
    return None

# Optimized flight parameter extraction
def extract_flight_parameters(query):
    """Extract flight search parameters from natural language query"""
    if query is None:
        return {}
        
    current_date = datetime.datetime.now()
    
    try:
        # Use a carefully constructed prompt to extract parameters
        system_prompt = """You are a flight parameter extraction system. Your ONLY task is to extract structured flight search parameters from user queries.
        
        DO NOT make up or invent information not explicitly stated.
        DO NOT include commentary or explanations.
        
        Output ONLY a valid JSON object with these fields if mentioned:
        - origin: Airport code or city name
        - destination: Airport code or city name
        - departure_date: The raw departure date mention (do not convert to a specific format)
        - return_date: The raw return date mention (do not convert to a specific format)
        - adults: Number of adult passengers (default to 1 if not specified)
        - cabin_class: One of "Economy", "Premium Economy", or "Business"
        - children: Number of children aged 2-12 (default to 0 if not specified)
        - infants: Number of infants under 2 (default to 0 if not specified)
        
        If a field is not mentioned in the query, DO NOT include it in the output.
        For round-trip queries, be sure to identify both departure and return dates.
        """
        
        prompt = f"Extract flight parameters from this query. Today's date is {current_date.strftime('%Y-%m-%d')}.\n\nQuery: {query}"
        
        # Use lower temperature for more deterministic results
        response = llm(prompt, system_prompt=system_prompt, temperature=0.1)
        
        # Extract JSON from response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            params = json.loads(json_match.group(1))
        else:
            # Fallback to regex extraction
            params = {}
            
            # Extract origin and destination
            if 'from' in query.lower() and 'to' in query.lower():
                parts = query.lower().split('from')[1].split('to')
                if len(parts) >= 2:
                    origin = parts[0].strip()
                    destination = parts[1].split()[0].strip()
                    params['origin'] = origin
                    params['destination'] = destination
            
            # Extract cabin class
            cabin_classes = ['economy', 'premium economy', 'business', 'first']
            for cabin in cabin_classes:
                if cabin.lower() in query.lower():
                    params['cabin_class'] = cabin.title()
                    break
            
            # Extract number of adults
            adults_match = re.search(r'(\d+)\s+adult', query, re.IGNORECASE)
            if adults_match:
                params['adults'] = int(adults_match.group(1))
            
            # Check for round-trip indicators
            is_round_trip = any(phrase in query.lower() for phrase in ["round trip", "round-trip", "return", "coming back"])
            
            # Try to extract dates
            date_keywords = ["today", "tomorrow", "next week", "monday", "tuesday", "wednesday", 
                           "thursday", "friday", "saturday", "sunday"]
            
            if is_round_trip:
                # Look for departure and return dates pattern
                departure_return_match = re.search(r'(?:leaving|departing|going)(?:\s+on)?\s+([^,]+)(?:,|\s+and)(?:\s+returning|\s+coming back)(?:\s+on)?\s+([^,\.]+)', query, re.IGNORECASE)
                if departure_return_match:
                    params['departure_date'] = departure_return_match.group(1).strip()
                    params['return_date'] = departure_return_match.group(2).strip()
                else:
                    # Try to find two date mentions
                    date_mentions = []
                    for keyword in date_keywords:
                        if keyword.lower() in query.lower():
                            date_mentions.append(keyword)
                    
                    if len(date_mentions) >= 2:
                        params['departure_date'] = date_mentions[0]
                        params['return_date'] = date_mentions[1]
                    elif len(date_mentions) == 1:
                        params['departure_date'] = date_mentions[0]
            else:
                # For one-way, just look for the first date mention
                for keyword in date_keywords:
                    if keyword.lower() in query.lower():
                        params['departure_date'] = keyword
                        break
        
        # Clean and validate parameters
        result = {}
        
        # Handle origin/destination
        if 'origin' in params and params['origin']:
            result['origin'] = params['origin'].upper() if len(params['origin']) == 3 else params['origin']
        if 'destination' in params and params['destination']:
            result['destination'] = params['destination'].upper() if len(params['destination']) == 3 else params['destination']
        
        # Parse departure date
        departure_date_text = params.get('departure_date')
        if departure_date_text:
            parsed_departure_date = parse_natural_date(departure_date_text, current_date)
            if parsed_departure_date:
                result['departure_date'] = parsed_departure_date
        
        # Parse return date if present
        return_date_text = params.get('return_date')
        if return_date_text:
            parsed_return_date = parse_natural_date(return_date_text, current_date)
            if parsed_return_date:
                result['return_date'] = parsed_return_date
        
        # Handle passenger counts
        for field in ['adults', 'children', 'infants']:
            if field in params and params[field] is not None:
                try:
                    result[field] = int(params[field])
                except:
                    result[field] = 1 if field == 'adults' else 0
                    
        # Handle cabin class
        if 'cabin_class' in params and params['cabin_class']:
            valid_classes = ['Economy', 'Premium Economy', 'Business', 'First']
            cabin = params['cabin_class'].strip().title()
            if cabin in valid_classes:
                result['cabin_class'] = cabin
            elif 'business' in cabin.lower():
                result['cabin_class'] = 'Business'
            elif 'first' in cabin.lower():
                result['cabin_class'] = 'Business'  # Map to Business as the API doesn't support First
            elif 'premium' in cabin.lower():
                result['cabin_class'] = 'Premium Economy'
            else:
                result['cabin_class'] = 'Economy'
        
        return result
    except Exception as e:
        print(f"Error extracting flight parameters: {str(e)}")
        return {}