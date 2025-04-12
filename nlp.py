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
        - date: The raw date mention (do not convert to a specific format)
        - adults: Number of adult passengers (default to 1 if not specified)
        - cabin_class: One of "Economy", "Premium Economy", or "Business"
        - children: Number of children aged 2-12 (default to 0 if not specified)
        - infants: Number of infants under 2 (default to 0 if not specified)
        
        If a field is not mentioned in the query, DO NOT include it in the output.
        """
        
        prompt = f"Extract flight parameters from this query. Today's date is {current_date.strftime('%Y-%m-%d')}.\n\nQuery: {query}"
        
        # Use lower temperature for more deterministic results
        response = llm(prompt, system_prompt=system_prompt, temperature=0.1)
        
        # Extract JSON from response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            params = json.loads(json_match.group(1))
        else:
            # Fallback to regex extraction for simple queries
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
            
            # Check for common date keywords
            date_keywords = ["today", "tomorrow", "next week", "monday", "tuesday", "wednesday", 
                           "thursday", "friday", "saturday", "sunday"]
            for keyword in date_keywords:
                if keyword.lower() in query.lower():
                    params['date'] = keyword
                    break
        
        # Clean and validate parameters
        result = {}
        
        # Handle origin/destination
        if 'origin' in params and params['origin']:
            result['origin'] = params['origin'].upper() if len(params['origin']) == 3 else params['origin']
        if 'destination' in params and params['destination']:
            result['destination'] = params['destination'].upper() if len(params['destination']) == 3 else params['destination']
        
        # Parse departure date
        date_text = params.get('date')
        if date_text:
            parsed_date = parse_natural_date(date_text, current_date)
            if parsed_date:
                result['date'] = parsed_date
        
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