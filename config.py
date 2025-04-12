import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set API keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SKYSCANNER_API_KEY = os.environ.get("SKYSCANNER_API_KEY")