# Flight Search AI Agent

A natural language flight search application that lets users search for flights using conversational language. The application uses an LLM to process user queries, extract flight parameters, and search for flights using the Skyscanner API.

## Features

- Natural language flight search queries
- Support for one-way and round-trip flights
- Multiple cabin classes (Economy, Premium Economy, Business)
- Web interface and command-line interface
- Conversational memory to maintain context

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YuhengZhang3/18786FinalProject.git
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
SKYSCANNER_API_KEY=your_skyscanner_api_key
```

## Usage

### Web Interface

Run the Flask web server:
```bash
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000/` to access the web interface.

### Command Line Interface

Run the CLI application:
```bash
python main.py
```

Example queries:
- "I need to find a business class flight from SFO to ORD next Friday for 2 adults"
- "Find me a round-trip flight from New York to Los Angeles, leaving next Friday and returning next Sunday, economy class"

## Evaluation & Benchmarking

The project includes tools for generating benchmark datasets and evaluating the agent's performance:

```bash
# Generate benchmark dataset
python benchmark_datasets.py --size 100 --output benchmark_data.json

# Run evaluation on benchmark dataset
python evaluation.py
```

## Future Optimizations
LLM Process Improvements

UI Enhancements
