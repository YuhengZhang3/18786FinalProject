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

### New Agent Pipeline with Memory and ReAct

Directly run `Agent_Langgraph.py`, you can find the chatbot in the terminal.

## Evaluation & Benchmarking

This project includes tools for generating benchmark datasets and evaluating the flight search agent's performance.

### Benchmark Dataset Generation

Generate test datasets with natural language queries and expected results:

```bash
# Generate default dataset (100 mock samples)
python benchmark_datasets.py

# Generate 100 samples with custom output file
python benchmark_datasets.py --size 100 --output synthetic_datasets.json

# Generate dataset using real Skyscanner API
python benchmark_datasets.py --size 100 --use-real-api --output real_datasets.json
```

**📁 Example Benchmark Datasets:**  
You can also download example benchmark datasets directly from Google Drive:  
[Click here to access the dataset folder](https://drive.google.com/drive/folders/1ijn1nGVkoxTJL18JtVLxJ24DsCu-TCOF?usp=sharing)

#### Command Line Options:
- `--size`: Number of samples to generate (default: 100)
- `--use-real-api`: Use real Skyscanner API instead of mock data
- `--output`: Output file path (default: benchmark_dataset.json)

### Agent Evaluation

Evaluate the agent's performance against benchmark datasets:

```bash
# Evaluate using default settings
python evaluation.py

# Evaluate with a custom dataset file
python evaluation.py --input synthetic_dataset.json

# Evaluate only the first 20 entries
python evaluation.py --max-entries 20

# Customize output files
python evaluation.py --input synthetic_datasets.json --predictions synthetic_predictions.json --results eval_synthetic_summary.json

# Set custom timeout and cache directory
python evaluation.py --timeout 60 --cache-dir custom_cache
```

#### Command Line Options:
- `--input`: Path to benchmark dataset (default: benchmark_dataset.json)
- `--max-entries`: Maximum number of entries to evaluate (default: all)
- `--predictions`: File to save individual predictions (default: predictions.json)
- `--results`: File to save summary results (default: evaluation_results.json)
- `--cache-dir`: Directory for cache files (default: eval_cache)
- `--timeout`: Timeout in seconds for each evaluation (default: 45)

#### Output Files:
- **Predictions file**: Contains detailed predictions and metrics for each query
- **Results file**: Contains summary statistics and overall performance metrics

The evaluation process automatically caches parameter extraction results to reduce LLM calls and saves progress after each entry, allowing for safe interruption and resumption.

#### Evaluation Metrics:

The evaluation measures three key metrics:

1. **Slot Filling Accuracy (SFA)**: How accurately the agent extracts flight parameters  
2. **Task Completion Rate (TCR)**: Whether the agent successfully completes the requested search  
3. **First-Turn Success Rate (FTSR)**: Whether the agent completes the task in the first interaction  

Results are saved to `evaluation_results_final.json` for further analysis.



## Future Optimizations

This section is continuously updated as new issues or improvement ideas are discovered.

### LLM Process Improvements

In some cases, the LLM enters a repetitive reasoning loop, repeatedly issuing the same action without recognizing that the task has been completed. This indicates a need for more robust loop detection and better handling of task state transitions.

### Additional Optimization Opportunities

Based on available data, the following fields beyond standard flight info can enhance LLM dialogue and reasoning:

- **Fare Policy Details**: Whether tickets are refundable, changeable, or flexible
```json
"farePolicy": {
    "isCancellationAllowed": false,
    "isChangeAllowed": false,
    "isPartiallyChangeable": false,
    "isPartiallyRefundable": false
}
```

- **Tag-based Ranking**: Labels like "shortest", "cheapest", or "best" can guide preference-based reasoning
```json
"tags": ["cheapest", "shortest", "best"]
```

- **Stopover Airports**: Locations of layovers may be relevant for user preferences
```json
"segments": [
    {
        "origin": {"city": "New York", "displayCode": "JFK"},
        "destination": {"city": "Chicago", "displayCode": "ORD"}
    },
    {
        "origin": {"city": "Chicago", "displayCode": "ORD"},
        "destination": {"city": "Honolulu", "displayCode": "HNL"}
    }
]
```

- **Marketing vs Operating Carrier**: Useful when different airlines operate the same route
```json
"carriers": {
    "marketing": [{"name": "jetBlue", "logoUrl": "URL"}],
    "operating": [{"name": "Hawaiian Airlines", "logoUrl": "URL"}],
    "operationType": "not_operated"
}
```

- **Time Delta Across Dates**: Multi-day travel detection can help LLM clarify date accuracy
```json
"timeDeltaInDays": 1
```

- **Scoring Metrics**: Score fields can aid in LLM-driven ranking or selection logic
```json
"score": 0.999
```

These metadata can be used to improve slot-filling accuracy, resolve ambiguities, and refine multi-turn interactions.

Based on available data, the following fields beyond standard flight info can enhance LLM dialogue and reasoning:

- **Fare Policy Details**: Whether tickets are refundable, changeable, or flexible
- **Tag-based Ranking**: Labels like "shortest", "cheapest", or "best" can guide preference-based reasoning
- **Stopover Airports**: Locations of layovers may be relevant for user preferences
- **Marketing vs Operating Carrier**: Useful when different airlines operate the same route
- **Time Delta Across Dates**: Multi-day travel detection can help LLM clarify date accuracy
- **Scoring Metrics**: Score fields can aid in LLM-driven ranking or selection logic

These metadata can be used to improve slot-filling accuracy, resolve ambiguities, and refine multi-turn interactions.

### UI Enhancements

Planned improvements include better feedback indicators for user input processing, cleaner presentation of search results, and enhanced error messaging when API responses fail.


