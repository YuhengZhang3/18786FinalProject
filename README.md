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

Run the CLI application, you can find the chatbot in the terminal:
```bash
python Agent_Langgraph.py
```

Example queries:
- "I need to find a business class flight from SFO to ORD next Friday for 2 adults"
- "Find me a round-trip flight from New York to Los Angeles, leaving next Friday and returning next Sunday, economy class"

## Evaluation & Benchmarking

This project includes tools for generating benchmark datasets and evaluating the flight search agent's performance.

### Benchmark Dataset Generation

Generate test datasets with natural language flight queries and expected results:

```bash
# Generate 100 samples with mock data
python benchmark_datasets.py --size 100 --output mock_benchmark_dataset.json

# Generate dataset using real Skyscanner API
python benchmark_datasets.py --size 100 --use-real-api --output real_benchmark_dataset.json
```

### Agent Evaluation

There are two ways to evaluate the agent's performance:

#### Standard Evaluation

For smaller datasets or when API rate limits are not a concern:

```bash
# Run evaluation on all benchmark samples
python evaluation.py --benchmark mock_benchmark_dataset.json --output mock_evaluation_results.json


# Evaluate only a subset of samples
python evaluation.py --benchmark real_benchmark_dataset.json --output real_evaluation_results.json

```

#### Batch Evaluation (Recommended for Large Datasets)

For evaluating large datasets while respecting API rate limits:

```bash
# Evaluate mock data in batches
python batch_evaluation.py --benchmark mock_benchmark_dataset.json --output mock_evaluation --batch-size 10 --api-delay 3 --batch-delay 180

# Evaluate real data in batches
python batch_evaluation.py --benchmark real_benchmark_dataset.json --output real_evaluation --batch-size 10 --api-delay 3 --batch-delay 180

# Merge results from an incomplete evaluation
python batch_evaluation.py --output mock_evaluation --merge-only

```


### Evaluation Metrics

The evaluation measures three key metrics:

1. **Successful Flight Acquisition (SFA)**: Whether the agent successfully retrieves flight information
2. **Task Completion Rate (TCR)**: Whether the agent completes the full task workflow (search, rate, recommend)
3. **Flight-Trip Success Rate (FTSR)**: How accurately the agent's response matches the expected flight parameters

# Evaluation Methodology

## Parameter Matching Algorithm

The evaluation framework employs a text-based parameter matching algorithm to assess the flight search agent's response accuracy. This section details the current approach and its operational characteristics.

### Algorithm Description

The parameter matching algorithm implements a string containment verification methodology to determine response accuracy. For each benchmark sample, the algorithm:

1. Extracts ground truth parameters from the benchmark dataset
2. Processes the agent's natural language response as unstructured text
3. Performs case-insensitive string matching between each ground truth parameter and the response text
4. Calculates a match ratio as the proportion of successfully matched parameters
5. Applies a threshold-based classification to determine overall success

### Evaluation Metrics

The evaluation framework assesses agent performance using three complementary metrics:

1. **Successful Flight Acquisition (SFA)**: Measures whether the agent successfully retrieves any flight information, determined by the presence of flight-related terminology in the response (e.g., "flight," "airline," "departure," "arrival").

2. **Task Completion Rate (TCR)**: Evaluates whether the agent presents flight information in a structured format. Success requires both flight information retrieval and proper presentation (typically through a markdown table or explicit flight recommendations).

3. **Flight-Trip Success Rate (FTSR)**: Quantifies the accuracy of retrieved flight information against ground truth parameters. Success is determined when the match ratio exceeds a predefined threshold of 70%.

### Matching Procedure

The current parameter matching procedure operates as follows:

1. For each ground truth parameter:
   - Skip non-string values for which string matching is not applicable
   - Convert both the parameter value and the agent's response to lowercase
   - Search for the parameter value as a substring within the response text
   - Increment the match counter if the parameter value is found

2. The match ratio is calculated as:
   ```
   match_ratio = matched_parameters / total_parameters
   ```

3. The FTSR metric is considered successful if:
   ```
   match_ratio ≥ 0.7
   ```

This approach implements a binary classification for each parameter (matched or unmatched) and applies a threshold-based determination of overall success, balancing precision requirements with allowance for acceptable variations in representation.


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


