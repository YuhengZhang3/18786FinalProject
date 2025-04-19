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
# Generate 50 samples with mock data
python benchmark_datasets.py --size 50 --output mock_benchmark_dataset.json

# Generate dataset using real Skyscanner API
python benchmark_datasets.py --size 50 --use-real-api --output real_benchmark_dataset.json
```

### Agent Evaluation

There are two ways to evaluate the agent's performance:

#### Standard Evaluation

For smaller datasets or when API rate limits are not a concern:

```bash

python evaluation.py --benchmark mock_benchmark_dataset.json --output mock_evaluation_results.json
```

#### Batch Evaluation (Recommended for Large Datasets)

For evaluating large datasets while respecting API rate limits:

```bash
# Evaluate mock data in batches
python batch_evaluation.py --benchmark mock_benchmark_dataset.json --output mock_evaluation --batch-size 10 --api-delay 3 --batch-delay 120 --use-cache --optimize-batches

# Evaluate real data in batches
python batch_evaluation.py --benchmark real_benchmark_dataset.json --output real_evaluation --batch-size 10 --api-delay 3 --batch-delay 120 --use-cache --optimize-batches


# Merge results from an incomplete evaluation
python batch_evaluation.py --output mock_evaluation --merge-only
python batch_evaluation.py --output real_evaluation --merge-only
```

### Evaluation Metrics on Real Data

The evaluation framework assesses agent performance using three complementary metrics:

1. **Slot Filling Accuracy (SFA)**

2. **Task Completion Rate (TCR)**

3. **First-Turn Success Rate (FTSR)**

### Evaluation Results

| **Metric**                     | **Value**  |
|-------------------------------|------------|
| Slot Filling Accuracy (SFA)   | 56.00%     |
| Task Completion Rate (TCR)    | 100.00%    |
| First-Turn Success Rate (FTSR)| 98.00%     |

## Future Optimizations

This section is continuously updated as new issues or improvement ideas are discovered.

### API Future Work

Future enhancements for this flight booking agent could leverage more granular flight data to improve interaction quality. While the current system successfully retrieves and ranks basic flight information, incorporating additional metadata such as fare flexibility details, preference-based categorization tags, comprehensive stopover information, distinctions between marketing and operating carriers, multi-day travel detection, and formalized scoring metrics would enable more sophisticated dialogue capabilities. These enriched data points would allow the agent to more accurately address user preferences, resolve ambiguities in natural language requests, and provide truly personalized recommendations. Additionally, implementing mechanisms to detect and prevent repetitive reasoning patterns would streamline the conversation flow. Collectively, these improvements would transform the system from a basic flight search tool into an intelligent travel assistant capable of nuanced, preference-aware flight recommendations through refined multi-turn interactions.
