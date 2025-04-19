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

## Flight Search Agent Evaluation & Benchmarking

This project includes tools for generating benchmark datasets and evaluating the flight search agent's performance.

### Benchmark Dataset Generation

Generate test datasets with natural language flight queries and expected results:

```bash
# Generate dataset using real Skyscanner API snapshots
python benchmark_datasets.py --size 50 --use-real-api --output real_benchmark_dataset.json --snapshot-date "2025-04-10"
```

The snapshot approach uses flight data captured at a specific point in time (2025-04-10 in this example), ensuring consistent evaluation despite the constantly changing nature of flight availability and pricing. This creates a stable benchmark that allows for reliable comparison between different agent versions.

### Agent Evaluation

#### Batch Evaluation (Recommended Method)

For evaluating datasets while respecting API rate limits:

```bash
# Evaluate real data in batches
python batch_evaluation.py --benchmark real_benchmark_dataset.json --output real_evaluation --batch-size 10 --api-delay 3 --batch-delay 120 --use-cache --optimize-batches

# Merge results from an incomplete evaluation
python batch_evaluation.py --output real_evaluation --merge-only
```

### Evaluation Metrics

The evaluation framework assesses agent performance using three complementary metrics:

1. **Slot Filling Accuracy (SFA)**: Measures how accurately the agent extracts key information from natural language queries (departure/arrival locations, dates, preferences).

2. **Task Completion Rate (TCR)**: Measures whether the agent successfully completes the entire search process and returns valid flight results.

3. **First-Turn Success Rate (FTSR)**: Measures how often the agent successfully completes the search without requiring additional clarification from the user.

### Comparative Evaluation Results

We've improved the evaluation methodology and metrics since the middle project report. The following results show performance improvements using the same evaluation algorithm and metrics on an identical real data benchmark dataset:

| **Metric**                     | **Current Method** | **Middle Report Method** |
|-------------------------------|-------------------|--------------------------|
| Slot Filling Accuracy (SFA)   | 56.00%            | 64.00%                  |
| Task Completion Rate (TCR)    | 100.00%           | 82.00%                  |
| First-Turn Success Rate (FTSR)| 98.00%            | 68.00%                  |

#### Analysis

- While our current method shows slightly lower Slot Filling Accuracy, it demonstrates significant improvements in both Task Completion Rate and First-Turn Success Rate.
- The 18% improvement in Task Completion Rate indicates our agent is now much more reliable at completing searches successfully.
- The 30% improvement in First-Turn Success Rate demonstrates our agent's enhanced ability to understand and process queries correctly on the first attempt, greatly improving the user experience.

## Future Optimizations

### API Future Work

Future flight booking systems need better data integration to transform from basic search tools to intelligent travel assistants. Enhanced metadata—including ticket fare flexibility details, preference-based categorization tags, comprehensive stopover information, distinctions between marketing and operating carriers, multi-day travel detection, and formalized scoring metrics—would enable more sophisticated conversations.

This improvement would allow for better preference interpretation, ambiguity resolution, and personalized recommendations while preventing repetitive reasoning patterns.
