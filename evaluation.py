# evaluation.py
import json
import datetime
import argparse
import logging
import re
import time
from tqdm import tqdm
from difflib import SequenceMatcher

# Import agent components from Agent_Langgraph.py
from Agent_Langgraph import graph, memory_store, SYSTEM_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResponseCache:
    """Simple cache for agent responses to save API calls"""
    
    def __init__(self, cache_file="agent_response_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cache from file"""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2)
    
    def get(self, query):
        """Get cached response for query"""
        query_key = query.strip().lower()
        return self.cache.get(query_key)
    
    def set(self, query, response):
        """Cache response for query"""
        query_key = query.strip().lower()
        self.cache[query_key] = response
        self._save_cache()

class FlightAgentEvaluator:
    """Evaluates a flight search agent using benchmark datasets"""
    
    def __init__(self, benchmark_file, output_file="evaluation_results.json", use_cache=False):
        """Initialize the evaluator
        
        Args:
            benchmark_file (str): Path to benchmark dataset file
            output_file (str): Path to save evaluation results
            use_cache (bool): Whether to use response caching
        """
        self.benchmark_file = benchmark_file
        self.output_file = output_file
        self.benchmark_data = self._load_benchmark_data()
        self.evaluation_results = {
            "metrics": {
                "sfa": 0,  # Slot Filling Accuracy
                "tcr": 0,  # Task Completion Rate
                "ftsr": 0  # First-Turn Success Rate
            },
            "detailed_results": []
        }
        self.use_cache = use_cache
        if use_cache:
            self.response_cache = ResponseCache()
    
    def _load_benchmark_data(self):
        """Load benchmark dataset from file"""
        try:
            with open(self.benchmark_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading benchmark file: {e}")
            return []
    
    def evaluate_agent(self, max_samples=None, start_index=0, api_delay=1):
        """Run evaluation on benchmark dataset
        
        Args:
            max_samples (int): Maximum number of samples to evaluate (None for all)
            start_index (int): Starting index for evaluation
            api_delay (float): Delay in seconds between API calls to avoid rate limits
        """
        if not self.benchmark_data:
            logger.error("No benchmark data available for evaluation")
            return
        
        # Limit samples if specified
        end_index = min(start_index + max_samples, len(self.benchmark_data)) if max_samples else len(self.benchmark_data)
        samples = self.benchmark_data[start_index:end_index]
        logger.info(f"Evaluating agent on {len(samples)} benchmark samples (from index {start_index} to {end_index-1})")
        
        # Run evaluation on each sample
        for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
            result = self._evaluate_sample(sample, api_delay)
            self.evaluation_results["detailed_results"].append(result)
            
            # Save intermediate results every 5 samples
            if (i + 1) % 5 == 0:
                self._save_results(f"{self.output_file}.interim")
                logger.info(f"Saved interim results after sample {start_index + i + 1}")
        
        # Calculate overall metrics
        total_samples = len(self.evaluation_results["detailed_results"])
        if total_samples > 0:
            self.evaluation_results["metrics"]["sfa"] = sum(1 for r in self.evaluation_results["detailed_results"] if r["sfa"]) / total_samples
            self.evaluation_results["metrics"]["tcr"] = sum(1 for r in self.evaluation_results["detailed_results"] if r["tcr"]) / total_samples
            self.evaluation_results["metrics"]["ftsr"] = sum(1 for r in self.evaluation_results["detailed_results"] if r["ftsr"]) / total_samples
        
        # Save results
        self._save_results()
        self._print_summary()
    
    def _evaluate_sample(self, sample, api_delay=1, max_retries=3):
        """Evaluate agent performance on a single benchmark sample
        
        Args:
            sample (dict): Benchmark sample with query and ground truth
            api_delay (float): Delay in seconds between API calls
            max_retries (int): Maximum number of retries for API calls
            
        Returns:
            dict: Evaluation results for this sample
        """
        sample_id = sample.get("id", "unknown")
        query = sample.get("query", "")
        ground_truth = sample.get("ground_truth", {})
        expected_results = sample.get("expected_results", {})
        
        logger.debug(f"Evaluating sample {sample_id}: {query}")
        
        # Check cache first if enabled
        agent_response = None
        if self.use_cache:
            cached_response = self.response_cache.get(query)
            if cached_response:
                logger.info(f"Using cached response for sample {sample_id}")
                agent_response = cached_response
        
        # If no cached response, call the agent
        if not agent_response:
            # Create a unique thread ID for this evaluation
            thread_id = f"eval-{sample_id}-{datetime.datetime.now().timestamp()}"
            
            # Prepare input for the agent
            inputs = {"messages": [("user", query)]}
            config = {"configurable": {"thread_id": thread_id}}
            
            # Call the agent with retry logic
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Add delay before API call to reduce rate limit issues
                    time.sleep(api_delay)
                    
                    result = graph.invoke(inputs, config=config)
                    agent_response = result["messages"][-1].content
                    
                    # Cache response if enabled
                    if self.use_cache:
                        self.response_cache.set(query, agent_response)
                    
                    break
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    # If it's a rate limit error, wait longer
                    if "429" in error_msg or "rate_limit" in error_msg.lower():
                        wait_time = 30 * (2 ** retry_count)  # Exponential backoff: 30s, 60s, 120s
                        logger.warning(f"Rate limit exceeded, waiting {wait_time}s before retry {retry_count}/{max_retries}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Error evaluating sample {sample_id}: {e}")
                        time.sleep(api_delay * 2)  # Wait a bit longer on other errors
        
        # If we still don't have a response after retries
        if not agent_response:
            logger.error(f"Failed to evaluate sample {sample_id} after {max_retries} retries")
            return {
                "sample_id": sample_id,
                "query": query,
                "ground_truth": ground_truth,
                "agent_response": f"ERROR: Evaluation failed after {max_retries} retries",
                "sfa": False,
                "tcr": False,
                "ftsr": False,
                "notes": f"Evaluation failed after {max_retries} retries"
            }
        
        # Evaluate results
        evaluation = self._analyze_agent_response(agent_response, ground_truth, expected_results)
        
        return {
            "sample_id": sample_id,
            "query": query,
            "ground_truth": ground_truth,
            "agent_response": agent_response,
            "sfa": evaluation["sfa"],
            "tcr": evaluation["tcr"],
            "ftsr": evaluation["ftsr"],
            "notes": evaluation["notes"]
        }
    
    def _analyze_agent_response(self, response, ground_truth, expected_results):
        """Analyze the agent's response against ground truth with improved algorithms
        
        Args:
            response (str): Agent's response text
            ground_truth (dict): Ground truth parameters
            expected_results (dict): Expected flight results
            
        Returns:
            dict: Analysis results with metrics
        """
        # Initialize results
        analysis = {
            "sfa": False,  # Slot Filling Accuracy
            "tcr": False,  # Task Completion Rate
            "ftsr": False, # First-Turn Success Rate
            "notes": []
        }
        
        # Normalize response text
        normalized_response = response.lower()
        
        # 1. SFA - Check if all required flight parameters were correctly extracted
        # Define required parameters
        required_params = ["origin", "destination", "departure_date"]
        if ground_truth.get("is_round_trip", False):
            required_params.append("return_date")
        
        # Extract and match parameters from the response
        extracted_values = {}
        for param in required_params:
            # Get ground truth value
            truth_value = str(ground_truth.get(param, "")).lower()
            if not truth_value:
                continue
                
            # Check if parameter was correctly extracted
            param_found = False
            
            # Direct match check
            if truth_value in normalized_response:
                param_found = True
                extracted_values[param] = truth_value
            else:
                # Fuzzy match for parameters
                best_match = 0
                best_text = ""
                
                # Split response into chunks for better matching
                chunks = re.split(r'[.,;:\n]', normalized_response)
                for chunk in chunks:
                    chunk = chunk.strip()
                    if len(chunk) < 3:  # Skip very short chunks
                        continue
                        
                    similarity = SequenceMatcher(None, truth_value, chunk).ratio()
                    if similarity > best_match and similarity > 0.7:  # 70% similarity threshold
                        best_match = similarity
                        best_text = chunk
                
                if best_match > 0.7:
                    param_found = True
                    extracted_values[param] = best_text
            
            if not param_found:
                analysis["notes"].append(f"Failed to extract parameter: {param}")
        
        # SFA is successful if all required parameters were found
        params_found = len(extracted_values)
        total_required = len(required_params)
        param_ratio = params_found / total_required if total_required > 0 else 0
        
        analysis["sfa"] = param_ratio >= 0.8  # At least 80% of parameters must be found
        
        # 2. TCR - Check for task completion
        has_flight_results = bool(re.search(r'(flight|airline|departure|arrival)', normalized_response))
        has_booking_info = "book" in normalized_response or "reservation" in normalized_response
        has_flight_table = "|" in response and ("-|-" in response or "---" in response)
        has_recommendations = "recommend" in normalized_response or "best option" in normalized_response
        
        # TCR is successful if flight results were found and organized
        analysis["tcr"] = has_flight_results and (has_flight_table or has_recommendations) and has_booking_info
        
        # 3. FTSR - Check if task was completed successfully on first turn
        # Look for indicators of follow-up requests or clarifications
        asks_for_clarification = bool(re.search(r'(could you|can you|would you|please) (clarify|provide|specify)', normalized_response))
        asks_follow_up = "follow-up" in normalized_response or "additional information" in normalized_response
        asks_more_details = "more details" in normalized_response or "need more information" in normalized_response
        
        # FTSR is successful if task was completed without asking for clarification
        analysis["ftsr"] = analysis["tcr"] and not (asks_for_clarification or asks_follow_up or asks_more_details)
        
        # Add notes about the evaluation
        if not analysis["sfa"]:
            analysis["notes"].append(f"Slot filling accuracy: {param_ratio:.2f} - below threshold")
        if not analysis["tcr"]:
            missing = []
            if not has_flight_results:
                missing.append("flight information")
            if not has_booking_info:
                missing.append("booking details")
            if not (has_flight_table or has_recommendations):
                missing.append("organized results or recommendations")
            if missing:
                analysis["notes"].append(f"Task not completed (missing: {', '.join(missing)})")
        if not analysis["ftsr"]:
            if not analysis["tcr"]:
                analysis["notes"].append("First-turn failure: task not completed")
            else:
                analysis["notes"].append("First-turn failure: required clarification or follow-up")
        
        return analysis
    
    def _save_results(self, output_file=None):
        """Save evaluation results to file"""
        try:
            file_to_save = output_file or self.output_file
            with open(file_to_save, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, indent=2)
            logger.info(f"Evaluation results saved to {file_to_save}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def _print_summary(self):
        """Print evaluation summary to console"""
        metrics = self.evaluation_results["metrics"]
        
        print("\n" + "="*50)
        print("FLIGHT SEARCH AGENT EVALUATION SUMMARY")
        print("="*50)
        print(f"Total samples evaluated: {len(self.evaluation_results['detailed_results'])}")
        print(f"Slot Filling Accuracy (SFA): {metrics['sfa']:.2%}")
        print(f"Task Completion Rate (TCR): {metrics['tcr']:.2%}")
        print(f"First-Turn Success Rate (FTSR): {metrics['ftsr']:.2%}")
        print("="*50)
        
        # Count success/failure by metric
        sfa_success = sum(1 for r in self.evaluation_results["detailed_results"] if r["sfa"])
        tcr_success = sum(1 for r in self.evaluation_results["detailed_results"] if r["tcr"])
        ftsr_success = sum(1 for r in self.evaluation_results["detailed_results"] if r["ftsr"])
        
        print(f"SFA: {sfa_success} succeeded, {len(self.evaluation_results['detailed_results']) - sfa_success} failed")
        print(f"TCR: {tcr_success} succeeded, {len(self.evaluation_results['detailed_results']) - tcr_success} failed")
        print(f"FTSR: {ftsr_success} succeeded, {len(self.evaluation_results['detailed_results']) - ftsr_success} failed")


def main():
    """Main function to run evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate flight search agent using benchmark dataset')
    parser.add_argument('--benchmark', type=str, default="benchmark_dataset.json", 
                        help='Path to benchmark dataset file (default: benchmark_dataset.json)')
    parser.add_argument('--output', type=str, default="evaluation_results.json", 
                        help='Path to save evaluation results (default: evaluation_results.json)')
    parser.add_argument('--samples', type=int, default=None, 
                        help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting index for evaluation (default: 0)')
    parser.add_argument('--api-delay', type=float, default=2.0,
                        help='Delay in seconds between API calls (default: 2.0)')
    parser.add_argument('--use-cache', action='store_true',
                        help='Use response caching to reduce API calls')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"- Benchmark file: {args.benchmark}")
    print(f"- Output file: {args.output}")
    print(f"- Max samples: {args.samples if args.samples else 'All'}")
    print(f"- Start index: {args.start_index}")
    print(f"- API delay: {args.api_delay}s")
    print(f"- Use cache: {args.use_cache}")
    
    # Run evaluation
    evaluator = FlightAgentEvaluator(args.benchmark, args.output, use_cache=args.use_cache)
    evaluator.evaluate_agent(max_samples=args.samples, start_index=args.start_index, api_delay=args.api_delay)
    
    print(f"Evaluation complete. Results saved to: {args.output}")


if __name__ == "__main__":
    main()