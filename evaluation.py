import json
import time
import signal
import argparse
import numpy as np
import os
from tqdm import tqdm
from agent import agent_execute_with_retry
from nlp import extract_flight_parameters

class TimeoutException(Exception):
    """Exception raised when evaluation takes too long"""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutException("Evaluation timed out")

# Define NLM caching classes to avoid repeated LLM calls
class NLMCache:
    """Cache for NLM function calls to avoid repeated API calls"""
    
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cache from file if it exists"""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def get(self, key):
        """Get cached result for key if it exists"""
        return self.cache.get(key)
    
    def set(self, key, value):
        """Set cache entry and save to file"""
        self.cache[key] = value
        self._save_cache()

class CachedNLMFunction:
    """Wrapper for NLM functions to cache results"""
    
    def __init__(self, function, cache_file):
        self.function = function
        self.cache = NLMCache(cache_file)
    
    def __call__(self, *args, **kwargs):
        """Call function with caching"""
        # Create cache key - simple serialization of args and kwargs
        args_str = json.dumps(args) if args else ""
        kwargs_str = json.dumps(kwargs, sort_keys=True) if kwargs else ""
        cache_key = f"{args_str}|{kwargs_str}"
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Call original function
        result = self.function(*args, **kwargs)
        
        # Cache result
        self.cache.set(cache_key, result)
        
        return result

class FlightSearchEvaluator:
    """Evaluator for flight search agent"""
    
    def __init__(self, benchmark_file="benchmark_dataset.json", max_entries=None, 
                predictions_file="predictions.json", results_file="evaluation_results.json", 
                cache_dir="eval_cache"):
        """Initialize with benchmark dataset
        
        Args:
            benchmark_file: Path to benchmark dataset JSON file
            max_entries: Maximum number of entries to evaluate (None for all)
            predictions_file: File to save individual predictions
            results_file: File to save summary results
            cache_dir: Directory for cache files
        """
        # Save file paths
        self.benchmark_file = benchmark_file
        self.predictions_file = predictions_file
        self.results_file = results_file
        self.cache_dir = cache_dir
        
        # Load benchmark dataset
        self.benchmark_data = self._load_benchmark_data(benchmark_file)
        
        # Apply max_entries filter if specified
        if max_entries and max_entries < len(self.benchmark_data):
            self.benchmark_data = self.benchmark_data[:max_entries]
            print(f"Limited to first {max_entries} entries for evaluation")
        
        # Set timeout
        self.timeout = 45  # seconds
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize cached extract_params function
        self.cached_extract_params = CachedNLMFunction(
            extract_flight_parameters,
            f"{cache_dir}/params_cache.json"
        )
        
        # Initialize metrics
        self.metrics = {
            "tcr": [],  # Task Completion Rate
            "sfa": [],  # Slot Filling Accuracy
            "ftsr": []  # First-Turn Success Rate
        }
        
        # Field weights for SFA calculation
        self.field_weights = {
            "origin": 0.25,
            "destination": 0.25,
            "departure_date": 0.2,
            "return_date": 0.1,
            "adults": 0.1,
            "cabin_class": 0.1
        }
        
        # Initialize predictions and results
        self.predictions = []
        self.results = {}
        
        # Hidden checkpoint file for resuming interrupted evaluations
        self.checkpoint_file = f"{cache_dir}/.evaluation_checkpoint.json"
        self._load_checkpoint()
    
    def _load_benchmark_data(self, file_path):
        """Load benchmark dataset from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} benchmark entries from {file_path}")
            return data
        except Exception as e:
            print(f"Error loading benchmark data: {e}")
            return []
    
    def _save_checkpoint(self):
        """Save current evaluation state to hidden checkpoint file"""
        checkpoint = {
            "predictions": self.predictions,
            "metrics": self.metrics,
            "benchmark_file": self.benchmark_file,
            "predictions_file": self.predictions_file,
            "results_file": self.results_file,
            "timestamp": time.time()
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def _load_checkpoint(self):
        """Load evaluation state from checkpoint if available"""
        if not os.path.exists(self.checkpoint_file):
            return False
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            # Only load checkpoint if it's for the same benchmark file
            if checkpoint.get("benchmark_file") != self.benchmark_file:
                print("Checkpoint is for a different benchmark file. Starting fresh.")
                return False
                
            self.predictions = checkpoint.get("predictions", [])
            self.metrics = checkpoint.get("metrics", {"tcr": [], "sfa": [], "ftsr": []})
            
            # Update output files from checkpoint if not specified
            if checkpoint.get("predictions_file") and self.predictions_file == "predictions.json":
                self.predictions_file = checkpoint.get("predictions_file")
            
            if checkpoint.get("results_file") and self.results_file == "evaluation_results.json":
                self.results_file = checkpoint.get("results_file")
            
            print(f"Loaded checkpoint with {len(self.predictions)} previously evaluated entries")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def _normalize_city_code(self, value):
        """Normalize city/airport code for comparison"""
        if not value:
            return ""
        
        value = str(value).strip().upper()
        # Remove non-alphanumeric characters
        return ''.join(c for c in value if c.isalnum())
    
    def _calculate_date_similarity(self, date1, date2):
        """Calculate similarity between two dates"""
        if not date1 or not date2:
            return 0.0
        
        try:
            from datetime import datetime
            date1_obj = datetime.strptime(date1, "%Y-%m-%d").date()
            date2_obj = datetime.strptime(date2, "%Y-%m-%d").date()
            
            # Calculate days difference
            days_diff = abs((date1_obj - date2_obj).days)
            
            # Convert to similarity (0-1)
            if days_diff == 0:
                return 1.0
            elif days_diff <= 1:
                return 0.8
            elif days_diff <= 3:
                return 0.5
            elif days_diff <= 7:
                return 0.2
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_slot_accuracy(self, extracted, ground_truth):
        """Calculate slot filling accuracy"""
        scores = {}
        
        # Origin
        if "origin" in extracted and "origin" in ground_truth:
            extracted_origin = self._normalize_city_code(extracted["origin"])
            ground_truth_origin = self._normalize_city_code(ground_truth["origin"])
            scores["origin"] = 1.0 if extracted_origin == ground_truth_origin else 0.0
        else:
            scores["origin"] = 0.0 if "origin" in ground_truth else 1.0
        
        # Destination
        if "destination" in extracted and "destination" in ground_truth:
            extracted_dest = self._normalize_city_code(extracted["destination"])
            ground_truth_dest = self._normalize_city_code(ground_truth["destination"])
            scores["destination"] = 1.0 if extracted_dest == ground_truth_dest else 0.0
        else:
            scores["destination"] = 0.0 if "destination" in ground_truth else 1.0
        
        # Departure date
        if "departure_date" in extracted and "departure_date" in ground_truth:
            scores["departure_date"] = self._calculate_date_similarity(
                extracted["departure_date"], ground_truth["departure_date"]
            )
        else:
            scores["departure_date"] = 0.0 if "departure_date" in ground_truth else 1.0
        
        # Return date (only if ground truth is round-trip)
        if ground_truth.get("is_round_trip", False):
            if "return_date" in extracted and "return_date" in ground_truth:
                scores["return_date"] = self._calculate_date_similarity(
                    extracted["return_date"], ground_truth["return_date"]
                )
            else:
                scores["return_date"] = 0.0
        else:
            scores["return_date"] = 1.0  # Not applicable for one-way
        
        # Adults
        if "adults" in extracted and "adults" in ground_truth:
            extracted_adults = int(extracted.get("adults", 1))
            ground_truth_adults = int(ground_truth.get("adults", 1))
            scores["adults"] = 1.0 if extracted_adults == ground_truth_adults else 0.0
        else:
            # Default to adult=1 if not specified
            scores["adults"] = 1.0 if ground_truth.get("adults", 1) == 1 else 0.0
        
        # Cabin class
        if "cabin_class" in extracted and "cabin_class" in ground_truth:
            extracted_cabin = extracted["cabin_class"].lower() if extracted["cabin_class"] else ""
            ground_truth_cabin = ground_truth["cabin_class"].lower() if ground_truth["cabin_class"] else ""
            scores["cabin_class"] = 1.0 if extracted_cabin == ground_truth_cabin else 0.0
        else:
            # Default to Economy if not specified
            scores["cabin_class"] = 1.0 if ground_truth.get("cabin_class", "Economy") == "Economy" else 0.0
        
        # Calculate weighted average
        weighted_score = 0.0
        total_weight = 0.0
        
        for field, score in scores.items():
            weight = self.field_weights.get(field, 0.0)
            weighted_score += score * weight
            total_weight += weight
        
        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return final_score, scores
    
    def _check_first_turn_success(self, result, history):
        """Check if query was successfully handled in first turn"""
        # If there's only one interaction, we succeeded in first turn
        if len(history) == 1:
            return True
        return False
    
    def _check_task_completion(self, result):
        """Check if the task was completed successfully"""
        # Check if result contains flight information
        try:
            if isinstance(result, str):
                # Success indicators - removed "booking" as requested
                success_phrases = [
                    "flight", "airline", "departure", "arrival", "ticket", 
                    "price", "duration", "found", "available"
                ]
                
                # Check if any success phrase is in the result
                if any(phrase in result.lower() for phrase in success_phrases):
                    # Check for error indicators even if success phrases exist
                    error_phrases = [
                        "couldn't find", "not able to", "try again", "no flights found",
                        "no results", "unable to find", "couldn't locate", "error performing"
                    ]
                    
                    # If error phrases are found, it's likely a failure
                    if any(phrase in result.lower() for phrase in error_phrases):
                        return False
                    
                    return True
                
                return False
            
            return False
        except:
            return False
    
    def _is_already_evaluated(self, entry_id):
        """Check if an entry has already been evaluated"""
        return any(pred.get("id") == entry_id for pred in self.predictions)
    
    def evaluate_entry(self, entry):
        """Evaluate a single benchmark entry"""
        # Skip if already evaluated
        if self._is_already_evaluated(entry.get("id", 0)):
            print(f"Entry {entry.get('id', 0)} already evaluated, skipping")
            return None
        
        try:
            query = entry["query"]
            ground_truth = entry["ground_truth"]
            
            # Extract parameters using cached function
            extracted_params = self.cached_extract_params(query)
            
            # Calculate slot filling accuracy
            sfa_score, slot_details = self._calculate_slot_accuracy(extracted_params, ground_truth)
            
            # Set up timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            
            try:
                # Execute agent with the query - limit retries
                start_time = time.time()
                success, result, history = agent_execute_with_retry(query, retry_times=1)
                processing_time = time.time() - start_time
                
                # Cancel timeout
                signal.alarm(0)
            except TimeoutException:
                print(f"Warning: Evaluation for entry {entry.get('id', 0)} timed out after {self.timeout} seconds")
                success = False
                result = "TIMEOUT: Evaluation took too long"
                history = [(query, result)]
                processing_time = self.timeout
            except Exception as e:
                print(f"Error executing agent for entry {entry.get('id', 0)}: {e}")
                success = False
                result = f"ERROR: {str(e)}"
                history = [(query, result)]
                processing_time = time.time() - start_time
            
            # Check task completion
            tcr_score = 1.0 if success and self._check_task_completion(result) else 0.0
            
            # Check first turn success
            ftsr_score = 1.0 if success and self._check_first_turn_success(result, history) else 0.0
            
            # Store metrics
            self.metrics["sfa"].append(sfa_score)
            self.metrics["tcr"].append(tcr_score)
            self.metrics["ftsr"].append(ftsr_score)
            
            # Store prediction (includes both input and response)
            prediction = {
                "id": entry.get("id", 0),
                "query": query,
                "ground_truth": ground_truth,
                "extracted_params": extracted_params,
                "metrics": {
                    "sfa_score": sfa_score,
                    "tcr_score": tcr_score,
                    "ftsr_score": ftsr_score,
                    "processing_time": processing_time
                },
                "response": result,
                "success": success,
                "timed_out": isinstance(result, str) and result.startswith("TIMEOUT"),
                "error": isinstance(result, str) and result.startswith("ERROR")
            }
            
            self.predictions.append(prediction)
            
            # Save checkpoint after each evaluation
            self._save_checkpoint()
            
            # Update and save predictions file after each evaluation
            with open(self.predictions_file, 'w', encoding='utf-8') as f:
                json.dump(self.predictions, f, ensure_ascii=False, indent=2)
            
            return {
                "sfa": sfa_score,
                "tcr": tcr_score,
                "ftsr": ftsr_score
            }
        
        except Exception as e:
            print(f"Error evaluating entry: {e}")
            return {
                "sfa": 0.0,
                "tcr": 0.0,
                "ftsr": 0.0
            }
        finally:
            # Ensure timeout is cancelled
            signal.alarm(0)
    
    def evaluate_dataset(self):
        """Evaluate the dataset"""
        # Filter out already evaluated entries
        entries_to_evaluate = []
        for entry in self.benchmark_data:
            if not self._is_already_evaluated(entry.get("id", 0)):
                entries_to_evaluate.append(entry)
        
        remaining = len(entries_to_evaluate)
        print(f"Found {len(self.benchmark_data) - remaining} already evaluated entries")
        print(f"Evaluating {remaining} remaining entries...")
        
        if remaining == 0:
            print("All entries already evaluated!")
            return self.get_results()
            
        try:
            for entry in tqdm(entries_to_evaluate):
                self.evaluate_entry(entry)
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user. Progress has been saved.")
        
        # Calculate overall metrics
        self.results = self.get_results()
        
        # Save final results
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"Final evaluation results saved to {self.results_file}")
        print(f"Predictions saved to {self.predictions_file}")
        
        return self.results
    
    def get_results(self):
        """Get overall evaluation results"""
        results = {}
        
        # Calculate averages for each metric
        for metric, values in self.metrics.items():
            if values:
                results[metric] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "std": float(np.std(values)),
                    "count": len(values)
                }
            else:
                results[metric] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "std": 0.0,
                    "count": 0
                }
        
        # Add overall success rate
        timed_out_count = sum(1 for r in self.predictions if r.get("timed_out", False))
        error_count = sum(1 for r in self.predictions if r.get("error", False))
        success_count = sum(1 for r in self.predictions if r.get("success", False))
        
        total = len(self.predictions)
        
        results["overall"] = {
            "total_entries": total,
            "success_rate": float(success_count / total) if total > 0 else 0.0,
            "timeout_rate": float(timed_out_count / total) if total > 0 else 0.0,
            "error_rate": float(error_count / total) if total > 0 else 0.0
        }
        
        return results
    
    def print_results(self):
        """Print evaluation results in a formatted way"""
        if not self.results:
            self.results = self.get_results()
        
        print("\n===== EVALUATION RESULTS =====")
        print(f"Total Entries Evaluated: {self.results['overall']['total_entries']}")
        print(f"Success Rate: {self.results['overall']['success_rate']:.2%}")
        print(f"Timeout Rate: {self.results['overall']['timeout_rate']:.2%}")
        print(f"Error Rate: {self.results['overall']['error_rate']:.2%}")
        
        for metric, stats in self.results.items():
            if metric == "overall":
                continue
                
            metric_name = {
                "tcr": "Task Completion Rate",
                "sfa": "Slot Filling Accuracy",
                "ftsr": "First-Turn Success Rate"
            }.get(metric, metric)
            
            print(f"\n{metric_name}:")
            print(f"  Mean:   {stats['mean']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Min:    {stats['min']:.4f}")
            print(f"  Max:    {stats['max']:.4f}")
            print(f"  StdDev: {stats['std']:.4f}")
        
        print("\n=============================")

def main():
    """Main function to run evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate flight search agent performance')
    parser.add_argument('--input', type=str, default="benchmark_dataset.json",
                      help='Path to benchmark dataset (default: benchmark_dataset.json)')
    parser.add_argument('--max-entries', type=int, default=None,
                      help='Maximum number of entries to evaluate (default: all)')
    parser.add_argument('--predictions', type=str, default="predictions.json",
                      help='File to save individual predictions (default: predictions.json)')
    parser.add_argument('--results', type=str, default="evaluation_results.json",
                      help='File to save summary results (default: evaluation_results.json)')
    parser.add_argument('--cache-dir', type=str, default="eval_cache",
                      help='Directory for cache files (default: eval_cache)')
    parser.add_argument('--timeout', type=int, default=10,
                      help='Timeout in seconds for each evaluation (default: 45)')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"- Input dataset: {args.input}")
    print(f"- Max entries: {'all' if args.max_entries is None else args.max_entries}")
    print(f"- Predictions file: {args.predictions}")
    print(f"- Results file: {args.results}")
    print(f"- Cache directory: {args.cache_dir}")
    print(f"- Timeout: {args.timeout} seconds")
    
    # Initialize evaluator
    evaluator = FlightSearchEvaluator(
        benchmark_file=args.input,
        max_entries=args.max_entries,
        predictions_file=args.predictions,
        results_file=args.results,
        cache_dir=args.cache_dir
    )
    
    # Set timeout from args
    evaluator.timeout = args.timeout
    
    # Evaluate dataset
    evaluator.evaluate_dataset()
    
    # Print results
    evaluator.print_results()

if __name__ == "__main__":
    main()