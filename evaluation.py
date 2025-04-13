import json
import time
import signal
import numpy as np
from tqdm import tqdm
from agent import agent_execute_with_retry
from nlp import extract_flight_parameters

class TimeoutException(Exception):
    """Exception raised when evaluation takes too long"""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutException("Evaluation timed out")

class FlightSearchEvaluator:
    """Evaluator for flight search agent"""
    
    def __init__(self, benchmark_file="benchmark_dataset.json"):
        """Initialize with benchmark dataset"""
        # Load benchmark dataset
        self.benchmark_data = self._load_benchmark_data(benchmark_file)
        
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
        
        # Initialize detailed results
        self.detailed_results = []
    
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
                # Success indicators
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
    
    def evaluate_entry(self, entry, timeout=60):
        """Evaluate a single benchmark entry"""
        try:
            query = entry["query"]
            ground_truth = entry["ground_truth"]
            
            # Extract parameters
            extracted_params = extract_flight_parameters(query)
            
            # Calculate slot filling accuracy
            sfa_score, slot_details = self._calculate_slot_accuracy(extracted_params, ground_truth)
            
            # Set up timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                # Execute agent with the query - limit to just 1 retry
                start_time = time.time()
                success, result, history = agent_execute_with_retry(query, retry_times=1)
                processing_time = time.time() - start_time
                
                # Cancel timeout
                signal.alarm(0)
            except TimeoutException:
                print(f"Warning: Evaluation for entry {entry.get('id', 0)} timed out after {timeout} seconds")
                success = False
                result = "TIMEOUT: Evaluation took too long"
                history = [(query, result)]
                processing_time = timeout
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
            
            # Store detailed results
            detail = {
                "id": entry.get("id", 0),
                "query": query,
                "ground_truth": ground_truth,
                "extracted_params": extracted_params,
                "slot_scores": slot_details,
                "sfa_score": sfa_score,
                "tcr_score": tcr_score,
                "ftsr_score": ftsr_score,
                "processing_time": processing_time,
                "response": result,
                "success": success,
                "timed_out": isinstance(result, str) and result.startswith("TIMEOUT"),
                "error": isinstance(result, str) and result.startswith("ERROR")
            }
            
            self.detailed_results.append(detail)
            
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
    
    def evaluate_in_batches(self, batch_size=10, max_entries=None, save_interim=True):
        """Evaluate the dataset in batches"""
        if max_entries and max_entries < len(self.benchmark_data):
            data_to_evaluate = self.benchmark_data[:max_entries]
        else:
            data_to_evaluate = self.benchmark_data
        
        total_entries = len(data_to_evaluate)
        num_batches = (total_entries + batch_size - 1) // batch_size
        
        print(f"Evaluating {total_entries} entries in {num_batches} batches of size {batch_size}...")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_entries)
            
            print(f"Processing batch {batch_idx+1}/{num_batches} (entries {start_idx+1}-{end_idx})...")
            
            batch_data = data_to_evaluate[start_idx:end_idx]
            for entry in tqdm(batch_data):
                self.evaluate_entry(entry)
            
            # Save interim results after each batch
            if save_interim:
                interim_file = f"evaluation_interim_batch_{batch_idx+1}.json"
                with open(interim_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "summary": self.get_results(),
                        "detailed_results": self.detailed_results,
                        "progress": f"{end_idx}/{total_entries} entries processed"
                    }, f, ensure_ascii=False, indent=2)
                
                print(f"Interim results saved to {interim_file}")
        
        # Calculate overall metrics
        results = self.get_results()
        
        # Save final results
        final_file = "evaluation_results_final.json"
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": results,
                "detailed_results": self.detailed_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Final evaluation results saved to {final_file}")
        
        return results
    
    def evaluate_dataset(self, max_entries=None):
        """Legacy method for evaluating the entire dataset at once"""
        return self.evaluate_in_batches(batch_size=10, max_entries=max_entries)
    
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
        timed_out_count = sum(1 for r in self.detailed_results if r.get("timed_out", False))
        error_count = sum(1 for r in self.detailed_results if r.get("error", False))
        success_count = sum(1 for r in self.detailed_results if r.get("success", False))
        
        total = len(self.detailed_results)
        
        results["overall"] = {
            "total_entries": total,
            "success_rate": float(success_count / total) if total > 0 else 0.0,
            "timeout_rate": float(timed_out_count / total) if total > 0 else 0.0,
            "error_rate": float(error_count / total) if total > 0 else 0.0
        }
        
        return results
    
    def print_results(self):
        """Print evaluation results in a formatted way"""
        results = self.get_results()
        
        print("\n===== EVALUATION RESULTS =====")
        print(f"Total Entries Evaluated: {results['overall']['total_entries']}")
        print(f"Success Rate: {results['overall']['success_rate']:.2%}")
        print(f"Timeout Rate: {results['overall']['timeout_rate']:.2%}")
        print(f"Error Rate: {results['overall']['error_rate']:.2%}")
        
        for metric, stats in results.items():
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
    evaluator = FlightSearchEvaluator()
    
    # Evaluate with batch processing for better stability
    evaluator.evaluate_in_batches(batch_size=5, max_entries=50)
    
    # Print results
    evaluator.print_results()

if __name__ == "__main__":
    main()