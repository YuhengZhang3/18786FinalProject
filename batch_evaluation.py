import argparse
import time
import subprocess
import os
import json
import glob
import hashlib

def calculate_similarity(query1, query2):
    """Calculate similarity between two queries"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, query1.lower(), query2.lower()).ratio()

def run_batch_evaluation(benchmark_file, output_prefix, start_idx, end_idx, api_delay=2, batch_delay=120):
    """Run evaluation for a specific range of samples with optimizations
    
    Args:
        benchmark_file (str): Benchmark dataset file
        output_prefix (str): Output file prefix
        start_idx (int): Starting sample index
        end_idx (int): Ending sample index
        api_delay (float): Delay between API calls (seconds)
        batch_delay (float): Delay between batches (seconds)
    """
    # Create output directory
    os.makedirs("batch_results", exist_ok=True)
    
    # Generate batch output filename
    output_file = f"batch_results/{output_prefix}_batch_{start_idx}_{end_idx}.json"
    
    # Build command with cache enabled
    cmd = [
        "python", "evaluation.py",
        "--benchmark", benchmark_file,
        "--output", output_file,
        "--samples", str(end_idx - start_idx),
        "--start-index", str(start_idx),
        "--api-delay", str(api_delay),
        "--use-cache"  # Enable caching to reduce API calls
    ]
    
    print("\n" + "="*60)
    print(f"Running batch evaluation [{start_idx}:{end_idx}]")
    print("="*60)
    print(f"Command: {' '.join(cmd)}")
    
    # Execute evaluation
    result = subprocess.run(cmd)
    
    # Check if successful
    if result.returncode != 0:
        print(f"Batch [{start_idx}:{end_idx}] evaluation failed with return code: {result.returncode}")
    else:
        print(f"Batch [{start_idx}:{end_idx}] evaluation completed successfully!")
    
    # Wait between batches if not the last batch
    if end_idx < 100:
        print(f"Waiting {batch_delay} seconds before next batch...")
        time.sleep(batch_delay)

def optimize_batches(benchmark_file, batch_size):
    """Optimize batch distribution based on query similarity to save API calls
    
    Args:
        benchmark_file (str): Path to benchmark dataset file
        batch_size (int): Default batch size
        
    Returns:
        list: List of (start_idx, end_idx) tuples for optimized batches
    """
    try:
        # Load benchmark data
        with open(benchmark_file, 'r') as f:
            benchmark_data = json.load(f)
        
        # If very small dataset, just use one batch
        if len(benchmark_data) <= batch_size:
            return [(0, len(benchmark_data))]
        
        # Group similar queries within the same batch
        batches = []
        current_batch = []
        
        # Sort samples by query similarity
        for i, sample in enumerate(benchmark_data):
            if len(current_batch) >= batch_size:
                # Start a new batch if current is full
                start_idx = min(s["id"] for s in current_batch) - 1  # Adjust for 0-indexing
                end_idx = max(s["id"] for s in current_batch)
                batches.append((start_idx, end_idx))
                current_batch = []
            
            current_batch.append(sample)
        
        # Add the final batch
        if current_batch:
            start_idx = min(s["id"] for s in current_batch) - 1  # Adjust for 0-indexing
            end_idx = max(s["id"] for s in current_batch)
            batches.append((start_idx, end_idx))
        
        print(f"Optimized {len(benchmark_data)} samples into {len(batches)} batches")
        return batches
        
    except Exception as e:
        print(f"Error optimizing batches: {e}")
        # Fall back to default batch distribution
        total_samples = 100  # Default assumption
        return [(i, min(i + batch_size, total_samples)) for i in range(0, total_samples, batch_size)]

def merge_results(output_prefix, final_output):
    """Merge all batch results
    
    Args:
        output_prefix (str): Output file prefix
        final_output (str): Final merged results file
    """
    print("\n" + "="*60)
    print(f"Merging batch results to: {final_output}")
    print("="*60)
    
    # Get all batch result files
    batch_files = glob.glob(f"batch_results/{output_prefix}_batch_*.json")
    batch_files.sort()
    
    if not batch_files:
        print(f"No batch result files found!")
        return
    
    print(f"Found {len(batch_files)} batch result files:")
    for file in batch_files:
        print(f"  - {file}")
    
    # Initialize merged results
    merged_results = {
        "metrics": {
            "sfa": 0,  # Slot Filling Accuracy
            "tcr": 0,  # Task Completion Rate
            "ftsr": 0  # First-Turn Success Rate
        },
        "detailed_results": []
    }
    
    # Merge all batch results
    total_samples = 0
    for file in batch_files:
        with open(file, 'r') as f:
            try:
                batch_results = json.load(f)
                
                # Add detailed results
                merged_results["detailed_results"].extend(batch_results["detailed_results"])
                total_samples += len(batch_results["detailed_results"])
                
                print(f"Added {len(batch_results['detailed_results'])} sample results from {file}")
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON data in {file}")
    
    # Recalculate overall metrics
    if total_samples > 0:
        merged_results["metrics"]["sfa"] = sum(1 for r in merged_results["detailed_results"] if r.get("sfa", False)) / total_samples
        merged_results["metrics"]["tcr"] = sum(1 for r in merged_results["detailed_results"] if r.get("tcr", False)) / total_samples
        merged_results["metrics"]["ftsr"] = sum(1 for r in merged_results["detailed_results"] if r.get("ftsr", False)) / total_samples
    
        # Save merged results
        with open(final_output, 'w') as f:
            json.dump(merged_results, f, indent=2)
        
        print(f"\nMerged {len(batch_files)} batch results, total {total_samples} samples")
        print(f"Overall metrics:")
        print(f"  - Slot Filling Accuracy (SFA): {merged_results['metrics']['sfa']:.2%}")
        print(f"  - Task Completion Rate (TCR): {merged_results['metrics']['tcr']:.2%}")
        print(f"  - First-Turn Success Rate (FTSR): {merged_results['metrics']['ftsr']:.2%}")
        print(f"Results saved to: {final_output}")
    else:
        print("Warning: No valid sample results found, skipping merge")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch run evaluation tasks')
    parser.add_argument('--benchmark', type=str, required=True, help='Path to benchmark dataset file')
    parser.add_argument('--output', type=str, required=True, help='Output file prefix for evaluation results')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of samples to process per batch')
    parser.add_argument('--api-delay', type=float, default=2.0, help='Delay between API calls (seconds)')
    parser.add_argument('--batch-delay', type=int, default=120, help='Delay between batches (seconds)')
    parser.add_argument('--merge-only', action='store_true', help='Only merge existing results, don\'t run new evaluations')
    parser.add_argument('--optimize-batches', action='store_true', help='Optimize batch distribution to save API calls')
    parser.add_argument('--use-cache', action='store_true', help='Use response caching to reduce API calls')
    args = parser.parse_args()
    
    # Final output file
    final_output = f"{args.output}_final.json"
    
    # If only merging results
    if args.merge_only:
        merge_results(args.output, final_output)
        return
    
    # Process in batches
    print(f"Configuration:")
    print(f"- Benchmark dataset: {args.benchmark}")
    print(f"- Output prefix: {args.output}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- API delay: {args.api_delay} seconds")
    print(f"- Batch delay: {args.batch_delay} seconds")
    print(f"- Optimize batches: {args.optimize_batches}")
    
    # Get batch ranges
    if args.optimize_batches:
        batch_ranges = optimize_batches(args.benchmark, args.batch_size)
    else:
        # Default batch distribution
        batch_ranges = [(i, min(i + args.batch_size, 100)) for i in range(0, 100, args.batch_size)]
    
    # Create response cache file if it doesn't exist
    if not os.path.exists("agent_response_cache.json"):
        with open("agent_response_cache.json", 'w') as f:
            json.dump({}, f)
    
    # Run each batch
    for start_idx, end_idx in batch_ranges:
        run_batch_evaluation(
            args.benchmark,
            args.output,
            start_idx,
            end_idx,
            args.api_delay,
            args.batch_delay
        )
    
    # Merge all batch results
    merge_results(args.output, final_output)
    
    print(f"All batch evaluations and merging complete!")

if __name__ == "__main__":
    main()