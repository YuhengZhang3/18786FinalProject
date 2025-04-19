import argparse
import time
import subprocess
import os
import json
import glob

def run_batch_evaluation(benchmark_file, output_prefix, start_idx, end_idx, api_delay=2, batch_delay=120):
    """Run evaluation for a specific range of samples
    
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
    
    # Build command
    cmd = [
        "python", "evaluation.py",
        "--benchmark", benchmark_file,
        "--output", output_file,
        "--samples", str(end_idx - start_idx),
        "--start-index", str(start_idx),
        "--api-delay", str(api_delay)
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
            "sfa": 0,
            "tcr": 0,
            "ftsr": 0
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
        print(f"  - SFA: {merged_results['metrics']['sfa']:.2%}")
        print(f"  - TCR: {merged_results['metrics']['tcr']:.2%}")
        print(f"  - FTSR: {merged_results['metrics']['ftsr']:.2%}")
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
    
    for start_idx in range(0, 100, args.batch_size):
        end_idx = min(start_idx + args.batch_size, 100)
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