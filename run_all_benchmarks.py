#!/usr/bin/env python3
"""
Wrapper script to run evaluation.py across all datasets and languages.
Discovers available datasets and languages by checking for manifest.json files.
"""

import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

def discover_benchmarks(benchmarks_root="/home/vistaar/benchmarks"):
    """
    Discover all available dataset/language combinations by looking for manifest.json files.
    
    Returns:
        dict: {dataset_name: [list of languages]}
    """
    benchmarks = defaultdict(list)
    
    benchmarks_path = Path(benchmarks_root)
    
    # Look for manifest.json files
    for manifest in benchmarks_path.glob("*/*/manifest.json"):
        parts = manifest.parent.parts
        dataset = parts[-2]
        language = parts[-1]
        benchmarks[dataset].append(language)
    
    # Sort for consistent ordering
    for dataset in benchmarks:
        benchmarks[dataset].sort()
    
    return dict(sorted(benchmarks.items()))

def run_evaluation(model_path, manifest_path, dataset_name, language, api_url="http://localhost:6769/v1/audio/transcriptions", num_workers=1, num_endpoints=8):
    """
    Run evaluation.py for a single dataset/language combination.
    """
    manifest_name = f"{dataset_name}_{language}"
    
    cmd = [
        "python", "/home/vistaar/evaluation.py",
        "--model_path", model_path,
        "--manifest_path", manifest_path,
        "--manifest_name", manifest_name,
        "--language", language,
        "--api_url", api_url,
        "--num_workers", str(num_workers),
        "--num_endpoints", str(num_endpoints),
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {dataset_name} / {language}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Evaluation failed for {dataset_name}/{language}")
        print(f"Command: {' '.join(cmd)}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation.py across all benchmarks")
    parser.add_argument(
        "--model_path",
        type=str,
        default="omniASR_LLM_Unlimited_7B_v2",
        help="Model name to use with the API",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:6769/v1/audio/transcriptions",
        help="API endpoint URL for transcription",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel API workers (default: 8)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run only a specific dataset (optional)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Run only a specific language (optional)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what will be processed without actually running evaluations",
    )
    parser.add_argument(
        "--num_endpoints",
        type=int,
        default=8,
        help="Number of parallel endpoints to distribute load across, starting from port 6769 (default: 8)",
    )
    
    args = parser.parse_args()
    
    # Discover available benchmarks
    benchmarks = discover_benchmarks()
    
    if not benchmarks:
        print("ERROR: No benchmarks found. Check /home/vistaar/benchmarks/")
        return 1
    
    print(f"Found {sum(len(langs) for langs in benchmarks.values())} benchmark combinations:")
    for dataset, languages in benchmarks.items():
        print(f"  {dataset}: {', '.join(languages)}")
    
    # Filter by dataset/language if specified
    if args.dataset:
        if args.dataset not in benchmarks:
            print(f"ERROR: Dataset '{args.dataset}' not found")
            return 1
        benchmarks = {args.dataset: benchmarks[args.dataset]}
    
    # Build the list of tasks to process
    tasks = []
    for dataset, languages in benchmarks.items():
        for language in languages:
            if args.language and language != args.language:
                continue
            tasks.append((dataset, language))
    
    # Dry run: just show what would be processed
    if args.dry_run:
        print(f"\nDRY RUN - Will process {len(tasks)} benchmark(s):")
        for dataset, language in tasks:
            manifest_path = f"/home/vistaar/benchmarks/{dataset}/{language}/manifest.json"
            print(f"  {dataset:20s} / {language:15s}  ->  {manifest_path}")
        print(f"\nRun without --dry_run to start evaluation")
        return 0
    
    # Run evaluations
    total = 0
    passed = 0
    failed = 0
    
    for dataset, language in tasks:
        total += 1
        manifest_path = f"/home/vistaar/benchmarks/{dataset}/{language}/manifest.json"
        
        if run_evaluation(
            model_path=args.model_path,
            manifest_path=manifest_path,
            dataset_name=dataset,
            language=language,
            api_url=args.api_url,
            num_workers=args.num_workers,
            num_endpoints=args.num_endpoints,
        ):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
