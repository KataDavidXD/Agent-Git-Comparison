#!/usr/bin/env python3
"""
Run framework evaluation script.

Direct parameter configuration - modify the parameters below:
"""

import sys
import glob
from pathlib import Path

# =============================================================================
# CONFIGURATION - Modify these parameters
# =============================================================================
TOPIC = "large_language_models"        # Topic name
FRAMEWORKS = ["agno", "autogen", "sections", "timetravel"]  # List of frameworks to evaluate
MODEL = "gpt-4o-mini"                  # Model for evaluation
SAMPLES = 5                            # Number of G-Eval samples
SKIP_EVAL = True                       # Set to True to skip evaluation, False to run it
# =============================================================================

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.evaluator import BranchEvaluator, ResultsAnalyzer


def find_latest_run_dir(results_base_dir, topic, framework):
    """Find the latest run directory for given topic and framework."""
    pattern = f"{topic}_{framework}_*"
    matching_dirs = list(results_base_dir.glob(pattern))
    
    if not matching_dirs:
        return None
    
    # Sort by directory name (which includes timestamp) to get the latest
    matching_dirs.sort(key=lambda x: x.name, reverse=True)
    return matching_dirs[0]


def main():
    """Main function to run evaluation and analysis."""
    print(f"\n🚀 Starting Framework Evaluation")
    print(f"{'='*60}")
    print(f"Topic:      {TOPIC}")
    print(f"Frameworks: {', '.join(FRAMEWORKS)}")
    print(f"Model:      {MODEL}")
    print(f"Samples:    {SAMPLES}")
    print(f"Skip Eval:  {SKIP_EVAL}")
    print(f"{'='*60}")
    
    # Construct paths
    project_root = Path(__file__).parent.parent
    results_base_dir = project_root / "results"
    
    # Process each framework
    for framework in FRAMEWORKS:
        print(f"\n{'='*60}")
        print(f"🔄 Processing Framework: {framework}")
        print(f"{'='*60}")
        
        print(f"\n[SEARCH] Looking for results: {TOPIC}_{framework}_*")
        print(f"         In directory: {results_base_dir}")
        
        # Find the latest run directory
        run_dir = find_latest_run_dir(results_base_dir, TOPIC, framework)
        
        if not run_dir:
            print(f"\n⚠️  WARNING: No run directory found for topic '{TOPIC}' and framework '{framework}'")
            print(f"Looking for pattern: {TOPIC}_{framework}_*")
            print(f"In directory: {results_base_dir}")
            
            # Show available directories for this framework
            available_dirs = [d.name for d in results_base_dir.iterdir() 
                            if d.is_dir() and framework in d.name]
            if available_dirs:
                print(f"\nAvailable directories for {framework}:")
                for dir_name in sorted(available_dirs):
                    print(f"  - {dir_name}")
            print(f"Skipping framework: {framework}")
            continue
        
        print(f"[FOUND] Using run directory: {run_dir.name}")
        
        # Find the raw results JSON file
        raw_dir = run_dir / "raw"
        if not raw_dir.exists():
            print(f"⚠️  WARNING: Raw directory not found: {raw_dir}")
            print(f"Skipping framework: {framework}")
            continue
        
        # Find JSON file in raw directory
        json_files = list(raw_dir.glob("results_*.json"))
        if not json_files:
            print(f"⚠️  WARNING: No results JSON file found in {raw_dir}")
            print(f"Skipping framework: {framework}")
            continue
        
        results_file = json_files[0]  # Take the first (should be only one)
        print(f"[INPUT] Found results file: {results_file.name}")
        
        # Check if evaluation should be run or skipped
        eval_file = run_dir / "eval" / f"evaluation_{results_file.stem}.json"
        
        if not SKIP_EVAL:
            # Run evaluation
            print(f"\n[EVAL] Starting evaluation with {MODEL} (samples={SAMPLES})")
            try:
                evaluator = BranchEvaluator(
                    model=MODEL,
                    num_samples=SAMPLES
                )
                
                eval_results = evaluator.evaluate_all_branches(str(results_file))
                print(f"[EVAL] Evaluation completed successfully for {framework}")
                
            except Exception as e:
                print(f"❌ ERROR during evaluation for {framework}: {e}")
                continue
                
        else:
            # Use existing evaluation file from eval/ folder
            if not eval_file.exists():
                print(f"⚠️  WARNING: Evaluation file not found: {eval_file}")
                print(f"Set SKIP_EVAL = False to generate evaluation first.")
                print(f"Skipping framework: {framework}")
                continue
            print(f"\n[SKIP EVAL] Using existing evaluation: {eval_file.name}")
        
        # Run analysis and visualization
        print(f"\n[VIZ] Creating visualizations for {framework}...")
        try:
            analyzer = ResultsAnalyzer(str(eval_file))
            analyzer.create_all_visualizations()
            print(f"[VIZ] Visualizations created successfully for {framework}")
            
        except Exception as e:
            print(f"❌ ERROR during visualization for {framework}: {e}")
            continue
        
        print(f"✅ Framework {framework} completed successfully!")
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ ALL FRAMEWORKS EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nTopic: {TOPIC}")
    print(f"Frameworks: {', '.join(FRAMEWORKS)}")
    print(f"\n📋 Results can be found in:")
    for framework in FRAMEWORKS:
        run_dir = find_latest_run_dir(results_base_dir, TOPIC, framework)
        if run_dir:
            print(f"  {framework}: {run_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())