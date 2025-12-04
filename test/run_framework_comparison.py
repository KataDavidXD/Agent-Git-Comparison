#!/usr/bin/env python3
"""
Example usage of the improved FrameworkComparator class.

This demonstrates how to use the new flexible multi-framework comparison
with configurable parameters.
"""

import sys
import os
import glob
from pathlib import Path

# Add src directory to path to import agentgit_paper modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
print(sys.path)

from experiments.evaluator import FrameworkComparator

def get_latest_eval_file(topic, framework, results_dir):
    """
    Find the latest evaluation file for a given topic and framework.
    
    Args:
        topic (str): The topic (e.g., 'large_language_models')
        framework (str): The framework name (e.g., 'sections', 'timetravel', 'autogen', 'agno')
        results_dir (str): Base results directory
        
    Returns:
        str: Path to the latest evaluation JSON file
        
    Raises:
        FileNotFoundError: If no matching directory is found
    """
    # Pattern to match: {topic}_{framework}_*
    pattern = f"{topic}_{framework}_*"
    search_path = os.path.join(results_dir, pattern)
    
    # Find all matching directories
    matching_dirs = glob.glob(search_path)
    
    if not matching_dirs:
        raise FileNotFoundError(f"No directories found matching pattern: {pattern}")
    
    # Sort to get the latest (timestamps are sortable)
    latest_dir = max(matching_dirs)
    
    # Construct the evaluation file path
    eval_file = os.path.join(latest_dir, "eval", f"evaluation_results_{os.path.basename(latest_dir)}.json")
    
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
    
    return eval_file

def build_framework_files(topic, frameworks, results_dir):
    """
    Build framework_files dictionary dynamically.
    
    Args:
        topic (str): The topic (e.g., 'large_language_models')
        frameworks (dict): Dictionary mapping framework names to their directory suffixes
                          e.g., {'agent_git': 'sections', 'langgraph': 'timetravel', ...}
        results_dir (str): Base results directory
        
    Returns:
        dict: Framework files dictionary ready for FrameworkComparator
    """
    framework_files = {}
    
    for framework_name, framework_suffix in frameworks.items():
        try:
            eval_file = get_latest_eval_file(topic, framework_suffix, results_dir)
            framework_files[framework_name] = eval_file
            print(f"Found {framework_name}: {eval_file}")
        except FileNotFoundError as e:
            print(f"Warning: Could not find evaluation file for {framework_name}: {e}")
    
    return framework_files

def example_basic_comparison():
    """Example: Basic comparison with multiple frameworks using dynamic file discovery."""
    print("="*80)
    print("EXAMPLE 1: Basic Multi-Framework Comparison (Dynamic)")
    print("="*80)
    
    # Define topic and framework mappings
    topic = "large_language_models"
    frameworks = {
        'agent_git': 'sections',
        'langgraph': 'timetravel', 
        'autogen': 'autogen',
        'agno': 'agno'
    }
    
    # Build framework files dynamically
    try:
        result_base_dir = str(Path(__file__).parent.parent / "results")
        framework_files = build_framework_files(topic, frameworks, result_base_dir)
        
        if not framework_files:
            print("No evaluation files found!")
            return
            
        print(f"\nFound {len(framework_files)} frameworks:")
        for name, path in framework_files.items():
            print(f"  {name}: {os.path.basename(os.path.dirname(os.path.dirname(path)))}")
        
        # Create comparator with dynamically found files
        comparator = FrameworkComparator(framework_files)
        comparator.compare_all()
        
    except Exception as e:
        print(f"Error during comparison: {e}")

def example_custom_dimensions():
    """Example: Comparison with custom quality dimensions."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Quality Dimensions")
    print("="*80)
    
    framework_files = {
        'framework_a': 'results/framework_a_eval.json',
        'framework_b': 'results/framework_b_eval.json'
    }
    
    # Custom dimensions - only compare fluency and coherence
    custom_dimensions = ['fluency', 'coherence']
    
    try:
        comparator = FrameworkComparator(
            framework_files,
            comparison_dimensions=custom_dimensions,
            comparison_name="fluency_coherence_comparison"
        )
        comparator.compare_all()
    except FileNotFoundError:
        print("Note: Example files not found - this demonstrates the API")

def example_custom_metrics():
    """Example: Comparison with custom execution metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Execution Metrics")
    print("="*80)
    
    framework_files = {
        'fast_framework': 'results/fast_eval.json',
        'accurate_framework': 'results/accurate_eval.json'
    }
    
    # Only compare time metrics, not token usage
    custom_metrics = ['total_time']
    
    try:
        comparator = FrameworkComparator(
            framework_files,
            execution_metrics=custom_metrics,
            comparison_name="speed_comparison"
        )
        # Only run specific comparisons 
        comparator.compare_all(include_metrics=['execution', 'quality'])
    except FileNotFoundError:
        print("Note: Example files not found - this demonstrates the API")

def main():
    """Run all examples."""
    # example_basic_comparison()
    comparator = FrameworkComparator({
        "LangGraph_+_Agent_Git": "/Users/ethan/Downloads/Paper/results/large_language_models/framework_experiments/agentgit_evaluation_summary.json",
        "LangGraph": "/Users/ethan/Downloads/Paper/results/large_language_models/framework_experiments/langgraph_time_travel_evaluation_summary.json",
        "AutoGen": "/Users/ethan/Downloads/Paper/results/large_language_models/framework_experiments/autogen_evaluation_summary.json",
        "Agno": "/Users/ethan/Downloads/Paper/results/large_language_models/framework_experiments/agno_evaluation_summary.json",
    })
    comparator.compare_all()

if __name__ == "__main__":
    main()