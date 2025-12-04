"""
Evaluator for paper summarization branches using G-Eval.

This module evaluates all branches from agentgitAgent.py using multi-dimensional
evaluation (fluency, coherence, consistency, relevance) and visualizes results.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import G-Eval components
import sys
# Add both parent and project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluators.geval_evaluator import GEvalEvaluator
from evaluators.llm_client import LLMClient
from evaluators.prompt_templates import (
    FLUENCY, COHERENCE, CONSISTENCY, RELEVANCE,
    DEFAULT_SUMMARIZATION_DIMENSIONS
)

load_dotenv()


class BranchEvaluator:
    """Evaluates all branches from paper summarization workflow."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4",
        num_samples: int = 20,
        temperature: float = 2.0
    ):
        """Initialize evaluator.
        
        Args:
            api_key: OpenAI API key (or from env)
            base_url: Base URL for API (or from env)
            model: Model to use for evaluation
            num_samples: Number of samples for G-Eval (default 20 from paper)
            temperature: Temperature for G-Eval (default 2.0 from paper)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        
        # Initialize G-Eval evaluator
        self.evaluator = GEvalEvaluator(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            dimensions=DEFAULT_SUMMARIZATION_DIMENSIONS,
            num_samples=num_samples,
            temperature=temperature
        )
        
        print(f"[INIT] BranchEvaluator initialized with model: {model}")
        print(f"       G-Eval settings: n={num_samples}, temp={temperature}")
    
    def load_results(self, results_file: str) -> Dict[str, Any]:
        """Load results from agentgitAgent.py or langgraph_time_travel_agent.py output."""
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        workflow_type = data.get('workflow_type', 'unified')
        
        print(f"\n[LOAD] Loaded results from: {results_file}")
        print(f"       Workflow type: {workflow_type}")
        print(f"       Run ID: {data.get('run_id', 'N/A')}")
        print(f"       Topic: {data['topic']}")
        print(f"       Total branches: {data['total_branches']}")
        
        # Handle different JSON structures
        if workflow_type == 'time_travel':
            # LangGraph time travel format - extract abstracts from first branch
            abstracts_count = data['branches'][0]['metadata']['num_papers'] if data['branches'] else 0
            print(f"       Papers analyzed: {abstracts_count}")
            # Convert to standard format by adding phase_1_state
            if 'phase_1_state' not in data:
                data['phase_1_state'] = self._extract_phase1_from_timetravel(data)
        else:
            # Standard format (section_based or unified)
            print(f"       Papers analyzed: {data['phase_1_state']['abstracts_count']}")
        
        return data
    
    def _extract_phase1_from_timetravel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract phase_1_state from LangGraph time travel results.
        
        Time travel JSON doesn't have phase_1_state, but we can reconstruct it
        from the branches since they all share the same abstracts.
        """
        print(f"       [CONVERT] Extracting phase_1_state from time travel format...")
        
        if not data.get('branches'):
            return {
                'search_results_count': 0,
                'abstracts_count': 0,
                'abstracts': []
            }
        
        # All branches share the same papers, so get from first branch's metadata
        first_branch = data['branches'][0]
        abstracts_count = first_branch['metadata']['num_papers']
        
        # Try to get abstracts from the saved data
        # Time travel saves abstracts in extract_checkpoint.values
        # But in the final JSON, we need to reconstruct from branches
        
        # For now, create a minimal phase_1_state
        # The evaluator will get abstracts from prepare_document call
        phase_1_state = {
            'search_results_count': 1,  # One search was performed
            'abstracts_count': abstracts_count,
            'abstracts': []  # Will be filled if available
        }
        
        # Try to extract abstracts if they're embedded in the JSON
        # Some versions might include them
        if 'phase_1_abstracts' in data:
            phase_1_state['abstracts'] = data['phase_1_abstracts']
        elif 'abstracts' in data:
            phase_1_state['abstracts'] = data['abstracts']
        
        print(f"       [CONVERT] Reconstructed phase_1_state with {abstracts_count} papers")
        
        return phase_1_state
    
    def prepare_document(self, abstracts: List[Dict[str, Any]]) -> str:
        """Prepare source document from abstracts."""
        doc_parts = []
        for i, paper in enumerate(abstracts, 1):
            title = paper.get('title', 'N/A')
            abstract = paper.get('abstract', 'N/A')
            authors = ', '.join(paper.get('authors', [])[:3])
            
            doc_parts.append(f"Paper {i}: {title}")
            doc_parts.append(f"Authors: {authors}")
            doc_parts.append(f"Abstract: {abstract}")
            doc_parts.append("")
        
        return "\n".join(doc_parts)
    
    def evaluate_section(
        self,
        section_name: str,
        section_content: str,
        source_document: str
    ) -> Dict[str, Any]:
        """Evaluate a single section using G-Eval.
        
        Args:
            section_name: Name of the section (e.g., 'introduction', 'analysis')
            section_content: Content of the section
            source_document: Source papers (abstracts)
            
        Returns:
            Evaluation results with scores for each dimension
        """
        result = self.evaluator.evaluate_multi_dimension(
            dimensions=DEFAULT_SUMMARIZATION_DIMENSIONS,
            Document=source_document,
            Summary=section_content
        )
        
        scores = {
            dim_name: dim_result.score 
            for dim_name, dim_result in result.dimensions.items()
        }
        
        return {
            "section": section_name,
            "scores": scores,
            "overall_score": result.overall_score,
            "dimension_details": {
                dim_name: {
                    "score": dim_result.score,
                    "raw_scores": dim_result.raw_scores,
                    "num_samples": len(dim_result.raw_scores)
                }
                for dim_name, dim_result in result.dimensions.items()
            }
        }
    
    def evaluate_branch(
        self,
        branch: Dict[str, Any],
        source_document: str,
        branch_idx: int,
        total_branches: int,
        evaluate_sections: bool = False
    ) -> Dict[str, Any]:
        """Evaluate a single branch using G-Eval.
        
        Args:
            branch: Branch data with summary/sections and metadata
            source_document: Source papers (abstracts)
            branch_idx: Index of this branch
            total_branches: Total number of branches
            evaluate_sections: If True, evaluate individual sections (for section-based workflow)
            
        Returns:
            Evaluation results with scores for each dimension
        """
        print(f"\n[EVAL {branch_idx}/{total_branches}] Branch: {branch['branch_id']}")
        
        # Detect workflow type
        if 'final_review' in branch:
            # Section-based workflow
            print(f"  Styles: {branch.get('prompt_intro_style', 'N/A')} + "
                  f"{branch.get('prompt_analysis_style', 'N/A')} + "
                  f"{branch.get('prompt_discussion_style', 'N/A')}")
            summary = branch['final_review']
        else:
            # Original workflow
            prompt_1_2 = branch.get('prompt_1_2', 'N/A')
            prompt_2_3 = branch.get('prompt_2_3', 'N/A')
            prompt_3_4 = branch.get('prompt_3_4', 'N/A') if 'prompt_3_4' in branch else None
            
            if prompt_3_4:
                print(f"  Prompts: {prompt_1_2} + {prompt_2_3} + {prompt_3_4}")
            else:
                print(f"  Prompts: {prompt_1_2} + {prompt_2_3}")
            
            summary = branch['summary']
        
        # Evaluate overall document
        result = self.evaluator.evaluate_multi_dimension(
            dimensions=DEFAULT_SUMMARIZATION_DIMENSIONS,
            Document=source_document,
            Summary=summary
        )
        
        # Extract scores
        scores = {
            dim_name: dim_result.score 
            for dim_name, dim_result in result.dimensions.items()
        }
        
        print(f"  Overall Scores: Fluency={scores['fluency']:.2f}, "
              f"Coherence={scores['coherence']:.2f}, "
              f"Consistency={scores['consistency']:.2f}, "
              f"Relevance={scores['relevance']:.2f}")
        print(f"  Overall: {result.overall_score:.2f}")
        
        evaluation_result = {
            "branch_id": branch['branch_id'],
            "scores": scores,
            "overall_score": result.overall_score,
            "dimension_details": {
                dim_name: {
                    "score": dim_result.score,
                    "raw_scores": dim_result.raw_scores,
                    "num_samples": len(dim_result.raw_scores)
                }
                for dim_name, dim_result in result.dimensions.items()
            },
            "metadata": branch['metadata']
        }
        
        # Add workflow-specific fields
        if 'final_review' in branch:
            evaluation_result.update({
                "prompt_intro": branch.get('prompt_intro', 'N/A'),
                "prompt_intro_style": branch.get('prompt_intro_style', 'N/A'),
                "prompt_analysis": branch.get('prompt_analysis', 'N/A'),
                "prompt_analysis_style": branch.get('prompt_analysis_style', 'N/A'),
                "prompt_discussion": branch.get('prompt_discussion', 'N/A'),
                "prompt_discussion_style": branch.get('prompt_discussion_style', 'N/A'),
            })
            
            # Optionally evaluate individual sections
            if evaluate_sections:
                print(f"  → Evaluating individual sections...")
                section_evaluations = {}
                
                for section_name in ['introduction', 'analysis', 'discussion']:
                    if section_name in branch and branch[section_name]:
                        section_eval = self.evaluate_section(
                            section_name=section_name,
                            section_content=branch[section_name],
                            source_document=source_document
                        )
                        section_evaluations[section_name] = section_eval
                        print(f"    {section_name.capitalize()}: {section_eval['overall_score']:.2f}")
                
                evaluation_result["section_evaluations"] = section_evaluations
        else:
            evaluation_result.update({
                "prompt_1_2": branch.get('prompt_1_2', 'N/A'),
                "prompt_2_3": branch.get('prompt_2_3', 'N/A'),
            })
            if 'prompt_3_4' in branch:
                evaluation_result["prompt_3_4"] = branch['prompt_3_4']
        
        return evaluation_result
    
    def evaluate_all_branches(
        self,
        results_file: str,
        output_file: Optional[str] = None,
        evaluate_sections: bool = False
    ) -> Dict[str, Any]:
        """Evaluate all branches from results file IN PARALLEL.
        
        Args:
            results_file: Path to results JSON from agentgitAgent.py or agentgitAgent_sections.py
            output_file: Path to save evaluation results (optional)
            evaluate_sections: If True, evaluate individual sections (for section-based workflow)
            
        Returns:
            Complete evaluation results
        """
        # Load results
        data = self.load_results(results_file)
        
        # Detect workflow type
        workflow_type = data.get('workflow_type', 'unified')
        if workflow_type == 'section_based':
            evaluate_sections = True  # Always evaluate sections for section-based workflow
        
        # Prepare source document
        abstracts = data['phase_1_state']['abstracts']
        source_document = self.prepare_document(abstracts)
        
        print(f"\n[SOURCE] Prepared source document ({len(source_document)} chars)")
        print(f"[WORKFLOW] Type: {workflow_type}")
        if evaluate_sections:
            print("[SECTIONS] Will evaluate individual sections")
        
        # Evaluate all branches IN PARALLEL
        print(f"\n{'='*80}")
        print("EVALUATING ALL BRANCHES (PARALLEL)")
        print(f"{'='*80}")
        
        total_branches = len(data['branches'])
        
        # Define evaluation task for parallel execution
        def evaluate_single_branch(i, branch):
            return self.evaluate_branch(
                branch=branch,
                source_document=source_document,
                branch_idx=i,
                total_branches=total_branches,
                evaluate_sections=evaluate_sections
            )
        
        # Run all evaluations in parallel
        branch_evaluations = []
        with ThreadPoolExecutor(max_workers=min(total_branches, 10)) as executor:
            futures = {executor.submit(evaluate_single_branch, i, branch): i 
                      for i, branch in enumerate(data['branches'], 1)}
            for future in as_completed(futures):
                branch_evaluations.append(future.result())
        
        # Sort by branch_id to maintain order
        branch_evaluations.sort(key=lambda x: x['branch_id'])
        
        # Compile results
        evaluation_results = {
            "run_id": data.get('run_id', 'N/A'),
            "topic": data['topic'],
            "timestamp": datetime.now().isoformat(),
            "evaluation_model": self.model,
            "source_file": results_file,
            "num_branches": total_branches,
            "num_papers": len(abstracts),
            "timing": data.get('timing', {}),
            "token_usage_overall": data.get('token_usage_overall', {}),
            "branch_evaluations": branch_evaluations,
            "summary_statistics": self._calculate_summary_stats(branch_evaluations)
        }
        
        # Save results to organized folder structure: results/{topic}_{run_id}/eval/
        if output_file is None:
            # Extract topic and run_id from results_file path
            results_path = Path(results_file)
            run_dir = results_path.parent.parent  # Go up from raw/ to {topic}_{run_id}/
            eval_dir = run_dir / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = eval_dir / f"evaluation_{results_path.stem}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_file}")
        
        return evaluation_results
    
    def _calculate_summary_stats(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics across all branches."""
        all_scores = {
            'fluency': [],
            'coherence': [],
            'consistency': [],
            'relevance': [],
            'overall': []
        }
        
        for eval_result in evaluations:
            for dim in ['fluency', 'coherence', 'consistency', 'relevance']:
                all_scores[dim].append(eval_result['scores'][dim])
            all_scores['overall'].append(eval_result['overall_score'])
        
        stats = {}
        for dim, scores in all_scores.items():
            stats[dim] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            }
        
        return stats


class ResultsAnalyzer:
    """Analyzes and visualizes evaluation results."""
    
    def __init__(self, evaluation_file: str):
        """Initialize analyzer with evaluation results.
        
        Args:
            evaluation_file: Path to evaluation JSON file
        """
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # set env:KMP_DUPLICATE_LIB_OK="TRUE"; 
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        # Create results folder structure: results/{topic}_{run_id}/visualization/
        eval_path = Path(evaluation_file)
        run_dir = eval_path.parent.parent  # Go up from eval/ to {topic}_{run_id}/
        self.output_dir = run_dir / "visualization"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        print(f"\n[ANALYZER] Loaded evaluation results")
        print(f"           Run ID: {self.data.get('run_id', 'N/A')}")
        print(f"           Output directory: {self.output_dir}")
    
    def create_all_visualizations(self):
        """Create all visualizations."""
        print(f"\n{'='*80}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*80}")
        
        workflow_type = self.data.get('workflow_type', 'unified')
        
        # Common visualizations for all workflow types
        self.plot_dimension_scores()
        self.plot_overall_scores_by_path()
        self.plot_heatmap()
        self.plot_score_distributions()
        self.plot_prompt_comparison()
        
        # Workflow-specific visualizations
        if workflow_type in ['section_based', 'unified']:
            # Branch tree only makes sense for explicit branching workflows
            self.plot_branch_tree()
            print(f"  [INFO] Generated branch tree (workflow: {workflow_type})")
        else:
            # Time travel - no branch tree needed
            print(f"  [SKIP] Branch tree not applicable for time travel workflow")
        
        self.create_summary_report()
        
        print(f"\n[DONE] All visualizations saved to: {self.output_dir}")
    
    def plot_dimension_scores(self):
        """Plot scores for each dimension across all branches."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        branches = [e['branch_id'] for e in self.data['branch_evaluations']]
        dimensions = ['fluency', 'coherence', 'consistency', 'relevance']
        
        x = np.arange(len(branches))
        width = 0.2
        
        for i, dim in enumerate(dimensions):
            scores = [e['scores'][dim] for e in self.data['branch_evaluations']]
            ax.bar(x + i * width, scores, width, label=dim.capitalize())
        
        ax.set_xlabel('Branch ID', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'G-Eval Scores by Dimension - {self.data["topic"]}', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(branches, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "dimension_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: dimension_scores.png")
    
    def plot_overall_scores_by_path(self):
        """Plot overall scores for each branch path."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        branches = [e['branch_id'] for e in self.data['branch_evaluations']]
        overall_scores = [e['overall_score'] for e in self.data['branch_evaluations']]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(branches)))
        bars = ax.bar(branches, overall_scores, color=colors)
        
        # Add value labels on bars with better positioning
        for bar in bars:
            height = bar.get_height()
            label_height = height + max(overall_scores) * 0.02  # Add 2% offset
            ax.text(bar.get_x() + bar.get_width()/2., label_height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Adjust y-axis to accommodate labels
        if max(overall_scores) > 0:
            ax.set_ylim(0, max(overall_scores) * 1.15)  # 15% extra space
        
        ax.set_xlabel('Branch Path', fontsize=12)
        ax.set_ylabel('Overall Score', fontsize=12)
        ax.set_title(f'Overall G-Eval Scores by Branch Path - {self.data["topic"]}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticklabels(branches, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean line
        mean_score = np.mean(overall_scores)
        ax.axhline(y=mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.2f}')
        ax.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Add margins for better spacing
        plt.savefig(self.output_dir / "overall_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: overall_scores.png")
    
    def plot_heatmap(self):
        """Create heatmap of scores across branches and dimensions."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        branches = [e['branch_id'] for e in self.data['branch_evaluations']]
        dimensions = ['fluency', 'coherence', 'consistency', 'relevance', 'overall_score']
        
        # Create matrix
        matrix = []
        for eval_result in self.data['branch_evaluations']:
            row = [eval_result['scores'][d] for d in dimensions[:-1]]
            row.append(eval_result['overall_score'])
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                   xticklabels=[d.capitalize() for d in dimensions],
                   yticklabels=branches,
                   cbar_kws={'label': 'Score'},
                   ax=ax)
        
        ax.set_title(f'Score Heatmap - {self.data["topic"]}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimension', fontsize=12)
        ax.set_ylabel('Branch Path', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "score_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: score_heatmap.png")
    
    def plot_score_distributions(self):
        """Plot distribution of scores for each dimension."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        dimensions = ['fluency', 'coherence', 'consistency', 'relevance', 'overall_score']
        
        for i, dim in enumerate(dimensions):
            if dim == 'overall_score':
                scores = [e['overall_score'] for e in self.data['branch_evaluations']]
                label = 'Overall'
            else:
                scores = [e['scores'][dim] for e in self.data['branch_evaluations']]
                label = dim.capitalize()
            
            axes[i].hist(scores, bins=10, edgecolor='black', alpha=0.7, color='skyblue')
            axes[i].axvline(np.mean(scores), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(scores):.2f}')
            axes[i].set_xlabel('Score', fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].set_title(f'{label} Distribution', fontsize=11, fontweight='bold')
            axes[i].legend()
            axes[i].grid(axis='y', alpha=0.3)
        
        # Remove extra subplot
        fig.delaxes(axes[5])
        
        fig.suptitle(f'Score Distributions - {self.data["topic"]}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "score_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: score_distributions.png")
    
    def plot_prompt_comparison(self):
        """Compare scores by prompt combinations (DYNAMIC)."""
        # Detect workflow type
        first_branch = self.data['branch_evaluations'][0]
        is_section_based = 'prompt_intro_style' in first_branch
        
        if is_section_based:
            # Section-based workflow: 3 sections
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Group by intro style
            intro_scores = {}
            for eval_result in self.data['branch_evaluations']:
                style = eval_result['prompt_intro_style']
                if style not in intro_scores:
                    intro_scores[style] = []
                intro_scores[style].append(eval_result['overall_score'])
            
            # Group by analysis style
            analysis_scores = {}
            for eval_result in self.data['branch_evaluations']:
                style = eval_result['prompt_analysis_style']
                if style not in analysis_scores:
                    analysis_scores[style] = []
                analysis_scores[style].append(eval_result['overall_score'])
            
            # Group by discussion style
            discussion_scores = {}
            for eval_result in self.data['branch_evaluations']:
                style = eval_result['prompt_discussion_style']
                if style not in discussion_scores:
                    discussion_scores[style] = []
                discussion_scores[style].append(eval_result['overall_score'])
            
            # Plot intro comparison
            styles_intro = list(intro_scores.keys())
            means_intro = [np.mean(intro_scores[s]) for s in styles_intro]
            stds_intro = [np.std(intro_scores[s]) for s in styles_intro]
            
            ax1.bar(styles_intro, means_intro, yerr=stds_intro, capsize=5, alpha=0.7, color='coral')
            ax1.set_xlabel('Introduction Style', fontsize=11)
            ax1.set_ylabel('Mean Overall Score', fontsize=11)
            ax1.set_title('Performance by Introduction', fontsize=12, fontweight='bold')
            ax1.set_xticklabels(styles_intro, rotation=45, ha='right')
            ax1.grid(axis='y', alpha=0.3)
            
            # Plot analysis comparison
            styles_analysis = list(analysis_scores.keys())
            means_analysis = [np.mean(analysis_scores[s]) for s in styles_analysis]
            stds_analysis = [np.std(analysis_scores[s]) for s in styles_analysis]
            
            ax2.bar(styles_analysis, means_analysis, yerr=stds_analysis, capsize=5, alpha=0.7, color='lightgreen')
            ax2.set_xlabel('Analysis Style', fontsize=11)
            ax2.set_ylabel('Mean Overall Score', fontsize=11)
            ax2.set_title('Performance by Analysis', fontsize=12, fontweight='bold')
            ax2.set_xticklabels(styles_analysis, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
            
            # Plot discussion comparison
            styles_discussion = list(discussion_scores.keys())
            means_discussion = [np.mean(discussion_scores[s]) for s in styles_discussion]
            stds_discussion = [np.std(discussion_scores[s]) for s in styles_discussion]
            
            ax3.bar(styles_discussion, means_discussion, yerr=stds_discussion, capsize=5, alpha=0.7, color='skyblue')
            ax3.set_xlabel('Discussion Style', fontsize=11)
            ax3.set_ylabel('Mean Overall Score', fontsize=11)
            ax3.set_title('Performance by Discussion', fontsize=12, fontweight='bold')
            ax3.set_xticklabels(styles_discussion, rotation=45, ha='right')
            ax3.grid(axis='y', alpha=0.3)
            
        else:
            # Original workflow: 2 or 3 prompts
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Group by prompt_1_2
            prompt_1_2_scores = {}
            for eval_result in self.data['branch_evaluations']:
                p1 = eval_result['prompt_1_2']
                if p1 not in prompt_1_2_scores:
                    prompt_1_2_scores[p1] = []
                prompt_1_2_scores[p1].append(eval_result['overall_score'])
            
            # Group by prompt_2_3
            prompt_2_3_scores = {}
            for eval_result in self.data['branch_evaluations']:
                p2 = eval_result['prompt_2_3']
                if p2 not in prompt_2_3_scores:
                    prompt_2_3_scores[p2] = []
                prompt_2_3_scores[p2].append(eval_result['overall_score'])
            
            # Plot prompt_1_2 comparison
            prompts_1_2 = list(prompt_1_2_scores.keys())
            means_1_2 = [np.mean(prompt_1_2_scores[p]) for p in prompts_1_2]
            stds_1_2 = [np.std(prompt_1_2_scores[p]) for p in prompts_1_2]
            
            ax1.bar(prompts_1_2, means_1_2, yerr=stds_1_2, capsize=5, alpha=0.7, color='coral')
            ax1.set_xlabel('Prompt 1-2', fontsize=11)
            ax1.set_ylabel('Mean Overall Score', fontsize=11)
            ax1.set_title('Performance by Prompt 1-2', fontsize=12, fontweight='bold')
            ax1.set_xticklabels(prompts_1_2, rotation=45, ha='right')
            ax1.grid(axis='y', alpha=0.3)
            
            # Plot prompt_2_3 comparison
            prompts_2_3 = list(prompt_2_3_scores.keys())
            means_2_3 = [np.mean(prompt_2_3_scores[p]) for p in prompts_2_3]
            stds_2_3 = [np.std(prompt_2_3_scores[p]) for p in prompts_2_3]
            
            ax2.bar(prompts_2_3, means_2_3, yerr=stds_2_3, capsize=5, alpha=0.7, color='lightgreen')
            ax2.set_xlabel('Prompt 2-3', fontsize=11)
            ax2.set_ylabel('Mean Overall Score', fontsize=11)
            ax2.set_title('Performance by Prompt 2-3', fontsize=12, fontweight='bold')
            ax2.set_xticklabels(prompts_2_3, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
        
        fig.suptitle(f'Prompt Performance Comparison - {self.data["topic"]}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "prompt_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: prompt_comparison.png")
    
    def plot_branch_tree(self):
        """Plot tree graph showing branch relationships and prompt variations (DYNAMIC)."""
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        
        # Detect workflow type
        first_branch = self.data['branch_evaluations'][0]
        is_section_based = 'prompt_intro' in first_branch
        
        # Get unique prompts for each level based on workflow type
        if is_section_based:
            # Section-based workflow: Introduction -> Analysis -> Discussion
            prompts_1_2 = sorted(set(e['prompt_intro_style'] for e in self.data['branch_evaluations']))
            prompts_2_3 = sorted(set(e['prompt_analysis_style'] for e in self.data['branch_evaluations']))
            prompts_3_4 = sorted(set(e['prompt_discussion_style'] for e in self.data['branch_evaluations']))
            has_prompt_3_4 = True  # Always 3 levels for section-based
            level_names = ['Introduction', 'Analysis', 'Discussion']
        else:
            # Original workflow: Prompt 1-2 -> Prompt 2-3 -> (optional) Prompt 3-4
            has_prompt_3_4 = 'prompt_3_4' in first_branch
            prompts_1_2 = sorted(set(e['prompt_1_2'] for e in self.data['branch_evaluations']))
            prompts_2_3 = sorted(set(e['prompt_2_3'] for e in self.data['branch_evaluations']))
            prompts_3_4 = sorted(set(e['prompt_3_4'] for e in self.data['branch_evaluations'])) if has_prompt_3_4 else []
            level_names = ['Prompt 1-2', 'Prompt 2-3', 'Prompt 3-4'] if has_prompt_3_4 else ['Prompt 1-2', 'Prompt 2-3']
        
        # Helper function to get prompt fields based on workflow type
        def get_prompts(eval_result):
            """Extract prompt values from evaluation result based on workflow type."""
            if is_section_based:
                return (
                    eval_result['prompt_intro_style'],
                    eval_result['prompt_analysis_style'],
                    eval_result['prompt_discussion_style'] if has_prompt_3_4 else None
                )
            else:
                return (
                    eval_result['prompt_1_2'],
                    eval_result['prompt_2_3'],
                    eval_result.get('prompt_3_4') if has_prompt_3_4 else None
                )
        
        # Adjust figure size and layout based on structure
        if has_prompt_3_4:
            fig, ax = plt.subplots(figsize=(20, 14))
            canvas_width = 20
            canvas_height = 13
        else:
            fig, ax = plt.subplots(figsize=(16, 12))
            canvas_width = 16
            canvas_height = 11
        
        # Color mapping by score
        score_colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 100))
        
        def get_color_for_score(score):
            min_score = min(e['overall_score'] for e in self.data['branch_evaluations'])
            max_score = max(e['overall_score'] for e in self.data['branch_evaluations'])
            normalized = (score - min_score) / (max_score - min_score) if max_score > min_score else 0.5
            return score_colors[int(normalized * 99)]
        
        # Node positions (adaptive)
        center_x = canvas_width / 2
        root_pos = (center_x, canvas_height - 1)
        phase1_pos = (center_x, canvas_height - 3)
        
        # Level 2 positions (prompts_1_2)
        level2_y = canvas_height - 5
        level2_spacing = max(4, canvas_width / (len(prompts_1_2) + 1))
        level2_start_x = center_x - (len(prompts_1_2) - 1) * level2_spacing / 2
        level2_positions = {p: (level2_start_x + i * level2_spacing, level2_y) 
                           for i, p in enumerate(prompts_1_2)}
        
        # Level 3 positions (prompts_2_3)
        level3_y = canvas_height - 7.5
        level3_spacing = level2_spacing / (len(prompts_2_3) + 0.5)
        
        # Level 4 positions (prompts_3_4) if exists
        if has_prompt_3_4:
            level4_y = canvas_height - 10.5
            level4_spacing = level3_spacing / (len(prompts_3_4) + 0.5)
        
        # Draw root node
        root_box = FancyBboxPatch((root_pos[0] - 1, root_pos[1] - 0.3), 2, 0.6,
                                  boxstyle="round,pad=0.1", 
                                  facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(root_box)
        ax.text(root_pos[0], root_pos[1], 'START', ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        # Draw Phase 1 node
        phase1_box = FancyBboxPatch((phase1_pos[0] - 1.5, phase1_pos[1] - 0.3), 3, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax.add_patch(phase1_box)
        ax.text(phase1_pos[0], phase1_pos[1], 'Search + Extract', ha='center', va='center',
               fontsize=9, fontweight='bold')
        
        # Arrow from root to phase1
        arrow = FancyArrowPatch(root_pos, phase1_pos,
                               arrowstyle='->', mutation_scale=20, linewidth=2,
                               color='gray', alpha=0.7)
        ax.add_patch(arrow)
        
        # Draw Level 2 nodes (prompts_1_2)
        for prompt, pos in level2_positions.items():
            box = FancyBboxPatch((pos[0] - 0.8, pos[1] - 0.3), 1.6, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor='lightyellow', edgecolor='black', linewidth=1.5)
            ax.add_patch(box)
            
            # Wrap text
            prompt_short = prompt.replace('_', '\n')[:20]
            ax.text(pos[0], pos[1], prompt_short, ha='center', va='center',
                   fontsize=7, fontweight='bold')
            
            # Arrow from phase1 to this node
            arrow = FancyArrowPatch(phase1_pos, pos,
                                   arrowstyle='->', mutation_scale=15, linewidth=1.5,
                                   color='gray', alpha=0.6)
            ax.add_patch(arrow)
        
        # Draw Level 3 and Level 4 nodes based on structure
        if has_prompt_3_4:
            # Build level 3 positions (intermediate nodes)
            level3_positions = {}
            for eval_result in self.data['branch_evaluations']:
                p1, p2, p3 = get_prompts(eval_result)
                key = (p1, p2)
                
                if key not in level3_positions:
                    parent_pos = level2_positions[p1]
                    p2_index = prompts_2_3.index(p2)
                    offset = (p2_index - (len(prompts_2_3) - 1) / 2) * level3_spacing
                    level3_positions[key] = (parent_pos[0] + offset, level3_y)
            
            # Draw Level 3 nodes (prompts_2_3 - intermediate)
            for (p1, p2), pos in level3_positions.items():
                box = FancyBboxPatch((pos[0] - 0.5, pos[1] - 0.25), 1.0, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor='lightcyan', edgecolor='black', linewidth=1.2)
                ax.add_patch(box)
                
                p2_short = p2.replace('_', '\n')[:15]
                ax.text(pos[0], pos[1], p2_short, ha='center', va='center',
                       fontsize=6, fontweight='bold')
                
                # Arrow from parent (level 2)
                parent_pos = level2_positions[p1]
                arrow = FancyArrowPatch(parent_pos, pos,
                                       arrowstyle='->', mutation_scale=12, linewidth=1.2,
                                       color='gray', alpha=0.5)
                ax.add_patch(arrow)
            
            # Draw Level 4 nodes (prompts_3_4 - final branches with scores)
            for eval_result in self.data['branch_evaluations']:
                p1, p2, p3 = get_prompts(eval_result)
                score = eval_result['overall_score']
                
                # Find parent position (level 3)
                parent_pos = level3_positions[(p1, p2)]
                
                # Calculate position for this final branch
                p3_index = prompts_3_4.index(p3)
                offset = (p3_index - (len(prompts_3_4) - 1) / 2) * level4_spacing
                child_pos = (parent_pos[0] + offset, level4_y)
                
                # Draw node with color based on score
                color = get_color_for_score(score)
                box = FancyBboxPatch((child_pos[0] - 0.4, child_pos[1] - 0.35), 0.8, 0.7,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(box)
                
                # Add text
                p3_short = p3.replace('_', '\n')[:12]
                ax.text(child_pos[0], child_pos[1] + 0.12, p3_short, ha='center', va='center',
                       fontsize=5, fontweight='bold')
                ax.text(child_pos[0], child_pos[1] - 0.15, f'{score:.2f}', ha='center', va='center',
                       fontsize=7, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.7))
                
                # Arrow from parent to this node
                arrow = FancyArrowPatch(parent_pos, child_pos,
                                       arrowstyle='->', mutation_scale=10, linewidth=0.8,
                                       color='gray', alpha=0.4)
                ax.add_patch(arrow)
        else:
            # 2-level structure - Draw Level 3 as final nodes
            for eval_result in self.data['branch_evaluations']:
                p1, p2, _ = get_prompts(eval_result)
                score = eval_result['overall_score']
                
                parent_pos = level2_positions[p1]
                p2_index = prompts_2_3.index(p2)
                offset = (p2_index - (len(prompts_2_3) - 1) / 2) * level3_spacing
                child_pos = (parent_pos[0] + offset, level3_y)
                
                color = get_color_for_score(score)
                box = FancyBboxPatch((child_pos[0] - 0.6, child_pos[1] - 0.4), 1.2, 0.8,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(box)
                
                p2_short = p2.replace('_', '\n')
                ax.text(child_pos[0], child_pos[1] + 0.15, p2_short, ha='center', va='center',
                       fontsize=6, fontweight='bold')
                ax.text(child_pos[0], child_pos[1] - 0.15, f'{score:.2f}', ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                
                arrow = FancyArrowPatch(parent_pos, child_pos,
                                       arrowstyle='->', mutation_scale=12, linewidth=1,
                                       color='gray', alpha=0.5)
                ax.add_patch(arrow)
        
        # Set axis limits and remove axes
        ax.set_xlim(0, canvas_width)
        ax.set_ylim(0, canvas_height)
        ax.axis('off')
        
        # Add title
        title_text = f'Branch Tree: {len(prompts_1_2)}x{len(prompts_2_3)}'
        if has_prompt_3_4:
            title_text += f'x{len(prompts_3_4)}'
        title_text += f' Structure - {self.data["topic"]}'
        ax.text(center_x, canvas_height - 0.2, title_text,
               ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Add legend for score colors
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                   norm=plt.Normalize(vmin=min(e['overall_score'] for e in self.data['branch_evaluations']),
                                                     vmax=max(e['overall_score'] for e in self.data['branch_evaluations'])))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.02, shrink=0.5)
        cbar.set_label('Overall Score', fontsize=10)
        
        # Add legend for node types (DYNAMIC)
        if has_prompt_3_4:
            if is_section_based:
                legend_elements = [
                    mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Start'),
                    mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Phase 1: Search+Extract'),
                    mpatches.Patch(facecolor='lightyellow', edgecolor='black', label=f'Level 2: {level_names[0]}'),
                    mpatches.Patch(facecolor='lightcyan', edgecolor='black', label=f'Level 3: {level_names[1]}'),
                    mpatches.Patch(facecolor='white', edgecolor='black', label=f'Level 4: {level_names[2]} (scored)')
                ]
            else:
                legend_elements = [
                    mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Start'),
                    mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Phase 1: Search+Extract'),
                    mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Level 2: Prompt 1-2'),
                    mpatches.Patch(facecolor='lightcyan', edgecolor='black', label='Level 3: Prompt 2-3'),
                    mpatches.Patch(facecolor='white', edgecolor='black', label='Level 4: Prompt 3-4 (scored)')
                ]
        else:
            if is_section_based:
                legend_elements = [
                    mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Start'),
                    mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Phase 1: Search+Extract'),
                    mpatches.Patch(facecolor='lightyellow', edgecolor='black', label=f'Level 2: {level_names[0]}'),
                    mpatches.Patch(facecolor='white', edgecolor='black', label=f'Level 3: {level_names[1]} (scored)')
                ]
            else:
                legend_elements = [
                    mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Start'),
                    mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Phase 1: Search+Extract'),
                    mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Level 2: Prompt 1-2'),
                    mpatches.Patch(facecolor='white', edgecolor='black', label='Level 3: Prompt 2-3 (scored)')
                ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "branch_tree.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: branch_tree.png")
    
    def create_summary_report(self):
        """Create a text summary report (DYNAMIC)."""
        report_file = self.output_dir / "evaluation_summary.txt"
        
        # Detect workflow type
        first_branch = self.data['branch_evaluations'][0]
        is_section_based = 'prompt_intro_style' in first_branch
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EVALUATION SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Topic: {self.data['topic']}\n")
            f.write(f"Evaluation Date: {self.data['timestamp']}\n")
            f.write(f"Evaluation Model: {self.data['evaluation_model']}\n")
            f.write(f"Number of Branches: {self.data['num_branches']}\n")
            f.write(f"Number of Papers: {self.data['num_papers']}\n")
            
            # Workflow type display
            workflow_type = self.data.get('workflow_type', 'unified')
            if workflow_type == 'time_travel':
                f.write(f"Workflow Type: LangGraph Time Travel (串行执行，checkpoint回溯)\n\n")
            elif is_section_based:
                f.write(f"Workflow Type: Agent_git Branching (并行分支，section-based)\n\n")
            else:
                f.write(f"Workflow Type: Original (Sequential Prompts)\n\n")
            
            f.write("-"*80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*80 + "\n\n")
            
            stats = self.data['summary_statistics']
            for dim, values in stats.items():
                f.write(f"{dim.upper()}:\n")
                f.write(f"  Mean:   {values['mean']:.3f}\n")
                f.write(f"  Std:    {values['std']:.3f}\n")
                f.write(f"  Min:    {values['min']:.3f}\n")
                f.write(f"  Max:    {values['max']:.3f}\n")
                f.write(f"  Median: {values['median']:.3f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("TOP 3 BRANCHES (by overall score)\n")
            f.write("-"*80 + "\n\n")
            
            sorted_branches = sorted(
                self.data['branch_evaluations'],
                key=lambda x: x['overall_score'],
                reverse=True
            )
            
            for i, branch in enumerate(sorted_branches[:3], 1):
                f.write(f"{i}. Branch {branch['branch_id']}\n")
                
                # Dynamic prompt display based on workflow type
                if is_section_based:
                    f.write(f"   Styles: {branch['prompt_intro_style']} + "
                           f"{branch['prompt_analysis_style']} + "
                           f"{branch['prompt_discussion_style']}\n")
                else:
                    if 'prompt_3_4' in branch:
                        f.write(f"   Prompts: {branch['prompt_1_2']} + {branch['prompt_2_3']} + {branch['prompt_3_4']}\n")
                    else:
                        f.write(f"   Prompts: {branch['prompt_1_2']} + {branch['prompt_2_3']}\n")
                
                f.write(f"   Overall Score: {branch['overall_score']:.3f}\n")
                f.write(f"   Fluency: {branch['scores']['fluency']:.3f}, ")
                f.write(f"Coherence: {branch['scores']['coherence']:.3f}, ")
                f.write(f"Consistency: {branch['scores']['consistency']:.3f}, ")
                f.write(f"Relevance: {branch['scores']['relevance']:.3f}\n")
                f.write(f"   Time: {branch['metadata']['branch_execution_time']:.2f}s, ")
                f.write(f"Tokens: {branch['metadata']['branch_token_usage']['total_tokens']}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("ALL BRANCHES RANKED\n")
            f.write("-"*80 + "\n\n")
            
            for i, branch in enumerate(sorted_branches, 1):
                f.write(f"{i:2d}. {branch['branch_id']:12s} | ")
                f.write(f"Score: {branch['overall_score']:.3f} | ")
                f.write(f"F:{branch['scores']['fluency']:.2f} ")
                f.write(f"Co:{branch['scores']['coherence']:.2f} ")
                f.write(f"Cs:{branch['scores']['consistency']:.2f} ")
                f.write(f"R:{branch['scores']['relevance']:.2f} | ")
                
                # Dynamic prompt display based on workflow type
                if is_section_based:
                    f.write(f"{branch['prompt_intro_style']} + "
                           f"{branch['prompt_analysis_style']} + "
                           f"{branch['prompt_discussion_style']}\n")
                else:
                    if 'prompt_3_4' in branch:
                        f.write(f"{branch['prompt_1_2']} + {branch['prompt_2_3']} + {branch['prompt_3_4']}\n")
                    else:
                        f.write(f"{branch['prompt_1_2']} + {branch['prompt_2_3']}\n")
        
        print(f"  [OK] Created: evaluation_summary.txt")


def main():
    """Main function to run evaluation and analysis."""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Evaluate paper summarization branches")
    parser.add_argument("run_dir", help="Run directory name (e.g., large_language_models_20251027_172013)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model for evaluation")
    parser.add_argument("--samples", type=int, default=5, help="Number of G-Eval samples")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, only analyze")
    
    args = parser.parse_args()
    
    # Construct full path to run directory
    project_root = Path(__file__).parent.parent.parent.parent
    run_dir = project_root / "results" / args.run_dir
    
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        print(f"Looking for: results/{args.run_dir}/")
        return
    
    # Find the raw results JSON file
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        print(f"ERROR: Raw directory not found: {raw_dir}")
        return
    
    # Find JSON file in raw directory
    json_files = list(raw_dir.glob("results_*.json"))
    if not json_files:
        print(f"ERROR: No results JSON file found in {raw_dir}")
        return
    
    results_file = json_files[0]  # Take the first (should be only one)
    print(f"\n[INPUT] Found results file: {results_file.name}")
    
    if not args.skip_eval:
        # Run evaluation
        print(f"\n[EVAL] Starting evaluation with {args.model} (samples={args.samples})")
        evaluator = BranchEvaluator(
            model=args.model,
            num_samples=args.samples
        )
        
        eval_results = evaluator.evaluate_all_branches(str(results_file))
        
        # Evaluation file is saved in results/{topic}_{run_id}/eval/ folder
        eval_file = run_dir / "eval" / f"evaluation_{results_file.stem}.json"
    else:
        # Use existing evaluation file from eval/ folder
        eval_file = run_dir / "eval" / f"evaluation_{results_file.stem}.json"
        if not eval_file.exists():
            print(f"ERROR: Evaluation file not found: {eval_file}")
            print(f"Run without --skip-eval to generate evaluation first.")
            return
        print(f"\n[SKIP EVAL] Using existing evaluation: {eval_file.name}")
    
    # Run analysis and visualization
    print(f"\n[VIZ] Creating visualizations...")
    analyzer = ResultsAnalyzer(str(eval_file))
    analyzer.create_all_visualizations()
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults organized in: {run_dir}")
    print(f"  - raw/           Raw results from agentgitAgent.py")
    print(f"  - eval/          Evaluation scores and statistics")
    print(f"  - visualization/ Charts and graphs")
    print(f"\nTo view results:")
    print(f"  - Summary: {run_dir / 'visualization' / 'evaluation_summary.txt'}")
    print(f"  - Charts: {run_dir / 'visualization'}/")


class FrameworkComparator:
    """Compare multiple framework evaluation results with configurable parameters."""
    
    def __init__(
        self, 
        framework_files: Dict[str, str],
        comparison_dimensions: Optional[List[str]] = None,
        execution_metrics: Optional[List[str]] = None,
        output_base_dir: Optional[str] = None,
        comparison_name: Optional[str] = None
    ):
        """Initialize comparator with evaluation files from multiple frameworks.
        
        Args:
            framework_files: Dict mapping framework names to evaluation file paths
                           e.g., {"agentgit": "path/to/agentgit.json", "timetravel": "path/to/timetravel.json"}
            comparison_dimensions: List of quality dimensions to compare 
                                 (default: ['fluency', 'coherence', 'consistency', 'relevance', 'overall'])
            execution_metrics: List of execution metrics to compare
                             (default: ['total_time', 'token_usage', 'branch_execution_time'])
            output_base_dir: Base directory for output (default: inferred from first file)
            comparison_name: Name for this comparison (default: auto-generated from framework names)
        """
        if len(framework_files) < 2:
            raise ValueError("Need at least 2 frameworks to compare")
        
        # Load framework data
        self.frameworks = {}
        self.framework_names = list(framework_files.keys())
        
        for name, file_path in framework_files.items():
            with open(file_path, 'r', encoding='utf-8') as f:
                self.frameworks[name] = json.load(f)
        
        # Set configurable parameters
        self.comparison_dimensions = comparison_dimensions or [
            'fluency', 'coherence', 'consistency', 'relevance', 'overall'
        ]
        self.execution_metrics = execution_metrics or [
            'total_time', 'token_usage', 'branch_execution_time'
        ]
        
        # Create output directory
        if output_base_dir:
            comparison_base = Path(output_base_dir)
        else:
            first_file_path = Path(list(framework_files.values())[0])
            comparison_base = first_file_path.parent.parent.parent / "comparisons"
        
        if comparison_name:
            self.comparison_name = comparison_name
        else:
            self.comparison_name = "_vs_".join(self.framework_names)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = comparison_base / f"{self.comparison_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[COMPARATOR] Initialized multi-framework comparison")
        print(f"             Frameworks: {', '.join(self.framework_names)}")
        print(f"             Dimensions: {', '.join(self.comparison_dimensions)}")
        print(f"             Metrics: {', '.join(self.execution_metrics)}")
        print(f"             Output: {self.output_dir}")
    
    def compare_all(self, include_metrics: Optional[List[str]] = None):
        """Run all comparisons and generate visualizations.
        
        Args:
            include_metrics: List of metrics to include in comparison
                           Options: ['execution', 'quality', 'efficiency', 'prompt_styles', 'detailed_analysis']
                           Default: all metrics
        """
        if include_metrics is None:
            include_metrics = ['execution', 'quality', 'efficiency', 'prompt_styles', 'detailed_analysis']
        
        print(f"\n{'='*80}")
        print(f"FRAMEWORK COMPARISON: {' vs '.join(self.framework_names)}")
        print(f"{'='*80}")
        
        if 'execution' in include_metrics and 'total_time' in self.execution_metrics:
            self.compare_execution_metrics()
        
        if 'quality' in include_metrics:
            self.compare_quality_metrics()
        
        if 'efficiency' in include_metrics:
            self.compare_efficiency()
        
        if 'prompt_styles' in include_metrics:
            self.compare_prompt_styles()
        
        if 'detailed_analysis' in include_metrics:
            self.generate_detailed_cot_analysis()
        
        self.create_comparison_report()
        
        print(f"\n[DONE] Comparison results saved to: {self.output_dir}")
    
    def compare_execution_metrics(self):
        """Compare execution time and token usage across all frameworks."""
        if 'total_time' not in self.execution_metrics and 'token_usage' not in self.execution_metrics:
            print("  [SKIP] No execution metrics requested")
            return
        
        # Determine which metrics to plot
        plot_time = 'total_time' in self.execution_metrics
        plot_tokens = 'token_usage' in self.execution_metrics
        
        if not (plot_time or plot_tokens):
            return
        
        # Color palette for frameworks
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.framework_names)))
        
        # Create separate charts for time and tokens
        if plot_time:
            self._plot_time_comparison(colors)
        
        if plot_tokens:
            self._plot_token_comparison(colors)
        
        # Create trade-off analysis if both metrics are available
        if plot_time and plot_tokens:
            self._plot_time_token_tradeoff(colors)
    
    def _plot_time_comparison(self, colors):
        """Create a separate chart for execution time comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = []
        labels = []
        colors_used = []
        
        for i, name in enumerate(self.framework_names):
            data = self.frameworks[name]
            timing = data.get('timing', {})
            
            # Try different time field names
            time_value = (timing.get('total_project_time') or 
                        timing.get('total_time') or 
                        timing.get('execution_time', 0))
            
            times.append(time_value)
            labels.append(name.replace('_', ' '))
            colors_used.append(colors[i])
        
        bars = ax.bar(labels, times, color=colors_used, alpha=0.7, 
                     edgecolor='black', linewidth=2)
        ax.set_ylabel('Total Execution Time (seconds)', fontsize=12, fontweight='bold')
        # ax.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels with better positioning
        for bar in bars:
            height = bar.get_height()
            label_height = height + max(times) * 0.05  # Add 5% of max for offset
            ax.text(bar.get_x() + bar.get_width()/2., label_height,
                   f'{height:.1f}s',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Adjust y-axis to accommodate labels
        if max(times) > 0:
            ax.set_ylim(0, max(times) * 1.2)  # 20% extra space for labels
        
        # Rotate labels if too many frameworks
        if len(self.framework_names) > 3:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "execution_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: execution_time.png")
    
    def _plot_token_comparison(self, colors):
        """Create a separate chart for token usage comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        total_tokens = []
        labels = []
        colors_used = []
        
        for i, name in enumerate(self.framework_names):
            data = self.frameworks[name]
            token_usage = data.get('token_usage_overall', {})
            
            total_tokens.append(token_usage.get('total_tokens', 0))
            
            labels.append(name.replace('_', ' '))
            colors_used.append(colors[i])
        
        bars = ax.bar(labels, total_tokens, color=colors_used, alpha=0.7, 
                     edgecolor='black', linewidth=2)
        ax.set_ylabel('Total Token Usage', fontsize=12, fontweight='bold')
        # ax.set_title('Token Usage Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels with better positioning
        for bar in bars:
            height = bar.get_height()
            label_height = height + max(total_tokens) * 0.05  # Add 5% of max for offset
            ax.text(bar.get_x() + bar.get_width()/2., label_height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Adjust y-axis to accommodate labels
        if max(total_tokens) > 0:
            ax.set_ylim(0, max(total_tokens) * 1.2)  # 20% extra space for labels
        
        # Rotate labels if too many frameworks
        if len(self.framework_names) > 3:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "token_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: token_usage.png")
    
    def _plot_time_token_tradeoff(self, colors):
        """Create a bar chart showing efficiency multipliers (1/(time * token)) relative to baseline."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = []
        raw_efficiency_scores = []
        colors_used = []
        
        for i, name in enumerate(self.framework_names):
            data = self.frameworks[name]
            timing = data.get('timing', {})
            token_usage = data.get('token_usage_overall', {})
            
            # Get time and token values
            time_value = (timing.get('total_project_time') or 
                        timing.get('total_time') or 
                        timing.get('execution_time', 0))
            token_value = token_usage.get('total_tokens', 0)
            
            if time_value > 0 and token_value > 0:
                # Calculate raw efficiency score: 1 / (time * tokens) (higher is better)
                raw_efficiency = 1.0 / (time_value * token_value)
                raw_efficiency_scores.append(raw_efficiency)
                labels.append(name.replace('_', ' '))
                colors_used.append(colors[i])
        
        if not raw_efficiency_scores:
            print("  [SKIP] No valid time/token data for trade-off analysis")
            return
        
        # Use minimum efficiency as baseline and calculate multipliers
        baseline_efficiency = min(raw_efficiency_scores)
        efficiency_multipliers = [score / baseline_efficiency for score in raw_efficiency_scores]
        
        # Create bar chart with efficiency multipliers
        bars = ax.bar(labels, efficiency_multipliers, color=colors_used, alpha=0.7, 
                     edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Efficiency Multiplier (×baseline)', fontsize=12, fontweight='bold')
        # ax.set_title('Framework Efficiency Comparison (Relative to Baseline)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels showing multipliers
        for bar in bars:
            height = bar.get_height()
            label_height = height + max(efficiency_multipliers) * 0.05  # Add 5% of max for offset
            ax.text(bar.get_x() + bar.get_width()/2., label_height,
                   f'{height:.2f}×',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Adjust y-axis to accommodate labels
        if max(efficiency_multipliers) > 0:
            ax.set_ylim(0, max(efficiency_multipliers) * 1.2)  # 20% extra space for labels
        
        # Rotate labels if too many frameworks
        if len(self.framework_names) > 3:
            ax.tick_params(axis='x', rotation=45)
        
        # # Add efficiency ranking as text showing multipliers
        # sorted_indices = sorted(range(len(efficiency_multipliers)), key=lambda i: efficiency_multipliers[i], reverse=True)
        # ranking_text = "Efficiency Ranking (Best to Worst):\n"
        # for rank, idx in enumerate(sorted_indices, 1):
        #     multiplier = efficiency_multipliers[idx]
        #     ranking_text += f"{rank}. {labels[idx]}: {multiplier:.2f}×\n"
        
        # ax.text(0.02, 0.98, ranking_text, transform=ax.transAxes, 
        #        fontsize=9, verticalalignment='top',
        #        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "time_token_tradeoff.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: time_token_tradeoff.png")
    
    def compare_quality_metrics(self):
        """Compare quality scores between frameworks using configurable dimensions."""
        # Calculate optimized subplot layout
        n_dims = len(self.comparison_dimensions)
        
        # Optimize layout for better visual balance
        if n_dims <= 2:
            n_cols, n_rows = n_dims, 1
        elif n_dims == 3:
            n_cols, n_rows = 3, 1
        elif n_dims == 4:
            n_cols, n_rows = 2, 2  # 2x2 grid for 4 dimensions
        elif n_dims <= 6:
            n_cols, n_rows = 3, 2  # 3x2 grid for 5-6 dimensions
        else:
            # For many dimensions, use balanced grid
            n_cols = min(4, int(np.ceil(np.sqrt(n_dims))))  # Max 4 columns
            n_rows = (n_dims + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

        # Handle single subplot case
        if n_dims == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten() if n_dims > 1 else axes
        
        # Color palette for frameworks
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.framework_names)))
        
        for i, dim in enumerate(self.comparison_dimensions):
            ax = axes_flat[i]
            
            means = []
            stds = []
            labels = []
            colors_used = []
            
            for j, name in enumerate(self.framework_names):
                data = self.frameworks[name]
                stats = data.get('summary_statistics', {})
                
                if dim in stats:
                    means.append(stats[dim]['mean'])
                    stds.append(stats[dim]['std'])
                else:
                    # Try to calculate from raw data if summary stats not available
                    branches = data.get('branch_evaluations', [])
                    if branches:
                        if dim == 'overall':
                            scores = [b.get('overall_score', 0) for b in branches]
                        else:
                            scores = [b.get('scores', {}).get(dim, 0) for b in branches]
                        
                        scores = [s for s in scores if s > 0]  # Filter out zero scores
                        if scores:
                            means.append(np.mean(scores))
                            stds.append(np.std(scores))
                        else:
                            means.append(0)
                            stds.append(0)
                    else:
                        means.append(0)
                        stds.append(0)
                
                labels.append(name.replace('_', ' '))
                colors_used.append(colors[j])
            
            bars = ax.bar(labels, means, yerr=stds, capsize=10,
                         color=colors_used, alpha=0.7, edgecolor='black', linewidth=2)
            
            ax.set_ylabel('Score', fontsize=11, fontweight='bold')
            ax.set_title(f'{dim.replace("_", " ")} Comparison', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Calculate appropriate y-axis range first
            max_height = max(means[i] + stds[i] for i in range(len(means)) if means[i] > 0) if any(means) else 5
            
            # Set y-axis range with generous space for labels
            if max_height > 0:
                # Remove the 5.0 cap and give more space for labels
                y_max = max_height * 1.3  # 30% extra space for labels
                ax.set_ylim(0, y_max)
            else:
                ax.set_ylim(0, 5)
            
            # Add value labels with better positioning
            for bar, std in zip(bars, stds):
                height = bar.get_height()
                # Simple positioning above the error bar
                label_height = height + std + 0.1
                ax.text(bar.get_x() + bar.get_width()/2., label_height,
                       f'{height:.3f}\n±{std:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Rotate labels if too many frameworks
            if len(self.framework_names) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        # Remove extra subplots if any
        for i in range(n_dims, len(axes_flat)):
            fig.delaxes(axes_flat[i])
        
        fig.suptitle(f'Quality Metrics Comparison: {" vs ".join(self.framework_names)}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])  # Give more space to chart area
        plt.savefig(self.output_dir / "quality_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: quality_metrics.png")
    
    def compare_efficiency(self):
        """Compute a single efficiency metric = time * tokens, standardize it (min-max),
        and create a single bar chart showing the standardized values.

        Saves: efficiency_comparison.png
        """
        print("\n[EFFICIENCY] Computing simple efficiency = time * tokens and plotting...")

        labels = []
        raw_efficiencies = []
        colors_used = []

        for i, name in enumerate(self.framework_names):
            data = self.frameworks[name]
            timing = data.get('timing', {})
            token_usage = data.get('token_usage_overall', {})

            # Get time and token values
            time_value = (timing.get('total_project_time') or
                          timing.get('total_time') or
                          timing.get('execution_time', 0))
            token_value = token_usage.get('total_tokens', 0)

            raw = time_value * token_value
            raw_efficiencies.append(raw)
            labels.append(name.replace('_', ' '))
            colors_used.append(plt.cm.Set3(i / max(1, len(self.framework_names) - 1)))

        if not raw_efficiencies:
            print("  [SKIP] No frameworks or no efficiency data available")
            return

        # Min-max standardize raw efficiencies to [0, 1]
        min_raw, max_raw = min(raw_efficiencies), max(raw_efficiencies)
        if max_raw > min_raw:
            norm_eff = [(r - min_raw) / (max_raw - min_raw) for r in raw_efficiencies]
        else:
            # All equal -> neutral value 0.5 for visualization
            norm_eff = [0.5] * len(raw_efficiencies)

        # Plot standardized efficiencies as a single clear bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, norm_eff, color=colors_used, alpha=0.85,
                      edgecolor='black', linewidth=1.2)

        ax.set_ylabel('Standardized Efficiency (min-max)', fontsize=12, fontweight='bold')
        ax.set_title('Efficiency Comparison — Efficiency = Time × Tokens (standardized)',
                     fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.25)

        # Add raw (time*tokens) labels above bars (human-friendly formatting)
        for bar, raw in zip(bars, raw_efficiencies):
            height = bar.get_height()
            label_text = f'{raw:,.0f}'
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.03,
                    label_text, ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Rotate labels if too many frameworks
        if len(labels) > 3:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Print a short summary
        print('  [EFFICIENCY SUMMARY]')
        sorted_idx = sorted(range(len(raw_efficiencies)), key=lambda i: raw_efficiencies[i], reverse=True)
        for rank, idx in enumerate(sorted_idx, start=1):
            name = self.framework_names[idx]
            raw = raw_efficiencies[idx]
            std = norm_eff[idx]
            print(f"    {rank}. {name}: raw={raw:,.0f}, standardized={std:.3f}")

        print("  [OK] Created: efficiency_comparison.png")
    
    def compare_prompt_styles(self):
        """Compare CoT vs Few-shot performance across all frameworks."""
        print("  [INFO] Analyzing CoT vs Few-shot prompt styles...")
        
        # Helper function to group by style
        def group_by_style_pattern(branches):
            """Group branches by prompt style pattern."""
            patterns = {}
            for branch in branches:
                if 'prompt_intro_style' not in branch:
                    continue
                
                intro = branch['prompt_intro_style']
                analysis = branch['prompt_analysis_style']
                discussion = branch['prompt_discussion_style']
                pattern = f"{intro}/{analysis}/{discussion}"
                
                if pattern not in patterns:
                    patterns[pattern] = []
                patterns[pattern].append(branch)
            
            return patterns
        
        # Color mapping for each unique pattern - distinct colors for each combination
        def get_pattern_color(pattern):
            """Assign a unique color to each pattern."""
            # Use a colorblind-friendly palette
            color_map = {
                'cot/cot/cot': '#d62728',                 # Red - All CoT
                'few_shot/few_shot/few_shot': '#1f77b4', # Blue - All Few-shot
                'cot/cot/few_shot': '#9467bd',           # Purple
                'cot/few_shot/cot': '#ff7f0e',           # Orange  
                'cot/few_shot/few_shot': '#2ca02c',      # Green
                'few_shot/cot/cot': '#8c564b',           # Brown
                'few_shot/cot/few_shot': '#e377c2',      # Pink
                'few_shot/few_shot/cot': '#17becf',      # Cyan
            }
            return color_map.get(pattern, '#7f7f7f')  # Gray for unknown patterns
        
        # Group branches by style for all frameworks
        framework_patterns = {}
        all_patterns = set()
        
        for name in self.framework_names:
            data = self.frameworks[name]
            patterns = group_by_style_pattern(data.get('branch_evaluations', []))
            framework_patterns[name] = patterns
            all_patterns.update(patterns.keys())
        
        # Create comparison visualization with optimized layout
        n_frameworks = len(self.framework_names)
        
        # Optimize layout for better visual balance
        if n_frameworks <= 2:
            n_cols, n_rows = n_frameworks, 1
        elif n_frameworks == 3:
            n_cols, n_rows = 3, 1
        elif n_frameworks == 4:
            n_cols, n_rows = 2, 2  # 2x2 grid for 4 frameworks
        elif n_frameworks <= 6:
            n_cols, n_rows = 3, 2  # 3x2 grid for 5-6 frameworks
        elif n_frameworks <= 8:
            n_cols, n_rows = 4, 2  # 4x2 grid for 7-8 frameworks
        else:
            # For many frameworks, use balanced grid
            n_cols = int(np.ceil(np.sqrt(n_frameworks)))
            n_rows = (n_frameworks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
        
        # Handle single subplot case
        if n_frameworks == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1 and n_rows > 1:
            axes = axes.reshape(-1, 1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten() if n_frameworks > 1 else axes
        
        for i, name in enumerate(self.framework_names):
            ax = axes_flat[i]
            patterns = framework_patterns[name]
            
            if patterns:
                patterns_sorted = sorted(patterns.items(), 
                                       key=lambda x: np.mean([b['overall_score'] for b in x[1]]), 
                                       reverse=True)
                
                # Extract original pattern strings BEFORE any replacement
                original_patterns = [p for p, _ in patterns_sorted]
                pattern_names = [p.replace('/', '\n') for p in original_patterns]
                if 'overall_score_stats' in patterns_sorted[0][1][0]:
                    # Use mean from stats if available
                    pattern_scores = [patterns_sorted[j][1][0]['overall_score_stats']['mean'] 
                                      for j in range(len(patterns_sorted))]
                    pattern_stds = [patterns_sorted[j][1][0]['overall_score_stats']['std'] 
                                    for j in range(len(patterns_sorted))]
                else:
                    pattern_scores = [np.mean([b['overall_score'] for b in branches]) 
                                    for _, branches in patterns_sorted]
                    pattern_stds = [np.std([b['overall_score'] for b in branches])
                              for _, branches in patterns_sorted]
                print(name, [[b['overall_score'] for b in branches]
                              for _, branches in patterns_sorted])
                pattern_colors = [get_pattern_color(p) for p in original_patterns]
                
                bars = ax.bar(range(len(pattern_names)), pattern_scores, 
                             yerr=pattern_stds, capsize=5,
                             color=pattern_colors,
                             alpha=0.7, edgecolor='black', linewidth=1.5)
                
                ax.set_xticks(range(len(pattern_names)))
                ax.set_xticklabels(pattern_names, rotation=0, fontsize=8)
                ax.set_ylabel('Overall Score', fontsize=11, fontweight='bold')
                ax.set_title(f'{name.replace("_", " ")}', 
                           fontsize=12, fontweight='bold')
                ax.set_ylim(0, 5)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels with better positioning
                for j, (score, std) in enumerate(zip(pattern_scores, pattern_stds)):
                    label_height = score + std + 0.1  # Add offset
                    ax.text(j, label_height, f'{score:.2f}', 
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                # Adjust y-axis to accommodate labels
                max_height = max(pattern_scores[j] + pattern_stds[j] for j in range(len(pattern_scores)) if pattern_scores[j] > 0)
                if max_height > 0:
                    ax.set_ylim(0, min(5, max_height * 1.25))  # 25% extra space for labels
            else:
                ax.text(0.5, 0.5, 'No prompt style data\navailable', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, style='italic')
                ax.set_title(f'{name.replace("_", " ")}: No Data', 
                           fontsize=12, fontweight='bold')
        
        # Remove extra subplots if any
        for i in range(n_frameworks, len(axes_flat)):
            fig.delaxes(axes_flat[i])
        
        # Create legend with all unique patterns
        from matplotlib.patches import Patch
        legend_elements = []
        
        # Sort patterns for better readability (all CoT first, then mixed, then all few-shot)
        def pattern_sort_key(p):
            if p == 'cot/cot/cot':
                return (0, p)
            elif p == 'few_shot/few_shot/few_shot':
                return (2, p)
            else:
                return (1, p)
        
        sorted_patterns = sorted(all_patterns, key=pattern_sort_key)
        
        for pattern in sorted_patterns:
            color = get_pattern_color(pattern)
            # Format label: capitalize and replace underscores, but keep / as separator
            parts = pattern.split('/')
            formatted_parts = [part.replace('_', '-').upper() if part else part for part in parts]
            label = ' / '.join(formatted_parts)
            legend_elements.append(Patch(facecolor=color, alpha=0.7, edgecolor='black', 
                                        linewidth=1, label=label))
        
        # Add legend below the plots with better spacing
        if legend_elements:
            n_legend_cols = min(4, len(legend_elements))
            fig.legend(handles=legend_elements, loc='lower center', ncol=n_legend_cols,
                      fontsize=8, bbox_to_anchor=(0.5, 0.03), frameon=True,
                      columnspacing=1.0, handlelength=2, handleheight=1.5)
        
        # fig.suptitle('Prompt Style Pattern Comparison Across Frameworks',
        #             fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.08, 1, 0.98])
        plt.savefig(self.output_dir / "prompt_style_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: prompt_style_comparison.png")
        
        # Find best combinations for each framework
        self._analyze_best_combinations(framework_patterns)
    
    def _analyze_best_combinations(self, framework_patterns):
        """Analyze and store best prompt combinations for each framework."""
        self.best_combinations = {}
        
        for name in self.framework_names:
            patterns = framework_patterns.get(name, {})
            
            if patterns:
                best_pattern = max(patterns.items(),
                                 key=lambda x: np.mean([b['overall_score'] for b in x[1]]))
                pattern_name, branches = best_pattern
                scores = [b['overall_score'] for b in branches]
                
                self.best_combinations[name] = {
                    'pattern': pattern_name,
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'branches': len(branches),
                    'best_branch': max(branches, key=lambda x: x['overall_score'])
                }
                
                print(f"  [BEST] {name.replace('_', ' ')}: {pattern_name} "
                      f"(Score: {np.mean(scores):.3f} ± {np.std(scores):.3f})")
            else:
                self.best_combinations[name] = None
                print(f"  [BEST] {name.replace('_', ' ')}: No pattern data available")
    
    def generate_detailed_cot_analysis(self):
        """Generate detailed CoT vs Few-shot analysis for each framework."""
        print("  [INFO] Generating detailed CoT vs Few-shot analysis...")
        
        for name in self.framework_names:
            data = self.frameworks[name]
            safe_name = name.replace(' ', '_').replace('/', '_')
            
            self._generate_framework_cot_report(
                data,
                name.replace('_', ' '),
                self.output_dir / f"{safe_name}_cot_vs_fewshot_analysis.txt"
            )
            
            print(f"  [OK] Created: {safe_name}_cot_vs_fewshot_analysis.txt")
    
    def _group_by_style(self, evaluations):
        """Group evaluations by prompt style (CoT vs Few-shot)."""
        groups = {
            "all_cot": [],
            "all_fewshot": [],
            "intro_cot": [],
            "intro_fewshot": [],
            "analysis_cot": [],
            "analysis_fewshot": [],
            "discussion_cot": [],
            "discussion_fewshot": [],
            "mixed": []
        }
        
        for eval_result in evaluations:
            if 'prompt_intro_style' not in eval_result:
                continue
            
            intro_style = eval_result.get('prompt_intro_style', '')
            analysis_style = eval_result.get('prompt_analysis_style', '')
            discussion_style = eval_result.get('prompt_discussion_style', '')
            
            # Check if all sections use same style
            if intro_style in ['cot', 'chain_of_thought'] and \
               analysis_style in ['cot', 'chain_of_thought'] and \
               discussion_style in ['cot', 'chain_of_thought']:
                groups["all_cot"].append(eval_result)
            elif intro_style == 'few_shot' and \
                 analysis_style == 'few_shot' and \
                 discussion_style == 'few_shot':
                groups["all_fewshot"].append(eval_result)
            else:
                groups["mixed"].append(eval_result)
            
            # Group by individual sections
            if intro_style in ['cot', 'chain_of_thought']:
                groups["intro_cot"].append(eval_result)
            elif intro_style == 'few_shot':
                groups["intro_fewshot"].append(eval_result)
            
            if analysis_style in ['cot', 'chain_of_thought']:
                groups["analysis_cot"].append(eval_result)
            elif analysis_style == 'few_shot':
                groups["analysis_fewshot"].append(eval_result)
            
            if discussion_style in ['cot', 'chain_of_thought']:
                groups["discussion_cot"].append(eval_result)
            elif discussion_style == 'few_shot':
                groups["discussion_fewshot"].append(eval_result)
        
        return groups
    
    def _compare_style_groups(self, group1, group2, group1_name, group2_name):
        """Compare two style groups statistically."""
        dimensions = ['fluency', 'coherence', 'consistency', 'relevance', 'overall_score']
        
        comparison = {
            "group1_name": group1_name,
            "group2_name": group2_name,
            "group1_size": len(group1),
            "group2_size": len(group2),
            "dimensions": {}
        }
        
        for dim in dimensions:
            if dim == 'overall_score':
                scores1 = [e['overall_score'] for e in group1]
                scores2 = [e['overall_score'] for e in group2]
            else:
                scores1 = [e['scores'][dim] for e in group1]
                scores2 = [e['scores'][dim] for e in group2]
            
            if not scores1 or not scores2:
                continue
            
            stats1 = {
                "mean": np.mean(scores1),
                "std": np.std(scores1),
                "min": np.min(scores1),
                "max": np.max(scores1)
            }
            stats2 = {
                "mean": np.mean(scores2),
                "std": np.std(scores2),
                "min": np.min(scores2),
                "max": np.max(scores2)
            }
            
            mean_diff = stats1['mean'] - stats2['mean']
            diff_percent = (mean_diff / stats2['mean'] * 100) if stats2['mean'] > 0 else 0
            
            comparison["dimensions"][dim] = {
                f"{group1_name}": stats1,
                f"{group2_name}": stats2,
                "mean_difference": mean_diff,
                "difference_percent": diff_percent,
                "winner": group1_name if mean_diff > 0 else group2_name
            }
        
        return comparison
    
    def _generate_framework_cot_report(self, data, framework_name, output_file):
        """Generate detailed CoT vs Few-shot report for a framework."""
        evaluations = data['branch_evaluations']
        groups = self._group_by_style(evaluations)
        
        report = []
        report.append("="*80)
        report.append(f"COT VS FEW-SHOT COMPARISON ANALYSIS: {framework_name}")
        report.append("="*80)
        report.append(f"\nRun ID: {data.get('run_id', 'N/A')}")
        report.append(f"Topic: {data['topic']}")
        report.append(f"Total Branches: {data['num_branches']}")
        report.append(f"Evaluation Model: {data.get('evaluation_model', 'gpt-4o-mini')}")
        report.append("")
        
        # Group sizes
        report.append("-"*80)
        report.append("GROUP SIZES")
        report.append("-"*80)
        report.append(f"All CoT (all 3 sections): {len(groups['all_cot'])}")
        report.append(f"All Few-shot (all 3 sections): {len(groups['all_fewshot'])}")
        report.append(f"Mixed styles: {len(groups['mixed'])}")
        report.append("")
        
        # ALL BRANCHES DETAILED ANALYSIS
        report.append("="*80)
        report.append("ALL BRANCHES DETAILED ANALYSIS")
        report.append("="*80)
        report.append("")
        
        # Sort branches by overall score
        sorted_branches = sorted(evaluations, key=lambda x: x['overall_score'], reverse=True)
        
        report.append(f"{'Rank':<6}{'Branch ID':<15}{'Styles':<30}{'Overall':<10}{'Fluency':<10}{'Coherence':<10}{'Consistency':<12}{'Relevance':<10}")
        report.append("-"*100)
        
        for i, branch in enumerate(sorted_branches, 1):
            branch_id = branch['branch_id']
            
            # Get style combination
            if 'prompt_intro_style' in branch:
                intro = branch['prompt_intro_style'][:3]
                analysis = branch['prompt_analysis_style'][:3]
                discussion = branch['prompt_discussion_style'][:3]
                styles = f"{intro}/{analysis}/{discussion}"
            else:
                styles = "N/A"
            
            scores = branch['scores']
            overall = branch['overall_score']
            
            report.append(f"{i:<6}{branch_id:<15}{styles:<30}{overall:<10.2f}{scores['fluency']:<10.2f}"
                         f"{scores['coherence']:<10.2f}{scores['consistency']:<12.2f}{scores['relevance']:<10.2f}")
        
        # Statistics by style pattern
        report.append("")
        report.append("-"*80)
        report.append("STATISTICS BY STYLE PATTERN")
        report.append("-"*80)
        report.append("")
        
        # Group by style pattern
        style_patterns = {}
        for branch in evaluations:
            if 'prompt_intro_style' in branch:
                pattern = f"{branch['prompt_intro_style']}/{branch['prompt_analysis_style']}/{branch['prompt_discussion_style']}"
                if pattern not in style_patterns:
                    style_patterns[pattern] = []
                style_patterns[pattern].append(branch['overall_score'])
        
        if style_patterns:
            for pattern, scores in sorted(style_patterns.items(), key=lambda x: np.mean(x[1]), reverse=True):
                mean_score = np.mean(scores)
                std_score = np.std(scores) if len(scores) > 1 else 0.0
                report.append(f"  {pattern:<40} -> {mean_score:.3f} +/- {std_score:.3f} (n={len(scores)})")
        
        report.append("")
        
        # Main comparison: All CoT vs All Few-shot
        if groups['all_cot'] and groups['all_fewshot']:
            report.append("="*80)
            report.append("MAIN COMPARISON: ALL-COT VS ALL-FEW-SHOT")
            report.append("="*80)
            
            comparison = self._compare_style_groups(
                groups['all_cot'], groups['all_fewshot'],
                "All-CoT", "All-Few-shot"
            )
            
            for dim, data_dim in comparison['dimensions'].items():
                report.append(f"\n{dim.upper().replace('_', ' ')}:")
                report.append(f"  All-CoT:     {data_dim['All-CoT']['mean']:.3f} +/- {data_dim['All-CoT']['std']:.3f}")
                report.append(f"  All-Few-shot: {data_dim['All-Few-shot']['mean']:.3f} +/- {data_dim['All-Few-shot']['std']:.3f}")
                report.append(f"  Difference:   {data_dim['mean_difference']:+.3f} ({data_dim['difference_percent']:+.1f}%)")
                report.append(f"  Winner:       {data_dim['winner']}")
            
            report.append("")
        
        # Section-level analysis
        report.append("="*80)
        report.append("SECTION-LEVEL ANALYSIS")
        report.append("="*80)
        
        section_pairs = [
            ("intro_cot", "intro_fewshot", "Introduction"),
            ("analysis_cot", "analysis_fewshot", "Analysis"),
            ("discussion_cot", "discussion_fewshot", "Discussion")
        ]
        
        for key1, key2, section_name in section_pairs:
            if groups[key1] and groups[key2]:
                report.append(f"\n{section_name.upper()}:")
                section_comp = self._compare_style_groups(
                    groups[key1], groups[key2], "CoT", "Few-shot"
                )
                
                if 'overall_score' in section_comp['dimensions']:
                    overall_data = section_comp['dimensions']['overall_score']
                    report.append(f"  CoT:      {overall_data['CoT']['mean']:.3f} +/- {overall_data['CoT']['std']:.3f}")
                    report.append(f"  Few-shot: {overall_data['Few-shot']['mean']:.3f} +/- {overall_data['Few-shot']['std']:.3f}")
                    report.append(f"  Difference: {overall_data['mean_difference']:+.3f} ({overall_data['difference_percent']:+.1f}%)")
                    report.append(f"  Winner: {overall_data['winner']}")
        
        report.append("")
        
        # Key findings
        report.append("="*80)
        report.append("KEY FINDINGS")
        report.append("="*80)
        
        if groups['all_cot'] and groups['all_fewshot']:
            main_comp = self._compare_style_groups(
                groups['all_cot'], groups['all_fewshot'],
                "All-CoT", "All-Few-shot"
            )
            
            if 'overall_score' in main_comp['dimensions']:
                overall_diff = main_comp['dimensions']['overall_score']['mean_difference']
                overall_pct = abs(main_comp['dimensions']['overall_score']['difference_percent'])
                
                if overall_pct > 5:
                    report.append(f"\n[SIGNIFICANT] DIFFERENCE DETECTED: {overall_pct:.1f}%")
                    if overall_diff > 0:
                        report.append("  -> CoT performs better overall")
                    else:
                        report.append("  -> Few-shot performs better overall")
                else:
                    report.append(f"\n[SMALL] DIFFERENCE: {overall_pct:.1f}%")
                    report.append("  -> Consider enhancing prompt differences further")
                
                # Dimension breakdown
                report.append("\nDimension Winners:")
                for dim in ['fluency', 'coherence', 'consistency', 'relevance']:
                    if dim in main_comp['dimensions']:
                        winner = main_comp['dimensions'][dim]['winner']
                        diff = abs(main_comp['dimensions'][dim]['difference_percent'])
                        report.append(f"  {dim.capitalize():15s}: {winner:15s} ({diff:.1f}%)")
        
        report.append("")
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
    
    def create_comparison_report(self):
        """Create detailed comparison report for all frameworks."""
        report_file = self.output_dir / "comparison_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("FRAMEWORK COMPARISON REPORT\n")
            f.write(f"{' vs '.join(self.framework_names)}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Comparison Date: {datetime.now().isoformat()}\n")
            f.write(f"Comparison Name: {self.comparison_name}\n")
            f.write(f"Frameworks: {len(self.framework_names)}\n")
            f.write(f"Quality Dimensions: {', '.join(self.comparison_dimensions)}\n")
            f.write(f"Execution Metrics: {', '.join(self.execution_metrics)}\n\n")
            
            # Framework Overview
            f.write("-"*80 + "\n")
            f.write("FRAMEWORK OVERVIEW\n")
            f.write("-"*80 + "\n\n")
            
            for name in self.framework_names:
                data = self.frameworks[name]
                timing = data.get('timing', {})
                
                f.write(f"{name.replace('_', ' ').upper()}:\n")
                f.write(f"  Run ID: {data.get('run_id', 'N/A')}\n")
                f.write(f"  Topic: {data.get('topic', 'N/A')}\n")
                f.write(f"  Total Branches: {data.get('num_branches', 0)}\n")
                
                # Try different time field names
                total_time = (timing.get('total_project_time') or 
                            timing.get('total_time') or 
                            timing.get('execution_time', 0))
                f.write(f"  Total Time: {total_time:.2f}s\n")
                f.write(f"  Papers Analyzed: {data.get('num_papers', 0)}\n")
                
                # Add paper titles if available
                if 'papers' in data and data['papers']:
                    f.write(f"  Papers:\n")
                    for i, paper in enumerate(data['papers'][:3], 1):  # Show first 3
                        title = paper.get('title', 'N/A')[:50]  # Truncate long titles
                        f.write(f"    {i}. {title}...\n")
                    if data.get('num_papers', 0) > 3:
                        f.write(f"    ... and {data['num_papers'] - 3} more\n")
                f.write("\n")
            
            # Performance Metrics
            if 'total_time' in self.execution_metrics or 'token_usage' in self.execution_metrics:
                f.write("-"*80 + "\n")
                f.write("PERFORMANCE METRICS\n")
                f.write("-"*80 + "\n\n")
                
                if 'total_time' in self.execution_metrics:
                    f.write("EXECUTION TIME:\n")
                    times = []
                    for name in self.framework_names:
                        timing = self.frameworks[name].get('timing', {})
                        time_val = (timing.get('total_project_time') or 
                                  timing.get('total_time') or 
                                  timing.get('execution_time', 0))
                        times.append(time_val)
                        f.write(f"  {name.replace('_', ' '):15s}: {time_val:.2f}s\n")
                    
                    # Calculate speedup if multiple frameworks
                    if len(times) > 1 and min(times) > 0:
                        fastest = min(times)
                        fastest_idx = times.index(fastest)
                        fastest_name = self.framework_names[fastest_idx]
                        f.write(f"\n  Fastest: {fastest_name.replace('_', ' ')}\n")
                        for i, name in enumerate(self.framework_names):
                            if i != fastest_idx:
                                ratio = times[i] / fastest if fastest > 0 else 0
                                f.write(f"  {name.replace('_', ' ')} is {ratio:.2f}x slower\n")
                    f.write("\n")
                
                if 'token_usage' in self.execution_metrics:
                    f.write("TOKEN USAGE (per branch average):\n")
                    for name in self.framework_names:
                        branches = self.frameworks[name].get('branch_evaluations', [])
                        if branches:
                            tokens = [b.get('metadata', {}).get('branch_token_usage', {}).get('total_tokens', 0) 
                                     for b in branches]
                            tokens = [t for t in tokens if t > 0]
                            if tokens:
                                f.write(f"  {name.replace('_', ' '):15s}: {np.mean(tokens):.0f} ± {np.std(tokens):.0f} tokens\n")
                            else:
                                f.write(f"  {name.replace('_', ' '):15s}: No token data\n")
                        else:
                            f.write(f"  {name.replace('_', ' '):15s}: No branch data\n")
                    f.write("\n")
            
            # Quality Metrics
            f.write("-"*80 + "\n")
            f.write("QUALITY METRICS COMPARISON\n")
            f.write("-"*80 + "\n\n")
            
            for dim in self.comparison_dimensions:
                f.write(f"{dim.upper().replace('_', ' ')}:\n")
                
                # Get stats for each framework
                framework_stats = {}
                for name in self.framework_names:
                    data = self.frameworks[name]
                    stats = data.get('summary_statistics', {})
                    
                    if dim in stats:
                        framework_stats[name] = stats[dim]
                    else:
                        # Try to calculate from raw data
                        branches = data.get('branch_evaluations', [])
                        if branches:
                            if dim == 'overall':
                                scores = [b.get('overall_score', 0) for b in branches]
                            else:
                                scores = [b.get('scores', {}).get(dim, 0) for b in branches]
                            
                            scores = [s for s in scores if s > 0]
                            if scores:
                                framework_stats[name] = {
                                    'mean': np.mean(scores),
                                    'std': np.std(scores)
                                }
                
                # Display stats
                for name in self.framework_names:
                    if name in framework_stats:
                        stats = framework_stats[name]
                        f.write(f"  {name.replace('_', ' '):15s}: {stats['mean']:.3f} ± {stats['std']:.3f}\n")
                    else:
                        f.write(f"  {name.replace('_', ' '):15s}: No data\n")
                
                # Calculate differences if we have data
                if len(framework_stats) > 1:
                    means = [framework_stats[name]['mean'] for name in self.framework_names if name in framework_stats]
                    if means:
                        best_mean = max(means)
                        worst_mean = min(means)
                        diff = best_mean - worst_mean
                        diff_pct = (diff / worst_mean * 100) if worst_mean > 0 else 0
                        f.write(f"  Range:            {diff:.3f} ({diff_pct:.1f}% variation)\n")
                
                f.write("\n")
            
            # Best Prompt Combinations
            if hasattr(self, 'best_combinations') and any(self.best_combinations.values()):
                f.write("-"*80 + "\n")
                f.write("BEST PROMPT COMBINATIONS ANALYSIS\n")
                f.write("-"*80 + "\n\n")
                
                for name in self.framework_names:
                    best_combo = self.best_combinations.get(name)
                    if best_combo:
                        f.write(f"{name.replace('_', ' ').upper()} BEST PATTERN:\n")
                        f.write(f"  Pattern:      {best_combo['pattern']}\n")
                        f.write(f"  Mean Score:   {best_combo['mean_score']:.3f} ± {best_combo['std_score']:.3f}\n")
                        f.write(f"  Branches:     {best_combo['branches']}\n")
                        f.write(f"  Best Branch:  {best_combo['best_branch']['branch_id']} "
                               f"(Score: {best_combo['best_branch']['overall_score']:.3f})\n")
                        
                        # Add dimension scores for best branch
                        best_scores = best_combo['best_branch'].get('scores', {})
                        if best_scores:
                            f.write(f"  Dimension Scores:\n")
                            for dim in self.comparison_dimensions:
                                if dim in best_scores and dim != 'overall':
                                    f.write(f"    - {dim.capitalize():12s}: {best_scores[dim]:.2f}\n")
                        f.write("\n")
                    else:
                        f.write(f"{name.replace('_', ' ').upper()}: No prompt pattern data\n\n")
                
                # Overall winner analysis
                valid_combos = {k: v for k, v in self.best_combinations.items() if v is not None}
                if len(valid_combos) > 1:
                    best_framework = max(valid_combos.items(), key=lambda x: x[1]['mean_score'])
                    best_name, best_data = best_framework
                    
                    f.write("OVERALL WINNER:\n")
                    f.write(f"  -> {best_name.replace('_', ' ')}: {best_data['pattern']}\n")
                    f.write(f"    Score: {best_data['mean_score']:.3f} ± {best_data['std_score']:.3f}\n")
                    
                    # Show differences
                    for name, data in valid_combos.items():
                        if name != best_name:
                            diff = best_data['mean_score'] - data['mean_score']
                            diff_pct = (diff / data['mean_score'] * 100) if data['mean_score'] > 0 else 0
                            f.write(f"    Better than {name.replace('_', ' ')} by: {diff:.3f} points ({diff_pct:.1f}%)\n")
                    f.write("\n")
                    
                    # Pattern consistency check
                    patterns = [data['pattern'] for data in valid_combos.values()]
                    unique_patterns = set(patterns)
                    
                    f.write("PATTERN CONSISTENCY:\n")
                    if len(unique_patterns) == 1:
                        f.write(f"  ✓ CONSISTENT: All frameworks identified the same best pattern\n")
                        f.write(f"    Pattern: {list(unique_patterns)[0]}\n\n")
                    else:
                        f.write(f"  ✗ DIFFERENT: Frameworks identified different best patterns\n")
                        for name, data in valid_combos.items():
                            f.write(f"    {name.replace('_', ' '):15s}: {data['pattern']}\n")
                        f.write(f"    This suggests framework implementation affects pattern performance\n\n")
            
            # Summary and Recommendations
            f.write("-"*80 + "\n")
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n\n")
            
            f.write("KEY FINDINGS:\n")
            
            # Performance summary
            if 'total_time' in self.execution_metrics:
                times = []
                for name in self.framework_names:
                    timing = self.frameworks[name].get('timing', {})
                    time_val = (timing.get('total_project_time') or 
                              timing.get('total_time') or 
                              timing.get('execution_time', 0))
                    times.append((name, time_val))
                
                times.sort(key=lambda x: x[1])
                if times and times[0][1] > 0:
                    f.write(f"  • Fastest execution: {times[0][0].replace('_', ' ')} ({times[0][1]:.2f}s)\n")
                    if len(times) > 1:
                        f.write(f"  • Slowest execution: {times[-1][0].replace('_', ' ')} ({times[-1][1]:.2f}s)\n")
                        ratio = times[-1][1] / times[0][1] if times[0][1] > 0 else 0
                        f.write(f"  • Speed difference: {ratio:.2f}x\n")
            
            # Quality summary
            if 'overall' in self.comparison_dimensions:
                overall_scores = []
                for name in self.framework_names:
                    data = self.frameworks[name]
                    stats = data.get('summary_statistics', {})
                    if 'overall' in stats:
                        overall_scores.append((name, stats['overall']['mean']))
                    else:
                        branches = data.get('branch_evaluations', [])
                        if branches:
                            scores = [b.get('overall_score', 0) for b in branches]
                            scores = [s for s in scores if s > 0]
                            if scores:
                                overall_scores.append((name, np.mean(scores)))
                
                overall_scores.sort(key=lambda x: x[1], reverse=True)
                if overall_scores:
                    f.write(f"  • Highest quality: {overall_scores[0][0].replace('_', ' ')} ({overall_scores[0][1]:.3f})\n")
                    if len(overall_scores) > 1:
                        f.write(f"  • Lowest quality: {overall_scores[-1][0].replace('_', ' ')} ({overall_scores[-1][1]:.3f})\n")
                        diff = overall_scores[0][1] - overall_scores[-1][1]
                        f.write(f"  • Quality difference: {diff:.3f} points\n")
            
            f.write("\n")
            
            f.write("RECOMMENDATIONS:\n")
            f.write("  • Choose framework based on your priorities:\n")
            f.write("    - Speed: Select the fastest execution framework\n")
            f.write("    - Quality: Select the highest scoring framework\n")
            f.write("    - Balance: Consider speed vs quality trade-offs\n")
            f.write("  • Consider the best prompt patterns identified for each framework\n")
            f.write("  • Framework differences may affect prompt style effectiveness\n")
        
        print("  [OK] Created: comparison_report.txt")
    
    def compare_branching_efficiency(self, new_node_name: str):
        """Compare branching efficiency across frameworks.
        
        Analyzes the cost of exploring new branches from a rollback point.
        Creates separate visualizations for time, tokens, and combined cost.
        
        Args:
            new_node_name: Name of the node to analyze branching from
        """
        print(f"\n[BRANCHING] Analyzing branching efficiency for node: {new_node_name}")
        
        # Color palette for frameworks
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.framework_names)))
        
        # Create three separate charts
        self._plot_branching_time(new_node_name, colors)
        self._plot_branching_tokens(new_node_name, colors) 
        
        print("  [OK] Created branching efficiency visualizations")
    
    def _plot_branching_time(self, new_node_name: str, colors):
        """Plot time cost for branching operations."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = []
        labels = []
        colors_used = []
        
        for i, name in enumerate(self.framework_names):
            data = self.frameworks[name]
            
            if "langgraph" in name.lower().replace(" ", "").replace("_", ""):
                # LangGraph: only incremental cost from rollback point
                time_usage = sum(
                    branch.get('metadata', {}).get("breakdown", {}).get(f'{new_node_name}_time', 0)
                    for branch in data.get('branch_evaluations', [])
                )
            elif "agentgit" in name.lower().replace(" ", "").replace("_", ""):
                time_usage = data.get('timing', {}).get('total_project_time', 0)
            else:
                # Other frameworks: full re-execution cost
                time_usage = sum(
                    sum(branch.get('metadata', {}).get("breakdown", {}).get(stat, 0) if 'time' in stat else 0
                    for stat in branch.get('metadata', {}).get("breakdown", {}))
                    for branch in data.get('branch_evaluations', [])
                )
            
            times.append(time_usage)
            labels.append(name.replace('_', ' '))
            colors_used.append(colors[i])
        
        bars = ax.bar(labels, times, color=colors_used, alpha=0.8,
                     edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Branching Time Cost (seconds)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(times) * 0.02,
                   f'{height:.1f}s', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        if max(times) > 0:
            ax.set_ylim(0, max(times) * 1.15)
        
        if len(labels) > 3:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "branching_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: branching_time.png")
    
    def _plot_branching_tokens(self, new_node_name: str, colors):
        """Plot token cost for branching operations."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tokens = []
        labels = []
        colors_used = []
        
        for i, name in enumerate(self.framework_names):
            data = self.frameworks[name]
            
            if "langgraph" in name.lower().replace(" ", "").replace("_", ""):
                # LangGraph: only incremental tokens from rollback point
                token_usage = sum(
                    branch.get('metadata', {}).get("breakdown", {}).get(f'{new_node_name}_tokens', {}).get('total_tokens', 0)
                    for branch in data.get('branch_evaluations', [])
                )
            elif "agentgit" in name.lower().replace(" ", "").replace("_", ""):
                token_usage = sum(
                    branch.get('metadata', {}).get("breakdown", {}).get(f'{new_node_name}_tokens', {}).get('total_tokens', 0)
                    for branch in data.get('branch_evaluations', [])
                )
            else:
                # Other frameworks: full re-execution tokens
                token_usage = sum(
                    sum(branch.get('metadata', {}).get("breakdown", {}).get(stat, {}).get('total_tokens', 0) if 'tokens' in stat else 0
                    for stat in branch.get('metadata', {}).get("breakdown", {}))
                    for branch in data.get('branch_evaluations', [])
                )
            
            tokens.append(token_usage)
            labels.append(name.replace('_', ' '))
            colors_used.append(colors[i])
        
        bars = ax.bar(labels, tokens, color=colors_used, alpha=0.8,
                     edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Branching Token Cost', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(tokens) * 0.02,
                   f'{int(height):,}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        
        if max(tokens) > 0:
            ax.set_ylim(0, max(tokens) * 1.15)
        
        if len(labels) > 3:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "branching_tokens.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  [OK] Created: branching_tokens.png")
         
if __name__ == "__main__":
    main()

