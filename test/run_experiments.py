from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
from pathlib import Path
import json
import numpy as np
import asyncio
from functools import partial
from dotenv import load_dotenv
from copy import deepcopy

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath('src')))

from experiments.evaluator import BranchEvaluator, FrameworkComparator, ResultsAnalyzer

global EXPERIMENT_NUM

def import_experiment_modules():
    if EXPERIMENT_NUM == 1:
        from experiments.agentgitAgent_sections import SectionBasedSummarizer
        from experiments.agnoAgent import run_all_experiments as run_agno_experiments
        from experiments.autogenAgent import run_all_experiments as run_autogen_experiments
        from experiments.langgraph_time_travel_agent import TimeTravelSummarizer
        return SectionBasedSummarizer, TimeTravelSummarizer, run_autogen_experiments, run_agno_experiments
    elif EXPERIMENT_NUM == 2:
        from experiments.agentgitAgent_exp2 import SectionBasedSummarizer
        from experiments.langgraph_exp2 import TimeTravelSummarizer
        from experiments.autogenAgent_exp2 import run_all_experiments as run_autogen_experiments
        from experiments.agnoAgent_exp2 import run_all_experiments as run_agno_experiments
        return SectionBasedSummarizer, TimeTravelSummarizer, run_autogen_experiments, run_agno_experiments
    else:
        raise ValueError(f"Unsupported experiment number: {EXPERIMENT_NUM}")


evaluator = BranchEvaluator(model='gpt-4o-mini', num_samples=10)

RESULT_TEMPLATE = {
    "topic": "",
    "timestamp": "",
    "evaluation_model": "gpt-4o-mini",
    "num_branches": 0,
    "num_papers": 0,
    "timing": {
        "total_project_time": 0
    },
    "token_usage_overall": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    },
    "branch_evaluations": [],
    "summary_statistics": {}
    }

def run_and_eval_agentgit_experiment(topic: str):
    agent = SectionBasedSummarizer()
    result =  agent.run_workflow(topic=topic)
    output_path = result[1]
    evaluator.evaluate_all_branches(output_path)
    eval_file_path = str(Path(output_path).parent.parent.joinpath('eval')) + f"/evaluation_{Path(output_path).stem}.json"
    return eval_file_path

def run_and_eval_langgraph_experiments(topic: str):
    agent = TimeTravelSummarizer()
    result = agent.run_workflow(topic=topic)
    output_path = result[1]
    evaluator.evaluate_all_branches(output_path)
    eval_file_path = str(Path(output_path).parent.parent.joinpath('eval')) + f"/evaluation_{Path(output_path).stem}.json"
    return eval_file_path

async def run_and_eval_autogen_experiments(topic: str):
    result = await run_autogen_experiments(topic=topic)
    output_path = result[1]
    evaluator.evaluate_all_branches(output_path)
    eval_file_path = str(Path(output_path).parent.parent.joinpath('eval')) + f"/evaluation_{Path(output_path).stem}.json"
    return eval_file_path

def run_and_eval_agno_experiments(topic: str):
    result = run_agno_experiments(topic=topic)
    output_path = result[1]
    evaluator.evaluate_all_branches(output_path)
    eval_file_path = str(Path(output_path).parent.parent.joinpath('eval')) + f"/evaluation_{Path(output_path).stem}.json"
    return eval_file_path

def retry_wrapper(func, retries=3):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt < retries - 1:
                print(f"Retrying due to error: {str(e)} (Attempt {attempt + 1} of {retries})")
            else:
                raise
            
async def async_retry_wrapper(func, retries=3):
    for attempt in range(retries):
        try:
            return await func()
        except Exception as e:
            if attempt < retries - 1:
                print(f"Retrying due to error: {str(e)} (Attempt {attempt + 1} of {retries})")
            else:
                raise

def run_experiments_in_parallel(experiment_func, max_workers=10, nums=20):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_experiment = {executor.submit(retry_wrapper, experiment_func): i for i in range(nums)}
        results = []
        for future in as_completed(future_to_experiment):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error occurred in experiment: {str(e)}")
                results.append(f"Error: {str(e)}")
    return results

def compute_average_stats(results, experiment_num=1):
    output_files = results
    result_data = deepcopy(RESULT_TEMPLATE)
    
    all_project_time = 0
    token_usage_overall = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    branch_stats = {}
    fluency_ls = []
    coherence_ls = []
    consistency_ls = []
    relevance_ls = []
    overall_score_ls = []
    for file in output_files:
        with open(file, 'r') as f:
            content = json.load(f)
        if result_data.get("num_branches") is None:
            result_data["num_branches"] = content["num_branches"]
        if result_data.get("num_papers") is None:
            result_data["num_papers"] = content["num_papers"]
            
        all_project_time += content["timing"]["total_project_time"]
        
        token_usage_overall["prompt_tokens"] += content.get("token_usage_overall", {}).get("prompt_tokens", 0)
        token_usage_overall["completion_tokens"] += content.get("token_usage_overall", {}).get("completion_tokens", 0)
        token_usage_overall["total_tokens"] += content.get("token_usage_overall", {}).get("total_tokens", 0)
        
        for branch_eval in content.get("branch_evaluations", []):
            branch_id = branch_eval["branch_id"]
            if branch_id not in branch_stats:
                branch_stats[branch_id] = {
                    "scores": {
                        "fluency": 0,
                        "coherence": 0,
                        "consistency": 0,
                        "relevance": 0
                    },
                    "overall_score": [],
                    "metadata": {
                        "branch_execution_time": 0,
                        "branch_token_usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                }
                branch_stats[branch_id]["prompt_intro"] = branch_eval.get("prompt_intro", "")
                branch_stats[branch_id]["prompt_intro_style"] = branch_eval.get("prompt_intro_style", "")
                branch_stats[branch_id]["prompt_analysis"] = branch_eval.get("prompt_analysis", "")
                branch_stats[branch_id]["prompt_analysis_style"] = branch_eval.get("prompt_analysis_style", "")
                branch_stats[branch_id]["prompt_discussion"] = branch_eval.get("prompt_discussion", "")
                branch_stats[branch_id]["prompt_discussion_style"] = branch_eval.get("prompt_discussion_style", "")
                if EXPERIMENT_NUM == 2:
                    branch_stats[branch_id]["prompt_limitations"] = branch_eval.get("prompt_limitations", "")
                    branch_stats[branch_id]["prompt_limitations_style"] = branch_eval.get("prompt_limitations_style", "")
            branch_stats[branch_id]["scores"]["fluency"] += branch_eval["scores"]["fluency"]
            branch_stats[branch_id]["scores"]["coherence"] += branch_eval["scores"]["coherence"]
            branch_stats[branch_id]["scores"]["consistency"] += branch_eval["scores"]["consistency"]
            branch_stats[branch_id]["scores"]["relevance"] += branch_eval["scores"]["relevance"]
            branch_stats[branch_id]["overall_score"].append(branch_eval["overall_score"])
            branch_stats[branch_id]["metadata"]["branch_execution_time"] += branch_eval.get("metadata", {}).get("branch_execution_time", 0)
            branch_stats[branch_id]["metadata"]["branch_token_usage"]["prompt_tokens"] += branch_eval.get("metadata", {}).get("branch_token_usage", {}).get("prompt_tokens", 0)
            branch_stats[branch_id]["metadata"]["branch_token_usage"]["completion_tokens"] += branch_eval.get("metadata", {}).get("branch_token_usage", {}).get("completion_tokens", 0)
            branch_stats[branch_id]["metadata"]["branch_token_usage"]["total_tokens"] += branch_eval.get("metadata", {}).get("branch_token_usage", {}).get("total_tokens", 0)

            fluency_ls.append(branch_eval["scores"]["fluency"])
            coherence_ls.append(branch_eval["scores"]["coherence"])
            consistency_ls.append(branch_eval["scores"]["consistency"])
            relevance_ls.append(branch_eval["scores"]["relevance"])
            overall_score_ls.append(branch_eval["overall_score"])
            

    result_data["timing"]["total_project_time"] = all_project_time / len(output_files)
    result_data["token_usage_overall"] = {
        "prompt_tokens": token_usage_overall["prompt_tokens"] / len(output_files),
        "completion_tokens": token_usage_overall["completion_tokens"] / len(output_files),
        "total_tokens": token_usage_overall["total_tokens"] / len(output_files)
    }
    
    for branch_id, stats in branch_stats.items():
        stats["scores"]["fluency"] /= len(output_files)
        stats["scores"]["coherence"] /= len(output_files)
        stats["scores"]["consistency"] /= len(output_files)
        stats["scores"]["relevance"] /= len(output_files)
        stats["metadata"]["branch_execution_time"] /= len(output_files)
        stats["metadata"]["branch_token_usage"]["prompt_tokens"] /= len(output_files)
        stats["metadata"]["branch_token_usage"]["completion_tokens"] /= len(output_files)
        stats["metadata"]["branch_token_usage"]["total_tokens"] /= len(output_files)

        tmp_result = {
            "branch_id": branch_id,
            "scores": stats["scores"],
            "overall_score": np.mean(stats["overall_score"]),
            "overall_score_stats": {
                "mean": np.mean(stats["overall_score"]),
                "std": np.std(stats["overall_score"]),
                "min": np.min(stats["overall_score"]),
                "max": np.max(stats["overall_score"])
            },
            "metadata": stats["metadata"],
            "prompt_intro": stats.get("prompt_intro", ""),
            "prompt_intro_style": stats.get("prompt_intro_style", ""),
            "prompt_analysis": stats.get("prompt_analysis", ""),
            "prompt_analysis_style": stats.get("prompt_analysis_style", ""),
            "prompt_discussion": stats.get("prompt_discussion", ""),
            "prompt_discussion_style": stats.get("prompt_discussion_style", "")
        }
        if EXPERIMENT_NUM == 2:
            tmp_result["prompt_limitations"] = stats.get("prompt_limitations", "")
            tmp_result["prompt_limitations_style"] = stats.get("prompt_limitations_style", "")
        result_data["branch_evaluations"].append(tmp_result)
        
    result_data["summary_statistics"] = {
        "fluency": {
            "mean": np.mean(fluency_ls),
            "std": np.std(fluency_ls),
            "median": np.median(fluency_ls),
            "min": np.min(fluency_ls),
            "max": np.max(fluency_ls)
        },
        "coherence": {
            "mean": np.mean(coherence_ls),
            "std": np.std(coherence_ls),
            "median": np.median(coherence_ls),
            "min": np.min(coherence_ls),
            "max": np.max(coherence_ls)
        },
        "consistency": {
            "mean": np.mean(consistency_ls),
            "std": np.std(consistency_ls),
            "median": np.median(consistency_ls),
            "min": np.min(consistency_ls),
            "max": np.max(consistency_ls)
        },
        "relevance": {
            "mean": np.mean(relevance_ls),
            "std": np.std(relevance_ls),
            "median": np.median(relevance_ls),
            "min": np.min(relevance_ls),
            "max": np.max(relevance_ls)
        },
        "overall_score": {
            "mean": np.mean(overall_score_ls),
            "std": np.std(overall_score_ls),
            "median": np.median(overall_score_ls),
            "min": np.min(overall_score_ls),
            "max": np.max(overall_score_ls)
        }
    }
    return result_data
        
    

def run_sync_frameworks(topic:str, output_dir: Path):
    RESULT_TEMPLATE["topic"] = topic
    
    print("Running AgentGit experiments...")
    # agentgit_results_files = run_experiments_in_parallel(partial(run_and_eval_agentgit_experiment, topic), nums=20)
    # base = str(output_dir.joinpath('sections', 'eval'))
    # agentgit_results_files = [base + '/' + f for f in os.listdir(base) if f.endswith('.json')]
    # agentgit_eval_data = compute_average_stats(agentgit_results_files)
    # with open(output_dir.joinpath(f"framework_experiments_{EXPERIMENT_NUM}", 'agentgit_evaluation_summary.json'), 'w') as f:
    #     json.dump(agentgit_eval_data, f, indent=4)
        
    print("Running LangGraph Time Travel experiments...")
    langgraph_results_files = run_experiments_in_parallel(partial(run_and_eval_langgraph_experiments, topic), nums=20)
    base = str(output_dir.joinpath('timetravel', 'eval'))
    langgraph_results_files = [base + '/' + f for f in os.listdir(base) if f.endswith('.json')]
    langgraph_eval_data = compute_average_stats(langgraph_results_files)
    with open(output_dir.joinpath(f"framework_experiments_{EXPERIMENT_NUM}", 'langgraph_time_travel_evaluation_summary.json'), 'w') as f:
        json.dump(langgraph_eval_data, f, indent=4)
        
    print("Running Agno experiments...")
    # agno_results_files = run_experiments_in_parallel(partial(run_and_eval_agno_experiments, topic), nums=20)
    # base = str(output_dir.joinpath('agno', 'eval'))
    # agno_results_files = [base + '/' + f for f in os.listdir(base) if f.endswith('.json')]
    # agno_eval_data = compute_average_stats(agno_results_files)
    # with open(output_dir.joinpath(f"framework_experiments_{EXPERIMENT_NUM}", 'agno_evaluation_summary.json'), 'w') as f:
    #     json.dump(agno_eval_data, f, indent=4)
        
    return {
        "agentgit": str(output_dir.joinpath(f"framework_experiments_{EXPERIMENT_NUM}", 'agentgit_evaluation_summary.json')),
        "langgraph (time_travel)": str(output_dir.joinpath(f"framework_experiments_{EXPERIMENT_NUM}", 'langgraph_time_travel_evaluation_summary.json')),
        "agno": str(output_dir.joinpath(f"framework_experiments_{EXPERIMENT_NUM}", 'agno_evaluation_summary.json'))
    }
    
async def run_async_frameworks(topic:str, output_dir: Path):
    print("Running AutoGen experiments...")
    # autogen_results_files = await asyncio.gather(*[asyncio.create_task(run_and_eval_autogen_experiments(topic)) for _ in range(20)])
    base = str(output_dir.joinpath('autogen', 'eval'))
    autogen_results_files = [base + '/' + f for f in os.listdir(base) if f.endswith('.json')]
    autogen_eval_data = compute_average_stats(autogen_results_files)
    with open(output_dir.joinpath(f"framework_experiments_{EXPERIMENT_NUM}", 'autogen_evaluation_summary.json'), 'w') as f:
        json.dump(autogen_eval_data, f, indent=4)
    return {
        "autogen": str(output_dir.joinpath(f"framework_experiments_{EXPERIMENT_NUM}", 'autogen_evaluation_summary.json')),
    }

if __name__ == "__main__":
    topic = "large language models"
    EXPERIMENT_NUM = 2
    output_dir = Path(__file__).parent.parent.joinpath('results', f'{topic.replace(" ", "_").lower()}')
    output_dir.mkdir(parents=True, exist_ok=True)
    SectionBasedSummarizer, TimeTravelSummarizer, run_autogen_experiments, run_agno_experiments = import_experiment_modules()
    sync_framework_files = run_sync_frameworks(topic, output_dir)
    # async_framework_files = asyncio.run(run_async_frameworks(topic, output_dir))
    # framework_files = {**sync_framework_files, **async_framework_files}
    
    # framework_files = {
    #     'agent_git': str(output_dir.joinpath("framework_experiments", 'agentgit_evaluation_summary.json')),
    #     'langgraph (time_travel)': str(output_dir.joinpath("framework_experiments", 'langgraph_time_travel_evaluation_summary.json')),
    #     'autogen': str(output_dir.joinpath("framework_experiments", 'autogen_evaluation_summary.json')),
    #     'agno': str(output_dir.joinpath("framework_experiments", 'agno_evaluation_summary.json'))
    # }
    # comparator = FrameworkComparator(framework_files)
    # comparator.compare_all()
    
    # analyzer = ResultsAnalyzer(str(str(output_dir.joinpath('agentgit_evaluation_summary.json'))))
    # analyzer.create_all_visualizations()
    
    print(f"\nAll framework evaluation summaries saved to: {output_dir}")