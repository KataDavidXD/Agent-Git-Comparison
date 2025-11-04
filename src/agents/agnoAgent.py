import itertools
import os
import time
import json
import uuid
import operator
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, Annotated, List, Dict, Any

from langchain_openai import ChatOpenAI

try:
    from agno.workflow.step import Step, StepInput, StepOutput
    from agno.workflow.workflow import Workflow
    from agno.db.sqlite import SqliteDb
except ImportError:
    raise ImportError("Agno library is required for this module. Please install it via 'pip install agno'.")

# Load environment variables from .env file
load_dotenv()

# 1. Define State for Paper Search and Summarization
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    topic: str
    date_filter: str   # Optional date filter for arXiv search
    max_results: int   # Number of papers to retrieve
    search_results: List[dict]
    abstracts: List[dict]
    introduction: str  # Final Introduction section
    analysis: str      # Final Analysis section
    discussion: str    # Final Discussion section
    final_review: str  # Concatenated final review
    
    prompt_introduction: Dict[str, Any]
    prompt_analysis: Dict[str, Any]
    prompt_discussion: Dict[str, Any]
    experiment_result: Dict[str, Any]

# 2. Helper Functions for Configuration
def load_prompts(prompt_file: str) -> List[Dict[str, Any]]:
    """Load prompt candidates from a JSON file"""
    prompt_path = Path(__file__).parent / prompt_file

    with open(prompt_path, 'r') as f:
        return json.load(f)
    
def track_tokens(response) -> Dict[str, int]:
    """Track token usage from LLM response."""
    token_usage = {}
    if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
        usage = response.response_metadata['token_usage']
        token_usage["prompt_tokens"] = usage.get('prompt_tokens', 0)
        token_usage["completion_tokens"] = usage.get('completion_tokens', 0)
        token_usage["total_tokens"] = usage.get('total_tokens', 0)
    return token_usage

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)

# 3. Define Processing Node Functions
def step_search_papers(step_input: StepInput, session_state: AgentState) -> StepOutput:
    """Step 1: Search for papers online using arXiv API"""
    topic = session_state["topic"]
    date_filter = session_state.get("date_filter", "")  # Optional: e.g., "submittedDate:[20230101 TO 20231231]"
    max_results = session_state.get("max_results", 50)
    
    print(f"[STEP] Searching arXiv for: {topic} (max_results={max_results})")
    
    # Retry logic with exponential backoff
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            base_url = "http://export.arxiv.org/api/query"
            
            # Build query with optional date filter
            if date_filter:
                query = f"search_query=all:{topic}+AND+{date_filter}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
            else:
                query = f"search_query=all:{topic}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
            
            if attempt == 0:
                print(f"  Query: {base_url}?{query[:100]}...")
            else:
                print(f"  Retry {attempt}/{max_retries-1}...")
                
            response = requests.get(f"{base_url}?{query}", timeout=15, verify=False)

            if response.status_code == 200:
                search_results = [{
                    "source": "arxiv",
                    "data": response.text,
                    "status": "success"
                }]
                print(f"  [OK] Got response data ({len(response.text)} bytes)")
                session_state.update({
                    "messages": [{"role": "ai", "content": f"Found papers on '{topic}'"}],
                    "search_results": search_results
                })
                return StepOutput(content=f"Found {len(search_results)} search results for topic {topic}: {search_results}")
            else:
                print(f"  [WARN] Status code: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    search_results = [{
                        "source": "arxiv",
                        "error": f"Status code: {response.status_code}",
                        "status": "failed"
                    }]
                    print(f"  [FAIL] All retries exhausted")
                    
        except Exception as e:
            print(f"  [ERROR] Attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            else:
                search_results = [{
                    "source": "arxiv",
                    "error": str(e),
                    "status": "failed"
                }]
                print(f"  [FAIL] All retries exhausted after error")

    session_state.update({
        "search_results": search_results
    })
    session_state["messages"].append({"role": "ai", "content": f"Searched papers error for topic: {topic}"})
    return StepOutput(content=f"Search failed for topic {topic}: {search_results}")

def step_extract_abstracts(step_input: StepInput, session_state: AgentState) -> StepOutput:
    """Step 2: Extract abstracts from search results"""
    import xml.etree.ElementTree as ET

    abstracts = []
    search_results = session_state.get("search_results", [])

    for idx, result in enumerate(search_results):
        if result.get("status") == "success" and "data" in result:
            try:
                data = result["data"]
                root = ET.fromstring(data)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}

                entries = root.findall('atom:entry', ns)
                
                for entry in entries:
                    title = entry.find('atom:title', ns)
                    abstract = entry.find('atom:summary', ns)
                    authors = entry.findall('atom:author/atom:name', ns)

                    if title is not None and abstract is not None:
                        abstracts.append({
                            "title": title.text.strip() if title.text else "N/A",
                            "abstract": abstract.text.strip() if abstract.text else "N/A",
                            "authors": [a.text for a in authors if a.text] if authors else []
                        })
                        
            except Exception as e:
                print(f"  ERROR: Failed to parse XML: {e}")
                abstracts.append({"error": f"Failed to parse: {str(e)}"})
    
    print(f"  [OK] Extracted {len(abstracts)} abstracts")

    session_state.update({
        "abstracts": abstracts
    })
    session_state["messages"].append({"role": "ai", "content": f"Extracted {len(abstracts)} abstracts from search results."})

    return StepOutput(content=f"Extracted {len(abstracts)} abstracts: {abstracts}")

def step_generate_introduction(step_input: StepInput, session_state: AgentState) -> StepOutput:
    """Step 3: Generate Introduction using LLM based on extracted abstracts"""
    start_time = time.time()
    prompt_intro = session_state["prompt_introduction"]
    print(f"[STEP] Generating Introduction with prompt {prompt_intro["id"]}")
    
    abstracts = session_state.get("abstracts", [])
    
    # Prepare abstracts text
    abstracts_text = "\n\n".join([
        f"Paper {i+1}: {abs.get('title', 'N/A')}\n"
        f"Authors: {', '.join(abs.get('authors', []))}\n"
        f"Abstract: {abs.get('abstract', 'N/A')}"
        for i, abs in enumerate(abstracts) if 'error' not in abs
    ])
    
    user_prompt = prompt_intro["user"].format(abstracts_text=abstracts_text)
    system_message = prompt_intro.get("system", "You are a helpful research assistant.")
    
    messages = [dict(role="human", content=f"System: {system_message}\n\nUser: {user_prompt}")]
    response = llm.invoke(messages)
    
    original_token_usage = session_state["experiment_result"].get("cumulative_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    current_token_usage = track_tokens(response)
    cum_token_usage = {
        "prompt_tokens": original_token_usage.get("prompt_tokens", 0) + current_token_usage.get("prompt_tokens", 0),
        "completion_tokens": original_token_usage.get("completion_tokens", 0) + current_token_usage.get("completion_tokens", 0),
        "total_tokens": original_token_usage.get("total_tokens", 0) + current_token_usage.get("total_tokens", 0),
    }

    print(f"  → Prompt: {prompt_intro['id']} (Style: {prompt_intro['style']})")
    print(f"  → Generated {len(response.content)} chars")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    session_state.update({
        "introduction": response.content
    })
    session_state["experiment_result"].update({
        "introduction_tokens": current_token_usage,
        "introduction_time": execution_time,
        "cumulative_time": session_state["experiment_result"].get("cumulative_time", 0) + execution_time,
        "cumulative_tokens": cum_token_usage
        })
    session_state["messages"].append({"role": "ai", "content": f"[{prompt_intro['id']}] Introduction generated"})
    return StepOutput(content=f"Generated introduction: {response.content}")

def step_generate_analysis(step_input: StepInput, session_state: AgentState) -> StepOutput:
    """Step 4: Generate Literature Analysis section."""

    start_time = time.time()
    prompt_ana = session_state["prompt_analysis"]
    print(f"[STEP] Generating Analysis with prompt {prompt_ana['id']}")
    

    abstracts = session_state.get("abstracts", [])
    introduction = session_state.get("introduction", "")

    # Prepare abstracts text
    abstracts_text = "\n\n".join([
        f"Paper {i+1}: {abs.get('title', 'N/A')}\n"
        f"Authors: {', '.join(abs.get('authors', []))}\n"
        f"Abstract: {abs.get('abstract', 'N/A')}"
        for i, abs in enumerate(abstracts) if 'error' not in abs
    ])
    
    user_prompt = prompt_ana["user"].format(
        abstracts_text=abstracts_text,
        introduction=introduction
    )
    system_message = prompt_ana.get("system", "You are a helpful research assistant.")
    
    messages = [dict(role="human", content=f"System: {system_message}\n\nUser: {user_prompt}")]
    response = llm.invoke(messages)

    original_token_usage = session_state["experiment_result"].get("cumulative_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    current_token_usage = track_tokens(response)
    cum_token_usage = {
        "prompt_tokens": original_token_usage.get("prompt_tokens", 0) + current_token_usage.get("prompt_tokens", 0),
        "completion_tokens": original_token_usage.get("completion_tokens", 0) + current_token_usage.get("completion_tokens", 0),
        "total_tokens": original_token_usage.get("total_tokens", 0) + current_token_usage.get("total_tokens", 0),
    }

    print(f"  → Prompt: {prompt_ana['id']} (Style: {prompt_ana['style']})")
    print(f"  → Generated {len(response.content)} chars")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    session_state.update({
        "analysis": response.content
    })
    session_state["experiment_result"].update({
            "analysis_tokens": current_token_usage,
            "analysis_time": execution_time,
            "cumulative_time": session_state["experiment_result"].get("cumulative_time", 0) + execution_time,
            "cumulative_tokens": cum_token_usage
        })
    session_state["messages"].append({"role": "ai", "content": f"[{prompt_ana['id']}] Analysis generated"})
    return StepOutput(content=f"Generated analysis: {response.content}")

def step_generate_discussion(step_input: StepInput, session_state: AgentState) -> StepOutput:
    """Step 5: Generate Critical Discussion section."""
    start_time = time.time()
    prompt_disc = session_state["prompt_discussion"]
    print(f"[STEP] Generating Discussion with prompt {prompt_disc["id"]}")

    abstracts = session_state.get("abstracts", [])
    introduction = session_state.get("introduction", "")
    analysis = session_state.get("analysis", "")
    
    # Prepare abstracts text
    abstracts_text = "\n\n".join([
        f"Paper {i+1}: {abs.get('title', 'N/A')}\n"
        f"Authors: {', '.join(abs.get('authors', []))}\n"
        f"Abstract: {abs.get('abstract', 'N/A')}"
        for i, abs in enumerate(abstracts) if 'error' not in abs
    ])
    
    user_prompt = prompt_disc["user"].format(
        abstracts_text=abstracts_text,
        introduction=introduction,
        analysis=analysis
    )
    system_message = prompt_disc.get("system", "You are a helpful research assistant.")

    messages = [dict(role="human", content=f"System: {system_message}\n\nUser: {user_prompt}")]
    response = llm.invoke(messages)

    original_token_usage = session_state["experiment_result"].get("cumulative_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    current_token_usage = track_tokens(response)
    cum_token_usage = {
        "prompt_tokens": original_token_usage.get("prompt_tokens", 0) + current_token_usage.get("prompt_tokens", 0),
        "completion_tokens": original_token_usage.get("completion_tokens", 0) + current_token_usage.get("completion_tokens", 0),
        "total_tokens": original_token_usage.get("total_tokens", 0) + current_token_usage.get("total_tokens", 0),
    }

    print(f"  → Prompt: {prompt_disc['id']} (Style: {prompt_disc['style']})")
    print(f"  → Generated {len(response.content)} chars")
    end_time = time.time()
    execution_time = end_time - start_time
    
    session_state.update({
        "discussion": response.content,
    })
    session_state["experiment_result"].update({
            "discussion_tokens": current_token_usage,
            "discussion_time": execution_time,
            "cumulative_time": session_state["experiment_result"].get("cumulative_time", 0) + execution_time,
            "cumulative_tokens": cum_token_usage
        })
    session_state["messages"].append({"role": "ai", "content": f"[{prompt_disc['id']}] Discussion generated"})
    return StepOutput(content=f"Generated discussion: {response.content}")

def step_concatenate(step_input: StepInput, session_state: AgentState) -> StepOutput:
    """Node 6: Concatenate sections into final review (NO regeneration)."""
    start_time = time.time()
    print(f"[NODE] Concatenating sections")

    topic = session_state.get("topic", "Unknown")
    introduction = session_state.get("introduction", "")
    analysis = session_state.get("analysis", "")
    discussion = session_state.get("discussion", "")

    # Simple concatenation with section headers and transitions
    final_review = f"""# Literature Review: {topic}

## 1. Introduction

{introduction}

## 2. Literature Analysis

{analysis}

## 3. Critical Discussion

{discussion}

---
*This review synthesizes findings from {len(session_state.get('abstracts', []))} recent papers.*
"""
    
    word_count = len(final_review.split())
    print(f"  → Final review: {word_count} words")
    end_time = time.time()
    execution_time = end_time - start_time
    
    session_state.update({
        "final_review": final_review,
    })
    session_state["experiment_result"].update({
        "cumulative_time": session_state["experiment_result"].get("cumulative_time", 0) + execution_time,
    })
    session_state["messages"].append({"role": "ai", "content": f"Final review concatenated"})
    return StepOutput(content=f"Final review generated with {word_count} words.")

# 4. Build Workflow Dynamically from Configuration
def run_workflow(initial_state: AgentState, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the workflow graph dynamically based on config"""
    session_id = workflow_config.get("session_id", "default_session")
    db = SqliteDb(
            session_table="workflow_session",
            db_file="tmp/workflow.db",
        )
    db.delete_session(session_id)

    workflow = Workflow(
        session_id=session_id,
        steps=[
            Step(name="Search Papers", executor=step_search_papers),
            Step(name="Extract Abstracts", executor=step_extract_abstracts),
            Step(name="Generate Introduction", executor=step_generate_introduction),
            Step(name="Generate Analysis", executor=step_generate_analysis),
            Step(name="Generate Discussion", executor=step_generate_discussion),
            Step(name="Concatenate Review", executor=step_concatenate),
        ],
        db=SqliteDb(
            session_table="workflow_session",
            db_file="tmp/workflow.db",
        ),
        session_state=initial_state
    )
    workflow.run()
    return workflow.get_session_state(session_id=workflow_config.get("session_id", "default_session"))

# 5. Run Workflow in different prompt configurations in parallel
def run_all_experiments(topic: str, run_id: str = None):
    
    # Running config
    project_start_time = time.time()
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + str(uuid.uuid4())[:8]
    
    
    prompt_intro_candidates = load_prompts("prompts_introduction.json")
    prompt_ana_candidates = load_prompts("prompts_analysis.json")
    prompt_disc_candidates = load_prompts("prompts_discussion.json")
    all_results = []
    for (prompt_intro_idx, prompt_intro), (prompt_ana_idx, prompt_ana), (prompt_disc_idx, prompt_disc) in \
        itertools.product(
            enumerate(prompt_intro_candidates),
            enumerate(prompt_ana_candidates),
            enumerate(prompt_disc_candidates)
        ):
        init_state = AgentState(
            messages=[],
            topic=topic,
            date_filter="submittedDate:[20230101 TO 20231231]",
            max_results=10,
            search_results=[],
            abstracts=[],
            introduction="",
            analysis="",
            discussion="",
            final_review="",
            prompt_introduction=prompt_intro,
            prompt_analysis=prompt_ana,
            prompt_discussion=prompt_disc,
            experiment_result={}
        )
        state = run_workflow(init_state, {"session_id": f"I{prompt_intro_idx}_A{prompt_ana_idx}_D{prompt_disc_idx}"})
        all_results.append({
            "branch_id": f"I{prompt_intro_idx}_A{prompt_ana_idx}_D{prompt_disc_idx}",
            "prompt_intro": state["prompt_introduction"]["id"],
            "prompt_intro_style": state["prompt_introduction"]["style"],
            "prompt_analysis": state["prompt_analysis"]["id"],
            "prompt_analysis_style": state["prompt_analysis"]["style"],
            "prompt_discussion": state["prompt_discussion"]["id"],
            "prompt_discussion_style": state["prompt_discussion"]["style"],
            "introduction": state.get("introduction", ""),
            "analysis": state.get("analysis", ""),
            "discussion": state.get("discussion", ""),
            "final_review": state["final_review"],
            "metadata": {
                "topic": topic,
                "num_papers": len(state.get("abstracts", [])),
                "word_count": len(state["final_review"].split()),
                "branch_execution_time": state["experiment_result"].get("cumulative_time", 0),
                "branch_token_usage": state["experiment_result"].get("cumulative_tokens", {}),
                "breakdown": {
                    "introduction_time": state["experiment_result"]["introduction_time"],
                    "analysis_time": state["experiment_result"]["analysis_time"],
                    "discussion_time": state["experiment_result"]["discussion_time"],
                    "introduction_tokens": state["experiment_result"]["introduction_tokens"],
                    "analysis_tokens": state["experiment_result"]["analysis_tokens"],
                    "discussion_tokens": state["experiment_result"]["discussion_tokens"],
                }
            }
        })
        
    total_project_time = time.time() - project_start_time
    # with ThreadPoolExecutor(max_workers=total_experiments) as executor:
    #     futures = {executor.submit(run_workflow, init_state, {"session_id": session_id}): session_id for init_state, session_id in zip(init_state_ls, session_ids)}
    #     for future in as_completed(futures):
    #         all_states.append(future.result())
    
    token_usage_overall = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    for result in all_results:
        branch_tokens = result["metadata"]["branch_token_usage"]
        token_usage_overall["prompt_tokens"] += branch_tokens.get("prompt_tokens", 0)
        token_usage_overall["completion_tokens"] += branch_tokens.get("completion_tokens", 0)
        token_usage_overall["total_tokens"] += branch_tokens.get("total_tokens", 0)
        
    all_results.sort(key=lambda x: (x["prompt_intro"], x["prompt_analysis"], x["prompt_discussion"]))
    
    print("\n" + "="*80)
    print(f"WORKFLOW COMPLETE - Generated {len(all_results)} literature reviews")
    print("="*80)
    print(f"\nTIMING SUMMARY:")
    print(f"  TOTAL PROJECT TIME: {total_project_time:.2f}s")
    print("="*80)

    # Save results
    topic_clean = topic.replace(' ', '_').lower()
    results_base = Path(__file__).parent.parent.parent.parent / "results"
    run_dir = results_base / f"{topic_clean}" / "agno"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = raw_dir / f"{run_id}.json"
    output_data = {
        "run_id": run_id,
        "topic": topic,
        "workflow_type": "section_based",
        "timestamp": datetime.now().isoformat(),
        "total_branches": len(all_results),
        "timing": {
            "total_project_time": total_project_time
        },
        "token_usage_overall": token_usage_overall,
        "phase_1_state": {
            "search_results_count": len(state.get("search_results", [])),
            "abstracts_count": len(state.get("abstracts", [])),
            "abstracts": state.get("abstracts", [])
        },
        "branches": all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SAVED] Results saved to: {output_file}")

    return all_results, str(output_file), run_id

# 6. Example Usage
if __name__ == "__main__":
    topic = "large language models"
    all_results, output_file, run_id = run_all_experiments(topic)
    
    print("\n" + "="*80)
    print(f"FINAL RESULTS - {len(all_results)} BRANCHES")
    print("="*80)

    for i, result in enumerate(all_results[:5], 1):
        print(f"\n[Branch {i}] {result['branch_id']}")
        print(f"  Styles: {result['prompt_intro_style']} + {result['prompt_analysis_style']} + {result['prompt_discussion_style']}")
        print(f"  Papers: {result['metadata']['num_papers']}")
        print(f"  Total Time: {result['metadata']['branch_execution_time']:.2f}s")
        print(f"  Total Tokens: {result['metadata']['branch_token_usage']['total_tokens']:,}")
        print(f"  Words: {result['metadata']['word_count']}")
    
    if len(all_results) > 5:
        print(f"\n... and {len(all_results) - 5} more branches")
    
    print(f"\n[OK] Generated {len(all_results)} different reviews via section-based branching!")
    print(f"[OK] Results saved to: {output_file}")
