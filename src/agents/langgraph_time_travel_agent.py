"""
Section-based paper summarization using LangGraph Time Travel.

DESIGN:
- Phase 1: Search + Extract (unified base)
- Phase 2-4: Use TIME TRAVEL to test all prompt combinations
- Phase 5: Concatenation

KEY DIFFERENCE from agentgitAgent_sections.py:
- NO explicit branching via agent_git
- Uses LangGraph's native checkpointing and time travel
- Rewind to checkpoints and continue execution with different prompts
"""

import sys
import json
import time
import uuid
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dotenv import load_dotenv
import operator
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# LangChain/LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

# Load environment
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)

# ==============================================================================
# STATE DEFINITION
# ==============================================================================

class SectionBasedState(TypedDict):
    """State for section-based paper summarization."""
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

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_prompts(prompt_file: str) -> List[Dict[str, Any]]:
    """Load prompt candidates from JSON file."""
    prompt_path = Path(__file__).parent / prompt_file
    with open(prompt_path, 'r', encoding='utf-8') as f:
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

# ==============================================================================
# NODES - PHASE 1: SEARCH AND EXTRACT
# ==============================================================================

def node_search_papers(state: SectionBasedState) -> Dict[str, Any]:
    """Node 1: Search for papers on arXiv with retry logic."""
    topic = state["topic"]
    date_filter = state.get("date_filter", "")
    max_results = state.get("max_results", 50)
    
    print(f"[NODE] Searching arXiv for: {topic} (max_results={max_results})")
    
    # Retry logic with exponential backoff
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            base_url = "http://export.arxiv.org/api/query"
            
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
                return {
                    "messages": [AIMessage(content=f"Found papers on '{topic}'")],
                    "search_results": search_results
                }
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
    
    return {
        "messages": [AIMessage(content=f"Search failed for '{topic}'")],
        "search_results": search_results
    }

def node_extract_abstracts(state: SectionBasedState) -> Dict[str, Any]:
    """Node 2: Extract abstracts from search results."""
    print(f"[NODE] Extracting abstracts")

    abstracts = []
    search_results = state.get("search_results", [])
    
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
    
    return {
        "messages": [AIMessage(content=f"Extracted {len(abstracts)} abstracts")],
        "abstracts": abstracts
    }

# ==============================================================================
# NODES - PHASE 2/3/4: SECTION GENERATION
# ==============================================================================

def node_generate_introduction(state: SectionBasedState) -> Dict[str, Any]:
    """Node 3: Generate Introduction section using selected prompt."""
    start_time = time.time()
    prompt_intro = state["prompt_introduction"]
    print(f"[STEP] Generating Introduction with prompt {prompt_intro["id"]}")
    
    abstracts = state.get("abstracts", [])
    topic = state.get("topic", "Unknown")
    
    # Prepare abstracts text
    abstracts_text = "\n\n".join([
        f"Paper {i+1}: {abs.get('title', 'N/A')}\n"
        f"Authors: {', '.join(abs.get('authors', []))}\n"
        f"Abstract: {abs.get('abstract', 'N/A')}"
        for i, abs in enumerate(abstracts) if 'error' not in abs
    ])
    
    user_prompt = prompt_intro["user"].format(abstracts_text=abstracts_text)
    system_message = prompt_intro.get("system", "You are a helpful research assistant.")
    
    messages = [HumanMessage(content=f"System: {system_message}\n\nUser: {user_prompt}")]
    response = llm.invoke(messages)
    
    original_token_usage = state["experiment_result"].get("cumulative_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
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
    
    state.update({
        "messages": [AIMessage(content=f"[{prompt_intro['id']}] Introduction generated")],
        "introduction": response.content
    })
    state["experiment_result"].update({
        "introduction_tokens": current_token_usage,
        "introduction_time": execution_time,
        "cumulative_time": state["experiment_result"].get("cumulative_time", 0) + execution_time,
        "cumulative_tokens": cum_token_usage
    })
    return state

def node_generate_analysis(state: SectionBasedState) -> Dict[str, Any]:
    """Node 4: Generate Literature Analysis section using selected prompt."""
    start_time = time.time()
    prompt_ana = state["prompt_analysis"]
    print(f"[STEP] Generating Analysis with prompt {prompt_ana['id']}")
    
    abstracts = state.get("abstracts", [])
    introduction = state.get("introduction", "")
    
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
    
    messages = [HumanMessage(content=f"System: {system_message}\n\nUser: {user_prompt}")]
    response = llm.invoke(messages)
    
    original_token_usage = state["experiment_result"].get("cumulative_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
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
    
    print(f"  [>] Prompt: {prompt_ana['id']} (Style: {prompt_ana['style']})")
    print(f"  [>] Generated {len(response.content)} chars")
    
    state.update({
        "messages": [AIMessage(content=f"[{prompt_ana['id']}] Analysis generated")],
        "analysis": response.content
    }
    )
    state["experiment_result"].update({
        "analysis_tokens": current_token_usage,
        "analysis_time": execution_time,
        "cumulative_time": state["experiment_result"].get("cumulative_time", 0) + execution_time,
        "cumulative_tokens": cum_token_usage
    })
    return state

def node_generate_discussion(state: SectionBasedState) -> Dict[str, Any]:
    """Node 5: Generate Critical Discussion section using selected prompt."""
    start_time = time.time()
    prompt_disc = state["prompt_discussion"]
    print(f"[STEP] Generating Discussion with prompt {prompt_disc["id"]}")
    
    abstracts = state.get("abstracts", [])
    introduction = state.get("introduction", "")
    analysis = state.get("analysis", "")
    
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
    
    messages = [HumanMessage(content=f"System: {system_message}\n\nUser: {user_prompt}")]
    response = llm.invoke(messages)
    
    original_token_usage = state["experiment_result"].get("cumulative_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
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
    
    print(f"  [>] Prompt: {prompt_disc['id']} (Style: {prompt_disc['style']})")
    print(f"  [>] Generated {len(response.content)} chars")
    
    state.update({
        "messages": [AIMessage(content=f"[{prompt_disc['id']}] Discussion generated")],
        "discussion": response.content
    })
    state["experiment_result"].update({
        "discussion_tokens": current_token_usage,
        "discussion_time": execution_time,
        "cumulative_time": state["experiment_result"].get("cumulative_time", 0) + execution_time,
        "cumulative_tokens": cum_token_usage
    })
    return state

# ==============================================================================
# NODES - PHASE 5: CONCATENATION
# ==============================================================================

def node_concatenate(state: SectionBasedState) -> Dict[str, Any]:
    """Node 6: Concatenate sections into final review (NO regeneration)."""
    print(f"[NODE] Concatenating sections")
    
    topic = state.get("topic", "Unknown")
    introduction = state.get("introduction", "")
    analysis = state.get("analysis", "")
    discussion = state.get("discussion", "")
    
    # Simple concatenation with section headers
    final_review = f"""# Literature Review: {topic}

## 1. Introduction

{introduction}

## 2. Literature Analysis

{analysis}

## 3. Critical Discussion

{discussion}

---
*This review synthesizes findings from {len(state.get('abstracts', []))} recent papers.*
"""
    
    word_count = len(final_review.split())
    print(f"  [>] Final review: {word_count} words")
    
    return {
        "messages": [AIMessage(content="Final review assembled")],
        "final_review": final_review
    }

# ==============================================================================
# BUILD LANGGRAPH WORKFLOW
# ==============================================================================

def build_workflow() -> StateGraph:
    """Build the LangGraph workflow."""
    workflow = StateGraph(SectionBasedState)
    
    # Add nodes
    workflow.add_node("search_papers", node_search_papers)
    workflow.add_node("extract_abstracts", node_extract_abstracts)
    workflow.add_node("generate_introduction", node_generate_introduction)
    workflow.add_node("generate_analysis", node_generate_analysis)
    workflow.add_node("generate_discussion", node_generate_discussion)
    workflow.add_node("concatenate", node_concatenate)
    
    # Add edges
    workflow.add_edge(START, "search_papers")
    workflow.add_edge("search_papers", "extract_abstracts")
    workflow.add_edge("extract_abstracts", "generate_introduction")
    workflow.add_edge("generate_introduction", "generate_analysis")
    workflow.add_edge("generate_analysis", "generate_discussion")
    workflow.add_edge("generate_discussion", "concatenate")
    workflow.add_edge("concatenate", END)
    
    return workflow

# ==============================================================================
# TIME TRAVEL WORKFLOW RUNNER
# ==============================================================================

class TimeTravelSummarizer:
    """Section-based summarization using LangGraph Time Travel."""
    
    def __init__(self):
        self.checkpointer = MemorySaver()
        self.workflow = build_workflow()
        self.app: CompiledStateGraph = self.workflow.compile(checkpointer=self.checkpointer)
    
    def run_workflow(self, topic: str, run_id: Optional[str] = None):
        """
        Run section-based workflow using TIME TRAVEL.
        
        Flow:
        1. Phase 1: Run search → extract ONCE (creates checkpoint)
        2. Phase 2-4: For each combination:
           - Rewind to extract checkpoint
           - Update prompt indices
           - Continue from introduction → analysis → discussion → concatenate
        3. Return ALL results
        """
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + str(uuid.uuid4())[:8]
        
        project_start_time = time.time()
        
        print("\n" + "="*80)
        print(f"LangGraph Time Travel Workflow: {topic}")
        print(f"Run ID: {run_id}")
        print("="*80)
        
        # Phase 1: Run search → extract ONCE
        print(f"\n[PHASE 1] Running: search → extract (BASE EXECUTION - ONLY ONCE)")
        thread_id = "base_thread"
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = SectionBasedState(
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
            experiment_result={}
        )
        
        # Run ONLY search and extract nodes
        nodes_executed = []
        for event in self.app.stream(initial_state, config, stream_mode="updates"):
            for node_name, node_output in event.items():
                nodes_executed.append(node_name)
                print(f"  [OK] {node_name}")
                if node_name == "extract_abstracts":
                    # IMPORTANT: Break after extract completes
                    # The checkpoint is now saved
                    break
            if "extract_abstracts" in nodes_executed:
                break
        
        print(f"\n[PHASE 1 COMPLETE] Nodes executed: {nodes_executed}")
        print(f"[IMPORTANT] Search and Extract will NOT be repeated")
        
        # Get the checkpoint after extract_abstracts
        state_history = list(self.app.get_state_history(config))
        print(f"\n[DEBUG] Total checkpoints in history: {len(state_history)}")
        
        # Debug: print all checkpoints
        for i, snapshot in enumerate(state_history):
            print(f"  Checkpoint {i}: next={snapshot.next}, checkpoint_id={snapshot.config['configurable'].get('checkpoint_id', 'N/A')[:8]}...")
        
        # Find checkpoint after extract_abstracts (next should be generate_introduction)
        extract_checkpoint = None
        for state_snapshot in state_history:
            if state_snapshot.next == ("generate_introduction",):
                extract_checkpoint = state_snapshot
                print(f"\n[CHECKPOINT FOUND] After extract_abstracts")
                print(f"  [>] Checkpoint ID: {state_snapshot.config['configurable']['checkpoint_id'][:16]}...")
                print(f"  [>] Next node will be: {state_snapshot.next}")
                break
        
        if not extract_checkpoint:
            raise RuntimeError("Could not find extract checkpoint. Check workflow execution.")
        
        abstracts_count = len(extract_checkpoint.values.get("abstracts", []))
        print(f"[OK] Phase 1 complete. Found {abstracts_count} papers")
        print(f"[OK] Checkpoint saved. Ready for time travel.")
        
        # Load prompt configurations
        prompts_intro = load_prompts("prompts_introduction.json")
        prompts_analysis = load_prompts("prompts_analysis.json")
        prompts_discussion = load_prompts("prompts_discussion.json")
        
        print(f"\n[PHASE 2-5] Testing ALL combinations using TIME TRAVEL")
        print(f"  Intro prompts: {len(prompts_intro)}")
        print(f"  Analysis prompts: {len(prompts_analysis)}")
        print(f"  Discussion prompts: {len(prompts_discussion)}")
        print(f"  Total combinations: {len(prompts_intro) * len(prompts_analysis) * len(prompts_discussion)}")
        print(f"\n[TIME TRAVEL MECHANISM]")
        print(f"  [+] Search and Extract executed ONCE in Phase 1")
        print(f"  [+] Each combination will TIME TRAVEL to checkpoint after extract")
        print(f"  [+] Only execute: introduction -> analysis -> discussion -> concatenate")
        
        all_results = []
        token_usage_overall = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        # Test all combinations using TIME TRAVEL
        for intro_idx, intro_prompt in enumerate(prompts_intro):
            
            # Run introduction step to get to next checkpoint
            intro_step_config = extract_checkpoint.config
            # intro_step_config["configurable"]["thread_id"] = f"branch_I{intro_idx}"s
            # checkpoint_state = extract_checkpoint.values.copy()
            # checkpoint_state["prompt_introduction"] = intro_prompt
            
            intro_step_config = self.app.update_state(
                intro_step_config,
                values={"prompt_introduction": intro_prompt}
            )
            
            nodes_before_intro = nodes_executed.copy()
            for event in self.app.stream(None, intro_step_config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    nodes_before_intro.append(node_name)
                    print(f"  [OK] {node_name}")
                    if node_name == "generate_introduction":
                        token_usage_overall["prompt_tokens"] += node_output.get("experiment_result", {}).get("introduction_tokens", {}).get("prompt_tokens", 0)
                        token_usage_overall["completion_tokens"] += node_output.get("experiment_result", {}).get("introduction_tokens", {}).get("completion_tokens", 0)
                        token_usage_overall["total_tokens"] += node_output.get("experiment_result", {}).get("introduction_tokens", {}).get("total_tokens", 0)
                if "generate_introduction" in nodes_before_intro:
                    break
                
            state_history = list(self.app.get_state_history(config))
            intro_checkpoint = None
            for state_snapshot in state_history:
                if state_snapshot.next == ("generate_analysis",):
                    intro_checkpoint = state_snapshot
                    break

            for analysis_idx, analysis_prompt in enumerate(prompts_analysis):
                
                # Run analysis step to get to next checkpoint
                analysis_step_config = intro_checkpoint.config
                # analysis_step_config["configurable"]["thread_id"] = f"branch_I{intro_idx}_A{analysis_idx}"
                # checkpoint_state = intro_checkpoint.values.copy()
                # checkpoint_state["prompt_analysis"] = analysis_prompt
                analysis_step_config = self.app.update_state(
                    analysis_step_config,
                    values={"prompt_analysis": analysis_prompt}
                )
                nodes_before_analysis = nodes_before_intro.copy()
                for event in self.app.stream(None, analysis_step_config, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        nodes_before_analysis.append(node_name)
                        print(f"  [OK] {node_name}")
                        if node_name == "generate_analysis":
                            token_usage_overall["prompt_tokens"] += node_output.get("experiment_result", {}).get("analysis_tokens", {}).get("prompt_tokens", 0)
                            token_usage_overall["completion_tokens"] += node_output.get("experiment_result", {}).get("analysis_tokens", {}).get("completion_tokens", 0)
                            token_usage_overall["total_tokens"] += node_output.get("experiment_result", {}).get("analysis_tokens", {}).get("total_tokens", 0)
                    if "generate_analysis" in nodes_before_analysis:
                        break

                state_history = list(self.app.get_state_history(config))
                analysis_checkpoint = None
                for state_snapshot in state_history:
                    if state_snapshot.next == ("generate_discussion",):
                        analysis_checkpoint = state_snapshot
                        break

                for discussion_idx, discussion_prompt in enumerate(prompts_discussion):
                    
                    # Run analysis step to get to next checkpoint
                    discussion_step_config = analysis_checkpoint.config
                    # discussion_step_config["configurable"]["thread_id"] = f"branch_I{intro_idx}_A{analysis_idx}_D{discussion_idx}"
                    # checkpoint_state = analysis_checkpoint.values.copy()
                    # checkpoint_state["prompt_discussion"] = discussion_prompt
                    discussion_step_config = self.app.update_state(
                        discussion_step_config,
                        values={"prompt_discussion": discussion_prompt}
                    )
                    final_state = None
                    nodes_all = nodes_before_analysis.copy()
                    for event in self.app.stream(None, discussion_step_config, stream_mode="updates"):
                        for node_name, node_output in event.items():
                            nodes_all.append(node_name)
                            print(f"  [OK] {node_name}")
                            if node_name == "generate_discussion":
                                token_usage_overall["prompt_tokens"] += node_output.get("experiment_result", {}).get("discussion_tokens", {}).get("prompt_tokens", 0)
                                token_usage_overall["completion_tokens"] += node_output.get("experiment_result", {}).get("discussion_tokens", {}).get("completion_tokens", 0)
                                token_usage_overall["total_tokens"] += node_output.get("experiment_result", {}).get("discussion_tokens", {}).get("total_tokens", 0)

                            # Verify we're NOT re-running search or extract
                            if node_name in ["search_papers", "extract_abstracts", "generate_introduction", "generate_analysis"]:
                                print(f"  [ERROR] CRITICAL: {node_name} should NOT execute again!")
                                raise RuntimeError(f"Time travel failed: {node_name} re-executed")
                            
                            if node_name == "concatenate":
                                # Get final state
                                final_snapshot = self.app.get_state(config)
                                final_state = final_snapshot.values
                    
                    print(f"\n[BRANCH COMPLETE] Nodes executed: {nodes_all}")
                    
                    # Verify no duplicate execution
                    if len(set(nodes_all)) < len(nodes_all):
                        print(f"[ERROR] Time travel FAILED - nodes re-executed")
                        raise RuntimeError("Time travel verification failed")
                    else:
                        print(f"[SUCCESS] Time travel worked - only new nodes executed")
                    
                    if final_state:
                        
                        result = {
                            "branch_id": f"I{intro_idx}_A{analysis_idx}_D{discussion_idx}",
                            "prompt_intro": final_state["prompt_introduction"]["id"],
                            "prompt_intro_style": final_state["prompt_introduction"]["style"],
                            "prompt_analysis": final_state["prompt_analysis"]["id"],
                            "prompt_analysis_style": final_state["prompt_analysis"]["style"],
                            "prompt_discussion": final_state["prompt_discussion"]["id"],
                            "prompt_discussion_style": final_state["prompt_discussion"]["style"],
                            "introduction": final_state.get("introduction", ""),
                            "analysis": final_state.get("analysis", ""),
                            "discussion": final_state.get("discussion", ""),
                            "final_review": final_state["final_review"],
                            "metadata": {
                                "topic": topic,
                                "num_papers": len(final_state.get("abstracts", [])),
                                "word_count": len(final_state["final_review"].split()),
                                "branch_execution_time": final_state["experiment_result"].get("cumulative_time", 0),
                                "branch_token_usage": final_state["experiment_result"].get("cumulative_tokens", {}),
                                "breakdown": {
                                    "introduction_time": final_state["experiment_result"]["introduction_time"],
                                    "analysis_time": final_state["experiment_result"]["analysis_time"],
                                    "discussion_time": final_state["experiment_result"]["discussion_time"],
                                    "introduction_tokens": final_state["experiment_result"]["introduction_tokens"],
                                    "analysis_tokens": final_state["experiment_result"]["analysis_tokens"],
                                    "discussion_tokens": final_state["experiment_result"]["discussion_tokens"],
                                }
                            }
                        }
                        all_results.append(result)
        
        total_time = time.time() - project_start_time
        
        print("\n" + "="*80)
        print(f"WORKFLOW COMPLETE - Generated {len(all_results)} literature reviews")
        print(f"TOTAL TIME: {total_time:.2f}s")
        print("="*80)
        
        # Save results
        topic_clean = topic.replace(' ', '_').lower()
        results_base = Path(__file__).parent.parent.parent.parent / "results"
        run_dir = results_base / f"{topic_clean}" / "timetravel"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        output_file = raw_dir / f"{run_id}.json"
        output_data = {
            "run_id": run_id,
            "topic": topic,
            "workflow_type": "time_travel",
            "timestamp": datetime.now().isoformat(),
            "total_branches": len(all_results),
            "timing": {
                "total_project_time": total_time,
                "note": "Uses LangGraph time travel to test all combinations"
            },
            "token_usage_overall": token_usage_overall,
            "phase_1_state": {
                "search_results_count": len(extract_checkpoint.values.get("search_results", [])),
                "abstracts_count": abstracts_count,
                "abstracts": extract_checkpoint.values.get("abstracts", [])
            },
            "branches": all_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVED] Results saved to: {output_file}")
        
        return all_results, str(output_file), run_id


if __name__ == "__main__":
    summarizer = TimeTravelSummarizer()
    
    topic = "large language models"
    all_results, output_file, run_id = summarizer.run_workflow(topic)
    
    print("\n" + "="*80)
    print(f"FINAL RESULTS - {len(all_results)} BRANCHES")
    print("="*80)
    
    for i, result in enumerate(all_results[:5], 1):
        print(f"\n[Branch {i}] {result['branch_id']}")
        print(f"  Styles: {result['prompt_intro_style']} + {result['prompt_analysis_style']} + {result['prompt_discussion_style']}")
        print(f"  Papers: {result['metadata']['num_papers']}")
        print(f"  Time: {result['metadata']['branch_execution_time']:.2f}s")
        print(f"  Tokens: {result['metadata']['branch_token_usage']['total_tokens']:,}")
        print(f"  Words: {result['metadata']['word_count']}")
    
    if len(all_results) > 5:
        print(f"\n... and {len(all_results) - 5} more branches")
    
    print(f"\n[OK] Generated {len(all_results)} different reviews using TIME TRAVEL!")
    print(f"[OK] Results saved to: {output_file}")

