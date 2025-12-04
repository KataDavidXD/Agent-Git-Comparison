"""
Section-based paper summarization with checkpoint branching.

NEW DESIGN (Refactored to use MDP Branching Helper):
- Phase 1: Search + Extract (Unified Base)
- Phase 2: Introduction Generation (Few-shot vs CoT differences start)
- Phase 3: Literature Analysis (Differences amplify)
- Phase 4: Critical Discussion (Differences maximize)
- Phase 5: Concatenation (Assembly only)

Architecture:
1. Uses MDP-style checkpoints (Node granularity) via MDPBranchHelper.
2. Parallel execution for all branching phases.
3. Preserves exact output structure of the original implementation.
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root/"src"))

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Agent_git imports
try:
    from agentgit.database.repositories.user_repository import UserRepository
    from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
    from agentgit.database.repositories.internal_session_repository import InternalSessionRepository
    from agentgit.database.repositories.checkpoint_repository import CheckpointRepository
    from agentgit.sessions.external_session_mdp import ExternalSession_mdp # Use MDP session
    from agentgit.sessions.internal_session_mdp import InternalSession_mdp
    from agentgit.managers.checkpoint_manager_mdp import CheckpointManager_mdp
    from agentgit.managers.tool_manager import ToolManager
    from agentgit.core.workflow import WorkflowGraph
    from agentgit.managers.node_manager import NodeManager
    
    # Import our new Helper
    from utils.mdp_branching import MDPBranchHelper
except ImportError:
    print("⚠️  agent_git not installed or import failed.")
    # For development/IDE support without full package
    pass

# Load environment
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"), temperature=0)

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
    limitations: str   # NEW: Limitations and Future Work
    final_review: str  # Concatenated final review
    token_usage: Dict[str, int]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_prompts(prompt_file: str) -> List[Dict[str, Any]]:
    """Load prompt candidates from a JSON file"""
    prompt_path = Path(__file__).parent.parent / "prompts" / prompt_file

    with open(prompt_path, 'r') as f:
        return json.load(f)

def track_tokens(state: SectionBasedState, response) -> Dict[str, int]:
    """Track token usage from LLM response."""
    token_usage = state.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
        usage = response.response_metadata['token_usage']
        token_usage["prompt_tokens"] += usage.get('prompt_tokens', 0)
        token_usage["completion_tokens"] += usage.get('completion_tokens', 0)
        token_usage["total_tokens"] += usage.get('total_tokens', 0)
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
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            base_url = "http://export.arxiv.org/api/query"
            if date_filter:
                query = f"search_query=all:{topic}+AND+{date_filter}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
            else:
                query = f"search_query=all:{topic}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
            
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
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
    
    return {
        "messages": [AIMessage(content=f"Search failed for '{topic}'")],
        "search_results": []
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
    
    print(f"  [OK] Extracted {len(abstracts)} abstracts")
    
    return {
        "messages": [AIMessage(content=f"Extracted {len(abstracts)} abstracts")],
        "abstracts": abstracts
    }

# ==============================================================================
# NODES - PHASE 2/3/4: SECTION GENERATION
# ==============================================================================

def node_generate_introduction(state: SectionBasedState, prompt_idx: int = 0) -> Dict[str, Any]:
    """Node 3: Generate Introduction section."""
    print(f"[NODE] Generating Introduction with prompt index {prompt_idx}")
    
    prompts = load_prompts("prompts_introduction.json")
    prompt_config = prompts[prompt_idx]
    
    abstracts = state.get("abstracts", [])
    
    abstracts_text = "\n\n".join([
        f"Paper {i+1}: {abs.get('title', 'N/A')}\n"
        f"Authors: {', '.join(abs.get('authors', []))}\n"
        f"Abstract: {abs.get('abstract', 'N/A')}"
        for i, abs in enumerate(abstracts) if 'error' not in abs
    ])
    
    user_prompt = prompt_config["user"].format(abstracts_text=abstracts_text)
    system_message = prompt_config.get("system", "You are a helpful research assistant.")
    
    messages = [HumanMessage(content=f"System: {system_message}\n\nUser: {user_prompt}")]
    response = llm.invoke(messages)
    
    token_usage = track_tokens(state, response)
    
    print(f"  → Prompt: {prompt_config['id']} (Style: {prompt_config['style']})")
    print(f"  → Generated {len(response.content)} chars")
    
    return {
        "messages": [AIMessage(content=f"[{prompt_config['id']}] Introduction generated")],
        "introduction": response.content,
        "token_usage": token_usage
    }

def node_generate_analysis(state: SectionBasedState, prompt_idx: int = 0) -> Dict[str, Any]:
    """Node 4: Generate Literature Analysis section."""
    print(f"[NODE] Generating Analysis with prompt index {prompt_idx}")
    
    prompts = load_prompts("prompts_analysis.json")
    prompt_config = prompts[prompt_idx]
    
    abstracts = state.get("abstracts", [])
    introduction = state.get("introduction", "")
    
    abstracts_text = "\n\n".join([
        f"Paper {i+1}: {abs.get('title', 'N/A')}\n"
        f"Authors: {', '.join(abs.get('authors', []))}\n"
        f"Abstract: {abs.get('abstract', 'N/A')}"
        for i, abs in enumerate(abstracts) if 'error' not in abs
    ])
    
    user_prompt = prompt_config["user"].format(
        abstracts_text=abstracts_text,
        introduction=introduction
    )
    system_message = prompt_config.get("system", "You are a helpful research assistant.")
    
    messages = [HumanMessage(content=f"System: {system_message}\n\nUser: {user_prompt}")]
    response = llm.invoke(messages)
    
    token_usage = track_tokens(state, response)
    
    print(f"  → Prompt: {prompt_config['id']} (Style: {prompt_config['style']})")
    print(f"  → Generated {len(response.content)} chars")
    
    return {
        "messages": [AIMessage(content=f"[{prompt_config['id']}] Analysis generated")],
        "analysis": response.content,
        "token_usage": token_usage
    }

def node_generate_discussion(state: SectionBasedState, prompt_idx: int = 0) -> Dict[str, Any]:
    """Node 5: Generate Critical Discussion section."""
    print(f"[NODE] Generating Discussion with prompt index {prompt_idx}")
    
    prompts = load_prompts("prompts_discussion.json")
    prompt_config = prompts[prompt_idx]
    
    abstracts = state.get("abstracts", [])
    introduction = state.get("introduction", "")
    analysis = state.get("analysis", "")
    
    abstracts_text = "\n\n".join([
        f"Paper {i+1}: {abs.get('title', 'N/A')}\n"
        f"Authors: {', '.join(abs.get('authors', []))}\n"
        f"Abstract: {abs.get('abstract', 'N/A')}"
        for i, abs in enumerate(abstracts) if 'error' not in abs
    ])
    
    user_prompt = prompt_config["user"].format(
        abstracts_text=abstracts_text,
        introduction=introduction,
        analysis=analysis
    )
    system_message = prompt_config.get("system", "You are a helpful research assistant.")
    
    messages = [HumanMessage(content=f"System: {system_message}\n\nUser: {user_prompt}")]
    response = llm.invoke(messages)
    
    token_usage = track_tokens(state, response)
    
    print(f"  → Prompt: {prompt_config['id']} (Style: {prompt_config['style']})")
    print(f"  → Generated {len(response.content)} chars")
    
    return {
        "messages": [AIMessage(content=f"[{prompt_config['id']}] Discussion generated")],
        "discussion": response.content,
        "token_usage": token_usage
    }

# ==============================================================================
# NODES - PHASE 4.5: LIMITATIONS (Experiment 1.2)
# ==============================================================================

def node_generate_limitations(state: SectionBasedState, prompt_idx: int = 0) -> Dict[str, Any]:
    """Node 5.5: Generate Limitations and Future Work section."""
    print(f"[NODE] Generating Limitations with prompt index {prompt_idx}")
    
    prompts = load_prompts("prompts_limitations.json")
    prompt_config = prompts[prompt_idx]
    
    abstracts = state.get("abstracts", [])
    introduction = state.get("introduction", "")
    analysis = state.get("analysis", "")
    discussion = state.get("discussion", "")
    
    abstracts_text = "\n\n".join([
        f"Paper {i+1}: {abs.get('title', 'N/A')}\n"
        f"Authors: {', '.join(abs.get('authors', []))}\n"
        f"Abstract: {abs.get('abstract', 'N/A')}"
        for i, abs in enumerate(abstracts) if 'error' not in abs
    ])
    
    user_prompt = prompt_config["user"].format(
        abstracts_text=abstracts_text,
        introduction=introduction,
        analysis=analysis,
        discussion=discussion
    )
    system_message = prompt_config.get("system", "You are a helpful research assistant.")
    
    messages = [HumanMessage(content=f"System: {system_message}\n\nUser: {user_prompt}")]
    response = llm.invoke(messages)
    
    token_usage = track_tokens(state, response)
    
    print(f"  → Prompt: {prompt_config['id']} (Style: {prompt_config['style']})")
    print(f"  → Generated {len(response.content)} chars")
    
    return {
        "messages": [AIMessage(content=f"[{prompt_config['id']}] Limitations generated")],
        "limitations": response.content,
        "token_usage": token_usage
    }

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
    limitations = state.get("limitations", "") # Added for Exp 1.2
    
    final_review = f"""# Literature Review: {topic}

## 1. Introduction

{introduction}

## 2. Literature Analysis

{analysis}

## 3. Critical Discussion

{discussion}
"""

    if limitations:
        final_review += f"""
## 4. Limitations and Future Work

{limitations}
"""

    final_review += f"""
---
*This review synthesizes findings from {len(state.get('abstracts', []))} recent papers.*
"""
    
    word_count = len(final_review.split())
    print(f"  → Final review: {word_count} words")
    
    return {
        "messages": [AIMessage(content="Final review assembled")],
        "final_review": final_review
    }

# ==============================================================================
# WRAPPER TASKS FOR MDP HELPER
# ==============================================================================

def task_wrapper_generic(state: Dict, context: Dict) -> Dict[str, Any]:
    """Generic wrapper to execute node functions and calculate token diff."""
    func = context["func"]
    kwargs = context.get("kwargs", {})
    
    # Capture baseline for token diff
    token_baseline = state.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}).copy()
    
    # Execute Node Function
    updates = func(state, **kwargs)
    
    # Calculate diff (requires 'token_usage' to be in updates or state)
    new_tokens = updates.get("token_usage", token_baseline)
    
    diff_tokens = {
        "prompt_tokens": new_tokens["prompt_tokens"] - token_baseline["prompt_tokens"],
        "completion_tokens": new_tokens["completion_tokens"] - token_baseline["completion_tokens"],
        "total_tokens": new_tokens["total_tokens"] - token_baseline["total_tokens"]
    }
    
    updates["_tokens"] = diff_tokens
    return updates

# ==============================================================================
# MAIN WORKFLOW WITH MDP BRANCHING
# ==============================================================================

class SectionBasedSummarizer:
    """Section-based paper summarization with checkpoint branching."""

    def __init__(self, db_path: str = str(project_root / "tmp" / "agentgit_exp2.db")):
        self.db_path = db_path

        # Initialize Repos & Helper
        self.user_repo = UserRepository(db_path)
        self.external_session_repo = ExternalSessionRepository(db_path)
        self.internal_session_repo = InternalSessionRepository(db_path)
        self.checkpoint_repo = CheckpointRepository(db_path)
        self.branch_helper = MDPBranchHelper(db_path)

        # Create external session (MDP Style)
        session_name = f"Section-Based Summarization - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        try:
            session = ExternalSession_mdp(
                id=None,
                user_id=1,
                session_name=session_name,
                created_at=datetime.now()
            )
            self.external_session = self.external_session_repo.create(session)
            self.external_session_id = self.external_session.id
        except Exception:
             print("Fallback: Using default ID 1 for external session")
             self.external_session_id = 1

    def _serialize_state(self, state: SectionBasedState) -> Dict[str, Any]:
        """Convert state to JSON-serializable format."""
        serialized = {}
        for key, value in state.items():
            if key == "messages":
                serialized[key] = [
                    {"type": msg.__class__.__name__, "content": msg.content}
                    for msg in value
                ]
            else:
                serialized[key] = value
        return serialized

    def _deserialize_state(self, serialized: Dict[str, Any]) -> SectionBasedState:
        """Convert serialized state back to SectionBasedState."""
        state = serialized.copy()
        if "messages" in state:
            messages = []
            for msg in state["messages"]:
                if msg["type"] == "HumanMessage":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["type"] == "AIMessage":
                    messages.append(AIMessage(content=msg["content"]))
            state["messages"] = messages
        return state

    def run_base_experiment(self, topic: str, run_id: Optional[str] = None):
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + str(uuid.uuid4())
        
        project_start_time = time.time()
        
        print("\n" + "="*80)
        print(f"Section-Based Summarization Workflow: {topic}")
        print(f"Run ID: {run_id}")
        print("="*80)

        # Phase 1: Search → Extract
        print(f"\n[PHASE 1] Running: search → extract")
        initial_state = {
            "messages": [HumanMessage(content=f"Search and extract papers on: {topic}")],
            "topic": topic,
            "date_filter": "submittedDate:[20230101 TO 20231231]",
            "max_results": 10,
            "search_results": [],
            "abstracts": [],
            "introduction": "",
            "analysis": "",
            "discussion": "",
            "limitations": "",
            "final_review": "",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        state = initial_state.copy()
        result = node_search_papers(state)
        state.update(result)
        result = node_extract_abstracts(state)
        state.update(result)
        
        print(f"[OK] Phase 1 complete. Found {len(state.get('abstracts', []))} papers")

        # Create Initial Checkpoint (Node Checkpoint)
        print(f"\n[CHECKPOINT] Creating checkpoint after extract...")
        
        cm = CheckpointManager_mdp(self.checkpoint_repo)
        tm = ToolManager(tools=[])
        
        # Create initial internal session
        int_sess = InternalSession_mdp(
            id=None,
            external_session_id=self.external_session_id,
            langgraph_session_id=f"mdp_main_{uuid.uuid4().hex[:12]}",
            current_node_id="Extract",
            workflow_variables={}
        )
        int_sess.session_state["graph_state"] = self._serialize_state(state)
        
        # Save session
        saved_sess = self.internal_session_repo.create(int_sess)
        
        # Create Node Checkpoint
        cp_extract = cm.create_node_checkpoint(
            internal_session=saved_sess,
            node_id="Extract",
            workflow_version="1.0",
            tool_track_position=tm.get_tool_track_position()
        )
        checkpoint_1_id = cp_extract.id
        print(f"  [OK] Checkpoint 'Extract' created (ID: {checkpoint_1_id})")

        # Phase 2: Branch for Introduction
        phase_2_start = time.time()
        print(f"\n[PHASE 2] Branching for ALL Introduction prompts (PARALLEL)...")
        prompts_intro = load_prompts("prompts_introduction.json")
        
        tasks_p2 = []
        for idx, prompt in enumerate(prompts_intro):
            tasks_p2.append({
                "parent_checkpoint_id": checkpoint_1_id,
                "func": task_wrapper_generic,
                "context": {
                    "func": node_generate_introduction,
                    "kwargs": {"prompt_idx": idx},
                    "prompt": prompt,
                    "idx": idx
                },
                "node_id": f"Intro_{idx}",
                "version": "1.0"
            })
            
        results_p2 = self.branch_helper.execute_branches(
            tasks=tasks_p2,
            external_session_id=self.external_session_id,
            serialize_state_func=self._serialize_state,
            deserialize_state_func=self._deserialize_state
        )
        
        branches_after_intro = []
        for res in results_p2:
            ctx = tasks_p2[res["task_index"]]["context"]
            if res.get("status") == "error":
                print(f"  [ERROR] Branch Intro_{ctx['idx']} failed: {res.get('error')}")
                continue
            
            branch_data = {
                "prompt_intro_id": ctx["prompt"]["id"],
                "prompt_intro_style": ctx["prompt"]["style"],
                "prompt_intro_idx": ctx["idx"],
                "checkpoint_id": res["checkpoint_id"],
                "state": res["state"],
                "branch_tokens": res["updates"].get("_tokens", {}),
                "branch_time": res["execution_time"]
            }
            branches_after_intro.append(branch_data)
            print(f"    → Checkpoint: {res['checkpoint_id']}, Time: {res['execution_time']:.2f}s")

        phase_2_time = time.time() - phase_2_start
        print(f"\n  [PHASE 2 COMPLETE] All {len(branches_after_intro)} branches completed")

        # Phase 3: Analysis
        phase_3_start = time.time()
        print(f"\n[PHASE 3] Branching EACH branch for ALL Analysis prompts (PARALLEL)...")
        prompts_analysis = load_prompts("prompts_analysis.json")
        
        tasks_p3 = []
        for branch_intro in branches_after_intro:
            for idx, prompt in enumerate(prompts_analysis):
                tasks_p3.append({
                    "parent_checkpoint_id": branch_intro["checkpoint_id"],
                    "func": task_wrapper_generic,
                    "context": {
                        "func": node_generate_analysis,
                        "kwargs": {"prompt_idx": idx},
                        "prompt": prompt,
                        "idx": idx,
                        "parent_branch": branch_intro
                    },
                    "node_id": f"Ana_{branch_intro['prompt_intro_idx']}_{idx}",
                    "version": "1.0"
                })
        
        results_p3 = self.branch_helper.execute_branches(
            tasks=tasks_p3,
            external_session_id=self.external_session_id,
            serialize_state_func=self._serialize_state,
            deserialize_state_func=self._deserialize_state
        )
        
        branches_after_analysis = []
        for res in results_p3:
            ctx = tasks_p3[res["task_index"]]["context"]
            if res.get("status") == "error":
                print(f"  [ERROR] Phase 3 task failed: {res.get('error')}")
                continue
                
            branch_tokens = res["updates"].get("_tokens", {})
            parent = ctx["parent_branch"]
            cumulative_tokens = {
                "prompt_tokens": parent["branch_tokens"].get("prompt_tokens", 0) + branch_tokens.get("prompt_tokens", 0),
                "completion_tokens": parent["branch_tokens"].get("completion_tokens", 0) + branch_tokens.get("completion_tokens", 0),
                "total_tokens": parent["branch_tokens"].get("total_tokens", 0) + branch_tokens.get("total_tokens", 0)
            }
            
            branches_after_analysis.append({
                "prompt_intro_id": parent["prompt_intro_id"],
                "prompt_intro_style": parent["prompt_intro_style"],
                "prompt_intro_idx": parent["prompt_intro_idx"],
                "prompt_analysis_id": ctx["prompt"]["id"],
                "prompt_analysis_style": ctx["prompt"]["style"],
                "prompt_analysis_idx": ctx["idx"],
                "checkpoint_id": res["checkpoint_id"],
                "state": res["state"],
                "branch_tokens": branch_tokens,
                "branch_time": res["execution_time"],
                "cumulative_time": parent["branch_time"] + res["execution_time"],
                "cumulative_tokens": cumulative_tokens
            })

        branches_after_analysis.sort(key=lambda x: (x["prompt_intro_idx"], x["prompt_analysis_idx"]))
        phase_3_time = time.time() - phase_3_start
        print(f"\n  [PHASE 3 COMPLETE] All {len(branches_after_analysis)} branches completed")

        # Phase 4: Discussion
        phase_4_start = time.time()
        print(f"\n[PHASE 4] Branching EACH branch for ALL Discussion prompts (PARALLEL)...")
        prompts_discussion = load_prompts("prompts_discussion.json")
        
        tasks_p4 = []
        for branch_ana in branches_after_analysis:
            for idx, prompt in enumerate(prompts_discussion):
                branch_id = f"I{branch_ana['prompt_intro_idx']}_A{branch_ana['prompt_analysis_idx']}_D{idx}"
                
                # Prepare checkpoint metadata for DB resumption
                checkpoint_metadata = {
                    "run_id": run_id,
                    "branch_id": branch_id,
                    "topic": topic,
                    "prompt_intro_id": branch_ana["prompt_intro_id"],
                    "prompt_intro_style": branch_ana["prompt_intro_style"],
                    "prompt_analysis_id": branch_ana["prompt_analysis_id"],
                    "prompt_analysis_style": branch_ana["prompt_analysis_style"],
                    "prompt_discussion_id": prompt["id"],
                    "prompt_discussion_style": prompt["style"],
                    # These will be filled in after execution
                    "branch_execution_time": 0,  # Placeholder
                    "branch_token_usage": {}     # Placeholder
                }
                
                tasks_p4.append({
                    "parent_checkpoint_id": branch_ana["checkpoint_id"],
                    "func": task_wrapper_generic,
                    "context": {
                        "func": node_generate_discussion,
                        "kwargs": {"prompt_idx": idx},
                        "prompt": prompt,
                        "idx": idx,
                        "parent_branch": branch_ana
                    },
                    "node_id": f"Disc_{branch_ana['prompt_intro_idx']}_{branch_ana['prompt_analysis_idx']}_{idx}",
                    "version": "1.0",
                    "checkpoint_metadata": checkpoint_metadata
                })

        results_p4 = self.branch_helper.execute_branches(
            tasks=tasks_p4,
            external_session_id=self.external_session_id,
            serialize_state_func=self._serialize_state,
            deserialize_state_func=self._deserialize_state
        )
        
        all_final_branches = []
        for res in results_p4:
            ctx = tasks_p4[res["task_index"]]["context"]
            if res.get("status") == "error":
                print(f"  [ERROR] Phase 4 task failed: {res.get('error')}")
                continue
                
            branch_tokens = res["updates"].get("_tokens", {})
            parent = ctx["parent_branch"]
            cumulative_tokens = {
                "prompt_tokens": parent["cumulative_tokens"].get("prompt_tokens", 0) + branch_tokens.get("prompt_tokens", 0),
                "completion_tokens": parent["cumulative_tokens"].get("completion_tokens", 0) + branch_tokens.get("completion_tokens", 0),
                "total_tokens": parent["cumulative_tokens"].get("total_tokens", 0) + branch_tokens.get("total_tokens", 0)
            }
            
            # Update checkpoint metadata with actual execution data
            checkpoint_id = res.get("checkpoint_id")
            if checkpoint_id:
                try:
                    # Get the checkpoint and update its metadata
                    checkpoint = self.checkpoint_repo.get_by_id(checkpoint_id)
                    if checkpoint and checkpoint.metadata:
                        checkpoint.metadata["branch_execution_time"] = parent["cumulative_time"] + res["execution_time"]
                        checkpoint.metadata["branch_token_usage"] = cumulative_tokens
                        self.checkpoint_repo.update_checkpoint_metadata(checkpoint_id, checkpoint.metadata)
                except Exception as e:
                    print(f"  [WARNING] Failed to update checkpoint metadata: {e}")
            
            all_final_branches.append({
                "prompt_intro_id": parent["prompt_intro_id"],
                "prompt_intro_style": parent["prompt_intro_style"],
                "prompt_intro_idx": parent["prompt_intro_idx"],
                "prompt_analysis_id": parent["prompt_analysis_id"],
                "prompt_analysis_style": parent["prompt_analysis_style"],
                "prompt_analysis_idx": parent["prompt_analysis_idx"],
                "prompt_discussion_id": ctx["prompt"]["id"],
                "prompt_discussion_style": ctx["prompt"]["style"],
                "prompt_discussion_idx": ctx["idx"],
                "checkpoint_id": res["checkpoint_id"], # IMPORTANT: Included for scalability extension
                "state": res["state"],
                "branch_tokens": branch_tokens,
                "branch_time": res["execution_time"],
                "cumulative_time": parent["cumulative_time"] + res["execution_time"],
                "cumulative_tokens": cumulative_tokens
            })
        
        all_final_branches.sort(key=lambda x: (x["prompt_intro_idx"], x["prompt_analysis_idx"], x["prompt_discussion_idx"]))
        phase_4_time = time.time() - phase_4_start
        print(f"\n  [PHASE 4 COMPLETE] All {len(all_final_branches)} branches completed")

        # Phase 5: Concatenation
        phase_5_start = time.time()
        print(f"\n[PHASE 5] Concatenating sections...")
        
        # Add visualization at end of workflow
        print(f"\n[VISUALIZATION] Workflow Tree Structure:")
        try:
            self.branch_helper.visualize_tree(checkpoint_1_id)
        except Exception as e:
            print(f"Visualization skipped due to error: {e}")
        
        all_results = []
        
        def process_concat(branch):
            start = time.time()
            res = node_concatenate(branch["state"])
            branch["state"].update(res)
            concat_time = time.time() - start
            
            total_time = branch["cumulative_time"] + concat_time
            total_tokens = branch["cumulative_tokens"]
            
            return {
                "branch_id": f"I{branch['prompt_intro_idx']}_A{branch['prompt_analysis_idx']}_D{branch['prompt_discussion_idx']}",
                "prompt_intro": branch["prompt_intro_id"],
                "prompt_intro_style": branch["prompt_intro_style"],
                "prompt_analysis": branch["prompt_analysis_id"],
                "prompt_analysis_style": branch["prompt_analysis_style"],
                "prompt_discussion": branch["prompt_discussion_id"],
                "prompt_discussion_style": branch["prompt_discussion_style"],
                "checkpoint_id_discussion": branch["checkpoint_id"], # Pass checkpoint ID to output
                "introduction": branch["state"].get("introduction", ""),
                "analysis": branch["state"].get("analysis", ""),
                "discussion": branch["state"].get("discussion", ""),
                "final_review": branch["state"]["final_review"],
                "metadata": {
                    "topic": topic,
                    "num_papers": len(branch["state"].get("abstracts", [])),
                    "word_count": len(branch["state"]["final_review"].split()),
                    "branch_execution_time": total_time,
                    "branch_token_usage": total_tokens,
                    "breakdown": {
                        "discussion_time": branch["branch_time"],
                        "concatenation_time": concat_time,
                        "discussion_tokens": branch["branch_tokens"]
                    }
                }
            }

        with ThreadPoolExecutor(max_workers=16) as executor:
             all_results = list(executor.map(process_concat, all_final_branches))
             
        all_results.sort(key=lambda x: (x["prompt_intro"], x["prompt_analysis"], x["prompt_discussion"]))
        phase_5_time = time.time() - phase_5_start
        
        total_project_time = time.time() - project_start_time
        
        print(f"\nTIMING SUMMARY:")
        print(f"  Phase 2: {phase_2_time:.2f}s")
        print(f"  Phase 3: {phase_3_time:.2f}s")
        print(f"  Phase 4: {phase_4_time:.2f}s")
        print(f"  Phase 5: {phase_5_time:.2f}s")
        print(f"  Total: {total_project_time:.2f}s")
        
        # Save results
        topic_clean = topic.replace(' ', '_').lower()
        results_base = Path(__file__).parent.parent.parent / "results"
        run_dir = results_base / f"{topic_clean}" / "sections"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = raw_dir / f"{run_id}.json"
        output_data = {
            "run_id": run_id,
            "topic": topic,
            "workflow_type": "section_based_mdp",
            "timestamp": datetime.now().isoformat(),
            "total_branches": len(all_results),
            "timing": {
                "total_project_time": total_project_time,
                "phase_2_time": phase_2_time,
                "phase_3_time": phase_3_time,
                "phase_4_time": phase_4_time,
                "phase_5_time": phase_5_time
            },
            "phase_1_state": {
                "search_results_count": len(state.get("search_results", [])),
                "abstracts_count": len(state.get("abstracts", [])),
                "abstracts": state.get("abstracts", [])
            },
            "branches": all_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        return all_results, str(output_file), run_id

    def _query_checkpoints_from_db(self, run_id_pattern: str) -> List[Dict[str, Any]]:
        """
        Query Discussion phase checkpoints from database based on run_id pattern.
        
        Args:
            run_id_pattern: Pattern to match against checkpoint metadata (e.g., "20251127_123456")
        
        Returns:
            List of checkpoint info dictionaries with checkpoint_id and metadata
        """
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query all checkpoints with node_id starting with "Disc_" (Discussion phase)
        query = """
        SELECT id, checkpoint_data
        FROM checkpoints
        WHERE checkpoint_data LIKE '%"node_id": "Disc_%'
        ORDER BY id
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        checkpoints = []
        for row in rows:
            checkpoint_id = row[0]
            checkpoint_data = json.loads(row[1])
            metadata = checkpoint_data.get("metadata", {})
            
            # Filter by run_id if present in metadata
            if "run_id" in metadata and run_id_pattern in metadata["run_id"]:
                checkpoints.append({
                    "checkpoint_id": checkpoint_id,
                    "node_id": metadata.get("node_id", ""),
                    "metadata": metadata,
                    "state": checkpoint_data.get("state", {})
                })
        
        return checkpoints

    def run_experiment_1_2_addon(self, prev_run_file: str = None, base_run_id: str = None, run_id_suffix: str = "_exp1_2"):
        """
        Experiment 1.2: Add-on Scalability.
        Resumes from Phase 4 checkpoints of a previous run, adds Limitations, and concatenates.
        
        Args:
            prev_run_file: Path to JSON file from previous run (optional)
            base_run_id: Run ID pattern to query from database (optional, e.g., "20251127_123456")
            run_id_suffix: Suffix to append to run_id
        
        Note: Either prev_run_file OR base_run_id must be provided.
        """
        project_start_time = time.time()
        
        if not prev_run_file and not base_run_id:
            raise ValueError("Either prev_run_file or base_run_id must be provided.")
        
        # 1. Load Previous Run Data (from JSON or DB)
        if prev_run_file:
            print(f"\n[LOADING] Reading from JSON file: {prev_run_file}")
            with open(prev_run_file, 'r', encoding='utf-8') as f:
                prev_data = json.load(f)
                
            topic = prev_data["topic"]
            branches = prev_data["branches"]
            base_run_id = prev_data["run_id"]
            run_id = base_run_id + run_id_suffix
            
            # Extract checkpoint IDs from JSON
            resume_points = []
            for b in branches:
                if "checkpoint_id_discussion" in b:
                    resume_points.append({
                        "checkpoint_id": b["checkpoint_id_discussion"],
                        "branch_info": b
                    })
                else:
                    print(f"Warning: Branch {b.get('branch_id')} missing checkpoint info. Skipping.")
                    
        else:  # base_run_id provided
            print(f"\n[LOADING] Querying database for run_id: {base_run_id}")
            checkpoints = self._query_checkpoints_from_db(base_run_id)
            
            if not checkpoints:
                raise ValueError(f"No Discussion checkpoints found in database for run_id: {base_run_id}")
            
            print(f"Found {len(checkpoints)} Discussion checkpoints in database.")
            
            # Reconstruct branch info from checkpoint metadata
            resume_points = []
            for cp in checkpoints:
                meta = cp["metadata"]
                resume_points.append({
                    "checkpoint_id": cp["checkpoint_id"],
                    "branch_info": {
                        "branch_id": meta.get("branch_id", "unknown"),
                        "prompt_intro_id": meta.get("prompt_intro_id", ""),
                        "prompt_intro_style": meta.get("prompt_intro_style", ""),
                        "prompt_analysis_id": meta.get("prompt_analysis_id", ""),
                        "prompt_analysis_style": meta.get("prompt_analysis_style", ""),
                        "prompt_discussion_id": meta.get("prompt_discussion_id", ""),
                        "prompt_discussion_style": meta.get("prompt_discussion_style", ""),
                        "metadata": {
                            "topic": meta.get("topic", ""),
                            "branch_execution_time": meta.get("branch_execution_time", 0),
                            "branch_token_usage": meta.get("branch_token_usage", {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            })
                        }
                    }
                })
            
            # Extract topic and construct run_id
            topic = checkpoints[0]["metadata"].get("topic", "unknown_topic")
            run_id = base_run_id + run_id_suffix
        
        print("\n" + "="*80)
        print(f"Experiment 1.2: Add-on Scalability Evaluation")
        print(f"Base Run ID: {base_run_id if base_run_id else prev_data['run_id']}")
        print(f"New Run ID: {run_id}")
        print(f"Topic: {topic}")
        print("="*80)
        
        print(f"\nFound {len(resume_points)} branches to extend.")
        
        # 3. Branch for Limitations (Phase 4.5)
        phase_4_5_start = time.time()
        print(f"\n[PHASE 4.5] Branching EACH branch for ALL Limitations prompts (PARALLEL)...")
        prompts_limitations = load_prompts("prompts_limitations.json")
        
        tasks_p4_5 = []
        for point in resume_points:
            cp_id = point["checkpoint_id"]
            b_info = point["branch_info"]
            
            for idx, prompt in enumerate(prompts_limitations):
                tasks_p4_5.append({
                    "parent_checkpoint_id": cp_id,
                    "func": task_wrapper_generic,
                    "context": {
                        "func": node_generate_limitations,
                        "kwargs": {"prompt_idx": idx},
                        "prompt": prompt,
                        "idx": idx,
                        "parent_branch_info": b_info
                    },
                    "node_id": f"Lim_{b_info['branch_id']}_{idx}",
                    "version": "1.2"
                })
                
        results_p4_5 = self.branch_helper.execute_branches(
            tasks=tasks_p4_5,
            external_session_id=self.external_session_id,
            serialize_state_func=self._serialize_state,
            deserialize_state_func=self._deserialize_state
        )
        
        final_branches = []
        for res in results_p4_5:
            ctx = tasks_p4_5[res["task_index"]]["context"]
            if res.get("status") == "error":
                print(f"  [ERROR] Limitation task failed: {res.get('error')}")
                continue
                
            branch_tokens = res["updates"].get("_tokens", {})
            parent_info = ctx["parent_branch_info"]
            parent_tokens = parent_info["metadata"]["branch_token_usage"]
            
            cumulative_tokens = {
                "prompt_tokens": parent_tokens.get("prompt_tokens", 0) + branch_tokens.get("prompt_tokens", 0),
                "completion_tokens": parent_tokens.get("completion_tokens", 0) + branch_tokens.get("completion_tokens", 0),
                "total_tokens": parent_tokens.get("total_tokens", 0) + branch_tokens.get("total_tokens", 0)
            }
            
            print(f"    → Prompt: {ctx['prompt']['id']}, Time: {res['execution_time']:.2f}s, Tokens: {branch_tokens.get('total_tokens', 0)}")
            
            final_branches.append({
                "branch_id": f"{parent_info['branch_id']}_L{ctx['idx']}",
                "parent_info": parent_info,
                "prompt_limitations_id": ctx["prompt"]["id"],
                "prompt_limitations_style": ctx["prompt"]["style"],
                "state": res["state"],
                "branch_tokens": branch_tokens,
                "branch_time": res["execution_time"],
                "cumulative_time": parent_info["metadata"]["branch_execution_time"] + res["execution_time"],
                "cumulative_tokens": cumulative_tokens
            })
            
        phase_4_5_time = time.time() - phase_4_5_start
        print(f"\n  [PHASE 4.5 COMPLETE] All {len(final_branches)} branches completed")
        
        # 4. Phase 5: Concatenation (Re-run with limitations)
        phase_5_start = time.time()
        print(f"\n[PHASE 5] Concatenating sections (New)...")
        
        all_results = []
        
        def process_concat_new(branch):
            start = time.time()
            res = node_concatenate(branch["state"])
            branch["state"].update(res)
            concat_time = time.time() - start
            
            total_time = branch["cumulative_time"] + concat_time
            
            return {
                "branch_id": branch["branch_id"],
                "parent_branch_id": branch["parent_info"]["branch_id"],
                "styles": {
                    "intro": branch["parent_info"]["prompt_intro_style"],
                    "analysis": branch["parent_info"]["prompt_analysis_style"],
                    "discussion": branch["parent_info"]["prompt_discussion_style"],
                    "limitations": branch["prompt_limitations_style"]
                },
                "final_review": branch["state"]["final_review"],
                "metadata": {
                    "topic": topic,
                    "word_count": len(branch["state"]["final_review"].split()),
                    "total_time": total_time,
                    "token_usage": branch["cumulative_tokens"],
                    "breakdown": {
                        "base_execution_time": branch["parent_info"]["metadata"]["branch_execution_time"],
                        "base_token_usage": branch["parent_info"]["metadata"]["branch_token_usage"],
                        "limitations_time": branch["branch_time"],
                        "limitations_tokens": branch["branch_tokens"],
                        "concatenation_time": concat_time
                    }
                }
            }
            
        with ThreadPoolExecutor(max_workers=16) as executor:
             all_results = list(executor.map(process_concat_new, final_branches))
             
        phase_5_time = time.time() - phase_5_start
        total_project_time = time.time() - project_start_time
        
        # Calculate total token usage for extension
        total_limitations_tokens = {
            "prompt_tokens": sum(b["branch_tokens"].get("prompt_tokens", 0) for b in final_branches),
            "completion_tokens": sum(b["branch_tokens"].get("completion_tokens", 0) for b in final_branches),
            "total_tokens": sum(b["branch_tokens"].get("total_tokens", 0) for b in final_branches)
        }
        
        print(f"\nTIMING SUMMARY (Extension):")
        print(f"  Phase 4.5 (Limitations): {phase_4_5_time:.2f}s")
        print(f"  Phase 5 (Concatenation): {phase_5_time:.2f}s")
        print(f"  Total Extension Time: {total_project_time:.2f}s")
        print(f"\nTOKEN USAGE (Limitations Phase):")
        print(f"  Total Tokens: {total_limitations_tokens['total_tokens']:,}")
        print(f"  Prompt Tokens: {total_limitations_tokens['prompt_tokens']:,}")
        print(f"  Completion Tokens: {total_limitations_tokens['completion_tokens']:,}")
        
        # Save results
        topic_clean = topic.replace(' ', '_').lower()
        results_base = Path(__file__).parent.parent.parent / "results"
        run_dir = results_base / f"{topic_clean}" / "sections"
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = raw_dir / f"{run_id}.json"
        output_data = {
            "run_id": run_id,
            "base_run_id": base_run_id,
            "topic": topic,
            "experiment": "1.2_scalability_extension",
            "timestamp": datetime.now().isoformat(),
            "total_branches": len(all_results),
            "timing": {
                "total_project_time": total_project_time,
                "phase_4_5_time": phase_4_5_time,
                "phase_5_time": phase_5_time
            },
            "token_usage_extension": total_limitations_tokens,
            "phase_1_state": {
                "search_results_count": len(final_branches[0]["state"].get("search_results", [])),
                "abstracts_count": len(final_branches[0]["state"].get("abstracts", [])),
                "abstracts": final_branches[0]["state"].get("abstracts", [])
            },
            "branches": all_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print(f"\n[SAVED] Extension results saved to: {output_file}")
        return all_results, str(output_file)

    def run_workflow(self, topic: str):
        """
        Run the full section-based summarization workflow.
        
        Args:
            topic: Topic to search and summarize
        Returns:
            Tuple of (results list, output file path, run_id)
        """
        base_result = self.run_base_experiment(topic=topic)
        addon_result = self.run_experiment_1_2_addon(base_run_id=base_result[2])
        return addon_result

if __name__ == "__main__":
    summarizer = SectionBasedSummarizer()
    topic = "large language models"
    
    # Experiment 1.1: Standard run
    results, output_file, run_id = summarizer.run_workflow(topic)
    
    # Experiment 1.2: Add-on scalability (two options)
    # Option 1: Resume from JSON file
    # summarizer.run_experiment_1_2_addon(prev_run_file=output_file)
    
    # Option 2: Resume from database by run_id (recommended for clean DB workflow)
    # summarizer.run_experiment_1_2_addon(base_run_id=run_id)
