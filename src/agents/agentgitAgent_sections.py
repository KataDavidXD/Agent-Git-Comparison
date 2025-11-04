"""
Section-based paper summarization with checkpoint branching.

NEW DESIGN:
- Phase 1: Search + Extract (统一基础)
- Phase 2: Introduction Generation (Few-shot vs CoT 差异开始)
- Phase 3: Literature Analysis (差异放大)
- Phase 4: Critical Discussion (差异最大化)
- Phase 5: Concatenation (仅拼接，不重新生成)

Architecture:
1. Each phase generates FINAL CONTENT (not intermediate analysis)
2. Few-shot and CoT differences accumulate across sections
3. Final concatenation preserves section-level differences
"""

import sys
import json
import uuid
import time
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
sys.path.insert(0, str(project_root))

# LangChain/LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Agent_git imports
try:
    from agentgit.agents.rollback_agent import RollbackAgent
    from agentgit.database.repositories.user_repository import UserRepository
    from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
    from agentgit.database.repositories.internal_session_repository import InternalSessionRepository
    from agentgit.database.repositories.checkpoint_repository import CheckpointRepository
    from agentgit.sessions.external_session import ExternalSession
except ImportError:
    print("⚠️  agent_git not installed. Install with: pip install agent_git_langchain-0.1.0-py3-none-any.whl")
    sys.exit(1)

# Load environment
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)

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
    token_usage: Dict[str, int]

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
    date_filter = state.get("date_filter", "")  # Optional: e.g., "submittedDate:[20230101 TO 20231231]"
    max_results = state.get("max_results", 50)
    
    print(f"[NODE] Searching arXiv for: {topic} (max_results={max_results})")
    
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

def node_generate_introduction(state: SectionBasedState, prompt_idx: int = 0) -> Dict[str, Any]:
    """Node 3: Generate Introduction section."""
    print(f"[NODE] Generating Introduction with prompt index {prompt_idx}")
    
    prompts = load_prompts("prompts_introduction.json")
    prompt_config = prompts[prompt_idx]
    
    abstracts = state.get("abstracts", [])
    topic = state.get("topic", "Unknown")
    
    # Prepare abstracts text
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
    
    token_usage = track_tokens(response)
    
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
    
    # Prepare abstracts text
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
    
    token_usage = track_tokens(response)
    
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
    
    # Prepare abstracts text
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
    
    token_usage = track_tokens(response)
    
    print(f"  → Prompt: {prompt_config['id']} (Style: {prompt_config['style']})")
    print(f"  → Generated {len(response.content)} chars")
    
    return {
        "messages": [AIMessage(content=f"[{prompt_config['id']}] Discussion generated")],
        "discussion": response.content,
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
    
    # Simple concatenation with section headers and transitions
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
    print(f"  → Final review: {word_count} words")
    
    return {
        "messages": [AIMessage(content="Final review assembled")],
        "final_review": final_review
    }

# ==============================================================================
# MAIN WORKFLOW WITH CHECKPOINT BRANCHING
# ==============================================================================

class SectionBasedSummarizer:
    """Section-based paper summarization with checkpoint branching."""

    def __init__(self, db_path: str = "./tmp/agentgit.db"):
        self.db_path = db_path

        # Initialize repositories
        self.user_repo = UserRepository(db_path)
        self.external_session_repo = ExternalSessionRepository(db_path)
        self.internal_session_repo = InternalSessionRepository(db_path)
        self.checkpoint_repo = CheckpointRepository(db_path)

        # Create external session
        session_name = f"Section-Based Summarization - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        session = ExternalSession(
            user_id=1,
            session_name=session_name,
            created_at=datetime.now()
        )
        self.external_session = self.external_session_repo.create(session)
        self.external_session_id = self.external_session.id

        self.model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"), temperature=0)
        self.agent: Optional[RollbackAgent] = None

    def run_workflow(self, topic: str, run_id: Optional[str] = None):
        """
        Run section-based workflow with checkpoint branching.
        
        Flow:
        1. Phase 1: Search → Extract → [CHECKPOINT]
        2. Phase 2: Branch for ALL Introduction prompts (CoT vs Few-shot) → [CHECKPOINT each]
        3. Phase 3: For each Intro branch, branch for ALL Analysis prompts → [CHECKPOINT each]
        4. Phase 4: For each Analysis branch, branch for ALL Discussion prompts
        5. Phase 5: For each final branch, concatenate sections (NO regeneration)
        6. Return ALL results with metrics
        """
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + str(uuid.uuid4())[:8]
        
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
            "date_filter": "submittedDate:[20230101 TO 20231231]",  # Optional: e.g., "submittedDate:[20230101 TO 20231231]"
            "max_results": 10,  # Retrieve 50 papers
            "search_results": [],
            "abstracts": [],
            "introduction": "",
            "analysis": "",
            "discussion": "",
            "final_review": "",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        state = initial_state.copy()
        result = node_search_papers(state)
        state.update(result)
        result = node_extract_abstracts(state)
        state.update(result)
        
        print(f"[OK] Phase 1 complete. Found {len(state.get('abstracts', []))} papers")

        # Create RollbackAgent and checkpoint
        print(f"\n[CHECKPOINT] Creating checkpoint after extract...")
        self.agent = RollbackAgent(
            external_session_id=self.external_session_id,
            model=self.model,
            tools=[],
            auto_checkpoint=False,
            internal_session_repo=self.internal_session_repo,
            checkpoint_repo=self.checkpoint_repo
        )

        self.agent.internal_session.session_state["graph_state"] = self._serialize_state(state)
        self.agent._save_internal_session()
        checkpoint_1_id = self._create_checkpoint("after_extract", state)

        # Phase 2: Branch for Introduction prompts (CoT vs Few-shot)
        phase_2_start = time.time()
        print(f"\n[PHASE 2] Branching for ALL Introduction prompts (PARALLEL)...")
        prompts_intro = load_prompts("prompts_introduction.json")
        
        def process_intro_branch(idx, prompt):
            branch_start_time = time.time()
            print(f"\n  [Branch Intro_{idx+1}] Testing: {prompt['id']} ({prompt['style']})")
            
            branch_agent = RollbackAgent.from_checkpoint(
                checkpoint_id=checkpoint_1_id,
                external_session_id=self.external_session_id,
                model=self.model,
                checkpoint_repo=self.checkpoint_repo,
                internal_session_repo=self.internal_session_repo,
                tools=[],
                auto_checkpoint=False
            )
            
            restored_state = self._restore_state_from_checkpoint(checkpoint_1_id)
            
            result = node_generate_introduction(restored_state, prompt_idx=idx)
            restored_state.update(result)
            
            branch_tokens = restored_state.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}).copy()
            
            branch_execution_time = time.time() - branch_start_time
            
            branch_agent.internal_session.session_state["graph_state"] = self._serialize_state(restored_state)
            branch_agent._save_internal_session()
            branch_agent.create_checkpoint_tool(name=f"after_intro_{idx}")
            
            branch_checkpoints = self.checkpoint_repo.get_by_internal_session(
                branch_agent.internal_session.id,
                auto_only=False
            )
            branch_checkpoint_id = None
            for cp in branch_checkpoints:
                if cp.checkpoint_name == f"after_intro_{idx}":
                    branch_checkpoint_id = cp.id
                    break
            
            print(f"    → Checkpoint: {branch_checkpoint_id}, Time: {branch_execution_time:.2f}s, Tokens: {branch_tokens['total_tokens']}")
            
            return {
                "prompt_intro_id": prompt["id"],
                "prompt_intro_style": prompt["style"],
                "prompt_intro_idx": idx,
                "checkpoint_id": branch_checkpoint_id,
                "state": restored_state,
                "branch_agent": branch_agent,
                "branch_tokens": branch_tokens,
                "branch_time": branch_execution_time
            }
        
        branches_after_intro = []
        with ThreadPoolExecutor(max_workers=len(prompts_intro)) as executor:
            futures = {executor.submit(process_intro_branch, idx, prompt): idx for idx, prompt in enumerate(prompts_intro)}
            for future in as_completed(futures):
                branches_after_intro.append(future.result())
        
        branches_after_intro.sort(key=lambda x: x["prompt_intro_idx"])
        phase_2_time = time.time() - phase_2_start
        print(f"\n  [PHASE 2 COMPLETE] All {len(branches_after_intro)} branches completed in {phase_2_time:.2f}s (parallel)")

        # Phase 3: For each Intro branch, branch for Analysis prompts
        phase_3_start = time.time()
        print(f"\n[PHASE 3] Branching EACH branch for ALL Analysis prompts (PARALLEL)...")
        prompts_analysis = load_prompts("prompts_analysis.json")
        
        def process_analysis_branch(branch_intro, idx, prompt):
            branch_start_time = time.time()
            print(f"    [Branch Intro_{branch_intro['prompt_intro_idx']+1}_Ana_{idx+1}] Testing: {prompt['id']} ({prompt['style']})")
            
            branch_agent = RollbackAgent.from_checkpoint(
                checkpoint_id=branch_intro["checkpoint_id"],
                external_session_id=self.external_session_id,
                model=self.model,
                checkpoint_repo=self.checkpoint_repo,
                internal_session_repo=self.internal_session_repo,
                tools=[],
                auto_checkpoint=False
            )
            
            restored_state = self._restore_state_from_checkpoint(branch_intro["checkpoint_id"])
            
            result = node_generate_analysis(restored_state, prompt_idx=idx)
            restored_state.update(result)
            
            branch_tokens = restored_state.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}).copy()
            
            branch_execution_time = time.time() - branch_start_time
            
            branch_agent.internal_session.session_state["graph_state"] = self._serialize_state(restored_state)
            branch_agent._save_internal_session()
            branch_agent.create_checkpoint_tool(name=f"after_ana_{branch_intro['prompt_intro_idx']}_{idx}")
            
            branch_checkpoints = self.checkpoint_repo.get_by_internal_session(
                branch_agent.internal_session.id,
                auto_only=False
            )
            branch_checkpoint_id = None
            for cp in branch_checkpoints:
                if cp.checkpoint_name == f"after_ana_{branch_intro['prompt_intro_idx']}_{idx}":
                    branch_checkpoint_id = cp.id
                    break
            
            print(f"      → Checkpoint: {branch_checkpoint_id}, Time: {branch_execution_time:.2f}s, Tokens: {branch_tokens['total_tokens']}")
            
            return {
                "prompt_intro_id": branch_intro["prompt_intro_id"],
                "prompt_intro_style": branch_intro["prompt_intro_style"],
                "prompt_intro_idx": branch_intro["prompt_intro_idx"],
                "prompt_analysis_id": prompt["id"],
                "prompt_analysis_style": prompt["style"],
                "prompt_analysis_idx": idx,
                "checkpoint_id": branch_checkpoint_id,
                "state": restored_state,
                "branch_agent": branch_agent,
                "branch_tokens": branch_tokens,
                "branch_time": branch_execution_time,
                "cumulative_time": branch_intro["branch_time"] + branch_execution_time,
                "cumulative_tokens": {
                    "prompt_tokens": branch_intro["branch_tokens"]["prompt_tokens"] + branch_tokens["prompt_tokens"],
                    "completion_tokens": branch_intro["branch_tokens"]["completion_tokens"] + branch_tokens["completion_tokens"],
                    "total_tokens": branch_intro["branch_tokens"]["total_tokens"] + branch_tokens["total_tokens"]
                }
            }
        
        branches_after_analysis = []
        tasks = [(branch_intro, idx, prompt) for branch_intro in branches_after_intro for idx, prompt in enumerate(prompts_analysis)]
        
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {executor.submit(process_analysis_branch, *task): task for task in tasks}
            for future in as_completed(futures):
                branches_after_analysis.append(future.result())
        
        branches_after_analysis.sort(key=lambda x: (x["prompt_intro_idx"], x["prompt_analysis_idx"]))
        phase_3_time = time.time() - phase_3_start
        print(f"\n  [PHASE 3 COMPLETE] All {len(branches_after_analysis)} branches completed in {phase_3_time:.2f}s (parallel)")

        # Phase 4: For each Analysis branch, branch for Discussion prompts
        phase_4_start = time.time()
        print(f"\n[PHASE 4] Branching EACH branch for ALL Discussion prompts (PARALLEL)...")
        prompts_discussion = load_prompts("prompts_discussion.json")
        
        def process_discussion_branch(branch_ana, idx, prompt):
            branch_start_time = time.time()
            print(f"      [Branch I_{branch_ana['prompt_intro_idx']+1}_A_{branch_ana['prompt_analysis_idx']+1}_D_{idx+1}] Testing: {prompt['id']} ({prompt['style']})")
            
            branch_agent = RollbackAgent.from_checkpoint(
                checkpoint_id=branch_ana["checkpoint_id"],
                external_session_id=self.external_session_id,
                model=self.model,
                checkpoint_repo=self.checkpoint_repo,
                internal_session_repo=self.internal_session_repo,
                tools=[],
                auto_checkpoint=False
            )
            
            restored_state = self._restore_state_from_checkpoint(branch_ana["checkpoint_id"])
            
            result = node_generate_discussion(restored_state, prompt_idx=idx)
            restored_state.update(result)
            
            branch_tokens = restored_state.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}).copy()
            
            branch_execution_time = time.time() - branch_start_time
            
            print(f"        → Time: {branch_execution_time:.2f}s, Tokens: {branch_tokens['total_tokens']}")
            
            return {
                "prompt_intro_id": branch_ana["prompt_intro_id"],
                "prompt_intro_style": branch_ana["prompt_intro_style"],
                "prompt_intro_idx": branch_ana["prompt_intro_idx"],
                "prompt_analysis_id": branch_ana["prompt_analysis_id"],
                "prompt_analysis_style": branch_ana["prompt_analysis_style"],
                "prompt_analysis_idx": branch_ana["prompt_analysis_idx"],
                "prompt_discussion_id": prompt["id"],
                "prompt_discussion_style": prompt["style"],
                "prompt_discussion_idx": idx,
                "state": restored_state,
                "branch_agent": branch_agent,
                "branch_tokens": branch_tokens,
                "branch_time": branch_execution_time,
                "cumulative_time": branch_ana["cumulative_time"] + branch_execution_time,
                "cumulative_tokens": {
                    "prompt_tokens": branch_ana["cumulative_tokens"]["prompt_tokens"] + branch_tokens["prompt_tokens"],
                    "completion_tokens": branch_ana["cumulative_tokens"]["completion_tokens"] + branch_tokens["completion_tokens"],
                    "total_tokens": branch_ana["cumulative_tokens"]["total_tokens"] + branch_tokens["total_tokens"]
                }
            }
        
        all_final_branches = []
        tasks = [(branch_ana, idx, prompt) for branch_ana in branches_after_analysis for idx, prompt in enumerate(prompts_discussion)]
        
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {executor.submit(process_discussion_branch, *task): task for task in tasks}
            for future in as_completed(futures):
                all_final_branches.append(future.result())
        
        all_final_branches.sort(key=lambda x: (x["prompt_intro_idx"], x["prompt_analysis_idx"], x["prompt_discussion_idx"]))
        phase_4_time = time.time() - phase_4_start
        print(f"\n  [PHASE 4 COMPLETE] All {len(all_final_branches)} branches completed in {phase_4_time:.2f}s (parallel)")

        # Phase 5: Concatenate sections for all branches
        phase_5_start = time.time()
        print(f"\n[PHASE 5] Concatenating sections for ALL {len(all_final_branches)} branches (PARALLEL)...")
        
        def process_concatenation(i, branch):
            concat_start_time = time.time()
            print(f"\n  [Concat {i+1}/{len(all_final_branches)}] Branch: {branch['prompt_intro_style']} + {branch['prompt_analysis_style']} + {branch['prompt_discussion_style']}")
            
            token_baseline = branch["state"].get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}).copy()
            
            final_result = node_concatenate(branch["state"])
            branch["state"].update(final_result)
            
            # Concatenation doesn't use tokens (just string ops)
            concat_tokens = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            
            concat_execution_time = time.time() - concat_start_time
            
            total_time = branch["cumulative_time"] + concat_execution_time
            total_tokens = branch["cumulative_tokens"]  # No tokens added in concatenation
            
            print(f"    → Final review: {len(branch['state']['final_review'].split())} words, Time: {concat_execution_time:.2f}s")
            
            return {
                "branch_id": f"I{branch['prompt_intro_idx']}_A{branch['prompt_analysis_idx']}_D{branch['prompt_discussion_idx']}",
                "prompt_intro": branch["prompt_intro_id"],
                "prompt_intro_style": branch["prompt_intro_style"],
                "prompt_analysis": branch["prompt_analysis_id"],
                "prompt_analysis_style": branch["prompt_analysis_style"],
                "prompt_discussion": branch["prompt_discussion_id"],
                "prompt_discussion_style": branch["prompt_discussion_style"],
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
                        "introduction_time": branches_after_intro[branch["prompt_intro_idx"]]["branch_time"],
                        "analysis_time": branches_after_analysis[branch["prompt_intro_idx"] * len(prompts_analysis) + branch["prompt_analysis_idx"]]["branch_time"],
                        "discussion_time": branch["branch_time"],
                        "concatenation_time": concat_execution_time,
                        "introduction_tokens": branches_after_intro[branch["prompt_intro_idx"]]["branch_tokens"],
                        "analysis_tokens": branches_after_analysis[branch["prompt_intro_idx"] * len(prompts_analysis) + branch["prompt_analysis_idx"]]["branch_tokens"],
                        "discussion_tokens": branch["branch_tokens"],
                        "concatenation_tokens": concat_tokens
                    }
                }
            }
        
        all_results = []
        with ThreadPoolExecutor(max_workers=len(all_final_branches)) as executor:
            futures = {executor.submit(process_concatenation, i, branch): i for i, branch in enumerate(all_final_branches)}
            for future in as_completed(futures):
                all_results.append(future.result())
        
        all_results.sort(key=lambda x: (x["prompt_intro"], x["prompt_analysis"], x["prompt_discussion"]))
        phase_5_time = time.time() - phase_5_start
        print(f"\n  [PHASE 5 COMPLETE] All {len(all_results)} reviews completed in {phase_5_time:.2f}s (parallel)")

        total_project_time = time.time() - project_start_time
        
        # Calculate overall token usage
        token_usage_overall = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
                        
        for branch_intro in branches_after_intro:
            token_usage_overall["prompt_tokens"] += branch_intro["branch_tokens"]["prompt_tokens"]
            token_usage_overall["completion_tokens"] += branch_intro["branch_tokens"]["completion_tokens"]
            token_usage_overall["total_tokens"] += branch_intro["branch_tokens"]["total_tokens"]
        for branch_ana in branches_after_analysis:
            token_usage_overall["prompt_tokens"] += branch_ana["branch_tokens"]["prompt_tokens"]
            token_usage_overall["completion_tokens"] += branch_ana["branch_tokens"]["completion_tokens"]
            token_usage_overall["total_tokens"] += branch_ana["branch_tokens"]["total_tokens"]
        for branch_disc in all_final_branches:
            token_usage_overall["prompt_tokens"] += branch_disc["branch_tokens"]["prompt_tokens"]
            token_usage_overall["completion_tokens"] += branch_disc["branch_tokens"]["completion_tokens"]
            token_usage_overall["total_tokens"] += branch_disc["branch_tokens"]["total_tokens"]

        print("\n" + "="*80)
        print(f"WORKFLOW COMPLETE - Generated {len(all_results)} literature reviews")
        print("="*80)
        print(f"\nTIMING SUMMARY:")
        print(f"  Phase 1 (Search + Extract): Baseline")
        print(f"  Phase 2 (Introduction): {phase_2_time:.2f}s (parallel, {len(branches_after_intro)} branches)")
        print(f"  Phase 3 (Analysis): {phase_3_time:.2f}s (parallel, {len(branches_after_analysis)} branches)")
        print(f"  Phase 4 (Discussion): {phase_4_time:.2f}s (parallel, {len(all_final_branches)} branches)")
        print(f"  Phase 5 (Concatenation): {phase_5_time:.2f}s (parallel, {len(all_results)} reviews)")
        print(f"  TOTAL PROJECT TIME: {total_project_time:.2f}s")
        print("="*80)
        print(f"\nTOKEN USAGE SUMMARY:")
        print(f"  Prompt Tokens: {token_usage_overall['prompt_tokens']:,}")
        print(f"  Completion Tokens: {token_usage_overall['completion_tokens']:,}")
        print(f"  Total Tokens: {token_usage_overall['total_tokens']:,}")

        # Save results
        topic_clean = topic.replace(' ', '_').lower()
        results_base = Path(__file__).parent.parent.parent.parent / "results"
        run_dir = results_base / f"{topic_clean}" / "sections"
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
                "total_project_time": total_project_time,
                "phase_2_time": phase_2_time,
                "phase_2_branches": len(branches_after_intro),
                "phase_3_time": phase_3_time,
                "phase_3_branches": len(branches_after_analysis),
                "phase_4_time": phase_4_time,
                "phase_4_branches": len(all_final_branches),
                "phase_5_time": phase_5_time,
                "note": "Phases 2-5 run branches in parallel. Sections generated independently."
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

    def _create_checkpoint(self, name: str, state: SectionBasedState) -> int:
        """Create checkpoint and return ID."""
        result = self.agent.create_checkpoint_tool(name=name)
        
        checkpoints = self.checkpoint_repo.get_by_internal_session(
            self.agent.internal_session.id,
            auto_only=False
        )
        for cp in checkpoints:
            if cp.checkpoint_name == name:
                print(f"  [OK] Checkpoint '{name}' created (ID: {cp.id})")
                return cp.id
        
        raise RuntimeError(f"Failed to create checkpoint '{name}'")

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

    def _restore_state_from_checkpoint(self, checkpoint_id: int) -> SectionBasedState:
        """Restore graph state from checkpoint."""
        checkpoint = self.checkpoint_repo.get_by_id(checkpoint_id)
        if not checkpoint:
            raise RuntimeError(f"Checkpoint {checkpoint_id} not found")
        
        internal_session = self.internal_session_repo.get_by_id(checkpoint.internal_session_id)
        if not internal_session or "graph_state" not in internal_session.session_state:
            raise RuntimeError("No graph_state in checkpoint")
        
        serialized_state = internal_session.session_state["graph_state"]
        return self._deserialize_state(serialized_state)


if __name__ == "__main__":
    summarizer = SectionBasedSummarizer()

    topic = "large language models"
    all_results, output_file, run_id = summarizer.run_workflow(topic)
    
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

