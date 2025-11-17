import itertools
import os
import time
import json
import uuid
import operator
import requests
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, List, Dict, Any, Mapping, Sequence

from langchain_openai import ChatOpenAI

try:
    from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
    from autogen_agentchat.base import Response
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.messages import BaseChatMessage, StructuredMessage
    from autogen_agentchat.teams import SelectorGroupChat
    from autogen_agentchat.ui import Console
    from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
    from autogen_core import CancellationToken
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ImportError:
    raise ImportError("⚠️  autogen_core or autogen_ext not found. Please install the required packages.")

# Load environment variables from .env file
load_dotenv()

class AgentMessage(BaseModel):
    messages: Annotated[List, operator.add]
    topic: str
    date_filter: str   # Optional date filter for arXiv search
    max_results: int   # Number of papers to retrieve
    search_results: List[dict]
    abstracts: List[dict]
    introduction: str  # Final Introduction section
    analysis: str      # Final Analysis section
    discussion: str    # Final Discussion section
    limitations: str   # Final Limitations section
    final_review: str  # Concatenated final review
    
    prompt_introduction: Dict[str, Any]
    prompt_analysis: Dict[str, Any]
    prompt_discussion: Dict[str, Any]
    prompt_limitations: Dict[str, Any]
    experiment_result: Dict[str, Any]

def load_prompts(prompt_file: str) -> List[Dict[str, Any]]:
    """Load prompt candidates from a JSON file"""
    prompt_path = Path(__file__).parent.parent / "prompts" / prompt_file

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

class BaseWorkflowAgent(BaseChatAgent):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description=description)
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (StructuredMessage,)
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant agent to its initialization state."""
        await self._model_context.clear()

class SearchPaperAgent(BaseWorkflowAgent):
    def __init__(self) -> None:
        super().__init__(name="search_paper_agent", description="A search paper agent")
        
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        latest_message_content: AgentMessage = messages[-1].content
        topic = latest_message_content.topic
        date_filter = latest_message_content.date_filter
        max_results = latest_message_content.max_results
        
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
                    response_message = StructuredMessage[AgentMessage](
                        content=latest_message_content.model_copy(update={
                        "messages": [{"role": "ai", "content": f"Found papers on '{topic}'"}],
                        "search_results": search_results
                    }),
                        source=self.name
                    )
                    return Response(chat_message=response_message)
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
        
        response_message = StructuredMessage[AgentMessage](
                content=latest_message_content.model_copy(update={
                    "messages": [{"role": "ai", "content": f"Searched papers error for topic: {topic}"}],
                    "search_results": search_results
                }),
                source=self.name
            )
        return Response(chat_message=response_message)

class ExtractAbstractsAgent(BaseWorkflowAgent):
    """Step 2: Extract abstracts from search results"""
    def __init__(self) -> None:
        super().__init__(name="extract_abstracts_agent", description="An agent to extract abstracts from search results")

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        latest_message_content: AgentMessage = messages[-1].content
        import xml.etree.ElementTree as ET

        abstracts = []
        search_results = latest_message_content.search_results

        for result in search_results:
            if result.get("status") == "success" and "data" in result:
                try:
                    # Parse arXiv XML response
                    root = ET.fromstring(result["data"])
                    # Define namespace
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
                    abstracts.append({"error": f"Failed to parse: {str(e)}"})
        print(f"  [OK] Extracted {len(abstracts)} abstracts")

        response_message = StructuredMessage[AgentMessage](
            content=latest_message_content.model_copy(update={
            "messages": latest_message_content.messages + [{"role": "ai", "content": f"Extracted {len(abstracts)} abstracts from search results."}],
            "abstracts": abstracts
        }),
            source=self.name
        )
        return Response(chat_message=response_message)

class GenerateIntroductionAgent(BaseWorkflowAgent):
    """Step 3: Generate Introduction using LLM based on extracted abstracts"""
    def __init__(self) -> None:
        super().__init__(name="generate_introduction_agent", description="An agent to generate introduction from abstracts")

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        latest_message_content: AgentMessage = messages[-1].content
        start_time = time.time()
        prompt_intro = latest_message_content.prompt_introduction
        print(f"[STEP] Generating Introduction with prompt {prompt_intro["id"]}")

        abstracts = latest_message_content.abstracts

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
        response = await llm.ainvoke(messages)
        
        original_token_usage = latest_message_content.experiment_result.get("cumulative_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
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
        latest_message_content.experiment_result.update({
            "introduction_tokens": current_token_usage,
            "introduction_time": execution_time,
            "cumulative_time": latest_message_content.experiment_result.get("cumulative_time", 0) + execution_time,
            "cumulative_tokens": cum_token_usage
            })

        response_message = StructuredMessage[AgentMessage](
            content=latest_message_content.model_copy(update={
                "messages": latest_message_content.messages + [{"role": "ai", "content": f"Generated introduction: {response.content}"}],
                "introduction": response.content
            }),
            source=self.name
        )

        return Response(chat_message=response_message)

class GenerateAnalysisAgent(BaseWorkflowAgent):
    """Step 4: Generate Literature Analysis section."""
    def __init__(self) -> None:
        super().__init__(name="generate_analysis_agent", description="An agent to generate analysis from abstracts")

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        latest_message_content: AgentMessage = messages[-1].content
        start_time = time.time()
        prompt_ana = latest_message_content.prompt_analysis
        print(f"[STEP] Generating Analysis with prompt {prompt_ana['id']}")

        abstracts = latest_message_content.abstracts
        introduction = latest_message_content.introduction

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
        response = await llm.ainvoke(messages)

        original_token_usage = latest_message_content.experiment_result.get("cumulative_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
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
        
        latest_message_content.experiment_result.update({
                "analysis_tokens": current_token_usage,
                "analysis_time": execution_time,
                "cumulative_time": latest_message_content.experiment_result.get("cumulative_time", 0) + execution_time,
                "cumulative_tokens": cum_token_usage
            })

        response_message = StructuredMessage[AgentMessage](
            content=latest_message_content.model_copy(update={
                "messages": latest_message_content.messages + [{"role": "ai", "content": f"Generated analysis: {response.content}"}],
                "analysis": response.content
            }),
            source=self.name
        )

        return Response(chat_message=response_message)

class GenerateDiscussionAgent(BaseWorkflowAgent):
    """Step 5: Generate Critical Discussion section."""
    
    def __init__(self) -> None:
        super().__init__(name="generate_discussion_agent", description="An agent to generate discussion from abstracts")

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        latest_message_content: AgentMessage = messages[-1].content
        start_time = time.time()
        prompt_disc = latest_message_content.prompt_discussion
        print(f"[STEP] Generating Discussion with prompt {prompt_disc['id']}")

        abstracts = latest_message_content.abstracts
        introduction = latest_message_content.introduction
        analysis = latest_message_content.analysis

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
        response = await llm.ainvoke(messages)

        original_token_usage = latest_message_content.experiment_result.get("cumulative_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
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
        
        latest_message_content.experiment_result.update({
                "discussion_tokens": current_token_usage,
                "discussion_time": execution_time,
                "cumulative_time": latest_message_content.experiment_result.get("cumulative_time", 0) + execution_time,
                "cumulative_tokens": cum_token_usage
            })
        response_message = StructuredMessage[AgentMessage](
            content=latest_message_content.model_copy(update={
                "messages": latest_message_content.messages + [{"role": "ai", "content": f"Generated discussion: {response.content}"}],
                "discussion": response.content
            }),
            source=self.name
        )
        return Response(chat_message=response_message)
    
class GenerateLimitationsAgent(BaseWorkflowAgent):
    """Step 6: Generate Limitations section."""
    
    def __init__(self) -> None:
        super().__init__(name="generate_limitations_agent", description="An agent to generate limitations from abstracts")

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        latest_message_content: AgentMessage = messages[-1].content
        start_time = time.time()
        prompt_limit = latest_message_content.prompt_limitations
        print(f"[STEP] Generating Limitations with prompt {prompt_limit['id']}")

        abstracts = latest_message_content.abstracts
        introduction = latest_message_content.introduction
        analysis = latest_message_content.analysis
        discussion = latest_message_content.discussion

        # Prepare abstracts text
        abstracts_text = "\n\n".join([
            f"Paper {i+1}: {abs.get('title', 'N/A')}\n"
            f"Authors: {', '.join(abs.get('authors', []))}\n"
            f"Abstract: {abs.get('abstract', 'N/A')}"
            for i, abs in enumerate(abstracts) if 'error' not in abs
        ])
        
        user_prompt = prompt_limit["user"].format(
            abstracts_text=abstracts_text,
            introduction=introduction,
            analysis=analysis,
            discussion=discussion
        )
        system_message = prompt_limit.get("system", "You are a helpful research assistant.")

        messages = [dict(role="human", content=f"System: {system_message}\n\nUser: {user_prompt}")]
        response = await llm.ainvoke(messages)

        original_token_usage = latest_message_content.experiment_result.get("cumulative_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        current_token_usage = track_tokens(response)
        cum_token_usage = {
            "prompt_tokens": original_token_usage.get("prompt_tokens", 0) + current_token_usage.get("prompt_tokens", 0),
            "completion_tokens": original_token_usage.get("completion_tokens", 0) + current_token_usage.get("completion_tokens", 0),
            "total_tokens": original_token_usage.get("total_tokens", 0) + current_token_usage.get("total_tokens", 0),
        }

        print(f"  → Prompt: {prompt_limit['id']} (Style: {prompt_limit['style']})")
        print(f"  → Generated {len(response.content)} chars")
        end_time = time.time()
        execution_time = end_time - start_time
        
        latest_message_content.experiment_result.update({
                "limitations_tokens": current_token_usage,
                "limitations_time": execution_time,
                "cumulative_time": latest_message_content.experiment_result.get("cumulative_time", 0) + execution_time,
                "cumulative_tokens": cum_token_usage
            })
        response_message = StructuredMessage[AgentMessage](
            content=latest_message_content.model_copy(update={
                "messages": latest_message_content.messages + [{"role": "ai", "content": f"Generated limitations: {response.content}"}],
                "limitations": response.content
            }),
            source=self.name
        )
        return Response(chat_message=response_message)

class ConcatenateReviewAgent(BaseWorkflowAgent):
    """Step 7: Concatenate sections into final review (NO regeneration)."""
    def __init__(self) -> None:
        super().__init__(name="concatenate_review_agent", description="An agent to concatenate review sections")
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        latest_message_content = messages[-1].content
        start_time = time.time()
        print(f"[NODE] Concatenating sections")

        topic = latest_message_content.topic
        introduction = latest_message_content.introduction
        analysis = latest_message_content.analysis
        discussion = latest_message_content.discussion
        limitations = latest_message_content.limitations

        # Simple concatenation with section headers and transitions
        final_review = f"""# Literature Review: {topic}

    ## 1. Introduction

    {introduction}

    ## 2. Literature Analysis

    {analysis}

    ## 3. Critical Discussion

    {discussion}

    ## 4. Limitations

    {limitations}

    ---
    *This review synthesizes findings from {len(latest_message_content.abstracts)} recent papers.*
    """
        
        word_count = len(final_review.split())
        print(f"  → Final review: {word_count} words")
        end_time = time.time()
        execution_time = end_time - start_time
        
        latest_message_content.experiment_result.update({
            "cumulative_time": latest_message_content.experiment_result.get("cumulative_time", 0) + execution_time,
        })

        response_message = StructuredMessage[AgentMessage](
            content=latest_message_content.model_copy(update={
                "messages": latest_message_content.messages + [{"role": "ai", "content": f"Final review concatenated."}],
                "final_review": final_review
            }),
            source=self.name
        )
        return Response(chat_message=response_message)

# 4. Build Workflow Dynamically from Configuration
async def run_workflow(initial_message: AgentMessage) -> Dict[str, Any]:
    """Build the workflow graph dynamically based on config"""

    search_paper_agent = SearchPaperAgent()
    extract_abstracts_agent = ExtractAbstractsAgent()
    generate_introduction_agent = GenerateIntroductionAgent()
    generate_analysis_agent = GenerateAnalysisAgent()
    generate_discussion_agent = GenerateDiscussionAgent()
    generate_limitations_agent = GenerateLimitationsAgent()
    concatenate_review_agent = ConcatenateReviewAgent()
    # Build the graph
    builder = DiGraphBuilder()
    builder.add_node(search_paper_agent)\
        .add_node(extract_abstracts_agent)\
        .add_node(generate_introduction_agent)\
        .add_node(generate_analysis_agent)\
        .add_node(generate_discussion_agent)\
        .add_node(generate_limitations_agent)\
        .add_node(concatenate_review_agent)
    builder.add_edge(search_paper_agent, extract_abstracts_agent)
    builder.add_edge(extract_abstracts_agent, generate_introduction_agent)
    builder.add_edge(generate_introduction_agent, generate_analysis_agent)
    builder.add_edge(generate_analysis_agent, generate_discussion_agent)
    builder.add_edge(generate_discussion_agent, generate_limitations_agent)
    builder.add_edge(generate_limitations_agent, concatenate_review_agent)

    graph = builder.build()

    # Create the flow
    flow = GraphFlow([search_paper_agent, extract_abstracts_agent, generate_introduction_agent, 
                      generate_analysis_agent, generate_discussion_agent, generate_limitations_agent, concatenate_review_agent], graph=graph,
                     custom_message_types=[StructuredMessage[AgentMessage]])

    task = StructuredMessage[AgentMessage](
        content=initial_message,
        source="user"
    )
    result = await flow.run(task=task)
    return result.messages[-1].content

# 5. Run Workflow in different prompt configurations in parallel
async def run_all_experiments(topic: str, run_id: str = None):
    
    # Running config
    project_start_time = time.time()
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + str(uuid.uuid4())
    
    
    prompt_intro_candidates = load_prompts("prompts_introduction.json")
    prompt_ana_candidates = load_prompts("prompts_analysis.json")
    prompt_disc_candidates = load_prompts("prompts_discussion.json")
    prompt_limit_candidates = load_prompts("prompts_limitations.json")
    all_results = []
    for (prompt_intro_idx, prompt_intro), (prompt_ana_idx, prompt_ana), (prompt_disc_idx, prompt_disc), (prompt_limit_idx, prompt_limit) in \
        itertools.product(
            enumerate(prompt_intro_candidates),
            enumerate(prompt_ana_candidates),
            enumerate(prompt_disc_candidates),
            enumerate(prompt_limit_candidates)
        ):
        init_message = AgentMessage(
            messages=[],
            topic=topic,
            date_filter="submittedDate:[20230101 TO 20231231]",
            max_results=10,
            search_results=[],
            abstracts=[],
            introduction="",
            analysis="",
            discussion="",
            limitations="",
            final_review="",
            prompt_introduction=prompt_intro,
            prompt_analysis=prompt_ana,
            prompt_discussion=prompt_disc,
            prompt_limitations=prompt_limit,
            experiment_result={}
        )
        # init_state_ls.append(init_message)
        state = await run_workflow(init_message)
        state = state.model_dump()
        all_results.append({
            "branch_id": f"I{prompt_intro_idx}_A{prompt_ana_idx}_D{prompt_disc_idx}_L{prompt_limit_idx}",
            "prompt_intro": state["prompt_introduction"]["id"],
            "prompt_intro_style": state["prompt_introduction"]["style"],
            "prompt_analysis": state["prompt_analysis"]["id"],
            "prompt_analysis_style": state["prompt_analysis"]["style"],
            "prompt_discussion": state["prompt_discussion"]["id"],
            "prompt_discussion_style": state["prompt_discussion"]["style"],
            "prompt_limitations": state["prompt_limitations"]["id"],
            "prompt_limitations_style": state["prompt_limitations"]["style"],
            "introduction": state.get("introduction", ""),
            "analysis": state.get("analysis", ""),
            "discussion": state.get("discussion", ""),
            "limitations": state.get("limitations", ""),
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
                    "limitations_time": state["experiment_result"]["limitations_time"],
                    "introduction_tokens": state["experiment_result"]["introduction_tokens"],
                    "analysis_tokens": state["experiment_result"]["analysis_tokens"],
                    "discussion_tokens": state["experiment_result"]["discussion_tokens"],
                    "limitations_tokens": state["experiment_result"]["limitations_tokens"],
                }
            }
        })
    all_results.sort(key=lambda x: (x["prompt_intro"], x["prompt_analysis"], x["prompt_discussion"]))

    total_project_time = time.time() - project_start_time
    
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
    
    print("\n" + "="*80)
    print(f"WORKFLOW COMPLETE - Generated {len(all_results)} literature reviews")
    print("="*80)
    print(f"\nTIMING SUMMARY:")
    print(f"  TOTAL PROJECT TIME: {total_project_time:.2f}s")
    print("="*80)

    # Save results
    topic_clean = topic.replace(' ', '_').lower()
    results_base = Path(__file__).parent.parent.parent / "results"
    run_dir = results_base / f"{topic_clean}" / "autogen"
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
    all_results, output_file, run_id = asyncio.run(run_all_experiments(topic))
    
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
