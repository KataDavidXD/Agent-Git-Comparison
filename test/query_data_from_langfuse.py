from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env file
from langfuse import get_client
langfuse = get_client()  # uses environment variables to authenticate
import json
traces = langfuse.api.trace.list(limit=100, filter=json.dumps([{"type": "string", "column": "name", "operator": "=", "value": "langgraph_agent_run_001"}]))  # pagination via cursor
observations = traces.data[0].observations
langgraph_trace_summary = {}
for obs in observations:
    obs_info = langfuse.api.observations.get(obs)
    if hasattr(obs_info, "metadata") and obs_info.metadata.get("langgraph_node"):
        step = obs_info.metadata["langgraph_step"]
        node_name = obs_info.metadata["langgraph_node"]
        if step not in langgraph_trace_summary:
            langgraph_trace_summary[step] = {
                "name": node_name
            }
        if obs_info.type.lower() == "generation":
            langgraph_trace_summary[step].update({
                "token_usage": obs_info.usage_details,
                "cost": obs_info.cost_details
            })
        elif obs_info.type.lower() == "chain":
            langgraph_trace_summary[step].update({
                "latency": obs_info.latency
            })
print(json.dumps(dict(sorted(langgraph_trace_summary.items())), indent=2))