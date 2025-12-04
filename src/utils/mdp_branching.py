"""
MDP Branching Helper for AgentGit.

Provides pluggable branching and parallel execution capabilities using 
Node and Tool/Message granularity checkpoints (MDP style).
"""

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, TypedDict
import uuid
import json

# AgentGit MDP imports
try:
    from agentgit.agents.rollback_agent_mdp import RollbackAgent_mdp
    from agentgit.managers.checkpoint_manager_mdp import CheckpointManager_mdp
    from agentgit.managers.node_manager import NodeManager
    from agentgit.managers.tool_manager import ToolManager
    from agentgit.sessions.internal_session_mdp import InternalSession_mdp
    from agentgit.sessions.external_session_mdp import ExternalSession_mdp
    from agentgit.database.repositories.checkpoint_repository import CheckpointRepository
    from agentgit.database.repositories.internal_session_repository import InternalSessionRepository
    from agentgit.core.workflow import WorkflowGraph
except ImportError:
    # Fallback or mock for development environments where package might not be fully installed
    print("Warning: AgentGit MDP modules not found. Ensure agentgit is installed.")

class BranchTask(TypedDict, total=False):
    parent_checkpoint_id: int
    func: Callable
    context: Any
    node_id: str
    version: str

class MDPBranchHelper:
    """
    Helper class to manage branching and parallel execution using MDP-style checkpoints.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        # Initialize Repositories
        self.checkpoint_repo = CheckpointRepository(db_path)
        self.internal_session_repo = InternalSessionRepository(db_path)
        self.external_session_repo = ExternalSession_mdp 
        
    def fork_branch(
        self,
        parent_checkpoint_id: int,
        task_func: Callable[[Dict[str, Any], Any], Dict[str, Any]],
        context: Any = None,
        node_id: str = "Node",
        checkpoint_version: str = "1.0",
        serialize_state_func: Callable[[Dict], Dict] = None,
        deserialize_state_func: Callable[[Dict], Dict] = None,
        tool_manager: Optional[ToolManager] = None,
        external_session_id: int = 1,  # Default if not provided
        checkpoint_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Forks a new branch from a parent checkpoint, executes a task, and saves the result.
        """
        start_time = time.time()
        
        # 1. Setup Managers
        checkpoint_manager = CheckpointManager_mdp(self.checkpoint_repo)
        if tool_manager is None:
            tool_manager = ToolManager(tools=[])
            
        # Dummy Node Manager if not provided (needed for RollbackAgent_mdp)
        graph = WorkflowGraph(id=f"branch_{uuid.uuid4().hex[:8]}")
        node_manager = NodeManager(graph, tool_manager)

        # 2. Rollback / Restore Session
        # Create temp session for initialization
        temp_internal_session = InternalSession_mdp(
            id=None, 
            external_session_id=external_session_id,
            current_node_id="INIT",
            workflow_variables={}
        )
        
        # Fetch real external session object if possible, or mock
        try:
             ext_session = ExternalSession_mdp(id=external_session_id)
        except:
             ext_session = ExternalSession_mdp(id=external_session_id)
        
        agent = RollbackAgent_mdp(
            external_session=ext_session,
            internal_session=temp_internal_session,
            node_manager=node_manager,
            checkpoint_manager=checkpoint_manager
        )
        
        # Perform Rollback -> This is where the Fork happens
        try:
            branched_session = agent.rollback_to_checkpoint(parent_checkpoint_id)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Rollback failed: {e}",
                "context": context,
                "execution_time": time.time() - start_time
            }
        
        if not branched_session:
             return {
                "status": "error",
                "error": f"Failed to rollback to checkpoint {parent_checkpoint_id} (returned None)",
                "context": context,
                "execution_time": time.time() - start_time
             }
             
        # Update agent to use the branched session
        agent.internal_session = branched_session
        
        # 3. Deserialize State
        current_state_raw = branched_session.session_state.get("graph_state", {})
        if deserialize_state_func:
            current_state = deserialize_state_func(current_state_raw)
        else:
            current_state = current_state_raw
            
        # 4. Execute Task
        try:
            updates = task_func(current_state, context)
            current_state.update(updates)
            status = "success"
            error = None
        except Exception as e:
            print(f"Error in branch execution: {e}")
            traceback.print_exc()
            status = "error"
            error = str(e)
            updates = {}
            
        # 5. Save State & Create Checkpoint
        if serialize_state_func:
            serialized_state = serialize_state_func(current_state)
        else:
            serialized_state = current_state
            
        branched_session.session_state["graph_state"] = serialized_state
        
        # Update/Save session in DB
        if branched_session.id is None:
            branched_session = self.internal_session_repo.create(branched_session)
        else:
            self.internal_session_repo.update(branched_session)
        
        # Create Node Checkpoint
        try:
            new_checkpoint = checkpoint_manager.create_node_checkpoint(
                internal_session=branched_session,
                node_id=node_id,
                workflow_version=checkpoint_version,
                tool_track_position=tool_manager.get_tool_track_position()
            )
            checkpoint_id = new_checkpoint.id
            
            # Update checkpoint metadata with custom fields
            if checkpoint_id and checkpoint_metadata:
                new_checkpoint.metadata.update(checkpoint_metadata)
                checkpoint_repo = self.checkpoint_repo
                checkpoint_repo.update_checkpoint_metadata(checkpoint_id, new_checkpoint.metadata)
                
        except Exception as e:
            status = "error"
            error = f"Checkpoint creation failed: {e}"
            checkpoint_id = None
        
        execution_time = time.time() - start_time
        
        return {
            "status": status,
            "error": error,
            "checkpoint_id": checkpoint_id,
            "session_id": branched_session.id,
            "context": context,
            "state": current_state,
            "updates": updates,
            "execution_time": execution_time
        }

    def execute_branches(
        self,
        tasks: List[BranchTask],
        max_workers: int = 16,
        external_session_id: int = 1,
        serialize_state_func: Callable = None,
        deserialize_state_func: Callable = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a list of branch tasks in parallel.
        Each task can have a DIFFERENT parent_checkpoint_id, allowing tree-like execution.
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for i, task in enumerate(tasks):
                parent_cp = task.get("parent_checkpoint_id")
                if not parent_cp:
                    results.append({
                        "task_index": i,
                        "status": "error",
                        "error": "Missing parent_checkpoint_id",
                        "context": task.get("context")
                    })
                    continue
                
                node_id = task.get("node_id", f"Branch_{i}")
                version = task.get("version", "1.0")
                checkpoint_metadata = task.get("checkpoint_metadata")
                
                future = executor.submit(
                    self.fork_branch,
                    parent_checkpoint_id=parent_cp,
                    task_func=task["func"],
                    context=task.get("context"),
                    node_id=node_id,
                    checkpoint_version=version,
                    serialize_state_func=serialize_state_func,
                    deserialize_state_func=deserialize_state_func,
                    external_session_id=external_session_id,
                    checkpoint_metadata=checkpoint_metadata
                )
                future_to_task[future] = i
                
            for future in as_completed(future_to_task):
                idx = future_to_task[future]
                try:
                    result = future.result()
                    result["task_index"] = idx
                    results.append(result)
                except Exception as e:
                    print(f"Parallel execution failed for task {idx}: {e}")
                    results.append({
                        "task_index": idx,
                        "status": "error",
                        "error": str(e),
                        "context": tasks[idx].get("context")
                    })
                    
        results.sort(key=lambda x: x["task_index"])
        return results

    def visualize_tree(self, root_checkpoint_id: int):
        """
        Simple tree visualization starting from a root checkpoint.
        """
        print(f"\n[Tree Visualization] Root Checkpoint: {root_checkpoint_id}")
        
        root_cp = self.checkpoint_repo.get_by_id(root_checkpoint_id)
        if not root_cp:
             print("Root checkpoint not found.")
             return

        # Get node_id from metadata if available
        node_label = root_cp.metadata.get("node_id", root_cp.checkpoint_name)
        print(f"└── [CP {root_checkpoint_id}] {node_label} (Session {root_cp.internal_session_id})")
        self._print_children(root_cp.internal_session_id, prefix="    ")

    def _print_children(self, parent_session_id: int, prefix: str):
        checkpoints = self.checkpoint_repo.get_by_internal_session(parent_session_id, auto_only=False)
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for cp in checkpoints:
                # Find sessions that branched from this checkpoint
                cursor.execute("SELECT id, state_data FROM internal_sessions WHERE branch_point_checkpoint_id = ?", (cp.id,))
                children = cursor.fetchall()
                
                if children:
                    cp_node_label = cp.metadata.get("node_id", cp.checkpoint_name)
                    print(f"{prefix}└── [Branch from CP {cp.id} '{cp_node_label}']")
                    for child_id, state_json in children:
                        # Try to parse node_id from state_data if possible
                        node_id_label = "?"
                        try:
                            if state_json:
                                state = json.loads(state_json)
                                node_id_label = state.get("current_node_id", "?")
                        except:
                            pass
                            
                        # Updated query: Select checkpoint_data instead of metadata
                        cursor.execute("SELECT id, checkpoint_data, checkpoint_name FROM checkpoints WHERE internal_session_id = ? ORDER BY id DESC LIMIT 1", (child_id,))
                        latest_cp = cursor.fetchone()
                        
                        if latest_cp:
                            cp_id, cp_data_json, cp_name = latest_cp
                            cp_data = json.loads(cp_data_json) if cp_data_json else {}
                            cp_meta = cp_data.get("metadata", {})
                            label = cp_meta.get("node_id", cp_name)
                            cp_label = f"CP {cp_id} {label}"
                        else:
                            cp_label = "No CP"
                            
                        print(f"{prefix}    └── Session {child_id} ({node_id_label}) -> {cp_label}")
                        
                        self._print_children(child_id, prefix + "        ")
            
            conn.close()
        except Exception as e:
            print(f"Visualization error: {e}")
