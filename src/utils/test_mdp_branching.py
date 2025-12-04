"""
Test for MDP Branching Helper with Tree Support.
"""

import sys
import os
from pathlib import Path

# Force path priority to current workspace
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pytest
import time
import tempfile
import sqlite3
from typing import Dict, Any
from agentgit.managers.tool_manager import ToolManager
from agentgit.managers.checkpoint_manager_mdp import CheckpointManager_mdp
from agentgit.sessions.internal_session_mdp import InternalSession_mdp
from agentgit.sessions.external_session_mdp import ExternalSession_mdp
from agentgit.database.repositories.checkpoint_repository import CheckpointRepository
from agentgit.database.repositories.internal_session_repository import InternalSessionRepository
from agentgit.database.repositories.user_repository import UserRepository
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from agentgit.core.workflow import WorkflowGraph, WorkflowNode

try:
    from agentgit_paper.utils.mdp_branching import MDPBranchHelper
except ImportError:
    from src.agentgit_paper.utils.mdp_branching import MDPBranchHelper

def serialize_state(state): return state
def deserialize_state(state): return state

def task_append_msg(state, context):
    msg = context.get("msg", "default")
    current_msgs = state.get("msgs", [])
    # Add a small delay to simulate work and test parallelism
    time.sleep(0.1)
    return {"msgs": current_msgs + [msg]}

def setup_test_data(db_path):
    user_repo = UserRepository(db_path)
    ext_repo = ExternalSessionRepository(db_path)
    # Initialize other repos to create tables
    InternalSessionRepository(db_path)
    CheckpointRepository(db_path)
    
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY)") 
    conn.execute("INSERT OR IGNORE INTO users (id) VALUES (1)")
    conn.commit()
    conn.close()
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("INSERT INTO external_sessions (id, user_id, session_name, created_at) VALUES (1, 1, 'Test', ?)", (time.time(),))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Setup warning: {e}")

def test_tree_branching(db_path):
    setup_test_data(db_path)
    helper = MDPBranchHelper(db_path)
    
    # 1. Create Root Checkpoint
    int_repo = InternalSession_mdp(id=None, external_session_id=1, current_node_id="ROOT", workflow_variables={})
    int_repo.session_state["graph_state"] = {"msgs": ["root"]}
    helper.internal_session_repo.create(int_repo)
    
    cm = CheckpointManager_mdp(helper.checkpoint_repo)
    tm = ToolManager(tools=[])
    cp_root = cm.create_node_checkpoint(int_repo, "Root", "1.0", tm.get_tool_track_position())
    print(f"Root Checkpoint: {cp_root.id}")
    
    # 2. Branch Layer 1
    tasks_layer_1 = [
        {"parent_checkpoint_id": cp_root.id, "func": task_append_msg, "context": {"msg": "A"}, "node_id": "Node_A"},
        {"parent_checkpoint_id": cp_root.id, "func": task_append_msg, "context": {"msg": "B"}, "node_id": "Node_B"}
    ]
    results_1 = helper.execute_branches(tasks=tasks_layer_1, external_session_id=1)
    
    for r in results_1:
        if r["status"] != "success":
            print(f"Layer 1 Error: {r}")
            
    res_a = next(r for r in results_1 if r["context"]["msg"] == "A")
    res_b = next(r for r in results_1 if r["context"]["msg"] == "B")
    
    assert res_a["status"] == "success"
    assert res_b["status"] == "success"
    assert res_a["state"]["msgs"] == ["root", "A"]
    
    # 3. Branch Layer 2 (Branch and Branch)
    tasks_layer_2 = [
        {"parent_checkpoint_id": res_a["checkpoint_id"], "func": task_append_msg, "context": {"msg": "A1"}, "node_id": "Node_A1"},
        {"parent_checkpoint_id": res_a["checkpoint_id"], "func": task_append_msg, "context": {"msg": "A2"}, "node_id": "Node_A2"},
        {"parent_checkpoint_id": res_b["checkpoint_id"], "func": task_append_msg, "context": {"msg": "B1"}, "node_id": "Node_B1"}
    ]
    results_2 = helper.execute_branches(tasks=tasks_layer_2, external_session_id=1)
    
    for r in results_2:
        if r["status"] != "success":
            print(f"Layer 2 Error: {r}")

    res_a1 = next(r for r in results_2 if r["context"]["msg"] == "A1")
    
    assert res_a1["status"] == "success"
    assert res_a1["state"]["msgs"] == ["root", "A", "A1"]
    
    print("\nVisualization (Tree):")
    helper.visualize_tree(root_checkpoint_id=cp_root.id)

def test_rollback_scenarios(db_path):
    """Test complex scenarios: Branch-Rollback, Branch-Rollback-Branch."""
    setup_test_data(db_path)
    helper = MDPBranchHelper(db_path)
    
    # Initial State
    int_repo = InternalSession_mdp(id=None, external_session_id=1, current_node_id="START", workflow_variables={})
    int_repo.session_state["graph_state"] = {"data": "init", "msgs": []} # Init msgs list
    helper.internal_session_repo.create(int_repo)
    
    cm = CheckpointManager_mdp(helper.checkpoint_repo)
    tm = ToolManager(tools=[])
    cp_start = cm.create_node_checkpoint(int_repo, "Start", "1.0", tm.get_tool_track_position())
    
    print("\n--- Scenario: Branch then Rollback ---")
    # 1. Branch A from Start
    res_a = helper.fork_branch(cp_start.id, task_append_msg, {"msg": "A"}, node_id="Node_A")
    assert res_a["status"] == "success"
    
    # 2. Rollback: Fork from A's checkpoint (effectively rolling back state to A and branching new path)
    # This creates Branch B which starts at state of Node_A
    res_b = helper.fork_branch(res_a["checkpoint_id"], task_append_msg, {"msg": "B_Rollback"}, node_id="Node_B_from_A")
    
    assert res_b["status"] == "success"
    # Check state
    assert "A" in res_b["state"]["msgs"]
    assert "B_Rollback" in res_b["state"]["msgs"]
    
    print("\n--- Scenario: Branch then Rollback then Branch ---")
    # 3. Branch C from B (Branch-Rollback-Branch)
    res_c = helper.fork_branch(res_b["checkpoint_id"], task_append_msg, {"msg": "C_from_B"}, node_id="Node_C_from_B")
    
    assert res_c["status"] == "success"
    assert "B_Rollback" in res_c["state"]["msgs"]
    assert "C_from_B" in res_c["state"]["msgs"]
    
    print("\nVisualization (Rollback Scenarios):")
    helper.visualize_tree(root_checkpoint_id=cp_start.id)

def test_parallel_rollback(db_path):
    """Test PARALLEL execution of rollback scenarios."""
    setup_test_data(db_path)
    helper = MDPBranchHelper(db_path)
    
    # Initial State
    int_repo = InternalSession_mdp(id=None, external_session_id=1, current_node_id="START", workflow_variables={})
    int_repo.session_state["graph_state"] = {"msgs": ["Start"]} 
    helper.internal_session_repo.create(int_repo)
    
    cm = CheckpointManager_mdp(helper.checkpoint_repo)
    tm = ToolManager(tools=[])
    cp_start = cm.create_node_checkpoint(int_repo, "Start", "1.0", tm.get_tool_track_position())
    
    # 1. Create Branch A (Sequential base)
    res_a = helper.fork_branch(cp_start.id, task_append_msg, {"msg": "A"}, node_id="Node_A")
    cp_a_id = res_a["checkpoint_id"]
    
    print("\n--- Parallel Scenarios ---")
    
    # We will execute 4 parallel tasks:
    # 1 & 2: Branch from Start (Rollback to Start -> Branch)
    # 3 & 4: Branch from A (Branch from A -> Branch)
    
    tasks = [
        # From Start
        {"parent_checkpoint_id": cp_start.id, "func": task_append_msg, "context": {"msg": "B_from_Start"}, "node_id": "Node_B"},
        {"parent_checkpoint_id": cp_start.id, "func": task_append_msg, "context": {"msg": "C_from_Start"}, "node_id": "Node_C"},
        # From A
        {"parent_checkpoint_id": cp_a_id, "func": task_append_msg, "context": {"msg": "D_from_A"}, "node_id": "Node_D"},
        {"parent_checkpoint_id": cp_a_id, "func": task_append_msg, "context": {"msg": "E_from_A"}, "node_id": "Node_E"}
    ]
    
    start_time = time.time()
    results = helper.execute_branches(tasks, external_session_id=1)
    total_time = time.time() - start_time
    
    print(f"Executed 4 branches in {total_time:.2f}s")
    
    # Verify all success
    for r in results:
        assert r["status"] == "success"
        
    # Verify contents
    res_b = next(r for r in results if r["context"]["msg"] == "B_from_Start")
    assert res_b["state"]["msgs"] == ["Start", "B_from_Start"]
    
    res_d = next(r for r in results if r["context"]["msg"] == "D_from_A")
    assert res_d["state"]["msgs"] == ["Start", "A", "D_from_A"]
    
    print("\nVisualization (Parallel Scenarios):")
    helper.visualize_tree(root_checkpoint_id=cp_start.id)


if __name__ == "__main__":
    # Run Test 1
    fd1, path1 = tempfile.mkstemp(suffix=".db")
    os.close(fd1)
    try:
        print("Test 1: Tree Branching")
        test_tree_branching(path1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Test 1 Failed")
    finally:
        if os.path.exists(path1):
            try: os.remove(path1)
            except: pass
            
    # Run Test 2
    fd2, path2 = tempfile.mkstemp(suffix=".db")
    os.close(fd2)
    try:
        print("\nTest 2: Complex Rollback Scenarios (Sequential)")
        test_rollback_scenarios(path2)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Test 2 Failed")
    finally:
        if os.path.exists(path2):
            try: os.remove(path2)
            except: pass
            
    # Run Test 3 (Parallel)
    fd3, path3 = tempfile.mkstemp(suffix=".db")
    os.close(fd3)
    try:
        print("\nTest 3: Parallel Rollback/Branch Scenarios")
        test_parallel_rollback(path3)
        print("\nAll Tests Passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Test 3 Failed")
    finally:
        if os.path.exists(path3):
            try: os.remove(path3)
            except: pass
