from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import coder_node, execution_node, planner_node, self_correct_node
from src.agent.state import AgentState

# ---------------------------------------------------------------------------
# Node 5: Reporter Node (Final Synthesis)
# ---------------------------------------------------------------------------

def reporter_node(state: AgentState) -> dict:
    """
    Compiles a final summary report after all plan steps complete successfully.
    """
    artifacts = state.get("artifacts", [])
    user_goal = state["user_goal"]
    plan = state["plan"]
    
    summary_lines = [
        f"### Agentic DevOps Engine — Execution Report",
        f"**Goal:** {user_goal}",
        f"**Steps Completed:** {len(plan)}/{len(plan)}",
        f"**Generated Artifacts:** {len(artifacts)}",
    ]
    
    if artifacts:
        summary_lines.append("\n**Artifact File Paths:**")
        for art in set(artifacts):
            summary_lines.append(f"- `{art}`")
            
    summary_lines.append("\nExecution workflow completed successfully.")
    
    return {
        "messages": [AIMessage(content="\n".join(summary_lines))]
    }


# ---------------------------------------------------------------------------
# Combined Conditional Routing Logic
# ---------------------------------------------------------------------------

def route_after_execution(state: AgentState) -> Literal["self_correct_node", "coder_node", "reporter_node", "__end__"]:
    """
    Evaluates execution outcomes to decide whether to:
    1. Retry/Self-correct on error (if under max retries)
    2. Advance to the next step in the plan (if steps remain)
    3. Proceed to final report synthesis (if all steps finished)
    4. Terminate early if retries are exhausted
    """
    error_tb = state.get("error_traceback")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    # 1. Error handling route
    if error_tb:
        if retry_count < max_retries:
            return "self_correct_node"
        # Retry limit reached with unhandled error -> Terminate safely
        return END

    # 2. Step advancement route
    plan = state["plan"]
    current_idx = state["current_step_index"] + 1
    state["current_step_index"] = current_idx
    
    if current_idx < len(plan):
        return "coder_node"
        
    return "reporter_node"


# ---------------------------------------------------------------------------
# StateGraph Construction & Compilation
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Constructs and compiles the deterministic LangGraph workflow engine.
    """
    workflow = StateGraph(AgentState)
    
    # 1. Add Execution Nodes
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("coder_node", coder_node)
    workflow.add_node("execution_node", execution_node)
    workflow.add_node("self_correct_node", self_correct_node)
    workflow.add_node("reporter_node", reporter_node)
    
    # 2. Define Static Edges
    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "coder_node")
    workflow.add_edge("coder_node", "execution_node")
    workflow.add_edge("self_correct_node", "execution_node")
    workflow.add_edge("reporter_node", END)
    
    # 3. Add Conditional Edge
    workflow.add_conditional_edges(
        "execution_node",
        route_after_execution,
        {
            "self_correct_node": "self_correct_node",
            "coder_node": "coder_node",
            "reporter_node": "reporter_node",
            END: END,
        }
    )
    
    return workflow.compile()


app_graph = build_graph()