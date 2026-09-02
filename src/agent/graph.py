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
# Conditional Edge Routers
# ---------------------------------------------------------------------------

def should_self_correct(state: AgentState) -> Literal["self_correct_node", "step_router"]:
    """
    Evaluates execution outcomes to decide whether to trigger self-correction
    or route to step progression.
    """
    error_tb = state.get("error_traceback")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    # If code crashed and retry limit isn't reached, loop to self_correct_node
    if error_tb and retry_count < max_retries:
        return "self_correct_node"
    
    # Otherwise, proceed to step router (which either advances step or ends execution)
    return "step_router"


def step_router(state: AgentState) -> Literal["coder_node", "reporter_node", "END"]:
    """
    Advances step index upon success or handles terminal state on retry exhaustion.
    """
    error_tb = state.get("error_traceback")
    
    # If error persists after max retries, terminate graph safely
    if error_tb:
        return END
        
    plan = state["plan"]
    current_idx = state["current_step_index"] + 1
    
    # Update current_step_index in state before next evaluation
    state["current_step_index"] = current_idx
    
    # If more steps remain, return to coder_node for next step
    if current_idx < len(plan):
        return "coder_node"
        
    # All steps completed successfully -> compile report
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
    
    # 2. Define Static Structural Edges
    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "coder_node")
    workflow.add_edge("coder_node", "execution_node")
    workflow.add_edge("self_correct_node", "execution_node")
    workflow.add_edge("reporter_node", END)
    
    # 3. Add Conditional Edge Loops
    workflow.add_conditional_edges(
        "execution_node",
        should_self_correct,
        {
            "self_correct_node": "self_correct_node",
            "step_router": step_router,
        }
    )
    
    return workflow.compile()


# Export compiled graph instance
app_graph = build_graph()