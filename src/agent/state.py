import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentPlanStep(TypedDict):
    """Represents a single step in the agent's overall execution plan."""
    step_number: int
    task_description: str
    tool_required: str  # e.g., "python_repl", "web_fetch", "file_io"
    status: str          # "pending", "in_progress", "completed", "failed"


class AgentState(TypedDict):
    """
    Central shared state passed through every node in the LangGraph execution engine.
    """
    # 1. Conversation & Message History
    # Uses LangGraph's built-in add_messages reducer to cleanly append new messages
    # and update existing ones by ID without overwriting conversation context.
    messages: Annotated[List[BaseMessage], add_messages]

    # 2. Plan Orchestration
    user_goal: str
    plan: List[AgentPlanStep]
    current_step_index: int

    # 3. Code Generation & Execution Sandbox State
    generated_code: Optional[str]
    execution_result: Optional[str]
    error_traceback: Optional[str]

    # 4. Self-Correction & Loop Control
    retry_count: int
    max_retries: int

    # 5. Output Artifacts
    # Uses operator.add to incrementally append paths of newly generated files/charts
    artifacts: Annotated[List[str], operator.add]

    # 6. Governance & Human-In-The-Loop (HITL) Controls
    requires_human_approval: bool
    human_approved: Optional[bool]