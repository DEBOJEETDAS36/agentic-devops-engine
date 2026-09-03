import os
import re
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.agent.state import AgentPlanStep, AgentState
from src.tools.sandbox import default_sandbox

# Load environment configuration (.env)
load_dotenv()

# Initialize LLM Client
MODEL_NAME = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.1,
    max_retries=2,
)

# ---------------------------------------------------------------------------
# Pydantic Schema for Structured Planning Output
# ---------------------------------------------------------------------------

class SingleStep(BaseModel):
    step_number: int = Field(description="Sequential step index starting at 1")
    task_description: str = Field(description="Detailed explanation of what this step accomplishes")
    tool_required: str = Field(description="Primary tool required: 'python_repl', 'web_fetch', or 'file_io'")

class ExecutionPlan(BaseModel):
    user_goal: str = Field(description="Summary of the user's overall goal")
    steps: List[SingleStep] = Field(description="Ordered steps required to achieve the goal")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def extract_python_code(text: str) -> str:
    """Extract clean Python code from Markdown backticks or raw output."""
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    pattern_generic = r"```\s*(.*?)\s*```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic:
        return match_generic.group(1).strip()
    
    return text.strip()


# ---------------------------------------------------------------------------
# Node 1: Planner Node
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    """Deconstructs the user goal into a structured sequence of actionable steps."""
    user_goal = state["user_goal"]
    
    planner_prompt = f"""
    You are an Expert DevOps & Analytics Lead Planner.
    Break down the following user goal into a clear, minimal set of sequential, executable steps.
    
    User Goal: {user_goal}
    
    Rules:
    - Keep steps actionable and concise (typically 2-4 steps).
    - If code execution or data processing is required, assign the tool 'python_repl'.
    - Assign step numbers starting sequentially from 1.
    """
    
    structured_llm = llm.with_structured_output(ExecutionPlan)
    plan_output: ExecutionPlan = structured_llm.invoke(planner_prompt)
    
    formatted_steps: List[AgentPlanStep] = [
        {
            "step_number": s.step_number,
            "task_description": s.task_description,
            "tool_required": s.tool_required,
            "status": "pending",
        }
        for s in plan_output.steps
    ]
    
    plan_summary_text = "\n".join(
        [f"Step {s['step_number']}: {s['task_description']} (Tool: {s['tool_required']})" for s in formatted_steps]
    )
    
    return {
        "plan": formatted_steps,
        "current_step_index": 0,
        "retry_count": 0,
        "messages": [
            AIMessage(content=f"**Execution Plan Generated:**\n{plan_summary_text}")
        ],
    }


# ---------------------------------------------------------------------------
# Node 2: Coder Node
# ---------------------------------------------------------------------------

def coder_node(state: AgentState) -> dict:
    """Generates executable Python code tailored to the current step of the plan."""
    plan = state["plan"]
    current_idx = state["current_step_index"]
    current_step = plan[current_idx]
    user_goal = state["user_goal"]
    
    coder_sys_prompt = """
    You are a Senior Python & DevOps Automation Engineer.
    Your task is to write standalone, robust, executable Python code to solve the assigned step.
    
    Requirements:
    1. Output ONLY executable Python code inside a single ```python ... ``` block.
    2. Import all necessary standard or popular libraries (e.g., pandas, matplotlib, os, sys, requests).
    3. If saving charts or datasets, save them inside the 'artifacts' directory.
    4. Include explicit print() statements so the execution stdout produces readable logs.
    """
    
    user_msg = f"""
    Overall Goal: {user_goal}
    Current Step ({current_idx + 1}/{len(plan)}): {current_step['task_description']}
    
    Write Python code to accomplish this step.
    """
    
    response = llm.invoke([SystemMessage(content=coder_sys_prompt), HumanMessage(content=user_msg)])
    code = extract_python_code(response.content)
    
    return {
        "generated_code": code,
        "messages": [
            AIMessage(content=f"**Code Generated for Step {current_idx + 1}:**\n```python\n{code}\n```")
        ],
    }


# ---------------------------------------------------------------------------
# Node 3: Execution Node
# ---------------------------------------------------------------------------

def execution_node(state: AgentState) -> dict:
    """Executes generated Python code inside the REPL sandbox and registers execution results."""
    code = state["generated_code"]
    
    success, output, new_artifacts = default_sandbox.run(code)
    
    if success:
        return {
            "execution_result": output,
            "error_traceback": None,
            "artifacts": new_artifacts,
            "messages": [
                AIMessage(content=f"**Step Execution Succeeded:**\n```\n{output}\n```")
            ],
        }
    else:
        return {
            "execution_result": None,
            "error_traceback": output,
            "messages": [
                AIMessage(content=f"**Step Execution Failed:**\n```\n{output}\n```")
            ],
        }


# ---------------------------------------------------------------------------
# Node 4: Self-Correct Node
# ---------------------------------------------------------------------------

def self_correct_node(state: AgentState) -> dict:
    """Analyzes error tracebacks, increments retry count, and rewrites failing code."""
    failing_code = state["generated_code"]
    error_trace = state["error_traceback"]
    retry_count = state.get("retry_count", 0) + 1
    current_idx = state["current_step_index"]
    
    fix_prompt = f"""
    You are a Lead Python Debugger. The following code failed during execution.
    
    --- FAILED CODE ---
    {failing_code}
    
    --- ERROR LOG / TRACEBACK ---
    {error_trace}
    
    Task:
    Analyze the traceback error, fix the root cause, and return the complete corrected Python code.
    Output ONLY the corrected code wrapped inside a ```python ... ``` code block.
    """
    
    response = llm.invoke([HumanMessage(content=fix_prompt)])
    corrected_code = extract_python_code(response.content)
    
    return {
        "generated_code": corrected_code,
        "retry_count": retry_count,
        "error_traceback": None,
        "messages": [
            AIMessage(
                content=f"**Self-Correction Attempt #{retry_count} for Step {current_idx + 1}:**\n```python\n{corrected_code}\n```"
            )
        ],
    }