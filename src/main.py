import os
import sys
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from src.agent.graph import app_graph
from src.agent.state import AgentState

# Load environment configuration
load_dotenv()


def run_cli():
    """
    Interactive CLI entry point for the DevOps & Analytics Agentic Engine.
    """
    print("=" * 65)
    print(" 🚀 DevOps & Analytics Agentic Engine (LangGraph Powered) ")
    print("=" * 65)

    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: No API key found in .env file.")
        print("Please configure GROQ_API_KEY or OPENAI_API_KEY in your .env file.")
        sys.exit(1)

    while True:
        try:
            user_input = input("\n📌 Enter your DevOps/Analytics Goal (or 'exit' to quit): ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nShutting down Agentic Engine. Goodbye! 👋")
                break

            print(f"\n[Engine] Initializing execution pipeline for: '{user_input}'...\n")
            
            # 1. Initialize Typed State
            initial_state: AgentState = {
                "messages": [HumanMessage(content=user_input)],
                "user_goal": user_input,
                "plan": [],
                "current_step_index": 0,
                "generated_code": None,
                "execution_result": None,
                "error_traceback": None,
                "retry_count": 0,
                "max_retries": 3,
                "artifacts": [],
                "requires_human_approval": False,
                "human_approved": None,
            }

            # 2. Stream Graph Node Execution
            # Recursion limit handles multi-step loops safely
            config = {"recursion_limit": 50}
            
            for event in app_graph.stream(initial_state, config=config):
                for node_name, node_state in event.items():
                    print(f"\n--- [NODE EXECUTED: {node_name.upper()}] ---")
                    
                    # Print latest message output from the node
                    if "messages" in node_state and node_state["messages"]:
                        latest_msg = node_state["messages"][-1]
                        print(latest_msg.content)
                    
                    # Log artifacts if newly generated
                    if "artifacts" in node_state and node_state["artifacts"]:
                        print(f"📦 Artifacts updated: {node_state['artifacts']}")

            print("\n" + "=" * 65)
            print(" ✅ Pipeline Execution Complete ")
            print("=" * 65)

        except KeyboardInterrupt:
            print("\n\nExecution interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    run_cli()