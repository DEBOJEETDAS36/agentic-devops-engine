import io
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple


class PythonREPLSandbox:
    """
    Isolated Python REPL tool execution environment with output capture,
    persisted local variables across state iterations, and artifact tracking.
    """

    def __init__(self, artifact_dir: str = "artifacts"):
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)
        
        # Persistent execution namespace across consecutive runs
        self.globals: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "os": os,
            "sys": sys,
        }
        self.locals: Dict[str, Any] = {}

    def run(self, code: str) -> Tuple[bool, str, List[str]]:
        """
        Executes a Python code block and captures output, exceptions, and created files.

        Returns:
            Tuple[bool, str, List[str]]: 
                - success (bool): True if executed cleanly without unhandled exceptions.
                - output (str): Captured stdout, return values, or raw traceback.
                - new_artifacts (List[str]): Absolute/relative paths to files created during execution.
        """
        # Snapshot existing files in artifact directory before execution
        initial_artifacts = set(os.listdir(self.artifact_dir))

        # Redirect standard output streams to capture execution prints
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_stdout = io.StringIO()
        redirected_stderr = io.StringIO()

        success = True
        result_output = ""

        try:
            sys.stdout = redirected_stdout
            sys.stderr = redirected_stderr

            # Execute code block within persistent sandbox globals/locals
            exec(code, self.globals, self.locals)
            
            stdout_str = redirected_stdout.getvalue()
            stderr_str = redirected_stderr.getvalue()

            combined_logs = []
            if stdout_str:
                combined_logs.append(f"[STDOUT]\n{stdout_str.strip()}")
            if stderr_str:
                combined_logs.append(f"[STDERR]\n{stderr_str.strip()}")
            
            result_output = "\n".join(combined_logs) if combined_logs else "Execution completed successfully with no output."

        except Exception as e:
            success = False
            tb_str = traceback.format_exc()
            stdout_str = redirected_stdout.getvalue()
            result_output = f"[EXECUTION FAILED]\nError: {str(e)}\n\n[TRACEBACK]\n{tb_str}"
            if stdout_str:
                result_output = f"[STDOUT BEFORE CRASH]\n{stdout_str}\n\n" + result_output

        finally:
            # Always restore standard output streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Detect newly generated files in artifacts folder
        current_artifacts = set(os.listdir(self.artifact_dir))
        new_files = list(current_artifacts - initial_artifacts)
        new_artifact_paths = [
            os.path.join(self.artifact_dir, f) for f in new_files
        ]

        return success, result_output, new_artifact_paths


# Singleton sandbox instance ready for node imports
default_sandbox = PythonREPLSandbox()