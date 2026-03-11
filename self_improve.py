"""
Coding Nova X — Self-Improving Loop
=====================================
The model generates code, runs it, detects errors,
and iteratively fixes them — without human intervention.

Pipeline:
1. Receive coding task
2. Generate initial code solution
3. Execute the code safely
4. If errors → analyze error + regenerate improved code
5. Repeat up to max_attempts times
6. Return best working solution

This is how models like AlphaCode and Codex improve:
try → fail → learn from failure → retry.
"""

import os
import sys
import re
import json
import time
import subprocess
import tempfile
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@dataclass
class ExecutionResult:
    """Result of running generated code."""
    success: bool
    stdout: str
    stderr: str
    runtime_ms: float
    exit_code: int
    
    @property
    def error_type(self) -> Optional[str]:
        """Extract error type from stderr."""
        if not self.stderr:
            return None
        for line in self.stderr.split('\n'):
            if 'Error' in line or 'Exception' in line:
                # Extract e.g. "SyntaxError", "IndexError", etc.
                match = re.search(r'(\w+Error|\w+Exception)', line)
                if match:
                    return match.group(1)
        return "RuntimeError"


@dataclass
class ImprovementSession:
    """Track one full self-improvement session."""
    task: str
    attempts: List[Dict] = field(default_factory=list)
    final_code: str = ""
    success: bool = False
    total_attempts: int = 0
    
    def add_attempt(self, code: str, result: ExecutionResult, error_analysis: str = ""):
        self.attempts.append({
            "attempt_num": len(self.attempts) + 1,
            "code": code,
            "success": result.success,
            "error_type": result.error_type,
            "stderr": result.stderr[:500],
            "stdout": result.stdout[:200],
            "error_analysis": error_analysis,
        })
        self.total_attempts = len(self.attempts)
    
    def to_dict(self) -> Dict:
        return {
            "task": self.task,
            "total_attempts": self.total_attempts,
            "success": self.success,
            "final_code": self.final_code,
            "attempts": self.attempts,
        }


class SafeCodeExecutor:
    """
    Execute code safely with timeout and resource limits.
    Never executes code in the same process — always uses subprocess.
    """
    
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
    
    def run(self, code: str, test_input: str = "") -> ExecutionResult:
        """Execute Python code safely."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        ) as f:
            f.write(code)
            tmp = f.name
        
        try:
            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, tmp],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            runtime = (time.time() - t0) * 1000
            
            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                runtime_ms=runtime,
                exit_code=proc.returncode,
            )
        
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"TimeoutError: Code took more than {self.timeout}s",
                runtime_ms=self.timeout * 1000,
                exit_code=-1,
            )
        except Exception as e:
            return ExecutionResult(
                success=False, stdout="", stderr=str(e),
                runtime_ms=0, exit_code=-1,
            )
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass
    
    def run_with_tests(self, code: str, tests: str) -> ExecutionResult:
        """Run code + test assertions."""
        return self.run(code + "\n\n# === Tests ===\n" + tests)


class ErrorAnalyzer:
    """
    Analyze code errors and generate debugging hints.
    
    Uses pattern matching + a second LLM call to understand
    what went wrong and how to fix it.
    """
    
    ERROR_HINTS = {
        "SyntaxError": [
            "Check for missing colons after if/for/def/class",
            "Check for mismatched parentheses or brackets",
            "Check for incorrect indentation",
            "Ensure strings are properly quoted",
        ],
        "IndentationError": [
            "Use consistent indentation (4 spaces recommended)",
            "Don't mix tabs and spaces",
            "Check that all code blocks are properly indented",
        ],
        "NameError": [
            "Variable may not be defined before use",
            "Check spelling of variable/function names",
            "Import required modules",
        ],
        "IndexError": [
            "Array index is out of bounds",
            "Check list length before indexing",
            "Use len(lst) - 1 for last element, not len(lst)",
        ],
        "TypeError": [
            "Check argument types match function signature",
            "Don't mix incompatible types (e.g., int + string)",
            "Check for None values where objects are expected",
        ],
        "AttributeError": [
            "Object doesn't have this attribute/method",
            "Check for typos in method names",
            "Import the right class/module",
        ],
        "ValueError": [
            "Function received argument with right type but wrong value",
            "Check range of acceptable values",
        ],
        "ZeroDivisionError": [
            "Check for division by zero",
            "Add a guard: if divisor != 0",
        ],
        "ImportError": [
            "Module not installed — pip install <module_name>",
            "Check module name spelling",
        ],
        "RecursionError": [
            "Recursion depth exceeded — check base case",
            "Consider converting to iterative solution",
        ],
        "TimeoutError": [
            "Code is taking too long — likely infinite loop",
            "Check loop termination conditions",
            "Consider a more efficient algorithm",
        ],
    }
    
    def analyze(self, error_text: str, code: str) -> str:
        """
        Analyze error and return actionable feedback.
        """
        error_type = "RuntimeError"
        error_line = ""
        error_msg = ""
        
        # Extract error information
        for line in error_text.split('\n'):
            match = re.search(r'(\w+Error|\w+Exception): (.+)', line)
            if match:
                error_type = match.group(1)
                error_msg = match.group(2)
            
            line_match = re.search(r'line (\d+)', line)
            if line_match:
                line_num = int(line_match.group(1))
                code_lines = code.split('\n')
                if 0 < line_num <= len(code_lines):
                    error_line = f"Line {line_num}: {code_lines[line_num-1].strip()}"
        
        # Get hints for this error type
        hints = self.ERROR_HINTS.get(error_type, ["Carefully review the error message and stack trace"])
        
        analysis = f"Error Type: {error_type}\n"
        if error_msg:
            analysis += f"Message: {error_msg}\n"
        if error_line:
            analysis += f"Location: {error_line}\n"
        analysis += f"\nDebugging hints:\n"
        for hint in hints[:3]:
            analysis += f"  - {hint}\n"
        
        return analysis


class SelfImprovingCoder:
    """
    The self-improving code generation system.
    
    Uses the inference engine to generate code, execute it,
    analyze errors, and iteratively improve until it works.
    """
    
    def __init__(
        self,
        engine,                  # NovaInferenceEngine
        max_attempts: int = 5,
        timeout: int = 15,
        save_sessions: bool = True,
        sessions_dir: str = "logs/sessions",
    ):
        self.engine = engine
        self.max_attempts = max_attempts
        self.executor = SafeCodeExecutor(timeout=timeout)
        self.analyzer = ErrorAnalyzer()
        self.save_sessions = save_sessions
        self.sessions_dir = sessions_dir
        
        if save_sessions:
            os.makedirs(sessions_dir, exist_ok=True)

    def solve(
        self,
        task: str,
        test_code: str = "",
        verbose: bool = True,
    ) -> ImprovementSession:
        """
        Attempt to solve a coding task with self-improvement.
        
        Args:
            task: The programming task description
            test_code: Optional test assertions to verify correctness
            verbose: Print progress
        
        Returns:
            ImprovementSession with all attempts and final solution
        """
        session = ImprovementSession(task=task)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🤖 Self-Improving Coder — Task:")
            print(f"   {task[:80]}")
            print(f"{'='*60}")
        
        context = ""  # Accumulate error context across attempts
        
        for attempt_num in range(1, self.max_attempts + 1):
            if verbose:
                print(f"\n[Attempt {attempt_num}/{self.max_attempts}]")
            
            # ── Generate code ─────────────────────────────────────────────
            prompt = self._build_prompt(task, context, attempt_num)
            
            try:
                generated = self.engine.generate(prompt)
            except Exception as e:
                if verbose:
                    print(f"  Generation error: {e}")
                continue
            
            # Extract Python code from response
            code = self._extract_code(generated, task)
            
            if verbose:
                print(f"  Generated code ({len(code)} chars):")
                print("  " + code[:200].replace('\n', '\n  '))
            
            # ── Execute code ──────────────────────────────────────────────
            if test_code:
                result = self.executor.run_with_tests(code, test_code)
            else:
                result = self.executor.run(code)
            
            # ── Analyze result ────────────────────────────────────────────
            error_analysis = ""
            if not result.success:
                error_analysis = self.analyzer.analyze(result.stderr, code)
                if verbose:
                    print(f"  ✗ Failed: {result.error_type}")
                    print(f"  {error_analysis[:150]}")
                
                # Build context for next attempt
                context += f"\nAttempt {attempt_num} failed:\n"
                context += f"Code:\n{code[:300]}\n"
                context += f"Error: {result.stderr[:200]}\n"
                context += f"Analysis: {error_analysis[:200]}\n"
            else:
                if verbose:
                    print(f"  ✓ SUCCESS! ({result.runtime_ms:.1f}ms)")
                    if result.stdout:
                        print(f"  Output: {result.stdout[:100]}")
                
                session.final_code = code
                session.success = True
                session.add_attempt(code, result, error_analysis)
                break
            
            session.add_attempt(code, result, error_analysis)
        
        if not session.success and session.attempts:
            # Return the last attempt even if it failed
            session.final_code = session.attempts[-1]["code"]
        
        # Summary
        if verbose:
            status = "✅ SOLVED" if session.success else "❌ UNSOLVED"
            print(f"\n{status} after {session.total_attempts} attempts")
        
        # Save session
        if self.save_sessions:
            self._save_session(session)
        
        return session

    def _build_prompt(self, task: str, error_context: str, attempt: int) -> str:
        """Build the generation prompt, incorporating previous errors."""
        if attempt == 1 or not error_context:
            return (
                f"Write a complete, working Python solution for this task:\n\n"
                f"{task}\n\n"
                f"Requirements:\n"
                f"- Write only the Python code, no explanations\n"
                f"- Include all necessary imports\n"
                f"- Code must be complete and runnable\n"
                f"- Handle edge cases"
            )
        else:
            return (
                f"Fix the Python code for this task:\n\n{task}\n\n"
                f"Previous attempts had these errors:\n{error_context}\n\n"
                f"Write a corrected, complete solution that avoids all previous errors."
            )

    def _extract_code(self, text: str, task: str) -> str:
        """Extract Python code from generated text."""
        # Try markdown code blocks first
        pattern = r"```(?:python)?\n?(.*?)```"
        blocks = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if blocks:
            return blocks[0].strip()
        
        # Try to find function definitions
        lines = text.split('\n')
        code_lines = []
        in_code = False
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('def ', 'class ', 'import ', 'from ')):
                in_code = True
            
            if in_code:
                # Stop if we hit non-code text
                if stripped and not stripped.startswith('#') and \
                   not any(c in stripped for c in ['(', ')', '=', ':', '[', ']', '+', '-', '*', '/', 'return', 'if', 'for', 'while', 'else', 'elif', 'try', 'except', 'True', 'False', 'None']):
                    if not stripped.startswith((' ', '\t')) and code_lines:
                        break
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # Fallback: return everything that looks like code
        return text.strip()

    def _save_session(self, session: ImprovementSession):
        """Save session to disk for analysis."""
        filename = f"session_{int(time.time())}.json"
        path = os.path.join(self.sessions_dir, filename)
        
        with open(path, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)

    def batch_solve(self, tasks: List[Dict], verbose: bool = True) -> List[ImprovementSession]:
        """Solve multiple tasks and return all sessions."""
        sessions = []
        passed = 0
        
        print(f"\n[Batch] Solving {len(tasks)} tasks...")
        
        for i, task_info in enumerate(tasks):
            task = task_info.get("task", task_info.get("instruction", ""))
            tests = task_info.get("test_code", "")
            
            print(f"\n[{i+1}/{len(tasks)}] {task[:50]}...")
            session = self.solve(task, tests, verbose=verbose)
            sessions.append(session)
            
            if session.success:
                passed += 1
        
        accuracy = passed / max(len(tasks), 1)
        print(f"\n{'='*60}")
        print(f"Batch complete: {passed}/{len(tasks)} solved ({accuracy*100:.1f}%)")
        print(f"{'='*60}")
        
        return sessions


# ── Built-in test tasks for self-improvement ──────────────────────────────────

SELF_IMPROVEMENT_TASKS = [
    {
        "task": "Write a Python function two_sum(nums, target) that returns indices of two numbers that add up to target.",
        "test_code": """
assert two_sum([2, 7, 11, 15], 9) == [0, 1]
assert two_sum([3, 2, 4], 6) == [1, 2]
print("PASS: two_sum")
"""
    },
    {
        "task": "Write a Python function is_prime(n) that returns True if n is prime, False otherwise.",
        "test_code": """
assert is_prime(2) == True
assert is_prime(3) == True
assert is_prime(4) == False
assert is_prime(17) == True
assert is_prime(1) == False
print("PASS: is_prime")
"""
    },
    {
        "task": "Write a Python function flatten(nested_list) that flattens a nested list to one dimension.",
        "test_code": """
assert flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]
assert flatten([]) == []
assert flatten([1, 2, 3]) == [1, 2, 3]
print("PASS: flatten")
"""
    },
]


if __name__ == "__main__":
    print("Coding Nova X — Self-Improvement System")
    print("Import SelfImprovingCoder and pass your inference engine.")
    print(f"Built-in test tasks: {len(SELF_IMPROVEMENT_TASKS)}")
