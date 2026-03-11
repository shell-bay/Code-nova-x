"""
Coding Nova X — Evaluation Framework
=======================================
Measures model performance across multiple dimensions:

1. Perplexity — how well model predicts text (lower = better)
2. Code generation accuracy — does it write correct code?
3. Functional correctness — does generated code actually run?
4. Task benchmarks — standard coding questions
"""

import os
import sys
import json
import math
import subprocess
import tempfile
import time
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Benchmark Tasks ───────────────────────────────────────────────────────────

BENCHMARK_TASKS = [
    {
        "id": "hello_world",
        "instruction": "Write a Python function that prints 'Hello, World!'",
        "test_code": """
result = hello_world()
assert result is None or isinstance(result, str), "Should print or return a string"
print("PASS: hello_world")
""",
        "expected_contains": ["def hello_world", "Hello"],
    },
    {
        "id": "fibonacci",
        "instruction": "Write a Python function fibonacci(n) that returns the nth Fibonacci number.",
        "test_code": """
assert fibonacci(0) == 0 or fibonacci(1) == 1, "fib(0)=0, fib(1)=1"
assert fibonacci(5) == 5, f"fib(5) should be 5, got {fibonacci(5)}"
assert fibonacci(10) == 55, f"fib(10) should be 55, got {fibonacci(10)}"
print("PASS: fibonacci")
""",
        "expected_contains": ["def fibonacci", "return"],
    },
    {
        "id": "reverse_string",
        "instruction": "Write a Python function reverse_string(s) that returns the reversed string.",
        "test_code": """
assert reverse_string("hello") == "olleh", "Should reverse hello"
assert reverse_string("") == "", "Should handle empty string"
assert reverse_string("a") == "a", "Should handle single char"
print("PASS: reverse_string")
""",
        "expected_contains": ["def reverse_string", "return"],
    },
    {
        "id": "is_palindrome",
        "instruction": "Write a Python function is_palindrome(s) that returns True if the string is a palindrome.",
        "test_code": """
assert is_palindrome("racecar") == True
assert is_palindrome("hello") == False
assert is_palindrome("a") == True
assert is_palindrome("") == True
print("PASS: is_palindrome")
""",
        "expected_contains": ["def is_palindrome", "return"],
    },
    {
        "id": "list_sum",
        "instruction": "Write a Python function list_sum(numbers) that returns the sum of all numbers in a list.",
        "test_code": """
assert list_sum([1, 2, 3]) == 6
assert list_sum([]) == 0
assert list_sum([-1, 1]) == 0
print("PASS: list_sum")
""",
        "expected_contains": ["def list_sum", "return"],
    },
]


class CodeExecutor:
    """
    Safely execute generated Python code in a subprocess.
    
    Uses subprocess with timeout for safety.
    Never exec() code directly — too dangerous!
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def execute(self, code: str, test_code: str = "") -> Dict:
        """
        Execute code + test_code and return result.
        
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str,
                "runtime_ms": float,
            }
        """
        full_code = code + "\n" + test_code
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        ) as f:
            f.write(full_code)
            tmp_path = f.name
        
        try:
            t0 = time.time()
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            runtime_ms = (time.time() - t0) * 1000
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "runtime_ms": runtime_ms,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Timeout after {self.timeout}s",
                "runtime_ms": self.timeout * 1000,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "runtime_ms": 0,
            }
        finally:
            os.unlink(tmp_path)


class NovaEvaluator:
    """Complete evaluation suite for Coding Nova X."""

    def __init__(self, model, tokenizer, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.executor = CodeExecutor(timeout=10)
        self.model.eval()

    @torch.no_grad()
    def compute_perplexity(self, texts: List[str], max_length: int = 512) -> float:
        """
        Compute perplexity on a set of texts.
        
        Perplexity = exp(average_loss)
        Lower perplexity = model is less "surprised" by the text
        A perfect model would have PPL = 1.0 (always predicts correctly)
        Random model with 32k vocab has PPL ≈ 32000
        Good LLMs achieve PPL < 10 on code
        """
        total_loss = 0.0
        total_tokens = 0
        
        for text in texts:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True, max_length=max_length)
            if len(ids) < 2:
                continue
            
            input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            
            outputs = self.model(input_tensor, labels=input_tensor)
            loss = outputs["loss"]
            
            num_tokens = len(ids) - 1  # Predict N-1 tokens
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))  # Cap at e^20 ≈ 485M
        return perplexity

    def generate_code(
        self,
        instruction: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,   # Lower temp for more deterministic code
        **kwargs,
    ) -> str:
        """Generate code for an instruction."""
        prompt = self.tokenizer.format_instruction(
            instruction=instruction,
            language="python",
        )
        ids = self.tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        
        output_ids = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_id,
            **kwargs,
        )
        
        # Only decode the new tokens
        new_ids = output_ids[0][len(ids):].tolist()
        return self.tokenizer.decode(new_ids, skip_special=True)

    def check_contains_function(self, code: str, expected: List[str]) -> bool:
        """Check if generated code contains expected elements."""
        code_lower = code.lower()
        return all(e.lower() in code_lower for e in expected)

    def run_benchmark(self, tasks: Optional[List[Dict]] = None) -> Dict:
        """
        Run all benchmark tasks and return results.
        
        For each task:
        1. Generate code with the model
        2. Check it contains expected elements
        3. Execute it with test cases
        4. Record pass/fail
        """
        if tasks is None:
            tasks = BENCHMARK_TASKS
        
        results = []
        passed = 0
        
        print("\n" + "=" * 60)
        print("  Coding Nova X — Benchmark Evaluation")
        print("=" * 60)
        
        for task in tasks:
            print(f"\n[Task: {task['id']}]")
            print(f"Instruction: {task['instruction'][:60]}...")
            
            # Generate code
            generated = self.generate_code(task['instruction'])
            print(f"Generated:\n{generated[:200]}")
            
            # Check expected content
            has_expected = self.check_contains_function(
                generated, task.get("expected_contains", [])
            )
            
            # Execute with tests
            exec_result = self.executor.execute(
                generated,
                task.get("test_code", ""),
            )
            
            task_passed = exec_result["success"] and has_expected
            if task_passed:
                passed += 1
                print(f"✓ PASSED ({exec_result['runtime_ms']:.1f}ms)")
            else:
                print(f"✗ FAILED")
                if exec_result["error"]:
                    print(f"  Error: {exec_result['error'][:100]}")
            
            results.append({
                "task_id": task["id"],
                "instruction": task["instruction"],
                "generated_code": generated,
                "has_expected_content": has_expected,
                "execution_success": exec_result["success"],
                "passed": task_passed,
                "output": exec_result["output"],
                "error": exec_result["error"],
                "runtime_ms": exec_result["runtime_ms"],
            })
        
        # Summary
        accuracy = passed / max(len(tasks), 1)
        print("\n" + "=" * 60)
        print(f"  Benchmark Results: {passed}/{len(tasks)} passed ({accuracy*100:.1f}%)")
        print("=" * 60 + "\n")
        
        return {
            "tasks_passed": passed,
            "tasks_total": len(tasks),
            "accuracy": accuracy,
            "results": results,
        }

    def full_evaluation(self, eval_texts: Optional[List[str]] = None) -> Dict:
        """Run complete evaluation suite."""
        
        print("\n[Evaluator] Starting full evaluation...")
        
        # 1. Perplexity
        if eval_texts is None:
            eval_texts = [
                "def hello_world():\n    print('Hello, World!')\n",
                "for i in range(10):\n    print(i)\n",
                "class MyClass:\n    def __init__(self):\n        self.x = 0\n",
            ]
        
        print("[Evaluator] Computing perplexity...")
        ppl = self.compute_perplexity(eval_texts)
        print(f"[Evaluator] Perplexity: {ppl:.2f}")
        
        # 2. Code benchmarks
        benchmark_results = self.run_benchmark()
        
        # 3. Summary report
        report = {
            "perplexity": ppl,
            "benchmark_accuracy": benchmark_results["accuracy"],
            "benchmark_passed": benchmark_results["tasks_passed"],
            "benchmark_total": benchmark_results["tasks_total"],
            "detailed_results": benchmark_results["results"],
        }
        
        # Save report
        os.makedirs("logs", exist_ok=True)
        with open("logs/evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("[Evaluator] Report saved to logs/evaluation_report.json")
        
        return report


if __name__ == "__main__":
    print("Coding Nova X Evaluator — import and call full_evaluation()")
