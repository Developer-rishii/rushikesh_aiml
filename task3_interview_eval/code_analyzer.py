"""
code_analyzer.py
================
Evaluates problem-solving quality by analyzing Python code statically and dynamically.
"""

import ast
import re
import subprocess
import textwrap
import tempfile
import os
import json
import warnings
from typing import Dict, Any, List

try:
    import radon.metrics
    import radon.complexity
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False


class CodeAnalyzer:
    """Analyzes Python code to score problem-solving quality."""
    
    def __init__(self) -> None:
        """Initialize the CodeAnalyzer."""
        pass
        
    def analyze(self, code: str, problem_description: str = "",
                test_cases: List[Dict[str, Any]] = None,
                time_limit_seconds: int = 5) -> Dict[str, Any]:
        """
        Analyze code statically and dynamically.
        
        Args:
            code (str): The Python code to evaluate.
            problem_description (str): Description of the problem.
            test_cases (list): Test cases to run against the code. 
                               Format: [{"input": "func(1)", "expected": 2}]
            time_limit_seconds (int): Timeout for subprocess.
            
        Returns:
            Dict[str, Any]: Correctness, complexity, style scores and signals.
        """
        signals: Dict[str, Any] = {}
        
        # 1. Syntax
        syntax_valid = self._check_syntax(code)
        signals["syntax_valid"] = syntax_valid
        
        # 2. Dynamic Testing
        test_results, runtime_error_flag = [], False
        if syntax_valid and test_cases:
            test_results, runtime_error_flag = self._run_tests(code, test_cases, time_limit_seconds)
        signals["test_results"] = test_results
        signals["runtime_errors"] = runtime_error_flag
        
        correctness_score = self._score_correctness(test_results, syntax_valid, runtime_error_flag)
        
        # 3. Complexity
        complexity_info = self._analyze_complexity(code) if syntax_valid else {"time": "Unknown", "space": "Unknown", "explanation": "Syntax error"}
        signals["time_complexity"] = complexity_info.get("time", "Unknown")
        signals["space_complexity"] = complexity_info.get("space", "Unknown")
        
        # 4. AST / Style metrics
        if syntax_valid:
            style_score, style_signals = self._score_style(code)
            signals.update(style_signals)
        else:
            style_score = 0.0
            signals.update({
                "cyclomatic_complexity": 0,
                "maintainability_index": 0,
                "has_docstring": False,
                "has_type_hints": False,
                "function_count": 0,
                "line_count": len(code.splitlines()),
                "pylint_score": 0.0
            })
            
        cyclomatic = signals.get("cyclomatic_complexity", 0)
        complexity_score = self._score_complexity(complexity_info, cyclomatic)
        
        # Compute total problem-solving score
        scores = {
            "correctness": correctness_score,
            "complexity": complexity_score,
            "style": style_score
        }
        
        ps_score = self._compute_problem_solving_score(scores)
        
        return {
            "correctness_score": round(correctness_score, 2),
            "complexity_score": round(complexity_score, 2),
            "style_score": round(style_score, 2),
            "problem_solving_score": round(ps_score, 2),
            "signals": signals
        }

    def _check_syntax(self, code: str) -> bool:
        """Check if code is syntactically valid."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _run_tests(self, code: str, test_cases: List[Dict[str, Any]], time_limit: int) -> tuple[List[Dict[str, Any]], bool]:
        """Run test cases safely using subprocess."""
        runner_script = textwrap.dedent("""
        import json
        import traceback
        import sys
        
        # --- USER CODE START ---
        {code}
        # --- USER CODE END ---
        
        test_cases = {test_cases_json}
        results = []
        has_runtime_error = False
        
        for tc in test_cases:
            tc_input = tc.get('input')
            expected = tc.get('expected')
            try:
                # Evaluate the input string (e.g. "two_sum([2,7], 9)")
                actual = eval(tc_input)
                passed = (actual == expected)
                results.append({
                    'input': tc_input,
                    'expected': expected,
                    'actual': actual,
                    'passed': passed
                })
            except Exception as e:
                has_runtime_error = True
                results.append({
                    'input': tc_input,
                    'expected': expected,
                    'actual': str(type(e).__name__) + ": " + str(e),
                    'passed': False
                })
                
        print("###RESULTS_START###")
        print(json.dumps({"results": results, "runtime_error": has_runtime_error}))
        print("###RESULTS_END###")
        """)
        
        script = runner_script.replace("{code}", code).replace("{test_cases_json}", json.dumps(test_cases))
        
        test_results = []
        runtime_error = False
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            temp_path = f.name
            
        try:
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=time_limit
            )
            
            output = result.stdout
            if "###RESULTS_START###" in output and "###RESULTS_END###" in output:
                json_str = output.split("###RESULTS_START###")[1].split("###RESULTS_END###")[0].strip()
                parsed = json.loads(json_str)
                test_results = parsed.get("results", [])
                runtime_error = parsed.get("runtime_error", False)
            else:
                runtime_error = True
                
        except subprocess.TimeoutExpired:
            runtime_error = True
        except Exception as e:
            runtime_error = True
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return test_results, runtime_error

    def _analyze_complexity(self, code: str) -> Dict[str, str]:
        """Estimate time/space complexity using AST traversal heuristics."""
        tree = ast.parse(code)
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.max_loop_depth = 0
                self.current_loop_depth = 0
                self.has_recursion = False
                self.has_sorting = False
                self.function_names = set()
                
            def visit_FunctionDef(self, node):
                self.function_names.add(node.name)
                self.generic_visit(node)
                
            def visit_For(self, node):
                self.current_loop_depth += 1
                self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
                self.generic_visit(node)
                self.current_loop_depth -= 1
                
            def visit_While(self, node):
                self.current_loop_depth += 1
                self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
                self.generic_visit(node)
                self.current_loop_depth -= 1
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.function_names:
                        self.has_recursion = True
                    if node.func.id == "sorted":
                        self.has_sorting = True
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr == "sort":
                        self.has_sorting = True
                self.generic_visit(node)

        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        time_comp = "O(1)"
        if visitor.has_recursion:
            time_comp = "O(log n) or O(2^n)"
        elif visitor.max_loop_depth >= 3:
            time_comp = "O(n³)"
        elif visitor.max_loop_depth == 2:
            time_comp = "O(n²)"
        elif visitor.max_loop_depth == 1:
            time_comp = "O(n)"
            
        if visitor.has_sorting and time_comp in ["O(1)", "O(n)"]:
            time_comp = "O(n log n)"
            
        return {
            "time": time_comp,
            "space": "O(n)",  # Estimating space statically is tough; defaulting to O(n)
            "explanation": f"Max loop depth: {visitor.max_loop_depth}, Recursion: {visitor.has_recursion}, Sorts: {visitor.has_sorting}"
        }

    def _score_correctness(self, test_results: List[Dict[str, Any]], syntax_valid: bool, runtime_error: bool) -> float:
        """Calculate correctness score based on tests."""
        if not syntax_valid:
            return 0.0
            
        if not test_results:
            # If there are no test results but it didn't crash
            return 10.0 if runtime_error else 50.0
            
        passed = sum(1 for tr in test_results if tr.get("passed"))
        total = len(test_results)
        
        score = (passed / total) * 100
        if runtime_error and passed == 0:
            return 10.0
        return score

    def _score_complexity(self, complexity_info: Dict[str, str], cyclomatic: int) -> float:
        """Map time complexity string to a numeric score."""
        tc = complexity_info.get("time", "")
        if tc in ["O(1)", "O(log n)"]:
            score = 100.0
        elif tc in ["O(n)", "O(n log n)"]:
            score = 85.0
        elif tc == "O(n²)":
            score = 60.0
        elif tc == "O(n³)":
            score = 30.0
        else:
            score = 50.0  # Fallback for recursion or unknown
            
        if cyclomatic > 10:
            score -= 10
            
        return max(0.0, score)

    def _score_style(self, code: str) -> tuple[float, Dict[str, Any]]:
        """Score code style using pylint and radon."""
        has_docstring = False
        has_type_hints = False
        function_count = 0
        cyclomatic = 0
        mi = 0.0
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_count += 1
                    if ast.get_docstring(node):
                        has_docstring = True
                    if node.returns or any(arg.annotation for arg in node.args.args):
                        has_type_hints = True
        except Exception:
            pass
            
        line_count = len(code.splitlines())
        
        if RADON_AVAILABLE:
            try:
                mi = radon.metrics.mi_visit(code, multi=True)
                # Cyclomatic: calculate avg or max
                blocks = radon.complexity.cc_visit(code)
                if blocks:
                    cyclomatic = max(b.complexity for b in blocks)
            except Exception:
                pass
                
        # Pylint
        pylint_score = 0.0
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
            
        try:
            result = subprocess.run(
                ["pylint", "--score=y", "--reports=n", temp_path],
                capture_output=True, text=True
            )
            # Parse pylint output: "Your code has been rated at X.XX/10"
            match = re.search(r"Your code has been rated at ([-0-9\.]+)/10", result.stdout)
            if match:
                pylint_score = max(0.0, float(match.group(1)))
        except Exception:
            pass
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        # Calculate style components
        score = 0.0
        if has_docstring:
            score += 25.0
        if has_type_hints:
            score += 20.0
            
        score += (pylint_score / 10.0) * 30.0
        
        if line_count < 50:
            score += 15.0
        elif 50 <= line_count <= 100:
            score += 10.0
        else:
            score += 5.0
            
        mi_normalized = max(0.0, min(100.0, mi))
        score += (mi_normalized / 100.0) * 10.0
        
        signals = {
            "cyclomatic_complexity": cyclomatic,
            "maintainability_index": round(mi, 2),
            "has_docstring": has_docstring,
            "has_type_hints": has_type_hints,
            "function_count": function_count,
            "line_count": line_count,
            "pylint_score": pylint_score
        }
        return min(100.0, score), signals

    def _compute_problem_solving_score(self, scores: Dict[str, float]) -> float:
        """Compute the weighted problem solving score."""
        weights = {
            "correctness": 0.45,
            "complexity": 0.35,
            "style": 0.20
        }
        total_score = sum(scores[k] * weights[k] for k in weights)
        return max(0.0, min(100.0, total_score))
