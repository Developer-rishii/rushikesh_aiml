"""
test_all.py
===========
Comprehensive unit tests for the AI Interview Evaluation System.
"""

import unittest
import sys
import os
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from speech_analyzer import SpeechAnalyzer
from code_analyzer import CodeAnalyzer
from interview_evaluator import InterviewEvaluator
from feedback_generator import FeedbackGenerator

class TestSpeechAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = SpeechAnalyzer()

    def test_analyze_returns_all_keys(self):
        result = self.analyzer.analyze("Hello world.")
        expected_keys = {
            "fluency_score", "clarity_score", "vocabulary_score",
            "confidence_score", "grammar_score", "communication_score", "signals"
        }
        for key in expected_keys:
            self.assertIn(key, result)

    def test_fluency_penalizes_fillers(self):
        clean_text = "I will solve this using a binary search tree."
        filler_text = "Um, I will, like, solve this using a, basically, binary search tree. Uh, yeah."
        
        clean_res = self.analyzer.analyze(clean_text)
        filler_res = self.analyzer.analyze(filler_text)
        
        self.assertLess(filler_res["fluency_score"], clean_res["fluency_score"])
        self.assertGreater(filler_res["signals"]["filler_word_count"], 0)

    def test_grammar_score_range(self):
        result = self.analyzer.analyze("He go to the store yesterday.")
        score = result["grammar_score"]
        self.assertTrue(0.0 <= score <= 100.0)

    def test_confidence_score_range(self):
        result = self.analyzer.analyze("I am absolutely certain that this is the correct optimal solution.")
        score = result["confidence_score"]
        self.assertTrue(0.0 <= score <= 100.0)

    def test_empty_transcript_handled(self):
        result = self.analyzer.analyze("")
        self.assertEqual(result["communication_score"], 0.0)
        self.assertEqual(result["signals"]["sentence_count"], 0)


class TestCodeAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = CodeAnalyzer()

    def test_syntax_error_returns_zero_correctness(self):
        bad_code = "def foo():\\n    return 1 + "
        bad_code = bad_code.replace('\\n', '\n')
        result = self.analyzer.analyze(bad_code)
        self.assertEqual(result["correctness_score"], 0.0)
        self.assertFalse(result["signals"]["syntax_valid"])

    def test_correct_solution_passes_tests(self):
        code = (
            "def two_sum(nums, target):\\n"
            "    for i in range(len(nums)):\\n"
            "        for j in range(i + 1, len(nums)):\\n"
            "            if nums[i] + nums[j] == target:\\n"
            "                return [i, j]\\n"
            "    return []\\n"
        ).replace('\\n', '\n')
        
        tests = [{"input": "two_sum([2,7,11,15], 9)", "expected": [0, 1]}]
        result = self.analyzer.analyze(code, test_cases=tests)
        
        self.assertEqual(result["correctness_score"], 100.0)
        self.assertTrue(all(tr["passed"] for tr in result["signals"]["test_results"]))

    def test_complexity_detection_nested_loops(self):
        code = (
            "def test():\\n"
            "    for i in range(10):\\n"
            "        for j in range(10):\\n"
            "            pass"
        ).replace('\\n', '\n')
        result = self.analyzer.analyze(code)
        self.assertEqual(result["signals"]["time_complexity"], "O(n²)")

    def test_complexity_detection_single_loop(self):
        code = (
            "def test():\\n"
            "    for i in range(10):\\n"
            "        pass"
        ).replace('\\n', '\n')
        result = self.analyzer.analyze(code)
        self.assertEqual(result["signals"]["time_complexity"], "O(n)")

    def test_style_score_with_docstring(self):
        code_no_doc = "def f(x):\n    return x"
        code_with_doc = 'def f(x):\n    """A func."""\n    return x'
        
        res_no = self.analyzer.analyze(code_no_doc)
        res_with = self.analyzer.analyze(code_with_doc)
        
        self.assertGreater(res_with["style_score"], res_no["style_score"])


class TestInterviewEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = InterviewEvaluator()

    def test_total_score_range(self):
        res = self.evaluator.evaluate("Hello.", "def f(): pass", time_taken_minutes=20.0)
        self.assertTrue(0.0 <= res["total_score"] <= 100.0)

    def test_weights_sum_to_one(self):
        total_weight = (
            self.evaluator.COMMUNICATION_WEIGHT +
            self.evaluator.PROBLEM_SOLVING_WEIGHT +
            self.evaluator.TIME_MANAGEMENT_WEIGHT
        )
        self.assertAlmostEqual(total_weight, 1.0)

    def test_grade_assignment(self):
        self.assertEqual(self.evaluator._assign_grade(92), "A")
        self.assertEqual(self.evaluator._assign_grade(76), "B")
        self.assertEqual(self.evaluator._assign_grade(61), "C")
        self.assertEqual(self.evaluator._assign_grade(46), "D")
        self.assertEqual(self.evaluator._assign_grade(30), "F")

    def test_passed_threshold(self):
        # We can just test the evaluate logic indirectly, but _assign_grade logic holds
        res_pass = self.evaluator.evaluate("Great answer. Perfect.", "def f(): pass", time_taken_minutes=20.0)
        res_fail = self.evaluator.evaluate("", "def f():", time_taken_minutes=90.0)
        # Even empty might pass if code scores slightly, but with syntax error and empty text, it should fail
        self.assertFalse(res_fail["passed"])

    def test_time_management_scoring(self):
        res = self.evaluator._score_time_management(27.0, 45.0)
        # 27 / 45 = 0.60
        self.assertEqual(res["score"], 100.0)


class TestFeedbackGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = FeedbackGenerator()
        cls.mock_eval = {
            "breakdown": {
                "communication": {"score": 50, "signals": {"filler_word_count": 10}, "sub_scores": {}},
                "problem_solving": {"score": 90, "signals": {"time_complexity": "O(n²)"}, "sub_scores": {}},
                "time_management": {"score": 40, "signals": {"time_ratio": 1.15}}
            }
        }

    def test_feedback_has_all_sections(self):
        res = self.generator.generate(self.mock_eval)
        expected = ["overall_summary", "strengths", "improvements", "category_feedback", "priority_focus", "study_plan", "encouragement"]
        for key in expected:
            self.assertIn(key, res)

    def test_strengths_and_improvements_populated(self):
        res = self.generator.generate(self.mock_eval)
        self.assertGreaterEqual(len(res["strengths"]), 2)
        self.assertGreaterEqual(len(res["improvements"]), 3)

    def test_study_plan_has_five_items(self):
        res = self.generator.generate(self.mock_eval)
        self.assertEqual(len(res["study_plan"]), 5)


if __name__ == '__main__':
    unittest.main()
