"""
interview_evaluator.py
======================
Orchestrates the interview evaluation by integrating speech, code, and time analysis.
"""

import json
from datetime import datetime
from typing import Dict, Any, List

from speech_analyzer import SpeechAnalyzer
from code_analyzer import CodeAnalyzer
from feedback_generator import FeedbackGenerator

class InterviewEvaluator:
    """Evaluates an entire interview across multiple dimensions."""
    
    COMMUNICATION_WEIGHT = 0.40
    PROBLEM_SOLVING_WEIGHT = 0.40
    TIME_MANAGEMENT_WEIGHT = 0.20
    
    def __init__(self) -> None:
        """Initialize all sub-analyzers and generators."""
        print("Initializing Interview Evaluator... (loading models)")
        self.speech_analyzer = SpeechAnalyzer()
        self.code_analyzer = CodeAnalyzer()
        self.feedback_generator = FeedbackGenerator()
        print("Initialization complete.")

    def evaluate(self, transcript: str, code_solution: str,
                 problem_description: str = "", test_cases: List[Dict[str, Any]] = None,
                 time_taken_minutes: float = None, time_limit_minutes: float = 45.0) -> Dict[str, Any]:
        """
        Evaluate the interview and return a comprehensive scoring report.
        """
        # 1. Communication Score
        comm_results = self.speech_analyzer.analyze(transcript)
        comm_score = comm_results.get("communication_score", 0.0)
        
        # 2. Problem Solving Score
        ps_results = self.code_analyzer.analyze(
            code_solution, problem_description, test_cases, time_limit_seconds=5
        )
        ps_score = ps_results.get("problem_solving_score", 0.0)
        
        # 3. Time Management Score
        time_results = self._score_time_management(time_taken_minutes, time_limit_minutes)
        time_score = time_results.get("score", 0.0)
        
        # Calculate Total Score
        total_score = (
            (comm_score * self.COMMUNICATION_WEIGHT) +
            (ps_score * self.PROBLEM_SOLVING_WEIGHT) +
            (time_score * self.TIME_MANAGEMENT_WEIGHT)
        )
        
        grade = self._assign_grade(total_score)
        passed = total_score >= 60.0
        
        # Build Breakdown
        breakdown = {
            "communication": {
                "score": comm_score,
                "weight": self.COMMUNICATION_WEIGHT,
                "weighted_score": comm_score * self.COMMUNICATION_WEIGHT,
                "signals": comm_results.get("signals"),
                "sub_scores": {k: v for k, v in comm_results.items() if k != "signals"}
            },
            "problem_solving": {
                "score": ps_score,
                "weight": self.PROBLEM_SOLVING_WEIGHT,
                "weighted_score": ps_score * self.PROBLEM_SOLVING_WEIGHT,
                "signals": ps_results.get("signals"),
                "sub_scores": {k: v for k, v in ps_results.items() if k != "signals"}
            },
            "time_management": {
                "score": time_score,
                "weight": self.TIME_MANAGEMENT_WEIGHT,
                "weighted_score": time_score * self.TIME_MANAGEMENT_WEIGHT,
                "signals": time_results
            }
        }
        
        # Create full evaluation object for feedback generation
        eval_results = {
            "total_score": round(total_score, 2),
            "grade": grade,
            "passed": passed,
            "breakdown": breakdown,
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate Feedback & Summary
        feedback = self.feedback_generator.generate(eval_results)
        summary = self._generate_summary(total_score, grade, breakdown)
        
        eval_results["feedback"] = feedback
        eval_results["summary"] = summary
        
        return eval_results

    def _score_time_management(self, time_taken: float, time_limit: float) -> Dict[str, Any]:
        """Score time management based on ratio."""
        if time_taken is None:
            return {"score": 50.0, "time_ratio": None, "time_taken": None, "time_limit": time_limit, "verdict": "No data"}
            
        time_ratio = time_taken / time_limit
        
        if time_ratio <= 0.60:
            score, verdict = 100.0, "Finished early, excellent pace."
        elif 0.60 < time_ratio <= 0.85:
            score, verdict = 90.0, "Good pace."
        elif 0.85 < time_ratio <= 1.00:
            score, verdict = 75.0, "Used full time."
        elif 1.00 < time_ratio <= 1.20:
            score, verdict = 40.0, "Slightly over time limit."
        else:
            score, verdict = 10.0, "Significantly over time limit."
            
        return {
            "score": score,
            "time_ratio": round(time_ratio, 2),
            "time_taken": time_taken,
            "time_limit": time_limit,
            "verdict": verdict
        }

    def _assign_grade(self, score: float) -> str:
        """Assign letter grade."""
        if score >= 90: return "A"
        if score >= 75: return "B"
        if score >= 60: return "C"
        if score >= 45: return "D"
        return "F"

    def _generate_summary(self, total_score: float, grade: str, breakdown: Dict[str, Any]) -> str:
        """Generate a 2-3 sentence summary based on scores."""
        comm_score = breakdown["communication"]["score"]
        ps_score = breakdown["problem_solving"]["score"]
        
        weakest = "communication"
        weakest_val = comm_score
        
        if ps_score < weakest_val:
            weakest = "problem solving"
            weakest_val = ps_score
            
        if grade in ["A", "B"]:
            perf = "Strong"
        elif grade == "C":
            perf = "Adequate"
        else:
            perf = "Weak"
            
        summary = f"{perf} technical performance ({total_score:.1f}/100) overall. "
        
        if weakest == "communication":
            summary += f"Communication skills ({comm_score:.1f}/100) need improvement — focus on reducing filler words and structuring responses more clearly."
        else:
            summary += f"Problem-solving skills ({ps_score:.1f}/100) need improvement — focus on algorithm optimization and writing clean code."
            
        return summary


if __name__ == "__main__":
    sample_transcript = (
        "Um, first I will use a dictionary to store the values. "
        "Basically, we can loop through the array and check if the complement is in the dictionary. "
        "If it is, like, we just return the indices. "
        "The time complexity is O(n) because we iterate once. "
        "Uh, I think maybe the space complexity is also O(n). "
        "Then, finally we handle the edge cases where the list is empty. "
        "This algorithm is very efficient for large datasets. "
        "Actually, that is my whole solution."
    )
    
    sample_code = (
        "def two_sum(nums, target):\\n"
        "    for i in range(len(nums)):\\n"
        "        for j in range(i + 1, len(nums)):\\n"
        "            if nums[i] + nums[j] == target:\\n"
        "                return [i, j]\\n"
        "    return []\\n"
    ).replace('\\n', '\n')
    
    sample_tests = [
        {"input": "two_sum([2,7,11,15], 9)", "expected": [0, 1]},
        {"input": "two_sum([3,2,4], 6)", "expected": [1, 2]},
        {"input": "two_sum([3,3], 6)", "expected": [0, 1]}
    ]
    
    evaluator = InterviewEvaluator()
    result = evaluator.evaluate(
        transcript=sample_transcript,
        code_solution=sample_code,
        problem_description="Find two numbers that add up to target.",
        test_cases=sample_tests,
        time_taken_minutes=35.0,
        time_limit_minutes=45.0
    )
    
    print("\n" + "="*50)
    print("      INTERVIEW EVALUATION REPORT")
    print("="*50)
    print(f"Total Score : {result['total_score']}/100  (Grade: {result['grade']})")
    print(f"Passed      : {'Yes' if result['passed'] else 'No'}")
    print(f"Summary     : {result['summary']}")
    print("\n--- CATEGORY BREAKDOWN ---")
    
    for cat, data in result["breakdown"].items():
        print(f"  * {cat.replace('_', ' ').title():<17}: {data['score']:>5.1f}/100 (Weight: {data['weight']:.2f})")
        
    print("\n--- ACTIONABLE FEEDBACK ---")
    print(f"Priority Focus: {result['feedback']['priority_focus']}")
    print("\nStrengths:")
    for s in result['feedback']['strengths']:
        print(f"  + {s}")
    print("\nImprovements:")
    for imp in result['feedback']['improvements']:
        print(f"  - {imp}")
    print("\nStudy Plan:")
    for item in result['feedback']['study_plan']:
        print(f"  > {item}")
    print("\n" + "="*50)
