"""
feedback_generator.py
=====================
Generates qualitative feedback based on quantitative evaluation results.
"""

from typing import Dict, Any, List
import warnings

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FeedbackGenerator:
    """Generates qualitative, actionable feedback from evaluation signals."""
    
    def __init__(self) -> None:
        """Initialize the feedback generator and NLP models."""
        self.zero_shot_pipeline = None
        if TRANSFORMERS_AVAILABLE:
            try:
                # Reuse bart-large-mnli for priority classification
                self.zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            except Exception as e:
                warnings.warn(f"FeedbackGenerator failed to load zero-shot pipeline: {e}")

    def generate(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete qualitative feedback based on evaluation results.
        
        Args:
            evaluation_results (dict): The output from InterviewEvaluator containing
                                       all breakdowns and signals.
                                       
        Returns:
            Dict[str, Any]: Structured feedback including strengths, improvements, and a study plan.
        """
        breakdown = evaluation_results.get("breakdown", {})
        comm_data = breakdown.get("communication", {})
        ps_data = breakdown.get("problem_solving", {})
        time_data = breakdown.get("time_management", {})
        
        # 1. Generate category feedback
        comm_feedback = self._generate_communication_feedback(comm_data.get("signals", {}), comm_data.get("sub_scores", {}))
        ps_feedback = self._generate_code_feedback(ps_data.get("signals", {}), ps_data.get("sub_scores", {}))
        time_feedback = self._generate_time_feedback(time_data.get("signals", {}))
        
        # 2. Extract strengths and improvements
        strengths = []
        improvements = []
        
        # Aggregate from categories
        if comm_data.get("score", 0) > 80:
            strengths.append("Excellent communication skills and articulation.")
        if ps_data.get("score", 0) > 80:
            strengths.append("Strong problem-solving and algorithmic knowledge.")
        if time_data.get("score", 0) >= 90:
            strengths.append("Great time management and pacing.")
            
        improvements.extend(comm_feedback.get("action_items", []))
        improvements.extend(ps_feedback.get("action_items", []))
        improvements.extend(time_feedback.get("action_items", []))
        
        # Ensure minimums
        if len(strengths) < 2:
            strengths.extend(["Completed the technical assessment.", "Showed willingness to engage with the problem."])
        if len(improvements) < 3:
            improvements.extend(["Continue practicing daily coding challenges.", "Review core CS fundamentals.", "Do more mock interviews."])
            
        # 3. Determine priority focus
        priority_focus = self._determine_priority(improvements)
        
        # 4. Generate Study Plan
        study_plan = self._build_study_plan(comm_data, ps_data, time_data)
        
        # 5. Encouragement
        encouragement = "Remember, interviewing is a skill that improves with practice. Keep building and learning!"
        
        return {
            "overall_summary": "Here is a detailed breakdown of your interview performance.",
            "strengths": strengths[:3],
            "improvements": improvements,
            "category_feedback": {
                "communication": comm_feedback,
                "problem_solving": ps_feedback,
                "time_management": time_feedback
            },
            "priority_focus": priority_focus,
            "study_plan": study_plan,
            "encouragement": encouragement
        }

    def _generate_communication_feedback(self, comm_signals: Dict[str, Any], sub_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate feedback for communication skills."""
        issues = []
        actions = []
        
        filler_count = comm_signals.get("filler_word_count", 0)
        grammar_errs = comm_signals.get("grammar_error_count", 0)
        vocab_score = sub_scores.get("vocabulary", 100)
        conf_score = sub_scores.get("confidence", 100)
        clarity_score = sub_scores.get("clarity", 100)
        
        if filler_count > 5:
            issues.append(f"High usage of filler words ({filler_count} detected).")
            actions.append(f"Reduce filler words: detected {filler_count} instances of 'um', 'uh', 'like'.")
            
        if grammar_errs > 3:
            issues.append(f"Grammatical errors present ({grammar_errs} detected).")
            actions.append(f"Fix grammar: {grammar_errs} errors detected. Practice writing before speaking.")
            
        if vocab_score < 50:
            issues.append("Limited technical vocabulary.")
            actions.append("Use more technical vocabulary. Study domain-specific terms.")
            
        if conf_score < 60:
            issues.append("Hesitant or uncertain tone detected.")
            actions.append("Sound more assertive. Avoid phrases like 'I think maybe' or 'I'm not sure'.")
            
        if clarity_score < 60:
            issues.append("Lack of structured responses.")
            actions.append("Structure answers: use 'First... Then... Finally...' format.")
            
        if not issues:
            interpretation = "Your communication was clear, confident, and professional."
        else:
            interpretation = "Your communication needs some polish in specific areas."
            
        return {
            "score_interpretation": interpretation,
            "specific_issues": issues,
            "action_items": actions
        }

    def _generate_code_feedback(self, code_signals: Dict[str, Any], sub_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate feedback for problem solving and code."""
        issues = []
        actions = []
        
        tc = code_signals.get("time_complexity", "")
        if "n²" in tc or "n³" in tc:
            issues.append(f"Suboptimal time complexity ({tc}).")
            actions.append(f"Optimize: your solution is {tc}. Consider using a hash map or pointers for O(n).")
            
        if not code_signals.get("has_docstring", True):
            issues.append("Missing docstrings.")
            actions.append("Add docstrings to all functions.")
            
        if not code_signals.get("has_type_hints", True):
            issues.append("Missing type hints.")
            actions.append("Add type hints: def func(x: int) -> str.")
            
        if code_signals.get("pylint_score", 10) < 7:
            issues.append(f"Poor code style (Pylint: {code_signals.get('pylint_score')}/10).")
            actions.append("Improve code style. Run pylint and fix reported issues.")
            
        test_results = code_signals.get("test_results", [])
        if test_results:
            passed = sum(1 for tr in test_results if tr.get("passed"))
            total = len(test_results)
            if passed < total:
                issues.append(f"Failing tests ({total - passed} failed).")
                actions.append(f"Fix failing tests: {passed}/{total} passed.")
                
        if not issues:
            interpretation = "Your code was optimal, clean, and passed all tests."
        else:
            interpretation = "Your code has areas for optimization and stylistic improvement."
            
        return {
            "score_interpretation": interpretation,
            "specific_issues": issues,
            "action_items": actions
        }

    def _generate_time_feedback(self, time_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate feedback for time management."""
        issues = []
        actions = []
        
        ratio = time_signals.get("time_ratio", 1.0)
        
        if ratio > 1.0:
            issues.append("Exceeded the time limit.")
            actions.append("Practice timed coding. Use 25-min Pomodoro sessions on LeetCode.")
        elif ratio < 0.5:
            issues.append("Finished unusually quickly, possibly rushing.")
            actions.append("Don't rush. Use remaining time to test edge cases and optimize.")
            
        if not issues:
            interpretation = "Excellent time management and pacing."
        else:
            interpretation = "Time management can be improved."
            
        return {
            "score_interpretation": interpretation,
            "specific_issues": issues,
            "action_items": actions
        }

    def _determine_priority(self, improvements: List[str]) -> str:
        """Determine the highest priority improvement area."""
        if not improvements:
            return "Keep practicing and maintaining your current skills."
            
        if self.zero_shot_pipeline and len(improvements) > 1:
            try:
                # Combine improvements into a single text
                text = ". ".join(improvements)
                labels = ["communication and soft skills", "code correctness", "code style", "algorithm optimization"]
                res = self.zero_shot_pipeline(text, candidate_labels=labels)
                top_label = res['labels'][0]
                return f"Your primary focus should be on {top_label}."
            except Exception:
                pass
                
        return f"Primary Focus: {improvements[0]}"

    def _build_study_plan(self, comm_data: Dict[str, Any], ps_data: Dict[str, Any], time_data: Dict[str, Any]) -> List[str]:
        """Build a 5-item weekly study plan."""
        plan = []
        comm_score = comm_data.get("score", 100)
        ps_score = ps_data.get("score", 100)
        
        if ps_score <= comm_score:
            plan.append("Day 1-2: Practice 10 LeetCode mediums focusing on hash maps and pointers.")
            plan.append("Day 3: Rewrite your last solution with proper docstrings + type hints.")
        else:
            plan.append("Day 1-2: Record yourself answering mock questions, review for filler words.")
            plan.append("Day 3: Practice answering behavioral questions using the STAR format.")
            
        plan.append("Day 4: Review time and space complexity analysis of 5 different algorithms.")
        plan.append("Day 5: Do a timed mock interview (45 min) with a friend or platform.")
        plan.append("Day 6-7: Take a rest and review your learnings for the week.")
        
        return plan
