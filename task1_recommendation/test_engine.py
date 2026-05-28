import unittest
import os
import pandas as pd

try:
    from task1_recommendation.data_loader import load_data, build_student_matrix, DEFAULT_CSV_PATH
    from task1_recommendation.recommendation_engine import SkillRecommender, DEFAULT_K_SIMILAR
except ImportError:
    from data_loader import load_data, build_student_matrix, DEFAULT_CSV_PATH
    from recommendation_engine import SkillRecommender, DEFAULT_K_SIMILAR

class TestSkillRecommender(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Ensure data is generated
        cls.df = load_data(DEFAULT_CSV_PATH)
        cls.recommender = SkillRecommender(DEFAULT_CSV_PATH)
        cls.recommender.fit()
        
    def test_data_loads_correctly(self):
        self.assertFalse(self.df.empty, "DataFrame should not be empty")
        expected_columns = ['student_id', 'level_id', 'score', 'time_spent_minutes', 'passed', 'timestamp']
        for col in expected_columns:
            self.assertIn(col, self.df.columns, f"Missing required column: {col}")
            
    def test_student_matrix_shape(self):
        matrix = build_student_matrix(self.df)
        num_unique_students = self.df['student_id'].nunique()
        num_unique_levels = self.df['level_id'].nunique()
        
        self.assertEqual(matrix.shape[0], num_unique_students, "Matrix rows should equal unique students")
        self.assertEqual(matrix.shape[1], num_unique_levels, "Matrix cols should equal unique levels")
        
    def test_recommendations_return_top_3(self):
        student_id = self.recommender.matrix.index[0]
        recs = self.recommender.recommend(student_id, top_n=3)
        self.assertEqual(len(recs), 3, "Should return exactly 3 recommendations")
        
    def test_no_already_completed_levels(self):
        student_id = self.recommender.matrix.index[0]
        recs = self.recommender.recommend(student_id, top_n=3)
        
        history = self.df[self.df['student_id'] == student_id]['level_id'].tolist()
        
        for rec in recs:
            self.assertNotIn(rec['level_id'], history, "Recommended level should not be already completed")
            
    def test_confidence_scores_in_range(self):
        student_id = self.recommender.matrix.index[0]
        recs = self.recommender.recommend(student_id, top_n=3)
        
        for rec in recs:
            score = rec['confidence_score']
            self.assertTrue(0 <= score <= 100, f"Confidence score {score} out of range [0, 100]")
            
    def test_pass_probability_above_threshold(self):
        student_id = self.recommender.matrix.index[0]
        recs = self.recommender.recommend(student_id, top_n=3)
        
        for rec in recs:
            prob = rec['predicted_pass_probability']
            self.assertTrue(prob > 0.70, f"Pass probability {prob} not > 0.70")
            
    def test_cold_start_student(self):
        recs = self.recommender.recommend("UNKNOWN_STUDENT_9999", top_n=3)
        self.assertEqual(len(recs), 3, "Cold start student should still get 3 recommendations")
        
    def test_similar_students_count(self):
        student_id = self.recommender.matrix.index[0]
        similar = self.recommender.get_similar_students(student_id, k=DEFAULT_K_SIMILAR)
        self.assertEqual(len(similar), DEFAULT_K_SIMILAR, "Should return exactly K similar students")
        
if __name__ == "__main__":
    unittest.main()
