from src.evaluate import precision_at_k, average_precision, ndcg_at_k

def test_precision_at_k():
    assert precision_at_k({1, 2}, [1, 3, 2, 4], k=2) == 0.5

def test_average_precision_perfect_rank():
    assert average_precision({1, 2}, [1, 2, 3]) == 1.0

def test_ndcg_bounds():
    score = ndcg_at_k({1}, [3, 1, 2], k=3)
    assert 0 <= score <= 1