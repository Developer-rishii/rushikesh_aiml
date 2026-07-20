"""
scripts/run_tests_no_pytest.py
Custom test runner for Phase 3 Task 1 that doesn't require pytest to be installed.
Mocks pytest features and runs tests/test_phase3.py.
"""
import sys
import json
import traceback
from pathlib import Path

# ── Mock pytest ───────────────────────────────────────────────────────────────
class MockPytest:
    @staticmethod
    def fixture(scope="function"):
        def decorator(func):
            return func
        return decorator
        
    class approx:
        def __init__(self, expected, rel=None, abs=None, nan_ok=False):
            self.expected = expected
            self.rel = rel or 1e-6
            self.abs = abs or 1e-12
            
        def __eq__(self, actual):
            return abs(self.expected - actual) <= max(self.abs, self.rel * abs(self.expected))
            
    class raises:
        def __init__(self, expected_exc, match=None):
            self.expected_exc = expected_exc
            self.match = match
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"DID NOT RAISE {self.expected_exc}")
            if issubclass(exc_type, self.expected_exc):
                if self.match and self.match not in str(exc_val):
                    raise AssertionError(f"Exception message '{exc_val}' did not match '{self.match}'")
                return True
            return False

sys.modules['pytest'] = MockPytest()

# ── Run Tests ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import test_phase3

def run_tests():
    print("=" * 60)
    print("  Running Phase 3 Task 1 Tests (No-Pytest Mode)")
    print("=" * 60)
    
    # 1. Initialize fixtures
    try:
        pred = test_phase3.pred()
        inter = test_phase3.inter()
        health = test_phase3.health()
        backlog = test_phase3.backlog()
    except Exception as e:
        print(f"Failed to initialize fixtures: {e}")
        traceback.print_exc()
        sys.exit(1)

    fixtures = {
        'pred': pred,
        'inter': inter,
        'health': health,
        'backlog': backlog
    }
    
    # 2. Find and run test functions
    import inspect
    test_funcs = [getattr(test_phase3, name) for name in dir(test_phase3) if name.startswith('test_') and callable(getattr(test_phase3, name))]
    
    passed = 0
    failed = 0
    
    for test in test_funcs:
        sig = inspect.signature(test)
        kwargs = {}
        for param in sig.parameters:
            if param in fixtures:
                kwargs[param] = fixtures[param]
                
        try:
            test(**kwargs)
            print(f" [PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f" [FAIL] {test.__name__}")
            traceback.print_exc()
            failed += 1
            
    print("-" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
