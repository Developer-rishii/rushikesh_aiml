"""Tiny dependency-free test runner (pytest unavailable, offline sandbox).
Run: python3 tests/run_tests.py
Each test_* function in test_isolation.py is called; failures print the
AssertionError and the run exits non-zero so this is CI-safe."""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import tests.test_isolation as mod

def main():
    fns = [(n, f) for n, f in vars(mod).items() if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            traceback.print_exc()
            failed.append(name)
    print(f"\n{passed}/{len(fns)} passed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)

if __name__ == "__main__":
    main()
