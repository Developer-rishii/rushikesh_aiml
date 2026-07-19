"""
Standalone test runner -- this sandbox has no network access, so
`pip install pytest` isn't available. Discovers every `test_*` function in
test_pipeline.py and executes it, reporting pass/fail per test just like
pytest would.
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))
import test_pipeline as tp

TESTS = [
    (name, obj) for name, obj in vars(tp).items()
    if name.startswith("test_") and callable(obj)
]

passed, failed = 0, []
for name, fn in TESTS:
    try:
        fn()
        print(f"PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"FAIL  {name}: {e}")
        traceback.print_exc()
        failed.append(name)

print(f"\n{passed}/{len(TESTS)} tests passed")
if failed:
    print("Failed:", failed)
    sys.exit(1)
