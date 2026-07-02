"""Test edge cases for the API."""
import requests
import json

BASE = "http://localhost:8000"

def test_edge_case(name, payload):
    r = requests.post(f"{BASE}/map-skills", json=payload)
    print(f"\n--- {name} ---")
    print(f"Status: {r.status_code}")
    body = r.json()
    if r.status_code == 200:
        print(f"Mapped: {body['total_mapped']}, Unmapped: {body['total_unmapped']}")
        for u in body.get("unmapped", []):
            print(f"  UNMAPPED: '{u['raw']}' - {u['reason']}")
    else:
        print(f"Error: {body.get('error', '')} - {body.get('hint', '')}")

# Test 1: Missing raw_terms field
test_edge_case("Missing raw_terms field", {"wrong": "field"})

# Test 2: Empty list
test_edge_case("Empty list", {"raw_terms": []})

# Test 3: All empty/whitespace strings  
test_edge_case("All empty/whitespace", {"raw_terms": ["", "   ", "\t"]})

# Test 4: Extremely long string
test_edge_case("Extremely long string (600 chars)", {"raw_terms": ["a" * 600]})

# Test 5: All unmappable
test_edge_case("All unmappable noise", {
    "raw_terms": ["skills:", "N/A", "xkq7z", "asdfgh", "12345", "Lorem ipsum"]
})

# Test 6: Duplicate terms
test_edge_case("Duplicate terms", {
    "raw_terms": ["Python", "python", "PYTHON", "Python"]
})

# Test 7: Not-found student
r = requests.get(f"{BASE}/match-preview/stu_999")
print(f"\n--- Non-existent student ---")
print(f"Status: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

print("\n--- All edge case tests completed ---")
