"""
Scripted dry-run harness: exercises the full system as a college officer and admin.
Logs every request/response with timestamps to reports/dry_run_transcript.json.
"""
from fastapi.testclient import TestClient
from datetime import datetime
import json
import os


def run_dry_run(reports_dir: str):
    """Execute dry-run against the API using TestClient."""
    # Import here so startup events fire with data already generated
    from .api import app

    transcript = []
    step_counter = [0]

    with TestClient(app, raise_server_exceptions=False) as client:
        def log_step(journey: str, description: str, method: str, url: str, expected_status: int = 200):
            step_counter[0] += 1
            response = getattr(client, method)(url)
            passed = response.status_code == expected_status

            entry = {
                "step": step_counter[0],
                "timestamp": datetime.now().isoformat(),
                "journey": journey,
                "description": description,
                "request": {"method": method.upper(), "url": url},
                "expected_status": expected_status,
                "actual_status": response.status_code,
                "passed": passed,
                "response_body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text[:500],
            }
            transcript.append(entry)
            status_icon = "PASS" if passed else "FAIL"
            print(f"  [{status_icon}] Step {step_counter[0]}: {description} "
                  f"(expected {expected_status}, got {response.status_code})")
            return entry

        print("\n" + "=" * 60)
        print("DRY RUN — College & Admin Journeys")
        print("=" * 60)

        # ── College journeys (3 colleges) ─────────────────────────────────────
        test_colleges = ["college_A", "college_B", "college_C"]

        for college_id in test_colleges:
            print(f"\n--- College journey: {college_id} ---")

            # 1. Fetch recommendation dashboard
            log_step(
                f"college_{college_id}", "Fetch recommendations dashboard",
                "get", f"/college/{college_id}/recommendations"
            )

            # 2. Drill into a student → job
            # Pick the first student from this college
            student_id = f"fresh_{college_id}_0"
            # We need a valid job_id. Use /college/{id}/recommendations to find one.
            rec_resp = client.get(f"/college/{college_id}/recommendations")
            if rec_resp.status_code == 200:
                recs = rec_resp.json().get("students", [])
                if recs and recs[0].get("recommendations"):
                    job_id = recs[0]["recommendations"][0]["job_id"]
                    student_id = recs[0]["student_id"]
                    log_step(
                        f"college_{college_id}",
                        f"Drill into student={student_id}, job={job_id}",
                        "get",
                        f"/college/{college_id}/student/{student_id}/job/{job_id}",
                    )
                else:
                    log_step(
                        f"college_{college_id}",
                        "Drill into student/job (no recs found)",
                        "get",
                        f"/college/{college_id}/student/{student_id}/job/job_0",
                    )

            # 3. Cross-college access (should be rejected)
            other_college = "college_B" if college_id != "college_B" else "college_C"
            other_student = f"fresh_{other_college}_0"
            log_step(
                f"college_{college_id}",
                f"Cross-college access: {college_id} tries to access {other_student}",
                "get",
                f"/college/{college_id}/student/{other_student}/job/job_0",
                expected_status=403,
            )

        # ── Admin journey ─────────────────────────────────────────────────────
        print("\n--- Admin journey ---")

        # 1. Aggregated report
        log_step("admin", "Fetch aggregated cross-college report", "get", "/admin/report")

        # 2. Drift detection result
        log_step("admin", "Fetch drift-detection result", "get", "/validation/drift")

        # 3. Go/No-Go summary
        log_step("admin", "Fetch Go/No-Go verdict", "get", "/validation/go-no-go")

        # 4. Deliberate failure: unknown college
        log_step(
            "admin", "Deliberate failure: unknown college",
            "get", "/college/college_NONEXISTENT/recommendations",
            expected_status=404,
        )

        # 5. Deliberate failure: unknown student
        log_step(
            "admin", "Deliberate failure: unknown student",
            "get", "/college/college_A/student/UNKNOWN_STUDENT/job/job_0",
            expected_status=404,
        )

        # 6. Deliberate failure: empty college
        log_step(
            "admin", "Deliberate failure: college with zero data",
            "get", "/college/college_EMPTY/recommendations",
            expected_status=404,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    total = len(transcript)
    passed = sum(1 for t in transcript if t["passed"])
    failed = total - passed

    isolation_steps = [t for t in transcript if "Cross-college" in t["description"]]
    isolation_passed = sum(1 for t in isolation_steps if t["passed"])

    failure_steps = [t for t in transcript if "Deliberate failure" in t["description"]]
    failure_handled = sum(1 for t in failure_steps if t["passed"])

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "total_steps": total,
        "passed": passed,
        "failed": failed,
        "isolation_checks": f"{isolation_passed}/{len(isolation_steps)}",
        "deliberate_failures_handled": f"{failure_handled}/{len(failure_steps)}",
        "all_passed": failed == 0,
        "transcript": transcript,
    }

    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "dry_run_transcript.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total: {total} steps, {passed} passed, {failed} failed")
    print(f"  Isolation: {isolation_passed}/{len(isolation_steps)}")
    print(f"  Deliberate failures handled: {failure_handled}/{len(failure_steps)}")
    print(f"  Transcript saved to reports/dry_run_transcript.json")

    return summary


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dry_run(os.path.join(base, "reports"))
