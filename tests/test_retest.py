import argparse
import json
from pathlib import Path
import requests
import sys

DEFAULT_INPUT = "tests/retest_payloads.json"
DEFAULT_BASE_URL = "http://localhost:5000"


def load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {path}. Please create it first.")
        sys.exit(1)


def save_json(path, data):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def as_list(data):
    return data if isinstance(data, list) else [data]


def parse_selection(selection_str, max_val):
    if not selection_str or selection_str.strip().lower() == "all":
        return list(range(max_val))

    indexes = set()
    for part in selection_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            indexes.update(range(int(start) - 1, int(end)))
        else:
            indexes.add(int(part) - 1)

    return sorted([i for i in indexes if 0 <= i < max_val])


def print_menu(items):
    print("\n--- Available Retest Scenarios ---")
    for i, item in enumerate(items):
        label = item.get("scenario_name", f"Scenario {i + 1}")
        history_length = len(item.get("student_history", []))
        print(f"[{i + 1}] {label} ({history_length} sessions logged)")
    print("----------------------------------")


def print_verification(body, scenario_history):
    """
    Runs post-response verification checks and prints a clear report.
    Mirrors the same checks the original test_retest.py described manually.
    """
    questions = body.get("retest_questions", [])
    based_on = body.get("based_on_session")
    total_sessions = body.get("total_sessions_in_history")
    report = body.get("_meta_validation_report", {})
    counts = report.get("counts", {})

    print(f"\n[INFO] Based on session:          {based_on}")
    print(f"[INFO] Total sessions in history: {total_sessions}")
    print(f"[OUTPUT] Questions returned:      {counts.get('returned', 0)}")
    print(f"[OUTPUT] Questions pruned:        {counts.get('pruned', 0)}")

    print("\n=== NEW RETEST QUESTIONS ===")
    print(json.dumps(body, indent=2))
    print("============================\n")

    math_errors = report.get("math_errors", [])
    pedagogy_errors = report.get("pedagogy_errors", [])
    schema_errors = report.get("schema_errors", [])

    if math_errors:
        print(f"[VALIDATION] Math errors:     {math_errors}")
    if pedagogy_errors:
        print(f"[VALIDATION] Pedagogy errors: {pedagogy_errors}")
    if schema_errors:
        print(f"[VALIDATION] Schema errors:   {schema_errors}")

    # Collect all problems previously seen across all sessions
    all_previous = set()
    for session in scenario_history:
        for q in session.get("questions_asked", []):
            prob = q.get("problem", "").strip()
            if prob:
                all_previous.add(prob)

    print(f"\n[VERIFICATION] Previous questions in history pool: {sorted(all_previous)}")

    repeated = []
    new_problems = []
    for q in questions:
        prob = q.get("problem", "").strip()
        if prob in all_previous:
            repeated.append(prob)
        else:
            new_problems.append(prob)

    if repeated:
        print(f"[VERIFICATION] FAIL — Repeated questions detected: {repeated}")
    else:
        print(f"[VERIFICATION] PASS — No repeated questions from history.")

    print(f"[VERIFICATION] New questions generated: {new_problems}")

    # Check that generation was based on the latest session
    latest_session_id = scenario_history[-1].get("session_id")
    if based_on == latest_session_id:
        print(f"[VERIFICATION] PASS — Generated from latest session (session {based_on}).")
    else:
        print(
            f"[VERIFICATION] FAIL — Expected session {latest_session_id}, "
            f"but based_on_session is {based_on}."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI tool to test longitudinal retest generation."
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help="JSON file containing retest scenarios."
    )
    parser.add_argument(
        "--select",
        help="Selection like 1, 1,2, or all. Omit for interactive picker."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--save",
        help="Optional path to save request/response results as JSON."
    )
    args = parser.parse_args()

    items = as_list(load_json(args.input))
    print_menu(items)

    selection = args.select or input("\nSelect scenarios to run (e.g., 1, 1-2, all): ")
    selected_indexes = parse_selection(selection, len(items))

    if not selected_indexes:
        print("No valid selection made. Exiting.")
        return

    print(f"\n[SYSTEM] Running {len(selected_indexes)} selected scenario(s)...")

    results = []

    for idx in selected_indexes:
        payload = items[idx]
        label = payload.get("scenario_name", f"Scenario {idx + 1}")
        scenario_history = payload.get("student_history", [])

        print(f"\n{'=' * 60}")
        print(f"EXECUTING: {label}")
        print(f"{'=' * 60}")

        endpoint = f"{args.base_url.rstrip('/')}/generate_retest"
        api_payload = {"student_history": scenario_history}

        try:
            response = requests.post(endpoint, json=api_payload, timeout=args.timeout)
            is_json = "application/json" in response.headers.get("Content-Type", "")
            body = response.json() if is_json else response.text

            result_data = {
                "scenario_name": label,
                "request": api_payload,
                "response": {
                    "status_code": response.status_code,
                    "ok": response.ok,
                    "body": body,
                },
            }
            results.append(result_data)

            if response.status_code == 200:
                print(f"[STATUS] 200 OK — FULL VALIDATION PASSED")
                print_verification(body, scenario_history)

            elif response.status_code == 207:
                print(f"[STATUS] 207 — PARTIAL RESULT (review validation report)")
                warning = body.get("warning", "")
                if warning:
                    print(f"[WARNING] {warning}")
                print_verification(body, scenario_history)

            else:
                print(f"[STATUS] {response.status_code} — FAILED")
                if is_json:
                    print(json.dumps(body, indent=2))
                else:
                    print(body)

        except requests.exceptions.ConnectionError:
            print("[CRITICAL] Connection refused. Ensure Flask is running on port 5000.")
            results.append({
                "scenario_name": label,
                "request": api_payload,
                "response": {"status_code": 0, "ok": False, "body": "Connection refused"},
            })
        except requests.exceptions.Timeout:
            print(f"[CRITICAL] Request timed out after {args.timeout}s.")
            results.append({
                "scenario_name": label,
                "request": api_payload,
                "response": {"status_code": 0, "ok": False, "body": "Timeout"},
            })
        except requests.exceptions.RequestException as e:
            print(f"[CRITICAL] Unexpected error: {e}")
            results.append({
                "scenario_name": label,
                "request": api_payload,
                "response": {"status_code": 0, "ok": False, "body": str(e)},
            })

    if args.save:
        save_json(args.save, results)
        print(f"\n[SYSTEM] Results saved to {args.save}")


if __name__ == "__main__":
    main()
