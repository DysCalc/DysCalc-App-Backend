import argparse
import json
from pathlib import Path
import requests
import sys

DEFAULT_INPUT    = "tests/retest_payloads.json"
DEFAULT_BASE_URL = "http://localhost:5000"


def load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {path}.")
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
        label          = item.get("scenario_name", f"Scenario {i + 1}")
        history_length = len(item.get("student_history", []))
        print(f"[{i + 1}] {label} ({history_length} sessions logged)")
    print("----------------------------------")


def collect_history_ids(scenario_history):
    """Return a set of all question IDs asked across the entire history."""
    ids = set()
    for session in scenario_history:
        for q in session.get("questions_asked", []):
            q_id = q.get("id", "")
            if q_id:
                ids.add(q_id)
    return ids


def collect_history_questions(scenario_history):
    """Return a set of all question strings asked across the entire history."""
    questions = set()
    for session in scenario_history:
        for q in session.get("questions_asked", []):
            prob = q.get("question", "").strip()
            if prob:
                questions.add(prob)
    return questions


def verify_response(body, scenario_history):
    """
    Print a structured verification report for the retest response.
    Checks:
      - Session routing (latest session used)
      - Total session count
      - No duplicates from history (by ID where available, else by string)
      - Per-task breakdown (from_bank / from_llm / excluded / target)
      - Validation report (math, pedagogy, schema errors)
    """
    based_on      = body.get("based_on_session")
    total_sessions = body.get("total_sessions_in_history")
    report        = body.get("_meta_validation_report", {})
    counts        = report.get("counts", {})
    task_meta     = report.get("tasks", {})
    retest_data   = body.get("retest_data", {})

    print(f"\n[INFO] Based on session:          {based_on}")
    print(f"[INFO] Total sessions in history: {total_sessions}")
    print(f"[OUTPUT] Questions returned:      {counts.get('returned', 0)}")
    print(f"[OUTPUT] Questions pruned:        {counts.get('pruned', 0)}")

    if task_meta:
        print("\n[TASK BREAKDOWN]")
        for task, meta in task_meta.items():
            print(
                f"  {task}: "
                f"target={meta.get('target')} | "
                f"from_bank={meta.get('from_bank')} | "
                f"from_gap_fill={meta.get('from_gap_fill')} | "
                f"excluded={meta.get('excluded_from_pool')}"
            )

    math_errors    = report.get("math_errors", [])
    pedagogy_errors= report.get("pedagogy_errors", [])
    schema_errors  = report.get("schema_errors", [])

    if math_errors:
        print(f"\n[VALIDATION] Math errors:     {math_errors}")
    if pedagogy_errors:
        print(f"[VALIDATION] Pedagogy errors: {pedagogy_errors}")
    if schema_errors:
        print(f"[VALIDATION] Schema errors:   {schema_errors}")

    history_ids       = collect_history_ids(scenario_history)
    history_questions = collect_history_questions(scenario_history)

    print(f"\n[VERIFICATION] History pool: {len(history_ids)} unique IDs, "
          f"{len(history_questions)} unique question strings")

    repeated_ids  = []
    repeated_strs = []
    all_new_ids   = []
    all_new_qs    = []

    for domain, content in retest_data.items():
        for item in content.get("tests", []):
            item_id  = item.get("id", "")
            item_q   = item.get("question", "").strip()

            if item_id and item_id in history_ids:
                repeated_ids.append(item_id)
            elif item_q in history_questions:
                repeated_strs.append(item_q)

            if item_id:
                all_new_ids.append(item_id)
            all_new_qs.append(item_q)

    if repeated_ids:
        print(f"[VERIFICATION] FAIL — Repeated IDs from history: {repeated_ids}")
    else:
        print(f"[VERIFICATION] PASS — No repeated question IDs from history.")

    if repeated_strs:
        print(f"[VERIFICATION] FAIL — Repeated question strings (gap-fill): {repeated_strs}")
    else:
        print(f"[VERIFICATION] PASS — No repeated question strings from history.")

    latest_session_id = scenario_history[-1].get("session_id")
    if based_on == latest_session_id:
        print(f"[VERIFICATION] PASS — Correctly based on latest session ({based_on}).")
    else:
        print(f"[VERIFICATION] FAIL — Expected session {latest_session_id}, got {based_on}.")

    actual_sessions = len(scenario_history)
    if total_sessions == actual_sessions:
        print(f"[VERIFICATION] PASS — Session count correct ({total_sessions}).")
    else:
        print(f"[VERIFICATION] FAIL — Expected {actual_sessions} sessions, got {total_sessions}.")

    print("\n[QUESTION COUNTS PER TASK]")
    for domain, content in retest_data.items():
        tests = content.get("tests", [])
        target = task_meta.get(domain, {}).get("target", "?")
        status = "OK" if len(tests) == target else f"SHORT by {target - len(tests)}"
        print(f"  {domain}: {len(tests)} / {target} [{status}]")

    if "--verbose" in sys.argv:
        print("\n[SAMPLE OUTPUT — first 3 questions per task]")
        for domain, content in retest_data.items():
            print(f"\n  {domain.upper()}")
            print(f"  Rationale: {content.get('rationale', '')}...")
            for item in content.get("tests", [])[:3]:
                print(
                    f"    [{item.get('id','')}] "
                    f"{item.get('question','')} = {item.get('correct','')} "
                    f"| hint: {str(item.get('hint',''))}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI tool to test the hybrid bank+LLM retest endpoint."
    )
    parser.add_argument("--input",    default=DEFAULT_INPUT)
    parser.add_argument("--select",   help="1, 1,2, or all")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout",  type=int, default=120)
    parser.add_argument("--save",     help="Path to save results as JSON")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print sample questions per task")
    args = parser.parse_args()

    items = as_list(load_json(args.input))
    print_menu(items)

    selection        = args.select or input("\nSelect scenarios to run (e.g., 1, 1-2, all): ")
    selected_indexes = parse_selection(selection, len(items))

    if not selected_indexes:
        print("No valid selection made. Exiting.")
        return

    print(f"\n[SYSTEM] Running {len(selected_indexes)} selected scenario(s)...")

    results = []

    for idx in selected_indexes:
        payload          = items[idx]
        label            = payload.get("scenario_name", f"Scenario {idx + 1}")
        scenario_history = payload.get("student_history", [])

        print(f"\n{'=' * 60}")
        print(f"EXECUTING: {label}")
        print(f"{'=' * 60}")

        endpoint   = f"{args.base_url.rstrip('/')}/generate_retest"
        api_payload = {"student_history": scenario_history}

        try:
            response = requests.post(endpoint, json=api_payload, timeout=args.timeout)
            is_json  = "application/json" in response.headers.get("Content-Type", "")
            body     = response.json() if is_json else response.text

            result_data = {
                "scenario_name": label,
                "request":  api_payload,
                "response": {
                    "status_code": response.status_code,
                    "ok":          response.ok,
                    "body":        body,
                },
            }
            results.append(result_data)

            if response.status_code == 200:
                print("[STATUS] 200 OK — FULL VALIDATION PASSED")
                verify_response(body, scenario_history)

            elif response.status_code == 207:
                print("[STATUS] 207 — PARTIAL RESULT (review validation report)")
                warning = body.get("warning", "") if isinstance(body, dict) else ""
                if warning:
                    print(f"[WARNING] {warning}")
                verify_response(body, scenario_history)

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
                "request":  api_payload,
                "response": {"status_code": 0, "ok": False, "body": "Connection refused"},
            })
        except requests.exceptions.Timeout:
            print(f"[CRITICAL] Request timed out after {args.timeout}s.")
            results.append({
                "scenario_name": label,
                "request":  api_payload,
                "response": {"status_code": 0, "ok": False, "body": "Timeout"},
            })
        except requests.exceptions.RequestException as e:
            print(f"[CRITICAL] Unexpected error: {e}")
            results.append({
                "scenario_name": label,
                "request":  api_payload,
                "response": {"status_code": 0, "ok": False, "body": str(e)},
            })

    if args.save:
        save_json(args.save, results)
        print(f"\n[SYSTEM] Results saved to {args.save}")

    total  = len(results)
    passed = sum(1 for r in results if r["response"]["status_code"] == 200)
    partial= sum(1 for r in results if r["response"]["status_code"] == 207)
    failed = total - passed - partial
    print(f"\n[SUMMARY] {total} scenario(s) | {passed} passed | {partial} partial | {failed} failed")


if __name__ == "__main__":
    main()