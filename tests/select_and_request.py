import argparse
import json
from pathlib import Path

import requests


DEFAULT_INPUT = "tests/generated_test_payloads.json"
DEFAULT_BASE_URL = "http://localhost:5000"
MODULE_REQUIRED_KEYS = {
    "predicted_class",
    "confidence",
    "decision_path",
    "domain_severity_scores",
    "task_importance_scores",
}
OPTIONAL_ML_KEYS = {
    "decision_path_readable",
    "leaf_distribution",
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def as_list(data):
    return data if isinstance(data, list) else [data]


def item_label(item, index):
    test_id = item.get("test_id", f"item-{index:04d}") if isinstance(item, dict) else f"item-{index:04d}"

    if isinstance(item, dict) and MODULE_REQUIRED_KEYS.issubset(item.keys()):
        predicted = item.get("predicted_class", "?")
        confidence = item.get("confidence", "?")
        return f"{test_id} | ML response | class={predicted} confidence={confidence}"

    if isinstance(item, dict) and "diagnostic_data" in item:
        diagnostic = item.get("diagnostic_data", {})
        predicted = diagnostic.get("predicted_class", "?")
        confidence = diagnostic.get("confidence", "?")
        return f"{test_id} | module payload | class={predicted} confidence={confidence}"

    if isinstance(item, dict):
        fields = [
            f"NC={item.get('number_comparison', '?')}",
            f"DM={item.get('dot_matching', '?')}",
            f"NS={item.get('number_series', '?')}",
            f"ADD={item.get('single_addition', '?')}",
            f"SUB={item.get('single_subtraction', '?')}",
            f"CA={item.get('complex_arithmetic', '?')}",
        ]
        return f"{test_id} | diagnostic input | " + ", ".join(fields)

    return f"item-{index:04d} | unsupported item"


def print_menu(items):
    print("\nAvailable test payloads:")
    for index, item in enumerate(items, start=1):
        print(f"{index:>3}. {item_label(item, index)}")
    print("\nSelect one or more: 1, 1,3,5, 2-6, or all")


def parse_selection(selection, item_count):
    selection = selection.strip().lower()
    if selection == "all":
        return list(range(item_count))

    selected = set()
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                start, end = end, start
            selected.update(range(start - 1, end))
        else:
            selected.add(int(part) - 1)

    invalid = [index + 1 for index in selected if index < 0 or index >= item_count]
    if invalid:
        raise ValueError(f"Selection out of range: {invalid}")

    return sorted(selected)


def post_json(url, payload, timeout):
    response = requests.post(url, json=payload, timeout=timeout)
    try:
        body = response.json()
    except ValueError:
        body = response.text

    return {
        "status_code": response.status_code,
        "ok": response.ok,
        "body": body,
    }


def make_module_payload(ml_response, test_id):
    diagnostic_data = {
        key: ml_response[key]
        for key in MODULE_REQUIRED_KEYS
    }

    for key in OPTIONAL_ML_KEYS:
        if key in ml_response:
            diagnostic_data[key] = ml_response[key]

    return {
        "test_id": test_id,
        "diagnostic_data": diagnostic_data,
    }


def can_call_module_directly(item):
    return isinstance(item, dict) and (
        "diagnostic_data" in item or MODULE_REQUIRED_KEYS.issubset(item.keys())
    )


def call_diagnostic(item, base_url, timeout):
    url = f"{base_url.rstrip('/')}/generate-diagnostic"
    return post_json(url, item, timeout)


def call_module(item, base_url, timeout, fallback_test_id):
    url = f"{base_url.rstrip('/')}/generate_module"

    if isinstance(item, dict) and "diagnostic_data" in item:
        payload = item
    elif isinstance(item, dict) and MODULE_REQUIRED_KEYS.issubset(item.keys()):
        payload = make_module_payload(item, item.get("test_id", fallback_test_id))
    else:
        raise ValueError("This item is raw diagnostic input. Run endpoint=both or endpoint=diagnostic first.")

    return payload, post_json(url, payload, timeout)


def run_selected(items, selected_indexes, endpoint, base_url, timeout):
    results = []

    for index in selected_indexes:
        item = items[index]
        test_id = item.get("test_id", f"test-{index + 1:04d}-0000-0000-0000")
        print(f"\n[RUN] {test_id}")

        diagnostic_result = None
        if endpoint in {"diagnostic", "both"} and not can_call_module_directly(item):
            print("[POST] /generate-diagnostic")
            diagnostic_result = call_diagnostic(item, base_url, timeout)
            results.append({
                "test_id": test_id,
                "endpoint": "/generate-diagnostic",
                "request": item,
                "response": diagnostic_result,
            })
            print_status(diagnostic_result)

        if endpoint in {"module", "both"}:
            module_source = item
            if diagnostic_result is not None:
                if not diagnostic_result["ok"]:
                    print("[SKIP] /generate_module because /generate-diagnostic failed.")
                    continue
                module_source = diagnostic_result["body"]

            print("[POST] /generate_module")
            module_payload, module_result = call_module(module_source, base_url, timeout, test_id)
            results.append({
                "test_id": test_id,
                "endpoint": "/generate_module",
                "request": module_payload,
                "response": module_result,
            })
            print_status(module_result)

    return results


def print_status(result):
    label = "OK" if result["ok"] else "FAILED"
    print(f"[{label}] HTTP {result['status_code']}")
    if not result["ok"]:
        body = result["body"]
        print(json.dumps(body, indent=2) if isinstance(body, dict) else body)


def main():
    parser = argparse.ArgumentParser(
        description="Pick test payloads from a JSON file and request /generate-diagnostic, /generate_module, or both."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="JSON file containing one payload object or a list.")
    parser.add_argument("--select", help="Selection like 1, 1,3,5, 2-6, or all. Omit for interactive picker.")
    parser.add_argument("--endpoint", choices=["diagnostic", "module", "both"], default="both")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--save", help="Optional path to save request/response results as JSON.")
    args = parser.parse_args()

    items = as_list(load_json(args.input))
    print_menu(items)

    selection = args.select or input("\nSelection: ")
    selected_indexes = parse_selection(selection, len(items))
    selected_labels = ", ".join(str(index + 1) for index in selected_indexes)
    print(f"\nSelected: {selected_labels}")

    results = run_selected(items, selected_indexes, args.endpoint, args.base_url, args.timeout)

    if args.save:
        save_json(args.save, results)
        print(f"\n[SAVED] {args.save}")


if __name__ == "__main__":
    main()
