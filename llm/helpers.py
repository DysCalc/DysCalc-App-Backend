import re
import ast
import requests
import copy
import math
import json
import difflib
import random

from llm.constants import (
    ALLOWED_OPS,
    DRAFT_MODELS_TO_TRY,
    FORMAT_MODELS_TO_TRY,
    DRAFT_URL,
    DRAFT_HEADERS,
    FORMAT_URL,
    FORMAT_HEADERS,
    BAD_HINT_PATTERNS,
    DOMAIN_GENERATION_RULES,
    ML_INTERPRETATION_MAP, 
    CLINICAL_COOCCURRENCE_MAP,
)

import logging

from llm.constants import ML_INTERPRETATION_MAP, CLINICAL_COOCCURRENCE_MAP

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_top_deficits(ml_data: dict, rid: str, max_domains: int = 3) -> list[str]:
    """
    Select intervention targets using both domain severity and task importance.

    Why:
    - Severity tells how concerning the domain is.
    - Importance tells how much the model relied on the feature.
    """
    try:
        severity_scores = ml_data.get("domain_severity_scores", {}) or {}
        importance_scores = ml_data.get("task_importance_scores", {}) or {}

        if not severity_scores:
            return ["General Arithmetic Fluency"]

        reverse_map = {v: k for k, v in ML_INTERPRETATION_MAP.items()}

        broad_domains = {
            "Overall Arithmetic Fluency",
            "General Arithmetic Fluency",
        }

        scored_domains = []

        for domain, severity in severity_scores.items():
            acronym = reverse_map.get(domain)
            importance = importance_scores.get(acronym, 0.0)

            # Weighted score: severity is primary, importance is supporting evidence.
            combined = (0.70 * float(severity)) + (0.30 * float(importance))

            # Prefer clinically specific derived domains over broad summaries.
            if domain in broad_domains:
                combined *= 0.60

            # Avoid tiny/noise domains unless the model importance is very high.
            if severity < 0.08 and importance < 0.12:
                continue

            scored_domains.append((domain, combined, severity, importance))

        scored_domains.sort(key=lambda x: x[1], reverse=True)

        selected = [domain for domain, _, _, _ in scored_domains[:max_domains]]

        return selected if selected else ["General Arithmetic Fluency"]

    except Exception as e:
        logger.error(f"[REQ {rid}] Deficit selection failed: {e}")
        return ["General Arithmetic Fluency"]

def calculate_practice_tiers_by_domain(severity_scores: dict, domains: list[str]) -> dict[str, int]:
    """
    Return practice count per selected domain.
    Mild: 2
    Moderate: 4
    High: 6
    """
    tiers = {}
    for domain in domains:
        score = float(severity_scores.get(domain, 0.0))

        if score >= 0.40:
            tiers[domain] = 6
        elif score >= 0.20:
            tiers[domain] = 4
        else:
            tiers[domain] = 2

    return tiers

def interpret_decision_path(ml_data: dict) -> str:
    """Synthesizes a root-cause clinical profile based on structured ML data."""
    try:
        predicted_class = str(ml_data.get("predicted_class", ""))
        is_at_risk = "At-Risk" in predicted_class or "1" in predicted_class
        confidence = float(ml_data.get("confidence", 0.50))

        if not is_at_risk:
            tone_desc = "mild, emerging variations"
            tone_impact = "suggesting"
        elif confidence > 0.75:
            tone_desc = "critical foundational gaps"
            tone_impact = "creating a severe bottleneck driven by"
        else:
            tone_desc = "significant cognitive difficulties"
            tone_impact = "indicating"

        severity_dict = ml_data.get("domain_severity_scores", {})
        if not severity_dict:
            return "Profile indicates general baseline performance."

        REVERSE_MAP = {v: k for k, v in ML_INTERPRETATION_MAP.items()}

        active_acronyms = set()
        top_domains = []

        for domain, score in sorted(severity_dict.items(), key=lambda item: item[1], reverse=True):
            if score >= 0.15:
                top_domains.append(domain)
                if domain in REVERSE_MAP:
                    active_acronyms.add(REVERSE_MAP[domain])

        if not top_domains:
            return "The student's profile indicates general baseline performance with no dominant cognitive bottlenecks."

        synthesis = f"The primary difficulty lies in {tone_desc} involving {top_domains[0].lower()}"
        if len(top_domains) > 1:
            synthesis += f" and {top_domains[1].lower()}"

        co_occurrences = []
        for (f1, f2), meaning in CLINICAL_COOCCURRENCE_MAP.items():
            if f1 in active_acronyms and f2 in active_acronyms:
                clean_meaning = meaning.replace("indicates", "").replace("suggests", "").replace("points to", "").strip()
                co_occurrences.append(clean_meaning)

        if co_occurrences:
            synthesis += f", {tone_impact} {', '.join(co_occurrences)}."
        else:
            synthesis += f", {tone_impact} challenges in progressing to more complex arithmetic tasks."

        return synthesis

    except Exception as e:
        logger.error(f"Interpretation failed: {e}")
        return "Profile indicates general baseline performance."

def validate_as_structure(practice_set):
    """Enforces the 4-subtraction, 2-addition ratio for AS modules."""
    problems = [str(p.get("problem", "")) for p in practice_set]
    add_count = sum(1 for p in problems if '+' in p)
    sub_count = sum(1 for p in problems if '-' in p)
    return sub_count >= 3 and add_count >= 1

def check_hint_diversity(practice_set):
    """Ensures the LLM isn't copy/pasting the exact same hint."""
    hints = [str(p.get("hint", "")).lower() for p in practice_set]
    if not hints: return False
    unique = len(set(hints))
    return unique >= (len(hints) * 0.7)

def check_answer_diversity(practice_set):
    answers = [str(p.get("expected_answer", "")).strip() for p in practice_set]
    if not answers: return False
    return len(set(answers)) >= max(2, math.ceil(len(answers) * 0.5))

def validate_conceptual_steps(text):
    """Forces the LLM to use the Step 1, Step 2, Step 3 format with flexible punctuation."""
    text_str = str(text)
    return all(re.search(rf"Step\s*{i}\s*[:\-]", text_str, re.IGNORECASE) for i in [1, 2, 3])

def safe_eval(expr):
    """Strictly evaluates pure mathematical strings. Rejects all word problems and code."""
    expr = expr.split('=')[0].strip()

    if not re.fullmatch(r"^[0-9+\-()\s]+$", expr):
        return None

    try:
        node = ast.parse(expr, mode='eval')
        def _eval(n):
            if isinstance(n, ast.Expression):
                return _eval(n.body)
            elif isinstance(n, ast.BinOp):
                if type(n.op) not in ALLOWED_OPS: raise ValueError("Operator not allowed")
                return ALLOWED_OPS[type(n.op)](_eval(n.left), _eval(n.right))
            elif isinstance(n, ast.UnaryOp):
                if type(n.op) not in ALLOWED_OPS: raise ValueError("Operator not allowed")
                return ALLOWED_OPS[type(n.op)](_eval(n.operand))
            elif isinstance(n, ast.Constant):
                return n.value
            elif isinstance(n, ast.Num):
                return n.n
            else:
                raise ValueError("Unsafe AST node")
        return _eval(node)
    except ZeroDivisionError:
        return None
    except Exception:
        return None

def contains_invalid_subtraction(text):
    """Pedagogy constraint: Catch intermediate negative subtraction steps."""
    matches = re.findall(r'(\d+)\s*-\s*(\d+)', text)
    for a, b in matches:
        if int(a) < int(b): return True
    return False

def validate_make_ten_reasoning(problem: str, text: str, context: str) -> tuple[bool, str | None]:
    """Catch make-ten explanations that subtract the wrong amount to reach 10."""
    clean_eq = re.sub(r"[^0-9+\-\s]", "", str(problem))
    nums = [int(n) for n in re.findall(r"\d+", clean_eq)]

    if "-" not in clean_eq or len(nums) < 2:
        return True, None

    minuend, subtrahend = nums[0], nums[1]
    if minuend <= 10 or minuend - subtrahend >= 10:
        return True, None

    first_takeaway = minuend - 10
    remaining_takeaway = subtrahend - first_takeaway
    text_l = str(text).lower()

    if not any(phrase in text_l for phrase in ["reach 10", "make 10", "down to 10"]):
        return True, None

    for amount_text in re.findall(r"(?:take away|subtract|remove)\s+(\d+)[^.]{0,45}(?:reach|make|down to)\s+10", text_l):
        amount = int(amount_text)
        if amount != first_takeaway:
            return (
                False,
                f"Incorrect make-10 reasoning in {context}: {problem} should take away "
                f"{first_takeaway} to reach 10, then {remaining_takeaway} more."
            )

    return True, None

def generate_crossing_ten_subtraction_hint(problem: str) -> str | None:
    clean_eq = re.sub(r"[^0-9+\-\s]", "", str(problem))
    nums = [int(n) for n in re.findall(r"\d+", clean_eq)]

    if "-" not in clean_eq or len(nums) < 2:
        return None

    a, b = nums[0], nums[1]
    result = a - b

    if a <= 10 or result >= 10 or result < 0:
        return None

    first = a - 10
    remaining = b - first

    return (
        f"Use {a} blocks. Take away {first} to reach 10, "
        f"then take away {remaining} more. Count {result} blocks left."
    )

def validate_stated_final_target(problem: str, text: str, expected_answer: int, context: str) -> tuple[bool, str | None]:
    """
    Catch hints/reasoning that state the wrong final target.
    Example: 12 - 5 = 7, but text says "then 3 more to reach 5".
    """
    text_l = str(text).lower()

    matches = re.findall(
        r"(?:then|and then)[^.]{0,80}?(?:reach|get|make)\s+(\d+)",
        text_l
    )

    if matches:
        stated_final = int(matches[-1])
        if stated_final != int(expected_answer):
            return (
                False,
                f"Incorrect stated final target in {context}: {problem} should end at "
                f"{expected_answer}, but text says it reaches {stated_final}."
            )

    return True, None

def validate_addition_hint_amount(problem: str, text: str, context: str) -> tuple[bool, str | None]:
    clean_eq = re.sub(r"[^0-9+\-\s]", "", str(problem))
    nums = [int(n) for n in re.findall(r"\d+", clean_eq)]

    if "+" not in clean_eq or len(nums) < 2:
        return True, None

    addend = nums[1]
    text_l = str(text).lower()
    add_matches = re.findall(r"\badd\s+(\d+)\b", text_l)

    for match in add_matches:
        amount = int(match)
        if amount > addend and not any(
            phrase in text_l
            for phrase in ["take back", "remove", "because you only"]
        ):
            return (
                False,
                f"Incorrect addition strategy in {context}: {problem} adds {addend}, "
                f"but text says to add {amount} without correction."
            )

    return True, None

def call_llm_gateway(payload, url, headers, rid="SYSTEM", temp=0.2, model_paths=None):
    models_to_try = model_paths or DRAFT_MODELS_TO_TRY
    for model_path in models_to_try:
        for use_json_mode in (True, False):
            try:
                current_payload = copy.deepcopy(payload)
                current_payload["model"] = model_path
                current_payload["temperature"] = temp
                current_payload["max_tokens"] = 3000

                if use_json_mode:
                    current_payload["response_format"] = {"type": "json_object"}

                mode_label = "json mode" if use_json_mode else "plain mode"
                logger.info(f"[REQ {rid}] Calling model: {model_path} ({mode_label})")
                response = requests.post(
                    url, headers=headers, json=current_payload, timeout=900
                )
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except requests.exceptions.Timeout:
                logger.warning(f"[REQ {rid}] Timeout on {model_path}. Trying next...")
                break
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else None
                error_body = e.response.text[:500] if e.response is not None else ""
                logger.warning(
                    f"[REQ {rid}] Model {model_path} failed in {mode_label}: "
                    f"{e}. Body: {error_body}"
                )
                if use_json_mode and status_code == 400:
                    logger.info(f"[REQ {rid}] Retrying {model_path} without response_format.")
                    continue
                break
            except requests.exceptions.RequestException as e:
                logger.warning(f"[REQ {rid}] Model {model_path} failed in {mode_label}: {e}")
                break
    raise Exception("LLM service unavailable.")

def extract_json_robustly(text):
    if not text: return None
    text = text.replace('```json', '').replace('```', '').strip()

    start = text.find('{')
    end = text.rfind('}')

    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                repaired_str = re.sub(r',\s*}', '}', json_str)
                repaired_str = re.sub(r',\s*]', ']', repaired_str)
                return json.loads(repaired_str)
            except:
                return None
    return None

def normalize(s):
    return re.sub(r'\W+', '', s.lower())



def normalize_equation(eq: str) -> str:
    return re.sub(r"\s+", "", str(eq).split("=")[0])


def is_addition_subtraction_asymmetry(domain_name: str) -> bool:
    ratio = difflib.SequenceMatcher(
        None,
        normalize(domain_name),
        normalize("Addition vs. Subtraction Asymmetry")
    ).ratio()
    return ratio >= 0.70


def validate_operand_bounds(problem: str, domain_name: str) -> tuple[bool, str | None]:
    nums = [int(n) for n in re.findall(r"\d+", str(problem))]

    if not nums:
        return False, f"No numeric operands found in: {problem}"

    if any(n > 20 for n in nums):
        return False, f"Operand above 20 in: {problem}"

    if "-" in str(problem) and len(nums) >= 2:
        subtrahend = nums[1]
        if subtrahend > 9 and domain_name != "Multi-Digit Addition and Subtraction":
            return False, f"Subtrahend too large for early crossing-ten practice: {problem}"

    return True, None


def is_typical_profile(predicted_class: str) -> bool:
    class_text = str(predicted_class).strip().lower()
    return class_text == "0" or "typical" in class_text or "(0)" in class_text


def validate_tone_for_status(text: str, predicted_class: str) -> tuple[bool, str | None]:
    if not is_typical_profile(predicted_class):
        return True, None

    banned = [
        "significant difficulty",
        "significant difficulties",
        "severe",
        "critical",
        "major deficit",
        "high-risk",
        "at-risk",
    ]
    text_l = str(text).lower()

    for phrase in banned:
        if phrase in text_l:
            return False, f"Tone too strong for Typical profile: '{phrase}'"

    return True, None


def math_validator(practice_set, rid, report, domain_name=""):
    """Validates practice problems and tracks failure rates to prevent silent data loss."""
    valid_practice = []
    seen_equations = set()

    for item in practice_set:
        problem_str = item.get("problem", "").split('=')[0]
        expected_str = str(item.get("expected_answer", ""))
        clean_eq = re.sub(r'[^0-9\+\-\*\/\.\(\)]', '', problem_str)
        normalized_problem = normalize_equation(problem_str)

        if normalized_problem in seen_equations:
            report["pedagogy_errors"].append(f"Repeated equation in same module practice set: {problem_str}")
            report["counts"]["pruned"] += 1
            continue

        if re.search(r'[\+\-\*\/]{2,}', clean_eq):
            report["math_errors"].append(f"Invalid operator sequence: {clean_eq}")
            report["counts"]["pruned"] += 1
            continue

        if any(op in clean_eq for op in ['*', '/']):
            report["pedagogy_errors"].append(f"Multiplication/Division out of scope: {problem_str}")
            report["counts"]["pruned"] += 1
            continue

        if not any(op in clean_eq for op in ['+', '-']):
            report["pedagogy_errors"].append(f"Non-symbolic problem: {problem_str}")
            report["counts"]["pruned"] += 1
            continue

        operands_ok, operands_error = validate_operand_bounds(problem_str, domain_name)
        if not operands_ok:
            report["pedagogy_errors"].append(operands_error)
            report["counts"]["pruned"] += 1
            continue

        computed = safe_eval(clean_eq)
        if computed is None:
            report["math_errors"].append(f"AST crash on: {clean_eq}")
            report["counts"]["pruned"] += 1
            continue

        ans_match = re.search(r'-?\d*\.?\d+', expected_str)
        if not ans_match:
            report["schema_errors"].append(f"No answer found in: {expected_str}")
            report["counts"]["pruned"] += 1
            continue

        try:
            if not math.isclose(computed, float(ans_match.group()), rel_tol=1e-9):
                report["math_errors"].append(f"{clean_eq} evaluated to {computed}, got {ans_match.group()}")
                report["counts"]["pruned"] += 1
                continue
            computed_int = int(computed)
        except:
            report["counts"]["pruned"] += 1
            continue

        hint = str(item.get("hint", "")).strip()

        if contains_invalid_subtraction(hint):
            report["pedagogy_errors"].append("Negative subtraction in hint.")
            report["counts"]["pruned"] += 1
            continue

        if not hint:
            report["pedagogy_errors"].append(f"Missing hint for practice problem: {problem_str}")
            report["counts"]["pruned"] += 1
            continue

        target_ok, target_error = validate_stated_final_target(problem_str, hint, computed_int, "hint")
        if not target_ok:
            report["pedagogy_errors"].append(target_error)
            report["counts"]["pruned"] += 1
            continue

        addition_ok, addition_error = validate_addition_hint_amount(problem_str, hint, "hint")
        if not addition_ok:
            report["pedagogy_errors"].append(addition_error)
            report["counts"]["pruned"] += 1
            continue

        hint_ok, hint_error = validate_hint_quality(problem_str, hint, domain_name)
        if not hint_ok:
            repaired_hint = generate_crossing_ten_subtraction_hint(problem_str)
            if repaired_hint:
                item["hint"] = repaired_hint
                hint = repaired_hint
                hint_ok, hint_error = validate_hint_quality(problem_str, hint, domain_name)

        if not hint_ok:
            report["pedagogy_errors"].append(hint_error)
            report["counts"]["pruned"] += 1
            continue

        valid_practice.append(item)
        seen_equations.add(normalized_problem)
        report["counts"]["returned"] += 1

    return valid_practice

def pedagogy_validator(example, rid, report):
    """Validates the worked example schema and math logic."""
    required_keys = {"problem", "reasoning_steps", "final_answer"}
    if not required_keys.issubset(set(example.keys())):
        report["schema_errors"].append(f"worked_example keys mismatch: missing {required_keys - set(example.keys())}")
        return False

    # Strip any extra keys the LLM may have added
    for extra_key in set(example.keys()) - required_keys:
        del example[extra_key]

    problem = example.get("problem", "").split('=')[0]
    clean_eq = re.sub(r'[^0-9\+\-\*\/\.\(\)]', '', problem)

    if re.search(r'[\+\-\*\/]{2,}', clean_eq):
         report["math_errors"].append(f"Invalid operator sequence in worked example: {clean_eq}")
         return False

    if any(op in clean_eq for op in ['*', '/']):
        report["pedagogy_errors"].append(f"Multiplication/Division out of scope in worked example: {clean_eq}")
        return False

    if not any(op in clean_eq for op in ['+', '-']):
        report["pedagogy_errors"].append(f"Non-symbolic worked example problem: {problem}")
        return False

    if clean_eq and any(op in clean_eq for op in ['+', '-']):
        computed = safe_eval(clean_eq)
        if computed is not None:
            nums = re.findall(r'-?\d*\.?\d+', str(example.get("final_answer", "")))
            if nums:
                try:
                    ans_val = float(nums[-1])
                    if not math.isclose(computed, ans_val, rel_tol=1e-9):
                        report["math_errors"].append(f"WORKED ERROR: {clean_eq} != {ans_val}")
                        return False

                    example["final_answer"] = int(ans_val)
                except Exception:
                    report["schema_errors"].append(f"Invalid final_answer: {example.get('final_answer')}")
                    return False
            else:
                report["schema_errors"].append(f"Invalid final_answer: {example.get('final_answer')}")
                return False
        else:
            report["math_errors"].append(f"AST crash in worked example: {clean_eq}")
            return False

    reasoning_steps = example.get("reasoning_steps", [])
    if not isinstance(reasoning_steps, list) or len(reasoning_steps) == 0:
        report["schema_errors"].append("Worked example reasoning_steps must be a non-empty list.")
        return False

    for idx, step in enumerate(reasoning_steps, start=1):
        if not str(step).strip().lower().startswith(f"step {idx}:"):
            report["pedagogy_errors"].append(
                f"Worked example step {idx} must start with 'Step {idx}:'"
            )
            return False

    if len(reasoning_steps) > 4:
        report["pedagogy_errors"].append("Worked example has too many steps (>4). Cognitive overload risk.")
        return False

    steps_text = " ".join(reasoning_steps)
    if contains_invalid_subtraction(steps_text):
        report["pedagogy_errors"].append("Negative subtraction logic in reasoning.")
        return False

    reasoning_ok, reasoning_error = validate_make_ten_reasoning(problem, steps_text, "worked example")
    if not reasoning_ok:
        report["pedagogy_errors"].append(reasoning_error)
        return False

    target_ok, target_error = validate_stated_final_target(problem, steps_text, int(example["final_answer"]), "worked example")
    if not target_ok:
        report["pedagogy_errors"].append(target_error)
        return False

    addition_ok, addition_error = validate_addition_hint_amount(problem, steps_text, "worked example")
    if not addition_ok:
        report["pedagogy_errors"].append(addition_error)
        return False

    return True

def validate_assessment_item(item: dict, assessment_equations=None) -> tuple[bool, str | None, dict | None]:
    q_text = str(item.get("question", item.get("problem", ""))).strip()
    ans_text = str(item.get("expected_answer", item.get("answer", ""))).strip()

    if not re.fullmatch(r"-?\d+", ans_text):
        return False, f"Assessment answer is not an integer: {item}", None

    q_lower = q_text.lower()
    conceptual_patterns = [
        "inverse operation",
        "what does",
        "represent",
        "number line",
        "first step",
        "how can",
        "explain",
        "strategy",
        "why",
    ]
    if any(pattern in q_lower for pattern in conceptual_patterns):
        return False, f"Assessment question is conceptual/open-ended, not a symbolic equation: {q_text}", None

    expr_match = re.search(r"-?\d+\s*[+\-]\s*-?\d+(?:\s*[+\-]\s*-?\d+)*", q_text)
    if not expr_match:
        return False, f"Assessment question is not a symbolic equation: {q_text}", None

    clean_eq = expr_match.group()
    normalized_eq = normalize_equation(clean_eq)
    if normalized_eq in set(assessment_equations or set()):
        return False, f"Repeated equation in formative assessment: {q_text}", None

    computed = safe_eval(clean_eq)
    if computed is None:
        return False, f"Assessment equation could not be evaluated: {q_text}", None

    if int(computed) != int(ans_text):
        return False, f"Assessment mismatch: {q_text} evaluated to {computed}, got {ans_text}", None

    return True, None, {"question": q_text, "expected_answer": int(ans_text)}

def generate_retest_items(
    task: str,
    count: int,
    used_ids: set,
    used_strs: set,
) -> list[dict]:
    """
    Deterministically generates gap-fill retest items when the item bank is
    exhausted. No LLM involvement. Returns at most `count` items.
 
    Covers: number_comparison, dot_matching, number_series,
            single_addition, single_subtraction.
 
    complex_arithmetic is NOT handled here — it falls through to the LLM
    in the endpoint because multi-digit questions require more varied structure.
 
    Each returned item has keys: id, question, correct.
    Hints are attached later in the endpoint from TASK_TEACHER_HINTS.
    """
 
    results = []
 
    if task == "number_comparison":
        candidates = []
        for a in range(2, 16):
            for b in range(2, 16):
                if a == b:
                    continue
                q_str = f"{a} vs {b}"
                q_id  = f"nc_gen_{a:02d}_{b:02d}"
                if q_id in used_ids or normalize_equation(q_str) in used_strs:
                    continue
                candidates.append({
                    "id":       q_id,
                    "question": q_str,
                    "correct":  max(a, b),
                })
        random.shuffle(candidates)
        results = candidates[:count]
 
    elif task == "dot_matching":
        match_candidates = []
        for n in range(1, 10):
            q_str = f"{n} vs {n}"
            q_id  = f"dm_gen_m_{n:02d}"
            if q_id in used_ids or normalize_equation(q_str) in used_strs:
                continue
            match_candidates.append({
                "id":       q_id,
                "question": q_str,
                "correct":  True,
            })
 
        non_match_candidates = []
        for a in range(1, 10):
            for b in range(1, 10):
                if a == b:
                    continue
                q_str = f"{a} vs {b}"
                q_id  = f"dm_gen_n_{a:02d}_{b:02d}"
                if q_id in used_ids or normalize_equation(q_str) in used_strs:
                    continue
                non_match_candidates.append({
                    "id":       q_id,
                    "question": q_str,
                    "correct":  False,
                })
 
        random.shuffle(match_candidates)
        random.shuffle(non_match_candidates)
 
        half = count // 2
        taken_match     = match_candidates[:half]
        taken_non_match = non_match_candidates[:count - half]
        combined = taken_match + taken_non_match
        random.shuffle(combined)
        results = combined[:count]
 
    elif task == "number_series":
        candidates = []
        steps = [1, 2, 3, 4, 5, 10]
 
        for step in steps:
            for start in range(1, 21):
                terms = [start + step * i for i in range(4)]
                answer = start + step * 4
                q_str = f"{terms[0]}, {terms[1]}, {terms[2]}, {terms[3]}, __"
                q_id  = f"ns_gen_a_{start:02d}_s{step}"
                if q_id not in used_ids and normalize_equation(q_str) not in used_strs:
                    candidates.append({
                        "id":       q_id,
                        "question": q_str,
                        "correct":  answer,
                    })
 
                if start > step * 4:
                    d_terms = [start - step * i for i in range(4)]
                    d_answer = start - step * 4
                    if d_answer > 0:
                        q_str_d = f"{d_terms[0]}, {d_terms[1]}, {d_terms[2]}, {d_terms[3]}, __"
                        q_id_d  = f"ns_gen_d_{start:02d}_s{step}"
                        if q_id_d not in used_ids and normalize_equation(q_str_d) not in used_strs:
                            candidates.append({
                                "id":       q_id_d,
                                "question": q_str_d,
                                "correct":  d_answer,
                            })
 
        random.shuffle(candidates)
        results = candidates[:count]
 
    elif task == "single_addition":
        candidates = []
        for a in range(1, 10):
            for b in range(1, 10):
                q_str = f"{a} + {b}"
                q_id  = f"add_gen_{a:02d}_{b:02d}"
                if q_id in used_ids or normalize_equation(q_str) in used_strs:
                    continue
                candidates.append({
                    "id":       q_id,
                    "question": q_str,
                    "correct":  a + b,
                })
        random.shuffle(candidates)
        results = candidates[:count]
 
    elif task == "single_subtraction":
        candidates = []
        for a in range(1, 19):   
            for b in range(1, 10):
                if a - b <= 0:
                    continue
                q_str = f"{a} - {b}"
                q_id  = f"sub_gen_{a:02d}_{b:02d}"
                if q_id in used_ids or normalize_equation(q_str) in used_strs:
                    continue
                candidates.append({
                    "id":       q_id,
                    "question": q_str,
                    "correct":  a - b,
                })
        random.shuffle(candidates)
        results = candidates[:count]
  
    return results

def get_lesson_from_qwen(
    ml_data,
    domains,
    count,
    rid,
    correction_prompt="",
    domain_explanations="",
    formative_assessment_count=2,
):
    print(f"[REQ {rid}] Pass 1: Drafting for {domains} (Target: {count} items)...")

    domain_rules = json.dumps(
        {d: DOMAIN_GENERATION_RULES.get(d, {}) for d in domains},
        indent=2
    )

    practice_count_rules = (
        json.dumps(count, indent=2)
        if isinstance(count, dict)
        else str(count)
    )

    inst = f"""
    <role>
    You are a Senior Special Education (SPED) Clinical Consultant in the Philippines specializing in early Dyscalculia intervention.
    </role>

    <mission>
    You MUST generate exactly {len(domains)} distinct intervention modules.
    TARGET DOMAINS: {", ".join(domains)}.
    SYNTHESIS: Do not just list features. Treat these domains as a single interacting cognitive profile. Explicitly acknowledge Addition as a relative strength if its severity is low.
    VOCABULARY: Use "may hinder", "slightly less consistent", or "emerging area" instead of "critical bottleneck" or "below average" for Typical profiles. Never use "multi-digit math" unless explicitly listed.
    </mission>

    <domain_explanations>
    {domain_explanations if domain_explanations else "No additional domain explanations provided."}
    </domain_explanations>

    <domain_specific_rules>
    {domain_rules if domain_rules else "No additional domain-specific rules provided."}
    </domain_specific_rules>

    <practice_count_rules>
    Each domain must use this exact number of practice items:
    {practice_count_rules}
    </practice_count_rules>

    <philippine_context_and_tone>
    - Context: Use locally familiar physical objects (e.g., mangoes, candies, blocks, peso coins, jeepney toys) for concrete hints.
    - Tone: Use empathetic, encouraging language. Write short, simple, child-friendly sentences.
    - If predicted_class is 0 or status is Typical, avoid diagnostic-sounding phrases like "significant difficulty", "severe", "critical", "major deficit", "high-risk", or "at-risk". Use softer phrases such as "relative weakness", "emerging need", or "may benefit from support".
    </philippine_context_and_tone>

    <clinical_precision>
    - SEVERITY WEIGHTING: You MUST look at the numerical 'Domain Severity' scores in the ML DATA.
    - PEDAGOGY: NEVER suggest "timed drills" or focus on speed. Dyscalculic students suffer from math anxiety. Focus on derived facts, making 10s, and chunking.
    - CRA FRAMEWORK: All hints MUST use Concrete-Representational-Abstract language. NEVER use abstract procedural terms like "carry over" or "borrow". ALWAYS anchor the hint in physical objects (e.g., "Count 9 peso coins, add 1 to make 10, then add the rest").
    - CROSSING-TEN SUBTRACTION FORMULA: For A - B where A > 10 and A - B < 10, first take away A - 10 to reach 10, then take away B - (A - 10). Example: 15 - 7 means take away 5 to reach 10, then 2 more. Example: 16 - 7 means take away 6 to reach 10, then 1 more. Never say "take away B to reach 10".
    - REASONING CONSISTENCY: The final number stated in hints or reasoning must match the actual answer. For addition, do not say to add more than the actual addend unless you explicitly take back/remove the extra amount.
    - If a domain has a HIGH severity (>0.20), provide "cognitive stretch" problems (e.g., crossing tens, 12+9 or 14-6). Avoid overly simple problems like 8-2 or 9-3.
    - If a domain is AS (Addition vs. Subtraction Asymmetry), the student likely relies on forward (addition-based) reasoning and has difficulty with mental inversion. To rehabilitate this, the practice set MUST contain BOTH addition (+) and subtraction (-) equations (e.g., at least 3 subtraction problems bridging tens, and at least 1 addition problem for scaffolding). For subtraction hints, teach make-10 or inverse-operation reasoning; avoid shallow "take away and count" hints.
    - If a domain is BC (Basic vs. Complex Arithmetic Contrast), you MUST explicitly focus on "crossing tens". Do not use generic "break down problems" phrasing. Explicitly teach breaking numbers into tens and ones (e.g., 15 + 7 -> 10 + 5 + 7).
    - If a domain is SN (Symbolic vs. Non-Symbolic), focus on translating physical objects/visuals into digits, but practice_set.problem must still be a pure symbolic equation. Put dots/objects only in hints or explanations, never in the problem field.
    - If a domain is Processing-Fluency Integration or Overall Processing Efficiency, focus on reducing cognitive load, using chunking strategies, and supporting flexible switching between addition and subtraction when arithmetic practice is appropriate.
    </clinical_precision>

    <module_structure>
    For EACH domain, your module MUST contain:
    1. Clinical Explanation: Why the student struggled. Use precise clinical language (e.g., "difficulty with mental inversion" or "forward vs. reverse reasoning").
    2. Learning Objectives: 3 specific, actionable goals.
    3. Conceptual Explanation: Explain the concept strictly in a procedural format.
       CRITICAL FORMAT: You MUST explicitly start each step with "Step 1:", "Step 2:", and "Step 3:". This applies to ALL domains, even abstract ones like Processing Efficiency.
       - FOR BC DOMAIN: "Step 1: Solve basic facts. Step 2: Break complex numbers into tens and ones. Step 3: Recombine to solve."
       - FOR OTHER DOMAINS: "Step 1: Understand... Step 2: Visualize... Step 3: Practice..."
    4. Teaching Strategy: Include at least TWO representation types (e.g., visual objects AND a number line).
    5. Worked Example: Provide "problem", "reasoning_steps" (ARRAY OF MAXIMUM 3 STEPS), and "final_answer" for a SINGLE simple math equation. Each reasoning step MUST start with "Step 1:", "Step 2:", and "Step 3:" in order.
    6. Practice Set: A JSON array with the exact number of symbolic equations specified for that domain in <practice_count_rules>.
    </module_structure>

    <schema_typing>
    - CRITICAL: The "expected_answer" field MUST ALWAYS BE A STRICT INTEGER (e.g., 8). Do NOT wrap it in quotes (e.g., "8" is FATAL).
    </schema_typing>

    <strict_rules>
    1. ANTI-REPETITION: Do not repeat the exact same equation within the same practice_set or formative_assessment array. Use varied number families to avoid redundancy.
    2. DIVERSITY RULE: Use at least 3 DIFFERENT fact families/number triplets in the practice set. Do NOT repeat the same base numbers.
    3. ANSWER VARIETY: Do not make most practice items have the same expected_answer. Use varied answers across each practice_set.
    4. HINT VARIETY: Each hint MUST use entirely different wording and objects. Do NOT repeat phrasing.
       - EXAMPLES: "Hold up 7 fingers, fold down 2", "Draw 9 dots, cross out 3", "Place 4 blocks, add 2 more".
    5. SYMBOLIC MATH ONLY: Every practice_set.problem MUST be a pure math equation with + or - (e.g., "17 - 9"). Never output object labels like "8 dots", "12 blocks", or text word problems in the problem field.
    6. TARGETING: If the domain is 'Addition vs. Subtraction Asymmetry', your practice_set MUST contain a mix of BOTH '+' and '-' operators. Do not generate only subtraction.
    7. BOUNDS: Keep math simple but appropriate to severity. Operands <= 20. For subtraction, keep the second operand <= 9 unless the target domain is Multi-Digit Addition and Subtraction.
    8. ASSESSMENT: Provide exactly {formative_assessment_count} Formative Assessment questions at the very end. Each assessment question MUST be a symbolic equation with + or - and a single integer answer. Do not ask conceptual/open-ended questions.
    </strict_rules>

    {correction_prompt}
    """

    payload = {
        "messages": [
            {"role": "system", "content": inst},
            {"role": "user", "content": f"<ml_data>\n{ml_data}\n</ml_data>\n\nGenerate the modules based ONLY on the target domains."}
        ]
    }
    return call_llm_gateway(payload, DRAFT_URL, DRAFT_HEADERS, rid, temp=0.0 if correction_prompt else 0.4, model_paths=DRAFT_MODELS_TO_TRY)

def format_with_cloud_llm(qwen_messy_text, ml_data_string, count, rid, formative_assessment_count = 3):
    print(f"[REQ {rid}] Pass 2: Converting draft to strict JSON...")

    practice_count_rules = (
        "\n".join(
            f'- "{domain}": exactly {item_count} practice_set items'
            for domain, item_count in count.items()
        )
        if isinstance(count, dict)
        else f"Every domain: exactly {count} practice_set items"
    )

    formative_assessments = ',\n            '.join([
        '{ "question": "<Symbolic equation question, e.g., What is 9 + 4?>", "expected_answer": <INTEGER ONLY, NO QUOTES> }'
        for _ in range(formative_assessment_count)
    ])

    architect_prompt = f"""
    You are a JSON formatting robot. Convert the Teacher Text into the EXACT JSON schema below.

    STRICT RULES:
    - DO NOT add extra keys. DO NOT omit required keys.
    - worked_example MUST include: "problem", "reasoning_steps", "final_answer". Each reasoning_steps item MUST start with "Step 1:", "Step 2:", "Step 3:" in order.
    - Every module object MUST include the exact key "practice_set". Do NOT rename it to "practice", "practice_items", "activities", or "questions".
    - practice_set counts by domain:
    {practice_count_rules}
    - Every practice_set item MUST have a "problem" that is a symbolic equation containing + or -. Convert any object-only draft item like "8 dots" into an equation such as "8 + 0" or a better grade-appropriate equation.
    - For Symbolic vs. Non-Symbolic modules, dots/objects belong in "hint", not in "problem".
    - Do not repeat the exact same equation within the same practice_set or formative_assessment array.
    - Keep expected_answer values varied within each practice_set; do not make every item answer to the same number.
    - Keep operands <= 20, and keep subtraction's second operand <= 9 unless the domain is Multi-Digit Addition and Subtraction.
    - For Typical (0) status, use soft support language. Avoid "significant difficulty", "severe", "critical", "major deficit", "high-risk", and "at-risk".
    - formative_assessment questions MUST be symbolic arithmetic equations with + or -, not conceptual questions about strategies, representations, or inverse operations.
    - CRITICAL: The "diagnostic_modules" array MUST contain ALL modules present in the Teacher Text. DO NOT stop after formatting just one module. If the Teacher Text has 3 domains, you MUST output 3 objects in the array.
    - FINAL STEP - MANDATORY: You MUST include a "formative_assessment" array with EXACTLY {formative_assessment_count} objects. This CANNOT be empty. If omitted, the output is rejected.
    - DO NOT wrap your output in ```json markdown blocks. Output the raw {{ JSON directly.

    REQUIRED SCHEMA:
    {{
        "status": "<Typical (0) or At-Risk (1)>",
        "decision_path_rationale": "<Summarize the underlying clinical rationale. FATAL ERROR: Do NOT mention or list the acronyms from the raw decision path (e.g., NC, DM, ADD, SUB). Only discuss the High Severity target domains.>",
        "overall_summary": "<Summarize in 2 sentences>",
        "diagnostic_modules": [
            {{
                "domain_name": "<Insert Exact Domain Name>",
                "clinical_explanation": "<Insert Clinical Explanation>",
                "learning_objectives": ["<Goal 1>", "<Goal 2>", "<Goal 3>"],
                "conceptual_explanation": "<Insert Child-Friendly Explanation>",
                "worked_example": {{
                    "problem": "<Insert Equation ONLY, e.g., 5+3>",
                    "reasoning_steps": ["Step 1: <Concrete action>", "Step 2: <Strategy>", "Step 3: <Answer check>"],
                    "final_answer": <Insert SINGLE NUMBER ONLY, e.g., 8>
                }},
                "teaching_strategy": "<Insert Strategy>",
                "practice_set": [
                    {{ "problem": "<Equation ONLY with + or -, e.g., 9 + 4>", "expected_answer": <INTEGER ONLY, NO QUOTES>, "hint": "<String max 120 chars; objects allowed here>" }}
                ]
            }}
        ],
        "formative_assessment": [
            {formative_assessments}
        ]
    }}
    """

    payload = {
        "messages": [
            {"role": "system", "content": architect_prompt},
            {"role": "user", "content": f"ML DATA:\n{ml_data_string}\n\nTEACHER TEXT:\n{qwen_messy_text}"}
        ]
    }
    raw_output = call_llm_gateway(payload, FORMAT_URL, FORMAT_HEADERS, rid, temp=0.0, model_paths=FORMAT_MODELS_TO_TRY)
    if not raw_output: return None

    parsed_json = extract_json_robustly(raw_output)
    if not parsed_json:
         print(f"[REQ {rid}] [ERROR] Python rejected the AI's JSON syntax.")
    return parsed_json

def schema_validator(data, expected_count, allowed_domains, rid, formative_assessment_count = 3, predicted_class=None):
    """Orchestrates the validation pipeline and builds the metrics report."""
    if not isinstance(data, dict): return None, {"fatal": "Root data is not a dictionary."}

    if isinstance(expected_count, dict):
        total_expected = sum(
            int(expected_count.get(domain, 4))
            for domain in allowed_domains
        )
    else:
        total_expected = int(expected_count) * len(allowed_domains)

    validation_report = {
        "counts": {"expected": total_expected, "returned": 0, "pruned": 0},
        "modules_expected": len(allowed_domains),
        "modules_passed": 0,
        "math_errors": [],
        "pedagogy_errors": [],
        "schema_errors": [],
        "warnings": []
    }

    required_top_keys = ["status", "decision_path_rationale", "overall_summary", "diagnostic_modules", "formative_assessment"]
    if not all(k in data for k in required_top_keys):
        validation_report["schema_errors"].append("Missing top-level schema keys.")
        return None, validation_report

    tone_class = predicted_class if predicted_class is not None else data.get("status", "")
    for key in ["decision_path_rationale", "overall_summary"]:
        tone_ok, tone_error = validate_tone_for_status(data.get(key, ""), tone_class)
        if not tone_ok:
            validation_report["pedagogy_errors"].append(f"{key}: {tone_error}")

    modules = data.get("diagnostic_modules", [])
    if len(modules) == 0:
        validation_report["schema_errors"].append("No diagnostic modules generated.")
        return None, validation_report

    allowed_norm = {normalize(d): d for d in allowed_domains}
    valid_modules = []

    for m in modules:
        required_keys = [
            "domain_name", "clinical_explanation", "learning_objectives",
            "conceptual_explanation", "worked_example", "practice_set", "teaching_strategy"
        ]

        m_cleaned = {k: v for k, v in m.items() if k in required_keys}
        if not set(required_keys).issubset(set(m_cleaned.keys())):
            validation_report["schema_errors"].append(f"Module keys mismatch in domain: {m.get('domain_name')}, missing: {set(required_keys) - set(m_cleaned.keys())}")
            continue

        norm_domain = normalize(m_cleaned.get("domain_name", ""))
        best_ratio = 0
        best_match_domain = None
        for a_norm, a_orig in allowed_norm.items():
            ratio = difflib.SequenceMatcher(None, norm_domain, a_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match_domain = a_orig

        if best_ratio < 0.70:
            validation_report["schema_errors"].append(f"Invalid domain name: '{m_cleaned.get('domain_name')}' (best match: '{best_match_domain}' at {best_ratio:.0%})")
            continue

        tone_ok, tone_error = validate_tone_for_status(m_cleaned.get("clinical_explanation", ""), tone_class)
        if not tone_ok:
            validation_report["pedagogy_errors"].append(f"clinical_explanation in domain {m_cleaned.get('domain_name')}: {tone_error}")

        m_cleaned["conceptual_explanation"] = m_cleaned.get("conceptual_explanation", "")\
            .replace("Child-friendly 'Why':", "").replace("Child-friendly \"Why\":", "").replace("**Why**:", "").replace("Why:", "").strip()

        if not validate_conceptual_steps(m_cleaned["conceptual_explanation"]):
            validation_report["pedagogy_errors"].append(f"Missing Step 1, 2, 3 format in domain: {m.get('domain_name')}")

        m_cleaned["teaching_strategy"] = m_cleaned.get("teaching_strategy", "").replace("High-level teacher approach:", "").strip()

        worked = m_cleaned.get("worked_example", {})
        if not pedagogy_validator(worked, rid, validation_report):
            continue

        domain_name = m_cleaned.get("domain_name")
        raw_practice = m_cleaned.get("practice_set", [])
        clean_practice = math_validator(raw_practice, rid, validation_report, domain_name)

        if len(clean_practice) == 0:
             validation_report["schema_errors"].append(f"No valid practice problems survived in domain: {m_cleaned.get('domain_name')}")
             continue

        if isinstance(expected_count, dict):
            domain_expected_count = int(expected_count.get(domain_name, 4))
        else:
            domain_expected_count = int(expected_count)

        if len(clean_practice) != domain_expected_count:
            validation_report["schema_errors"].append(
                f"Practice count mismatch in domain: {domain_name}. "
                f"Expected {domain_expected_count}, got {len(clean_practice)}."
            )
            continue

        if not check_hint_diversity(clean_practice):
            validation_report["pedagogy_errors"].append(f"Low hint diversity in domain: {m.get('domain_name')}")

        if not check_answer_diversity(clean_practice):
            validation_report["pedagogy_errors"].append(f"Low answer diversity in domain: {m.get('domain_name')}")

        if is_addition_subtraction_asymmetry(m_cleaned.get("domain_name", "")):
            problems = [str(p.get("problem", "")) for p in clean_practice]
            sub_count = sum(1 for p in problems if '-' in p)
            add_count = sum(1 for p in problems if '+' in p)
            if sub_count < 2 or add_count < 1:
                validation_report["schema_errors"].append(f"AS module failed subtraction/addition ratio (sub={sub_count}, add={add_count}).")
                continue

        m_cleaned["practice_set"] = clean_practice
        valid_modules.append(m_cleaned)

    validation_report["modules_passed"] = len(valid_modules)

    if len(valid_modules) != len(allowed_domains):
        validation_report["schema_errors"].append(
            f"Expected {len(allowed_domains)} modules, got {len(valid_modules)} valid modules."
        )
        return None, validation_report

    data["diagnostic_modules"] = valid_modules

    assessment = data.get("formative_assessment", [])
    if not isinstance(assessment, list) or len(assessment) == 0:
        validation_report["schema_errors"].append("Formative assessment missing or empty.")
        data["formative_assessment"] = []
    else:
        valid_assessments = []
        assessment_equations = set()
        for item in assessment:
            is_valid, error, cleaned = validate_assessment_item(item, assessment_equations)
            if is_valid:
                 valid_assessments.append(cleaned)
                 expr_match = re.search(r"-?\d+\s*[+\-]\s*-?\d+(?:\s*[+\-]\s*-?\d+)*", cleaned["question"])
                 if expr_match:
                     assessment_equations.add(normalize_equation(expr_match.group()))
            else:
                 validation_report["schema_errors"].append(error)

        if len(valid_assessments) != formative_assessment_count:
            validation_report["schema_errors"].append(
                f"Formative assessment has {len(valid_assessments)} valid items instead of {formative_assessment_count}."
            )
        data["formative_assessment"] = valid_assessments

    validation_report["success_rate"] = (
        validation_report["counts"]["returned"] /
        max(1, validation_report["counts"]["expected"])
    )

    if (
        validation_report["math_errors"] or
        validation_report["pedagogy_errors"] or
        validation_report["schema_errors"]
    ):
        return None, validation_report

    return data, validation_report


def validate_hint_quality(problem: str, hint: str, domain_name: str = "") -> tuple[bool, str | None]:
    """
    Reject mathematically correct but pedagogically poor hints.
    """
    hint_l = str(hint).lower()

    for pattern in BAD_HINT_PATTERNS:
        if re.search(pattern, hint_l):
            return False, f"Bad hint pattern '{pattern}' in hint: {hint}"

    # Reject weird decomposition language like "break 7 into 9 and -2"
    if re.search(r"break\s+\d+\s+into\s+\d+\s+and\s+-\d+", hint_l):
        return False, f"Invalid decomposition with negative part: {hint}"

    # For subtraction crossing ten, require clearer language.
    # Example: 14 - 7 should say take away enough to reach 10, then take the rest.
    clean_eq = re.sub(r"[^0-9+\-\s]", "", str(problem))
    nums = [int(n) for n in re.findall(r"\d+", clean_eq)]

    if "-" in clean_eq and len(nums) >= 2:
        a, b = nums[0], nums[1]
        if a > 10 and a - b < 10:
            has_bridge_language = any(
                phrase in hint_l
                for phrase in [
                    "reach 10",
                    "make 10",
                    "down to 10",
                    "take away",
                    "remaining",
                    "left"
                ]
            )
            if not has_bridge_language:
                return False, f"Subtraction crossing-ten hint is unclear: {hint}"

            reasoning_ok, reasoning_error = validate_make_ten_reasoning(problem, hint, "hint")
            if not reasoning_ok:
                return False, reasoning_error

        if is_addition_subtraction_asymmetry(domain_name):
            has_as_strategy = any(
                phrase in hint_l
                for phrase in [
                    "reach 10",
                    "make 10",
                    "related addition fact",
                    "what plus",
                    "fact family"
                ]
            )
            if not has_as_strategy:
                return False, f"AS subtraction hint lacks make-10 or inverse-operation strategy: {hint}"

    return True, None

def get_single_pass_lesson(ml_data_string, domains, count, rid, correction_prompt="", domain_explanations="", formative_assessment_count=3):
    print(f"[REQ {rid}] EXP 1: SINGLE-PASS GENERATION...")
    
    domain_rules = json.dumps({d: DOMAIN_GENERATION_RULES.get(d, {}) for d in domains}, indent=2)
    
    formative_assessments = ',\n            '.join([
        '{ "question": "<Symbolic equation, e.g., 9 + 4>", "expected_answer": <INTEGER> }'
        for _ in range(formative_assessment_count)
    ])

    inst = f"""
    You are a SPED Teacher and a strict JSON formatting robot. 
    You MUST generate {len(domains)} intervention modules based on the ML data.
    TARGET DOMAINS: {", ".join(domains)}.

    <pedagogy_rules>
    1. Use Philippine context (mangoes, peso coins).
    2. Every conceptual_explanation must contain Step 1:, Step 2:, and Step 3:.
    3. Keep operands <= 20.
    {domain_rules}
    </pedagogy_rules>

    <json_rules>
    You MUST output RAW JSON EXACTLY matching this schema. NO markdown formatting.
    {{
        "status": "<Typical (0) or At-Risk (1)>",
        "decision_path_rationale": "<Summarize rationale>",
        "overall_summary": "<Summarize in 2 sentences>",
        "diagnostic_modules": [
            {{
                "domain_name": "<Insert Exact Domain Name>",
                "clinical_explanation": "<Explanation>",
                "learning_objectives": ["<Goal 1>", "<Goal 2>", "<Goal 3>"],
                "conceptual_explanation": "<Step 1: ... Step 2: ... Step 3: ...>",
                "worked_example": {{
                    "problem": "<Equation ONLY>",
                    "reasoning_steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
                    "final_answer": <INTEGER ONLY>
                }},
                "teaching_strategy": "<Strategy>",
                "practice_set": [
                    {{ "problem": "<Equation ONLY>", "expected_answer": <INTEGER ONLY>, "hint": "<Hint text>" }}
                ]
            }}
        ],
        "formative_assessment": [
            {formative_assessments}
        ]
    }}
    </json_rules>
    {correction_prompt}
    """

    payload = {
        "messages": [
            {"role": "system", "content": inst},
            {"role": "user", "content": f"ML DATA:\n{ml_data_string}\nGenerate the JSON."}
        ]
    }
    
    raw_output = call_llm_gateway(payload, DRAFT_URL, DRAFT_HEADERS, rid, temp=0.3, model_paths=DRAFT_MODELS_TO_TRY)
    if not raw_output: return None
    
    return extract_json_robustly(raw_output)
