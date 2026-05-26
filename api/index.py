from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import sys
import uuid
from dotenv import load_dotenv
import time
import difflib
import logging
import random
from collections import defaultdict
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.constants import (
    OPENROUTER_TOKEN,
    DOMAIN_EXPLANATIONS,
    EXPERIMENT_MODE,
    DRAFT_URL,
    DRAFT_HEADERS,
    DRAFT_MODELS_TO_TRY,
    TASK_ITEM_BANK,
    TASK_QUESTION_COUNTS,
    ACRONYM_TO_TASK,
    TASK_TEACHER_HINTS,
    RETEST_SAMPLE_SIZE

)
from llm.helpers import (
    get_top_deficits,
    calculate_practice_tiers_by_domain,
    interpret_decision_path,
    call_llm_gateway,
    extract_json_robustly,
    schema_validator,
    math_validator,
    get_lesson_from_qwen,
    format_with_cloud_llm,
    get_single_pass_lesson,
    validate_hint_quality,            
    validate_tone_for_status,       
    normalize_equation,
    generate_retest_items              
)

from ml.helpers import (
    process_and_predict_diagnostic
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

if not OPENROUTER_TOKEN:
    raise ValueError("Missing OPENROUTER_TOKEN in environment variables")

@app.route('/generate_module', methods=['POST'])
def generate_module():
    """
        Generate learning path from ML diagnostic data.

        Request Format:
        {
            "test_id": {"<string>"},
            "diagnostic_data": {
                "predicted_class": {"<string>"},
                "confidence": {<float>},
                "decision_path": [("<string>", <float>, "<string>")],
                "domain_severity_scores": {"<string>": <float>},
                "task_importance_scores": {"<string>": <float>}
            }
        }
    """
    try:
        request_data = request.json
        if not request_data: return jsonify({"error": "Invalid request"}), 400

        rid = request_data.get("test_id", uuid.uuid4().hex[:6])
        ml_data = request_data.get("diagnostic_data", None)

        if not ml_data:
            return jsonify({"error": "No ML data provided"}), 400

        logger.info(f"[REQ {rid}] Decision Path: {ml_data}")

        required_fields = [
            "domain_severity_scores",
            "task_importance_scores",
            "confidence",
            "predicted_class",
            "decision_path",
        ]

        missing_fields = [
            field for field in required_fields
            if field not in ml_data or ml_data[field] is None
        ]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        domain_severity_scores = ml_data["domain_severity_scores"]
        task_importance_scores = ml_data["task_importance_scores"]
        confidence = ml_data["confidence"]
        predicted_class = ml_data["predicted_class"]
        decision_path = ml_data["decision_path"]

        if not isinstance(domain_severity_scores, dict) or not domain_severity_scores:
            return jsonify({"error": "domain_severity_scores must be a non-empty object"}), 400

        if not isinstance(task_importance_scores, dict):
            return jsonify({"error": "task_importance_scores must be an object"}), 400

        if not isinstance(decision_path, list):
            return jsonify({"error": "decision_path must be a list"}), 400
        if decision_path and not all(
            isinstance(item, (list, tuple)) and len(item) == 3 and isinstance(item[1], (int, float))
            for item in decision_path
        ):
            return jsonify({"error": "Each item in decision_path must be a tuple/list of (domain, score, reason)"}), 400

        top_domains = get_top_deficits(ml_data, rid, max_domains=3)
        practice_tiers = calculate_practice_tiers_by_domain(domain_severity_scores, top_domains)
        interpretation = interpret_decision_path(ml_data)
        domain_explanations = json.dumps(DOMAIN_EXPLANATIONS, indent=2)

        correction_prompt = ""
        formative_assessment_count = 5
        best_candidate = None
        best_report = None
        best_score = -1
        for attempt in range(5):
            print(f"\n[REQ {rid}] Pipeline Loop: Attempt {attempt + 1}")
            try:
                # --- THE EXPERIMENT ROUTER ---
                if str(EXPERIMENT_MODE) == "1":
                    perfect_json = get_single_pass_lesson(
                        json.dumps(ml_data), top_domains, practice_tiers, rid, 
                        correction_prompt, domain_explanations, formative_assessment_count
                    )
                    if not perfect_json: raise Exception("Single Pass JSON extraction failed")
                else:
                    draft = get_lesson_from_qwen(
                        json.dumps(ml_data), top_domains, practice_tiers, rid, 
                        correction_prompt, domain_explanations, formative_assessment_count
                    )
                    if not draft: raise Exception("Pass 1 yielded empty draft")

                    perfect_json = format_with_cloud_llm(
                        draft, json.dumps(ml_data), practice_tiers, rid, formative_assessment_count
                    )
                    if not perfect_json: raise Exception("Pass 2 JSON extraction failed")

                validated_data, validation_report = schema_validator(
                    perfect_json,
                    practice_tiers,
                    top_domains,
                    rid,
                    formative_assessment_count,
                    predicted_class,
                )

                score = (
                    validation_report.get("counts", {}).get("returned", 0) -
                    validation_report.get("counts", {}).get("pruned", 0)
                )
                if score > best_score:
                    best_score = score
                    best_candidate = perfect_json
                    best_report = validation_report

                if validation_report["math_errors"] or validation_report["pedagogy_errors"] or validation_report["schema_errors"]:
                     logger.warning(f"[REQ {rid}] [VALIDATION REPORT]: {json.dumps(validation_report, indent=2)}")

                if validated_data:
                    logger.info(f"[REQ {rid}] [SUCCESS] VALIDATION PASSED")
                    validated_data["_meta_validation_report"] = validation_report
                    validated_data["decision_path_interpretation"] = interpretation
                    return jsonify(validated_data), 200
                else:
                    logger.warning(f"[REQ {rid}] [FAIL] Triggering Repair Loop.")

                    math_errors = validation_report["math_errors"]
                    pedagogy_errors = validation_report["pedagogy_errors"]
                    schema_errors = validation_report["schema_errors"]

                    correction_prompt = f"\nCRITICAL CORRECTION FROM SYSTEM:\nYour last output failed validation. Fix these exact errors: "
                    if math_errors: correction_prompt += f"Math Errors: {math_errors}. "
                    if pedagogy_errors: correction_prompt += f"Pedagogy Errors: {pedagogy_errors}. "
                    if schema_errors: correction_prompt += f"Schema Errors: {schema_errors}. "

                    correction_prompt += (
                        "Every practice item must include a non-empty hint. "
                        "For subtraction crossing-ten items, the hint must say how to reach 10 first, "
                        "then subtract the remaining amount. "
                        "Use this exact rule for A - B when A > 10 and A - B < 10: "
                        "first take away A - 10 to reach 10, then take away B - (A - 10). "
                        "For example, 15 - 7 takes away 5 to reach 10, then 2 more; "
                        "16 - 7 takes away 6 to reach 10, then 1 more. "
                        "Every conceptual_explanation must contain Step 1:, Step 2:, and Step 3:. "
                    )

                    if any("No answer found" in e for e in schema_errors):
                        correction_prompt += (
                            "Every practice_set item must include expected_answer as an integer. "
                            "Example: {\"problem\": \"8 + 4\", \"expected_answer\": 12, "
                            "\"hint\": \"Use 8 blocks and add 4 more.\"}. "
                        )

                    if any("AS module failed" in e for e in schema_errors):
                        correction_prompt += (
                            "The Addition vs. Subtraction Asymmetry module must include at least "
                            "3 subtraction problems and at least 1 addition problem. "
                        )

                    if any("Practice count mismatch" in e for e in schema_errors):
                        correction_prompt += (
                            "Do not remove or shorten practice_set arrays. Each module must contain "
                            "exactly the requested number of items. "
                        )

                    if any("Repeated equation" in e for e in pedagogy_errors):
                        correction_prompt += (
                            "Replace only the repeated equation with a different equation and keep "
                            "all other valid items. "
                        )

                    if any("Low answer diversity" in e for e in pedagogy_errors):
                        correction_prompt += (
                            "Vary the expected_answer values within each practice_set; do not make "
                            "most items solve to the same number. "
                        )

                    if any("Tone too strong" in e for e in pedagogy_errors):
                        correction_prompt += (
                            "For Typical (0) profiles, soften the language. Use phrases like "
                            "\"relative weakness\", \"emerging need\", or \"may benefit from support\" "
                            "instead of severe or at-risk language. "
                        )

                    if any("Subtrahend too large" in e or "Operand above 20" in e for e in pedagogy_errors):
                        correction_prompt += (
                            "Keep operands <= 20 and keep subtraction's second operand <= 9 unless "
                            "the target domain is Multi-Digit Addition and Subtraction. "
                        )

                    logger.info(f"[REQ {rid}] [REPAIR] Adding correction prompt: {correction_prompt}")
                    time.sleep(2 ** attempt)

            except Exception as e:
                logger.exception(f"[REQ {rid}] [RETRY] Pipeline step failed")
                time.sleep(2 ** attempt)

        return jsonify({
            "error": "Failed to generate fully valid module after retries",
            "best_validation_report": best_report,
            "best_candidate": best_candidate
        }), 500

    except Exception as e:
        logger.exception(f"[REQ {rid}] [ERROR] Critical Failure")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_retest', methods=['POST'])
def generate_retest():
    """
    Generate a retest set for a student's top 3 deficit tasks.
 
    Strategy (hybrid bank + algorithmic + LLM rationale):
      1. Identify top 3 tasks from task_importance_scores (latest session).
      2. For each task, sample the full original question count from the item
         bank, excluding question IDs already asked across ALL sessions.
      3. If the bank is exhausted:
           - NC, DM, NS, ADD, SUB: fill algorithmically via generate_retest_items()
           - CA only: fill via LLM gap-fill
      4. LLM generates ONLY the rationale strings (one per task).
      5. Teacher hints come from TASK_TEACHER_HINTS — not LLM-generated.
 
    Input/output schema: unchanged. See docstring of previous version.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request"}), 400
 
        student_history = data.get('student_history', [])
        if not student_history or not isinstance(student_history, list):
            return jsonify({"error": "Missing or empty student_history array"}), 400
 
        latest_session  = student_history[-1]
        ml_data         = latest_session.get('diagnostic_data', {})
        predicted_class = ml_data.get('predicted_class', '0')
        rid             = "RETEST_" + uuid.uuid4().hex[:4]
 
        task_scores = ml_data.get("task_importance_scores", {})
        if not task_scores:
            return jsonify({"error": "Missing task_importance_scores in diagnostic_data"}), 400
 
        filtered = {
            ACRONYM_TO_TASK[k]: v
            for k, v in task_scores.items()
            if k in ACRONYM_TO_TASK
        }
        top_tasks = sorted(filtered, key=filtered.get, reverse=True)[:3]
 
        logger.info(
            f"[{rid}] Retest | session={latest_session.get('session_id','?')} "
            f"| top_tasks={top_tasks}"
        )
 
        TASK_TO_ACRONYM = {v: k for k, v in ACRONYM_TO_TASK.items()}
 
        used_ids_by_task = defaultdict(set)
        used_str_by_task = defaultdict(set)
 
        for session in student_history:
            for q in session.get('questions_asked', []):
                q_id  = q.get('id', '')
                q_str = normalize_equation(q.get('question', ''))
 
                for task, acronym in TASK_TO_ACRONYM.items():
                    prefix = acronym.lower() + "_"
                    if q_id.lower().startswith(prefix):
                        if q_id:
                            used_ids_by_task[task].add(q_id)
                        used_str_by_task[task].add(q_str)
                        break
 
        bank_results = {}
        llm_needed   = {}
 
        for task in top_tasks:
            bank     = TASK_ITEM_BANK.get(task, [])
            target   = TASK_QUESTION_COUNTS.get(task, 10)
            used_ids = used_ids_by_task[task]
            used_strs = used_str_by_task[task]
 
            available = [
                item for item in bank
                if item.get('id', '') not in used_ids
                and normalize_equation(
                    item.get('question') or item.get('sequence', '')
                ) not in used_strs
            ]
 
            random.shuffle(available)
 
            if task == "complex_arithmetic":
                if len(available) >= target:
                    bank_results[task] = available[:target]
                    llm_needed[task]   = 0
                else:
                    bank_results[task] = available
                    llm_needed[task]   = target - len(available)
            else:
                if len(available) >= target:
                    bank_results[task] = available[:target]
                else:
                    used_in_bank = [
                        item for item in bank
                        if item.get('id', '') in used_ids
                        or normalize_equation(item.get('question') or item.get('sequence', '')) in used_strs
                    ]
                    random.shuffle(used_in_bank)
                    bank_results[task] = available + used_in_bank[:target - len(available)]
                llm_needed[task] = 0
 
            logger.info(
                f"[{rid}] {task}: bank_available={len(available)} "
                f"target={target} from_bank={len(bank_results[task])} "
                f"gap_needed={llm_needed[task]} excluded={len(used_ids)}"
            )
 
        gap_fill = {t: [] for t in top_tasks}
 
        for task in top_tasks:
            shortfall = llm_needed.get(task, 0)
            if shortfall == 0:
                continue
 
            if task != "complex_arithmetic":
                generated = generate_retest_items(
                    task,
                    shortfall,
                    used_ids_by_task[task],
                    used_str_by_task[task],
                )
                gap_fill[task] = generated
                logger.info(
                    f"[{rid}] {task}: algorithmic gap-fill generated "
                    f"{len(generated)} / {shortfall} needed"
                )
 
        ca_shortfall = llm_needed.get("complex_arithmetic", 0)
        ca_in_top    = "complex_arithmetic" in top_tasks
 
        ca_gap_instructions = ""
        if ca_in_top and ca_shortfall > 0:
            already_asked_ca = (
                [item.get('question', '') for item in TASK_ITEM_BANK.get("complex_arithmetic", [])]
                + list(used_str_by_task["complex_arithmetic"])
            )
            ca_generate_count = ca_shortfall + 5
            ca_gap_instructions = f"""
 
GAP_FILL_TASKS:
complex_arithmetic — generate exactly {ca_generate_count} NEW multi-digit addition
or subtraction questions.
Format: symbolic equation only (e.g., "25 + 12", "150 - 80").
correct must be the integer answer.
hint: teacher observation guide, CRA-grounded, max 120 chars.
Do NOT repeat: {json.dumps(already_asked_ca[:30])}
"""
 
        rationale_prompt = f"""
You are a Senior Special Education (SPED) Clinical Consultant in the Philippines.
 
Generate a JSON object with:
1. A "rationale" for EACH of the following tasks explaining WHY it is being
   retested based on the student's latest diagnostic profile (2-3 sentences,
   reference specific scores).
2. If GAP_FILL_TASKS are listed below, generate fresh questions under "gap_fill".
 
LATEST DIAGNOSTIC PROFILE:
{json.dumps(ml_data, indent=2)}
 
TASKS TO RATIONALE: {', '.join(top_tasks)}
{ca_gap_instructions}
 
CLINICAL TONE:
- If predicted_class is Typical (0), avoid "severe", "critical", "high-risk".
- Keep rationales concise. Reference specific task importance scores.
 
OUTPUT RAW JSON ONLY. No markdown fences. Schema:
{{
    "rationales": {{
        "<task_name>": "<rationale string>"
    }},
    "gap_fill": {{
        "complex_arithmetic": [
            {{
                "question": "<symbolic equation>",
                "correct": <integer>,
                "hint": "<teacher observation hint>"
            }}
        ]
    }}
}}
"""
 
        rationale_payload = {
            "messages": [
                {"role": "system", "content": rationale_prompt},
                {"role": "user",   "content": "Generate the rationales now."}
            ]
        }
 
        rationales = {}
        ca_llm_raw = []
 
        for attempt in range(3):
            try:
                raw = call_llm_gateway(
                    rationale_payload, DRAFT_URL, DRAFT_HEADERS,
                    rid, temp=0.4, model_paths=DRAFT_MODELS_TO_TRY
                )
                parsed = extract_json_robustly(raw)
                if parsed:
                    rationales  = parsed.get("rationales", {})
                    ca_llm_raw  = parsed.get("gap_fill", {}).get("complex_arithmetic", [])
                    break
            except Exception as e:
                logger.warning(f"[{rid}] LLM rationale attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)

        report = {
            "counts":          {"returned": 0, "pruned": 0},
            "tasks":           {},
            "math_errors":     [],
            "pedagogy_errors": [],
            "schema_errors":   [],
        }
 
        if ca_in_top and ca_shortfall > 0:
            validated_ca = []
            for item in ca_llm_raw:
                question = item.get("question", "")
                correct  = item.get("correct")
                hint     = str(item.get("hint", "")).strip()
 
                if normalize_equation(question) in used_str_by_task["complex_arithmetic"]:
                    report["counts"]["pruned"] += 1
                    continue
 
                probe = {
                    "problem":         question,
                    "expected_answer": correct,
                    "hint":            hint or TASK_TEACHER_HINTS.get("complex_arithmetic", "")
                }
                validated = math_validator([probe], rid, report, domain_name="complex_arithmetic")
                if not validated:
                    continue
 
                tone_ok, tone_error = validate_tone_for_status(hint, predicted_class)
                if not tone_ok:
                    report["pedagogy_errors"].append(f"[complex_arithmetic] {tone_error}")
                    report["counts"]["pruned"] += 1
                    continue
 
                validated_ca.append({
                    "question": validated[0].get("problem", question),
                    "correct":  validated[0].get("expected_answer", correct),
                    "hint":     validated[0].get("hint", hint),
                })
 
                if len(validated_ca) >= ca_shortfall:
                    break
 
            gap_fill["complex_arithmetic"] = validated_ca
 
        retest_data = {}
 
        for task in top_tasks:
            bank_items = bank_results[task]
            algo_items = gap_fill.get(task, [])
 
            algo_items = algo_items[:llm_needed.get(task, 0)]
 
            bank_tests = []
            for item in bank_items:
                question = item.get("question") or item.get("sequence", "")
                correct  = (
                    item.get("correct")
                    if item.get("correct") is not None
                    else item.get("match")
                )
                bank_tests.append({
                    "id":       item.get("id", ""),
                    "question": question,
                    "correct":  correct,
                    "hint":     TASK_TEACHER_HINTS.get(task, "Observe the student's strategy."),
                })
 
            algo_tests = []
            for i, item in enumerate(algo_items):
                item_id = item.get("id") or (
                    f"{TASK_TO_ACRONYM.get(task, 'xx').lower()}_llm_{i+1:02d}"
                )
                algo_tests.append({
                    "id":       item_id,
                    "question": item.get("question", ""),
                    "correct":  item.get("correct"),
                    "hint":     item.get("hint", TASK_TEACHER_HINTS.get(task, "Observe the student's strategy.")),
                })
 
            all_tests = bank_tests + algo_tests
            from_bank = len(bank_tests)
            from_algo = len(algo_tests)
            excluded  = len(used_ids_by_task[task])
 
            report["tasks"][task] = {
                "from_bank":           from_bank,
                "from_gap_fill":            from_algo,
                "excluded_from_pool":  excluded,
                "target":              TASK_QUESTION_COUNTS.get(task, 0),
            }
            report["counts"]["returned"] += len(all_tests)
 
            retest_data[task] = {
                "rationale": rationales.get(
                    task,
                    f"Retesting {task} based on latest diagnostic profile."
                ),
                "tests": all_tests,
            }
 
        short_tasks = [
            t for t in top_tasks
            if len(retest_data[t]["tests"]) < TASK_QUESTION_COUNTS.get(t, 0)
        ]
 
        has_blocking_errors = (
            report["math_errors"]
            or report["schema_errors"]
            or [e for e in report["pedagogy_errors"] if "Low" not in e]
        )
 
        base_response = {
            "retest_data":               retest_data,
            "_meta_validation_report":   report,
            "based_on_session":          latest_session.get("session_id", "Unknown"),
            "based_on_session_date":     latest_session.get("date", "Unknown"),
            "total_sessions_in_history": len(student_history),
        }
 
        if not short_tasks:
            logger.info(
                f"[{rid}] Retest PASSED — "
                f"{report['counts']['returned']} questions across {len(top_tasks)} tasks"
            )
            return jsonify(base_response), 200
 
        logger.warning(
            f"[{rid}] Retest PARTIAL — short tasks: {short_tasks}"
        )
        base_response["warning"] = (
            f"Some tasks returned fewer questions than the target. "
            f"Short: {short_tasks}. Review _meta_validation_report."
        )
        return jsonify(base_response), 207
 
    except Exception as e:
        logger.exception(f"[RETEST] Critical failure: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/generate-diagnostic", methods=["POST"])
def generate_diagnostic():
    """
        Generate learning path from ML diagnostic data.

        Request Format:
        {
            "test_id": {"<string>"},
            "number_comparison": {"<float>"},
            "dot_matching": {"<float>"},
            "number_series": {"<int>"},
            "single_addition": {"<int>"},
            "single_subtraction": {"<int>"},
            "complex_arithmetic": {"<int>"}
        }
    """
    try:
        test_results = request.json
        if not test_results: return jsonify({"error": "Invalid request"}), 400

        test_id = test_results.pop("test_id", None)

        # 1-Line Data Processing & Prediction
        diagnostic_result = process_and_predict_diagnostic(test_results)

        logger.info(f"[REQ {test_id}] Diagnostics: {diagnostic_result}")
        return jsonify(diagnostic_result), 200
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    try:
        flask_env = os.getenv("FLASK_ENV")
        env_available = bool(flask_env) and bool(OPENROUTER_TOKEN) and bool(EXPERIMENT_MODE)
        message = "API is running. Environment variables loaded successfully." if env_available else "API is running. Some environment variables are missing."
        
        return jsonify({
            "status": "healthy", 
            "message": message,
            "env_status": {
                "FLASK_ENV": bool(flask_env),
                "OPENROUTER_TOKEN": bool(OPENROUTER_TOKEN),
                "EXPERIMENT_MODE": bool(EXPERIMENT_MODE)
            }
        }), 200
    except Exception as e:
        logger.exception("Health check failed")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    debug = os.getenv("FLASK_ENV") == "development"
    app.run(port=5000, debug=debug)
