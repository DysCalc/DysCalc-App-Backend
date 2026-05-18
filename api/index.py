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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.constants import (
    OPENROUTER_TOKEN,
    DOMAIN_EXPLANATIONS,
    EXPERIMENT_MODE,
    DRAFT_URL,
    DRAFT_HEADERS,
    DRAFT_MODELS_TO_TRY,
    DOMAIN_GENERATION_RULES

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
    normalize_equation              
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
    Generate a fresh, fully validated set of retest questions targeting the
    top 3 deficit tasks from a student's longitudinal session history.

    Always uses the LATEST session's diagnostic profile as the generation basis.
    Pools ALL previous questions across ALL sessions for deduplication.
    Requires task_importance_scores in the latest session's diagnostic_data.

    Request Format:
    {
        "student_history": [
            {
                "session_id": <int>,
                "date": "<string>",
                "diagnostic_data": {
                    "predicted_class": "<string>",
                    "domain_severity_scores": { "<domain_name>": <float> },
                    "task_importance_scores": { "<task_key>": <float> }
                },
                "questions_asked": [
                    {"question": "<string>", "correct": <int>}
                ]
            }
        ]
    }

    Success Response (HTTP 200):
    {
        "retest_data": {
            "<task_name>": {
                "rationale": "<string>",
                "tests": [
                    {
                        "question": "<string>",
                        "correct": <int or bool>,
                        "hint": "<string>"
                    }
                ]
            }
        },
        "_meta_validation_report": {
            "counts": {"returned": <int>, "pruned": <int>},
            "math_errors": [],
            "pedagogy_errors": [],
            "schema_errors": []
        },
        "based_on_session": <int>,
        "based_on_session_date": "<string>",
        "total_sessions_in_history": <int>
    }

    Partial Response (HTTP 207):
    {
        "retest_data": { ... },
        "_meta_validation_report": { ... },
        "based_on_session": <int>,
        "based_on_session_date": "<string>",
        "total_sessions_in_history": <int>,
        "warning": "<string>"
    }

    Failure Response (HTTP 500):
    {
        "error": "<string>",
        "best_validation_report": { ... }
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request"}), 400

        student_history = data.get('student_history', [])
        if not student_history or not isinstance(student_history, list):
            return jsonify({"error": "Missing or empty student_history array"}), 400

        latest_session = student_history[-1]
        ml_data = latest_session.get('diagnostic_data', {})
        predicted_class = ml_data.get('predicted_class', '0')

        rid = "RETEST_" + uuid.uuid4().hex[:4]
        
        task_scores = ml_data.get("task_importance_scores", {})
        if not task_scores:
            return jsonify({"error": "Missing task_importance_scores in diagnostic_data"}), 400
            
        ACRONYM_TO_TASK = {
            "NC": "number_comparison",
            "DM": "dot_matching",
            "NS": "number_series",
            "ADD": "single_addition",
            "SUB": "single_subtraction",
            "CA": "complex_arithmetic"
        }
        
        filtered_task_scores = {
            ACRONYM_TO_TASK[k]: v 
            for k, v in task_scores.items() 
            if k in ACRONYM_TO_TASK
        }
            
        top_tasks = sorted(filtered_task_scores, key=filtered_task_scores.get, reverse=True)[:3]
        
        TASK_TO_DOMAIN = {
            "number_comparison": "Number Comparison",
            "dot_matching": "Digit-Dot Matching",
            "number_series": "Number Series",
            "single_addition": "Single-Digit Addition",
            "single_subtraction": "Single-Digit Subtraction",
            "complex_arithmetic": "Multi-Digit Addition and Subtraction",
        }

        domain_rules_dict = {t: DOMAIN_GENERATION_RULES.get(TASK_TO_DOMAIN.get(t, t), {}) for t in top_tasks}
        domain_rules = json.dumps(domain_rules_dict, indent=2)

        # Strict History Tracking & Deduplication (Pooling all sessions)
        all_previous_questions = []
        for session in student_history:
            all_previous_questions.extend(session.get('questions_asked', []))

        logger.info(
            f"[{rid}] Retest | session={latest_session.get('session_id', '?')} "
            f"| top_tasks={top_tasks} | history_pool={len(all_previous_questions)} questions"
        )

        retest_prompt = f"""
        <role>
        You are a Senior Special Education (SPED) Clinical Consultant in the Philippines specializing in early Dyscalculia intervention.
        </role>

        <mission>
        Generate a FRESH set of retest practice problems for the student's TOP 3 deficit tasks based on the LATEST diagnostic profile below. 
        You MUST generate exactly 3 questions per task.

        PRIMARY TASKS TO RETEST: {', '.join(top_tasks)}
        </mission>

        <latest_diagnostic_profile>
        {json.dumps(ml_data, indent=2)}
        </latest_diagnostic_profile>

        <history_pool_do_not_repeat>
        The following problems have already been asked. Do NOT generate any problem that is exactly the same as these:
        {json.dumps([q.get("question", "") for q in all_previous_questions])}
        </history_pool_do_not_repeat>

        <domain_specific_rules>
        {domain_rules if domain_rules else "No additional task rules provided."}
        </domain_specific_rules>

        <clinical_precision>
        - PHILIPPINE CONTEXT: Use locally familiar objects in hints (mangoes, peso coins, blocks).
        - RATIONALE: Provide a brief clinical explanation of why this task is being retested based on the student's profile.
        - HINTS: Provide a teacher guide/hint for every question using Concrete-Representational-Abstract language.
        - TONE: If predicted_class is 0 (Typical), avoid terms like "severe", "critical", or "high-risk".
        - ASYMMETRY RULE: For any subtraction problems crossing ten, do NOT combine addition and subtraction into a single hint. You MUST strictly follow the CROSSING-TEN SUBTRACTION FORMULA (e.g., "take away X to reach 10, then take away Y").
        - TRUE/FALSE BALANCE: For tasks where the answer is boolean (True/False) or matching, you MUST provide a mix of both outcomes. Do not make all 3 answers the same.
        </clinical_precision>

        <output_format>
        Output RAW JSON only. No markdown fences. Ensure you generate exactly 3 questions per task.
        Do NOT append "= ?" to the end of equations. Output the raw equation only (e.g., "14 + 7").

        TASK FORMAT RULES:
        - number_comparison: question must be "X vs Y" (e.g., "5 vs 8"). The two numbers MUST BE DIFFERENT SIZES (one strictly larger than the other). The 'correct' field must be the LARGER integer. NEVER use equal numbers like "6 vs 6".
        - dot_matching: question must be "X vs Y" using INTEGERS ONLY (e.g., "4 vs 4"). Do NOT write "dots" in the question field.
        - number_series: question must be the sequence string (e.g., "2, 4, 6, 8, __"), correct must be the missing integer
        - single_addition / single_subtraction / complex_arithmetic: question must be a symbolic equation (e.g., "7 + 5"), correct must be the integer answer
        {{
            "TASK_NAME_1": {{
                "rationale": "<Brief clinical explanation why this task is being retested>",
                "tests": [
                    {{
                        "question": "<string, e.g., '2 vs 3' or '1+4'>",
                        "correct": <integer or boolean depending on the task format>,
                        "hint": "<string, teacher observation guide/hint>"
                    }}
                ]
            }},
            "TASK_NAME_2": {{ ... }},
            "TASK_NAME_3": {{ ... }}
        }}
        </output_format>
        """

        RETEST_TARGET_PER_DOMAIN = 3
        best_retest_data = {}
        best_report = None
        correction_prompt = ""

        for attempt in range(3):
            logger.info(f"[{rid}] Retest attempt {attempt + 1}")
            try:
                payload = {
                    "messages": [
                        {"role": "system", "content": retest_prompt + correction_prompt},
                        {"role": "user",   "content": "Generate the retest JSON now."}
                    ]
                }

                raw_output = call_llm_gateway(
                    payload, DRAFT_URL, DRAFT_HEADERS, rid,
                    temp=0.4, model_paths=DRAFT_MODELS_TO_TRY
                )
                if not raw_output:
                    raise Exception("No response from LLM")

                parsed = extract_json_robustly(raw_output)
                if not parsed or not isinstance(parsed, dict):
                    raise Exception("JSON extraction failed or output is not a dictionary")

                report = {
                    "counts": {"returned": 0, "pruned": 0},
                    "math_errors": [],
                    "pedagogy_errors": [],
                    "schema_errors": []
                }

                current_retest_data = {}
                
                # Schema Integrity: Bypass AST Math Validator for non-equation tasks
                non_math_tasks = ["number_comparison", "dot_matching", "Number Comparison", "Digit-Dot Matching", "number_series", "Number Series"]

                # Iterate through the generated tasks
                for task_name, task_data in parsed.items():
                    rationale = task_data.get("rationale", "")
                    tests = task_data.get("tests", [])
                    
                    seen_in_batch = set()
                    unique_questions = []
                    
                    # Deduplication Loop against History
                    for new_q in tests:
                        prob = new_q.get("question", "")
                        norm = normalize_equation(prob)

                        is_history_duplicate = any(
                            difflib.SequenceMatcher(None, prob, old_q.get("question", "")).ratio() > 0.85
                            for old_q in all_previous_questions
                        )
                        if is_history_duplicate or norm in seen_in_batch:
                            continue

                        seen_in_batch.add(norm)
                        
                        is_non_math = any(t.lower() in task_name.lower() for t in non_math_tasks)
                        new_q["problem"] = "" if is_non_math else new_q["question"]
                        new_q["expected_answer"] = new_q.get("correct")
                        if "match" in new_q:
                            new_q["expected_answer"] = new_q["match"]
                            
                        unique_questions.append(new_q)

                    if any(t.lower() in task_name.lower() for t in non_math_tasks):
                        math_valid = unique_questions[:] 
                    else:
                        math_valid = math_validator(unique_questions, rid, report, domain_name=task_name)

                    # Validation Loop
                    for item in math_valid[:]:
                        hint = str(item.get("hint", "")).strip()

                        if not any(t.lower() in task_name.lower() for t in non_math_tasks):
                            hint_ok, hint_error = validate_hint_quality(item.get("problem", ""), hint, task_name)
                            if not hint_ok:
                                report["pedagogy_errors"].append(f"[{task_name}] {hint_error}")
                                report["counts"]["pruned"] += 1
                                math_valid.remove(item)
                                continue

                        tone_ok, tone_error = validate_tone_for_status(hint, predicted_class)
                        if not tone_ok:
                            report["pedagogy_errors"].append(f"[{task_name}] {tone_error}")
                            report["counts"]["pruned"] += 1
                            math_valid.remove(item)
                            continue

                        # Guardrail: Prevent equal numbers in number_comparison
                        if "comparison" in task_name.lower():
                            q_str = item.get("question", "")
                            parts = [p.strip() for p in q_str.lower().split("vs")]
                            if len(parts) == 2 and parts[0] == parts[1]:
                                report["pedagogy_errors"].append(f"[{task_name}] Cannot use equal numbers ({q_str}). X and Y must be different.")
                                report["counts"]["pruned"] += 1
                                math_valid.remove(item)
                                continue

                        # Guardrail: Force boolean types for dot_matching
                        if "dot_matching" in task_name.lower() or "digit-dot" in task_name.lower():
                            if not isinstance(item.get("expected_answer"), bool):
                                report["schema_errors"].append(f"[{task_name}] 'correct' field must be a boolean (true/false), not an integer.")
                                report["counts"]["pruned"] += 1
                                math_valid.remove(item)
                                continue
                        
                        item.pop("problem", None)
                        item.pop("expected_answer", None)

                    if "asymmetry" in task_name.lower() and math_valid:
                        sub_count = sum(1 for q in math_valid if "-" in q.get("question", ""))
                        add_count = sum(1 for q in math_valid if "+" in q.get("question", ""))
                        if sub_count < 1 or add_count < 1:
                            report["pedagogy_errors"].append(
                                f"[{task_name}] AS retest must include both addition and subtraction."
                            )

                    report["counts"]["returned"] += len(math_valid)
                    current_retest_data[task_name] = {
                        "rationale": rationale,
                        "tests": math_valid
                    }

                total_valid_questions = sum(len(d.get("tests", [])) for d in current_retest_data.values())
                best_valid_count = sum(len(d.get("tests", [])) for d in best_retest_data.values()) if best_retest_data else 0
                
                if total_valid_questions > best_valid_count:
                    best_retest_data = current_retest_data
                    best_report = report

                blocking_errors = report["math_errors"] + report["schema_errors"] + [e for e in report["pedagogy_errors"] if "Low" not in e]
                
                missing_domains = [t for t in top_tasks if len(current_retest_data.get(t, {}).get("tests", [])) < RETEST_TARGET_PER_DOMAIN]

                if not blocking_errors and not missing_domains:
                    logger.info(f"[{rid}] Retest VALIDATION PASSED on attempt {attempt + 1}")
                    response = {
                        "retest_data": current_retest_data,
                        "_meta_validation_report": report,
                        "based_on_session": latest_session.get("session_id", "Unknown"),
                        "based_on_session_date": latest_session.get("date", "Unknown"),
                        "total_sessions_in_history": len(student_history)
                    }
                    return jsonify(response), 200

                repair_notes = []
                if missing_domains:
                    repair_notes.append(f"Missing valid questions for domains: {missing_domains}. You MUST provide exactly {RETEST_TARGET_PER_DOMAIN} questions per domain.")
                
                if report["pedagogy_errors"] and any("make-10 or inverse-operation strategy" in e for e in report["pedagogy_errors"]):
                    repair_notes.append(
                        "Your subtraction hints were rejected. You MUST use the exact make-10 formula "
                        "(e.g., 'take away X to reach 10, then take away Y') for subtraction problems crossing ten. "
                        "Do NOT combine addition and subtraction in the same hint."
                    )
                
                if report["math_errors"]:
                    repair_notes.append(f"Math errors: {report['math_errors']}.")
                if report["pedagogy_errors"]:
                    repair_notes.append(f"Pedagogy errors: {report['pedagogy_errors']}.")
                
                if repair_notes:
                    correction_prompt = "\n\nCRITICAL CORRECTION:\n" + " ".join(repair_notes)
                    logger.warning(f"[{rid}] Retest attempt {attempt + 1} FAILED — retrying.")

                time.sleep(2 ** attempt)

            except Exception as e:
                logger.exception(f"[{rid}] Retest attempt {attempt + 1} exception: {e}")
                time.sleep(2 ** attempt)

        if best_retest_data:
            logger.warning(f"[{rid}] Retest returning best partial result after all retries.")
            return jsonify({
                "retest_data": best_retest_data,
                "_meta_validation_report": best_report,
                "based_on_session": latest_session.get("session_id", "Unknown"),
                "based_on_session_date": latest_session.get("date", "Unknown"),
                "total_sessions_in_history": len(student_history),
                "warning": "Partial validation failure. Review _meta_validation_report."
            }), 207

        return jsonify({
            "error": "Failed to generate valid retest questions after retries.",
            "best_validation_report": best_report,
        }), 500

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

if __name__ == '__main__':
    debug = os.getenv("FLASK_ENV") == "development"
    app.run(port=5000, debug=debug)
