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
    check_answer_diversity,         
    check_hint_diversity,           
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
    Generate a fresh, fully validated set of retest questions from a student's
    longitudinal session history.

    Always uses the LATEST session's diagnostic profile as the generation basis.
    Pools ALL previous questions across ALL sessions for deduplication.

    Request Format:
    {
        "student_history": [
            {
                "session_id": <int>,
                "date": "<string>",
                "diagnostic_data": {
                    "predicted_class": "<string>",
                    "domain_severity_scores": { "<domain_name>": <float> },
                    "task_importance_scores": { "<acronym>": <float> }
                },
                "questions_asked": [
                    {"problem": "<string>", "expected_answer": <int>}
                ]
            }
        ]
    }
    
    Success Response (HTTP 200):
    {
        "retest_questions": [
            {
                "problem": "<string>",
                "hint": "<string>",
                "expected_answer": <int>
            }
        ],
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
        "retest_questions": [...],
        "_meta_validation_report": {...},
        "based_on_session": <int>,
        "based_on_session_date": "<string>",
        "total_sessions_in_history": <int>,
        "warning": "<string>"
    }

    Failure Response (HTTP 500):
    {
        "error": "<string>",
        "best_validation_report": {...}
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
        top_domains = get_top_deficits(ml_data, rid, max_domains=1)
        top_domain = top_domains[0] if top_domains else ""
        domain_rules = json.dumps(
            {top_domain: DOMAIN_GENERATION_RULES.get(top_domain, {})} if top_domain else {},
            indent=2
        )

        all_previous_questions = []
        for session in student_history:
            all_previous_questions.extend(session.get('questions_asked', []))

        logger.info(
            f"[{rid}] Retest | session={latest_session.get('session_id', '?')} "
            f"| top_domain={top_domain} | history_pool={len(all_previous_questions)} questions"
        )

        retest_prompt = f"""
<role>
You are a Senior Special Education (SPED) Clinical Consultant in the Philippines
specializing in early Dyscalculia intervention.
</role>

<mission>
Generate a FRESH set of 5 retest practice problems targeting the student's highest-priority
deficit domain based on the LATEST diagnostic profile below.
PRIMARY DOMAIN: {top_domain if top_domain else "General Arithmetic"}
</mission>

<latest_diagnostic_profile>
{json.dumps(ml_data, indent=2)}
</latest_diagnostic_profile>

<history_pool_do_not_repeat>
The following problems have already been asked across all previous sessions.
Do NOT generate any problem that is the same as or similar to these:
{json.dumps([q.get("problem", "") for q in all_previous_questions])}
</history_pool_do_not_repeat>

<domain_specific_rules>
{domain_rules if domain_rules else "No additional domain rules provided."}
</domain_specific_rules>

<clinical_precision>
- PHILIPPINE CONTEXT: Use locally familiar objects in hints (mangoes, peso coins, blocks, jeepney toys).
- CRA FRAMEWORK: All hints MUST use Concrete-Representational-Abstract language.
  NEVER use abstract procedural terms like "carry over" or "borrow".
  Always anchor the hint in physical objects (e.g., "Count 9 peso coins, add 1 to make 10, then add the rest").
- CROSSING-TEN SUBTRACTION FORMULA: For A - B where A > 10 and A - B < 10,
  first take away A - 10 to reach 10, then take away B - (A - 10).
  Example: 15 - 7 → take away 5 to reach 10, then 2 more. Final answer is 8.
  NEVER say "take away B to reach 10".
- REASONING CONSISTENCY: The final value stated in any hint MUST match the actual answer.
  For addition, do not say to add more than the actual addend.
- If the domain is Addition vs. Subtraction Asymmetry, include BOTH addition (+) and
  subtraction (-) problems (at least 3 subtraction, at least 1 addition).
- Tone: If predicted_class is 0 (Typical), avoid "significant difficulty", "severe",
  "critical", "major deficit", "high-risk", or "at-risk" in hint language.
</clinical_precision>

<strict_rules>
1. BOUNDS: Keep all operands <= 20. Subtraction second operand <= 9 unless the domain
   is Multi-Digit Addition and Subtraction.
2. SYMBOLIC MATH ONLY: Every problem MUST be a pure arithmetic equation using + or -.
   No word problems, no object labels in the problem field.
3. ANTI-REPETITION: Do not repeat the same equation within the 5 new questions.
4. ANSWER DIVERSITY: The 5 questions must not all produce the same expected_answer.
5. HINT DIVERSITY: Each hint must use different wording and different physical objects.
6. expected_answer MUST be a strict integer (e.g., 8). NOT a string ("8").
</strict_rules>

<output_format>
Output RAW JSON only. No markdown fences. No extra keys.
{{
    "retest_questions": [
        {{
            "problem": "<symbolic equation, e.g. 13 - 6>",
            "hint": "<CRA-grounded hint referencing a concrete object, max 120 chars>",
            "expected_answer": <integer>
        }}
    ]
}}
</output_format>
"""

        RETEST_TARGET = 5
        best_questions = []
        best_report = None
        correction_prompt = ""

        for attempt in range(3):
            logger.info(f"[{rid}] Retest attempt {attempt + 1}")
            try:
                payload = {
                    "messages": [
                        {"role": "system", "content": retest_prompt + correction_prompt},
                        {"role": "user",   "content": "Generate the 5 retest questions now."}
                    ]
                }

                raw_output = call_llm_gateway(
                    payload,
                    DRAFT_URL,
                    DRAFT_HEADERS,
                    rid,
                    temp=0.4,
                    model_paths=DRAFT_MODELS_TO_TRY
                )
                if not raw_output:
                    raise Exception("No response from LLM")

                parsed = extract_json_robustly(raw_output)
                if not parsed:
                    raise Exception("JSON extraction failed")

                new_questions = parsed.get("retest_questions", [])
                if not isinstance(new_questions, list) or len(new_questions) == 0:
                    raise Exception("Invalid or empty retest_questions structure")

                seen_in_batch = set()
                unique_questions = []
                for new_q in new_questions:
                    prob = new_q.get("problem", "")
                    norm = normalize_equation(prob)

                    # Check against history pool
                    is_history_duplicate = any(
                        difflib.SequenceMatcher(
                            None, prob, old_q.get("problem", "")
                        ).ratio() > 0.85
                        for old_q in all_previous_questions
                    )
                    if is_history_duplicate:
                        continue

                    if norm in seen_in_batch:
                        continue

                    seen_in_batch.add(norm)
                    unique_questions.append(new_q)

                report = {
                    "counts": {"returned": 0, "pruned": 0},
                    "math_errors": [],
                    "pedagogy_errors": [],
                    "schema_errors": []
                }

                math_valid = math_validator(unique_questions, rid, report, domain_name=top_domain)

                for item in math_valid[:]:
                    hint = str(item.get("hint", "")).strip()

                    hint_ok, hint_error = validate_hint_quality(
                        item.get("problem", ""), hint, top_domain
                    )
                    if not hint_ok:
                        report["pedagogy_errors"].append(hint_error)
                        report["counts"]["pruned"] += 1
                        math_valid.remove(item)
                        continue

                    tone_ok, tone_error = validate_tone_for_status(hint, predicted_class)
                    if not tone_ok:
                        report["pedagogy_errors"].append(tone_error)
                        report["counts"]["pruned"] += 1
                        math_valid.remove(item)
                        continue

                if math_valid and not check_answer_diversity(math_valid):
                    report["pedagogy_errors"].append(
                        "Low answer diversity across retest questions."
                    )

                if math_valid and not check_hint_diversity(math_valid):
                    report["pedagogy_errors"].append(
                        "Low hint diversity across retest questions."
                    )

                if top_domain and "asymmetry" in top_domain.lower() and math_valid:
                    sub_count = sum(1 for q in math_valid if "-" in q.get("problem", ""))
                    add_count = sum(1 for q in math_valid if "+" in q.get("problem", ""))
                    if sub_count < 1 or add_count < 1:
                        report["pedagogy_errors"].append(
                            f"AS retest must include both addition and subtraction "
                            f"(sub={sub_count}, add={add_count})."
                        )

                # best batch across attempts
                report["counts"]["returned"] = len(math_valid)
                if len(math_valid) > len(best_questions):
                    best_questions = math_valid
                    best_report = report

                blocking_errors = (
                    report["math_errors"] +
                    report["schema_errors"] +
                    [e for e in report["pedagogy_errors"]
                     if "Low" not in e]   
                )
                if len(math_valid) >= RETEST_TARGET and not blocking_errors:
                    logger.info(f"[{rid}] Retest VALIDATION PASSED on attempt {attempt + 1}")
                    response = {
                        "retest_questions": math_valid[:RETEST_TARGET],
                        "_meta_validation_report": report,
                        "based_on_session": latest_session.get("session_id", "Unknown"),
                        "based_on_session_date": latest_session.get("date", "Unknown"),
                        "total_sessions_in_history": len(student_history),
                    }
                    return jsonify(response), 200

                # targeted correction prompt for the next attempt
                repair_notes = []
                if report["math_errors"]:
                    repair_notes.append(f"Math errors to fix: {report['math_errors']}.")
                if report["pedagogy_errors"]:
                    repair_notes.append(f"Pedagogy errors to fix: {report['pedagogy_errors']}.")
                if report["schema_errors"]:
                    repair_notes.append(f"Schema errors to fix: {report['schema_errors']}.")
                shortfall = RETEST_TARGET - len(math_valid)
                if shortfall > 0:
                    repair_notes.append(
                        f"Only {len(math_valid)} valid questions were produced. "
                        f"You need exactly {RETEST_TARGET}. "
                        f"Add {shortfall} more distinct questions."
                    )
                if any("AS retest" in e for e in report["pedagogy_errors"]):
                    repair_notes.append(
                        "The Addition vs. Subtraction Asymmetry domain requires BOTH "
                        "addition (+) and subtraction (-) problems in the retest set."
                    )
                if repair_notes:
                    correction_prompt = (
                        "\n\nCRITICAL CORRECTION FROM SYSTEM:\n" + " ".join(repair_notes)
                    )
                    logger.warning(f"[{rid}] Retest attempt {attempt + 1} FAILED — retrying. {correction_prompt}")

                time.sleep(2 ** attempt)

            except Exception as e:
                logger.exception(f"[{rid}] Retest attempt {attempt + 1} exception: {e}")
                time.sleep(2 ** attempt)

        if best_questions:
            logger.warning(
                f"[{rid}] Retest returning best partial result "
                f"({len(best_questions)} questions) after all retries."
            )
            response = {
                "retest_questions": best_questions,
                "_meta_validation_report": best_report,
                "based_on_session": latest_session.get("session_id", "Unknown"),
                "based_on_session_date": latest_session.get("date", "Unknown"),
                "total_sessions_in_history": len(student_history),
                "warning": (
                    f"Only {len(best_questions)} of {RETEST_TARGET} questions passed full "
                    "validation. Review _meta_validation_report for details."
                ),
            }
            return jsonify(response), 207

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
