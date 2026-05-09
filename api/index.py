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
    EXPERIMENT_MODE

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
    get_single_pass_lesson
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
        Generate a fresh set of retest questions based on the student's full session history.
        Always uses the LATEST session as the diagnostic basis.
        Pools ALL previous questions from ALL sessions to prevent duplicates.
 
        Request Format:
        {
            "student_history": [
                {
                    "session_id": <int>,
                    "date": "<string>",
                    "diagnostic_data": { ... },
                    "questions_asked": [
                        {"problem": "<string>", "expected_answer": <int>}
                    ]
                }
            ]
        }
    """
    try:
        data = request.json
        if not data: return jsonify({"error": "Invalid request"}), 400
 
        student_history = data.get('student_history', [])
        if not student_history or not isinstance(student_history, list) or len(student_history) == 0:
            return jsonify({"error": "Missing or empty student_history array"}), 400
 
        # 1. Route to the LATEST profile for generation
        latest_session = student_history[-1]
        ml_data = latest_session.get('diagnostic_data', {})
 
        # 2. Pool ALL previous questions from the ENTIRE history to prevent duplicates
        all_previous_questions = []
        for session in student_history:
            questions = session.get('questions_asked', [])
            all_previous_questions.extend(questions)
 
        rid = "RETEST_" + uuid.uuid4().hex[:4]
        logger.info(f"[{rid}] Retest based on session {latest_session.get('session_id', '?')} | {len(all_previous_questions)} total past questions pooled")
 
        retest_prompt = f"""
        You are a SPED Teacher. Generate a FRESH set of 5 practice problems based on this ML data.
 
        LATEST DIAGNOSTIC PROFILE:
        {json.dumps(ml_data)}
 
        ALL PREVIOUS QUESTIONS ASKED IN HISTORY (DO NOT REPEAT THESE):
        {json.dumps(all_previous_questions)}
 
        RULES:
        1. Ensure the new problems use DIFFERENT numbers and scenarios from the previous questions.
        2. Maintain pedagogical difficulty. Do not generate operands larger than 20.
        3. ALL problems MUST be symbolic math equations (e.g., "5 + 3", NOT word problems).
        4. Output RAW JSON EXACTLY matching this schema:
        {{
            "retest_questions": [
                {{"problem": "...", "hint": "<Must reference a physical action>", "expected_answer": <integer>}}
            ]
        }}
        """
 
        payload = {
            "messages": [{"role": "user", "content": retest_prompt}]
        }
 
        raw_output = call_llm_gateway(payload, rid, temp=0.7)
        if not raw_output: raise Exception("No response from retest")
 
        parsed = extract_json_robustly(raw_output)
        if not parsed: raise Exception("No JSON bounds")
 
        try:
            new_questions = parsed.get("retest_questions", [])
            if not isinstance(new_questions, list): raise Exception("Invalid retest structure")
 
            # Deterministic deduplication — check against ALL historical questions
            unique_questions = []
            for new_q in new_questions:
                is_duplicate = False
                for old_q in all_previous_questions:
                    if difflib.SequenceMatcher(None, new_q.get("problem", ""), old_q.get("problem", "")).ratio() > 0.85:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_questions.append(new_q)
 
            validation_report = {"counts": {"returned": 0, "pruned": 0}, "math_errors": [], "pedagogy_errors": [], "schema_errors": []}
            valid_math_questions = math_validator(unique_questions, rid, validation_report)
 
            if len(valid_math_questions) == 0:
                raise Exception(f"All generated retest questions were invalid. Report: {validation_report}")
 
            parsed["retest_questions"] = valid_math_questions
            parsed["_meta_validation_report"] = validation_report
 
            # Metadata for the frontend — tells it which session this was based on
            parsed["based_on_session"] = latest_session.get("session_id", "Unknown")
            parsed["based_on_session_date"] = latest_session.get("date", "Unknown")
            parsed["total_sessions_in_history"] = len(student_history)
 
            return jsonify(parsed), 200
 
        except Exception as e:
            return jsonify({"error": f"Invalid retest output: {e}"}), 500
 
    except Exception as e:
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
