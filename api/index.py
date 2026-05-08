from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import uuid
from dotenv import load_dotenv
import time
import difflib
import logging

from llm.constants import (
    OPENROUTER_TOKEN,
    DOMAIN_EXPLANATIONS
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
        for attempt in range(5):
            print(f"\n[REQ {rid}] Pipeline Loop: Attempt {attempt + 1}")
            try:
                draft = get_lesson_from_qwen(
                    json.dumps(ml_data),
                    top_domains,
                    practice_tiers,
                    rid,
                    correction_prompt,
                    domain_explanations,
                    formative_assessment_count,
                )
                if not draft: raise Exception("Pass 1 yielded empty draft")

                perfect_json = format_with_cloud_llm(draft, json.dumps(ml_data), practice_tiers, rid, formative_assessment_count)
                if not perfect_json: raise Exception("Pass 2 JSON extraction failed")

                validated_data, validation_report = schema_validator(perfect_json, practice_tiers, top_domains, rid, formative_assessment_count)

                if validation_report["math_errors"] or validation_report["pedagogy_errors"] or validation_report["schema_errors"]:
                     logger.warning(f"[REQ {rid}] [VALIDATION REPORT]: {json.dumps(validation_report, indent=2)}")

                if validated_data:
                    logger.info(f"[REQ {rid}] [SUCCESS] VALIDATION PASSED")
                    validated_data["_meta_validation_report"] = validation_report
                    validated_data["decision_path_interpretation"] = interpretation
                    return jsonify(validated_data), 200
                else:
                    logger.warning(f"[REQ {rid}] [FAIL] Triggering Repair Loop.")

                    correction_prompt = f"\nCRITICAL CORRECTION FROM SYSTEM:\nYour last output failed validation. Do not repeat these errors: "
                    if validation_report["math_errors"]: correction_prompt += f"Math Errors: {validation_report['math_errors'][:2]}. "
                    if validation_report["pedagogy_errors"]: correction_prompt += f"Pedagogy Errors: {validation_report['pedagogy_errors'][:2]}. "
                    if validation_report["schema_errors"]: correction_prompt += f"Schema Errors: {validation_report['schema_errors'][:2]}. "

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

                    logger.info(f"[REQ {rid}] [REPAIR] Adding correction prompt: {correction_prompt}")
                    time.sleep(2 ** attempt)

            except Exception as e:
                logger.exception(f"[REQ {rid}] [RETRY] Pipeline step failed")
                time.sleep(2 ** attempt)

        return jsonify({"error": "Failed to generate valid module after retries"}), 500

    except Exception as e:
        logger.exception(f"[REQ {rid}] [ERROR] Critical Failure")
        return jsonify({"error": str(e)}), 500

# @app.route('/generate_retest', methods=['POST'])
# def generate_retest():
#     try:
#         data = request.json
#         if not data: return jsonify({"error": "Invalid request"}), 400

#         ml_data = data.get('diagnostic_data', {})
#         previous_problems = data.get('previous_questions', [])
#         rid = "RETEST_" + uuid.uuid4().hex[:4]

#         retest_prompt = f"""
#         You are a SPED Teacher. Generate a FRESH set of 5 practice problems based on this ML data.

#         DIAGNOSTIC PROFILE:
#         {json.dumps(ml_data)}

#         PREVIOUS QUESTIONS (DO NOT REPEAT THESE):
#         {json.dumps(previous_problems)}

#         RULES:
#         1. Ensure the new problems use DIFFERENT numbers and scenarios.
#         2. Maintain pedagogical difficulty. Do not generate operands larger than 20.
#         3. ALL problems MUST be symbolic math equations (e.g., "5 + 3", NOT word problems).
#         4. Output RAW JSON EXACTLY matching this schema:
#         {{
#             "retest_questions": [
#                 {{"problem": "...", "hint": "<Must reference a physical action>", "expected_answer": "..."}}
#             ]
#         }}
#         """

#         payload = {
#             "messages": [{"role": "user", "content": retest_prompt}]
#         }

#         raw_output = call_llm_gateway(payload, rid, temp=0.7)
#         if not raw_output: raise Exception("No response from retest")

#         parsed = extract_json_robustly(raw_output)
#         if not parsed: raise Exception("No JSON bounds")

#         try:
#             new_questions = parsed.get("retest_questions", [])
#             if not isinstance(new_questions, list): raise Exception("Invalid retest structure")

#             unique_questions = []
#             for new_q in new_questions:
#                 is_duplicate = False
#                 for old_q in previous_problems:
#                     if difflib.SequenceMatcher(None, new_q.get("problem", ""), old_q.get("problem", "")).ratio() > 0.85:
#                         is_duplicate = True
#                         break
#                 if not is_duplicate:
#                     unique_questions.append(new_q)

#             validation_report = {"counts": {"returned": 0, "pruned": 0}, "math_errors": [], "pedagogy_errors": [], "schema_errors": []}
#             valid_math_questions = math_validator(unique_questions, rid, validation_report)

#             if len(valid_math_questions) == 0:
#                 raise Exception(f"All generated retest questions were invalid. Report: {validation_report}")

#             parsed["retest_questions"] = valid_math_questions
#             parsed["_meta_validation_report"] = validation_report
#             return jsonify(parsed), 200

#         except Exception as e:
#             return jsonify({"error": f"Invalid retest output: {e}"}), 500

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

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
