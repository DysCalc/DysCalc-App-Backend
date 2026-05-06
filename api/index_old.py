from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from dotenv import load_dotenv
import uuid
import difflib
import time
import logging


from .constants_old import OPENROUTER_TOKEN, DOMAIN_EXPLANATIONS
from .helpers_old import (
    get_top_deficits,
    calculate_practice_tiers,
    interpret_decision_path,
    call_llm_gateway,
    extract_json_robustly,
    schema_validator,
    math_validator,
    get_lesson_from_qwen,
    format_with_cloud_llm,
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
    rid = uuid.uuid4().hex[:6]
    try:
        data = request.json
        if not data: return jsonify({"error": "Invalid request"}), 400

        ml_data = data.get('diagnostic_data', '')
        if not ml_data: return jsonify({"error": "No ML data provided"}), 400
        
        logger.info(f"[REQ {rid}] Decision Path: {ml_data}")

        top_domains = get_top_deficits(ml_data, rid)
        req_count = calculate_practice_tiers(ml_data)
        interpretation = interpret_decision_path(ml_data)
        domain_explanations = json.dumps(DOMAIN_EXPLANATIONS, indent=2)

        correction_prompt = "" 

        for attempt in range(5):
            print(f"\n[REQ {rid}] Pipeline Loop: Attempt {attempt + 1}")
            try:
                draft = get_lesson_from_qwen(ml_data, top_domains, req_count, rid, correction_prompt, domain_explanations)
                if not draft: raise Exception("Pass 1 yielded empty draft")
                
                perfect_json = format_with_cloud_llm(draft, ml_data, req_count, rid)
                if not perfect_json: raise Exception("Pass 2 JSON extraction failed")

                validated_data, validation_report = schema_validator(perfect_json, req_count, top_domains, rid)
                
                if validation_report["math_errors"] or validation_report["pedagogy_errors"] or validation_report["schema_errors"]:
                     print(f"[REQ {rid}] [VALIDATION REPORT]: {json.dumps(validation_report, indent=2)}")

                if validated_data:
                    print(f"[REQ {rid}] [SUCCESS] VALIDATION PASSED")
                    validated_data["_meta_validation_report"] = validation_report
                    validated_data["decision_path_interpretation"] = interpretation
                    return jsonify(validated_data), 200
                else:
                    print(f"[REQ {rid}] [FAIL] Triggering Repair Loop.")
                    
                    correction_prompt = f"\nCRITICAL CORRECTION FROM SYSTEM:\nYour last output failed validation. Do not repeat these errors: "
                    if validation_report["math_errors"]: correction_prompt += f"Math Errors: {validation_report['math_errors'][:2]}. "
                    if validation_report["pedagogy_errors"]: correction_prompt += f"Pedagogy Errors: {validation_report['pedagogy_errors'][:2]}. "
                    
                    time.sleep(2 ** attempt)

            except Exception as e:
                print(f"[REQ {rid}] [RETRY] Pipeline step failed: {e}")
                time.sleep(2 ** attempt)

        return jsonify({"error": "Failed to generate valid module after retries"}), 500

    except Exception as e:
        print(f"[REQ {rid}] [ERROR] Critical Failure: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/generate_retest', methods=['POST'])
def generate_retest():
    try:
        data = request.json
        if not data: return jsonify({"error": "Invalid request"}), 400
        
        ml_data = data.get('diagnostic_data', '')
        previous_problems = data.get('previous_questions', []) 
        rid = "RETEST_" + uuid.uuid4().hex[:4]

        retest_prompt = f"""
        You are a SPED Teacher. Generate a FRESH set of 5 practice problems based on this ML data.
        
        DIAGNOSTIC PROFILE:
        {ml_data}
        
        PREVIOUS QUESTIONS (DO NOT REPEAT THESE):
        {json.dumps(previous_problems)}
        
        RULES:
        1. Ensure the new problems use DIFFERENT numbers and scenarios.
        2. Maintain pedagogical difficulty. Do not generate operands larger than 20.
        3. ALL problems MUST be symbolic math equations (e.g., "5 + 3", NOT word problems).
        4. Output RAW JSON EXACTLY matching this schema:
        {{
            "retest_questions": [
                {{"problem": "...", "hint": "<Must reference a physical action>", "expected_answer": "..."}}
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

            unique_questions = []
            for new_q in new_questions:
                is_duplicate = False
                for old_q in previous_problems:
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
            return jsonify(parsed), 200

        except Exception as e:
            return jsonify({"error": f"Invalid retest output: {e}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    debug = os.getenv("FLASK_ENV") == "development"
    app.run(port=5000, debug=debug)