from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
from dotenv import load_dotenv
import copy
import re
import ast
import uuid
import difflib
import operator
import math
import time
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN")

if not OPENROUTER_TOKEN:
    raise ValueError("Missing OPENROUTER_TOKEN in environment variables")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_TOKEN}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:5000", 
    "X-Title": "Dyscalculia Thesis App"
}

MODELS_TO_TRY = ["qwen/qwen-2.5-math-7b-instruct", "qwen/qwen-2.5-7b-instruct"]

ALLOWED_OPS = {
    ast.Add: operator.add, 
    ast.Sub: operator.sub,
    ast.Mult: operator.mul, 
    ast.Div: operator.truediv,
    ast.USub: operator.neg
}

def safe_eval(expr):
    """Strictly evaluates pure mathematical strings. Rejects all word problems and code."""
    expr = expr.split('=')[0].strip()
    
    if not re.fullmatch(r"^[0-9+\-*/().\s]+$", expr):
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

def call_llm_gateway(payload, rid="SYSTEM", temp=0.2):
    for model_path in MODELS_TO_TRY:
        try:
            current_payload = copy.deepcopy(payload)  
            current_payload["model"] = model_path
            current_payload["provider"] = {"allow_fallbacks": False}
            current_payload["temperature"] = temp

            logger.info(f"[REQ {rid}] Calling model: {model_path}")
            response = requests.post(
                OPENROUTER_URL, headers=HEADERS, json=current_payload, timeout=90  
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.warning(f"[REQ {rid}] Model {model_path} failed: {e}")
            continue 
    raise Exception("LLM service unavailable.")

def extract_json_robustly(text):
    text = text.replace('```json', '').replace('```', '').strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            repaired_str = re.sub(r',\s*}', '}', json_str)
            repaired_str = re.sub(r',\s*]', ']', repaired_str)
            try:
                return json.loads(repaired_str)
            except:
                return None
    return None

def normalize(s):
    return re.sub(r'\W+', '', s.lower())

def get_top_deficits(ml_data_string, rid):
    try:
        match = re.search(r"Domain Severity\s*:\s*(\{.*?\})", ml_data_string)
        if not match: return ["General Arithmetic Fluency"]
        severity_dict = ast.literal_eval(match.group(1))
        sorted_deficits = sorted(severity_dict.items(), key=lambda x: x[1], reverse=True)
        top = [d[0] for d in sorted_deficits if d[1] > 0.10][:2]
        return top if top else ["General Arithmetic Fluency"]
    except Exception as e:
        logger.error(f"[REQ {rid}] ML Data extraction failed: {e}")
        return ["General Arithmetic Fluency"]

def calculate_practice_tiers(ml_data_string):
    try:
        scores = re.findall(r': (0\.\d+)', ml_data_string)
        max_s = max([float(s) for s in scores]) if scores else 0.0
        if max_s > 0.40: return 6
        if max_s > 0.20: return 4
        return 2
    except:
        return 4

def math_validator(practice_set, rid, report):
    """Validates practice problems and tracks failure rates to prevent silent data loss."""
    valid_practice = []
    
    for item in practice_set:
        problem_str = item.get("problem", "").split('=')[0]
        expected_str = str(item.get("expected_answer", ""))
        clean_eq = re.sub(r'[^0-9\+\-\*\/\.]', '', problem_str)
        
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
        except:
            report["counts"]["pruned"] += 1
            continue
        
        if contains_invalid_subtraction(item.get("hint", "")):
            report["pedagogy_errors"].append("Negative subtraction in hint.")
            report["counts"]["pruned"] += 1
            continue
            
        valid_practice.append(item)
        report["counts"]["returned"] += 1
    
    return valid_practice

def pedagogy_validator(example, rid, report):
    """Validates the worked example schema and math logic."""
    required_keys = {"problem", "reasoning_steps", "final_answer"}
    if set(example.keys()) != required_keys:
        report["schema_errors"].append(f"worked_example keys mismatch")
        return False

    problem = example.get("problem", "").split('=')[0]
    clean_eq = re.sub(r'[^0-9\+\-\*\/\.]', '', problem)
    
    if re.search(r'[\+\-\*\/]{2,}', clean_eq):
         report["math_errors"].append(f"Invalid operator sequence in worked example: {clean_eq}")
         return False

    if any(op in clean_eq for op in ['*', '/']):
        report["pedagogy_errors"].append(f"Multiplication/Division out of scope in worked example: {clean_eq}")
        return False
        
    if clean_eq and any(op in clean_eq for op in ['+', '-']):
        computed = safe_eval(clean_eq)
        if computed is not None:
            ans_match = re.search(r'-?\d*\.?\d+', example.get("final_answer", ""))
            if ans_match:
                try:
                    ans_val = float(ans_match.group())
                    if not math.isclose(computed, ans_val, rel_tol=1e-9):
                        report["math_errors"].append(f"WORKED ERROR: {clean_eq} != {ans_val}")
                        return False
                except:
                    pass

    reasoning_steps = example.get("reasoning_steps", [])
    if len(reasoning_steps) > 4:
        report["pedagogy_errors"].append("Worked example has too many steps (>4). Cognitive overload risk.")
        return False

    steps_text = " ".join(reasoning_steps)
    if contains_invalid_subtraction(steps_text):
        report["pedagogy_errors"].append("Negative subtraction logic in reasoning.")
        return False
        
    return True

def get_lesson_from_qwen(ml_data, domains, count, rid, correction_prompt=""):
    print(f"[REQ {rid}] Pass 1: Drafting for {domains} (Target: {count} items)...")
    
    inst = f"""
    You are a Senior SPED Clinical Consultant specializing in Dyscalculia. 
    TARGET DOMAINS: {", ".join(domains)}.

    For EACH domain, you MUST write a complete draft containing:
    1. Clinical Explanation: Why the student struggled based on the glossary.
    2. Learning Objectives: 3 specific goals.
    3. Conceptual Explanation: Child-friendly "Why".
    4. Teaching Strategy: High-level teacher approach.
    5. Worked Example: MUST include "problem", "reasoning_steps", and "final_answer".
    6. Practice Set: MUST be a JSON array with EXACTLY {count} objects in this format:
       [
         {{ "problem": "5 + 3", "expected_answer": "8", "hint": "Use blocks" }}
       ]

    FEATURE GLOSSARY:
    - NC: Difficulty distinguishing which number/group is larger.
    - DM: Trouble mapping numerals to quantities.
    - NS: Difficulty recognizing patterns and sequences.
    - ADD: Challenges with forward counting and combining.
    - SUB: Difficulty with the conceptual act of 'taking away'.
    - AS: Gaps in understanding inverse operations.
    - SN: Gap between written numbers and physical quantities.
    - AF: Issues with speed and accuracy of basic facts.
    - BC: Struggle with the transition to multi-digit math.
    - PF: Slower cognitive speed for numerical tasks.

    RULES:
    - ALL practice problems MUST be symbolic math equations (e.g., "5 + 3", NOT word problems).
    - Use EXACT domain names: {", ".join(domains)}.
    - Pedagogy: NEVER use negative intermediate subtraction steps (e.g., No "2 - 5"). Explain borrowing.
    - Metaphor: Use physical objects (blocks, fingers, dots).
    - NUMERICAL BOUNDS: Do not generate operands larger than 20 unless specifically addressing the 'Basic vs. Complex Contrast' domain. Prevent cognitive overload.
    {correction_prompt}
    """
    
    payload = {
        "messages": [
            {"role": "system", "content": inst},
            {"role": "user", "content": f"Process this data:\n{ml_data}"}
        ]
    }
    return call_llm_gateway(payload, rid, temp=0.0 if correction_prompt else 0.4)

def format_with_cloud_llm(qwen_messy_text, ml_data_string, count, rid):
    print(f"[REQ {rid}] Pass 2: Converting draft to strict JSON...")
    
    architect_prompt = f"""
    You are a JSON formatting robot. Convert the Teacher Text into the EXACT JSON schema below. 

    STRICT RULES:
    - DO NOT add extra keys. DO NOT omit required keys.
    - worked_example MUST include: "problem", "reasoning_steps", "final_answer"
    - practice_set MUST contain EXACTLY {count} items.
    - DO NOT wrap your output in ```json markdown blocks. Output the raw {{ JSON directly.

    REQUIRED SCHEMA:
    {{
        "status": "<Typical (0) or At-Risk (1)>",
        "decision_path_rationale": "<LITERAL STRING FROM ML DATA>",
        "overall_summary": "<Summarize in 2 sentences>",
        "diagnostic_modules": [
            {{
                "domain_name": "<Insert Exact Domain Name>",
                "clinical_explanation": "<Insert Clinical Explanation>",
                "learning_objectives": ["<Goal 1>", "<Goal 2>", "<Goal 3>"],
                "conceptual_explanation": "<Insert Child-Friendly Explanation>",
                "worked_example": {{
                    "problem": "<Insert Math Problem>",
                    "reasoning_steps": ["<Step 1>", "<Step 2>"],
                    "final_answer": "<Insert Number or Statement>"
                }},
                "teaching_strategy": "<Insert Strategy>",
                "practice_set": [ 
                    {{ "problem": "<Equation>", "expected_answer": "<Number>", "hint": "<Must reference a physical action: e.g., 'Draw 4 dots and cross out 2' or 'Hold up 5 fingers'>" }} 
                ] 
            }}
        ],
        "formative_assessment": [
            {{ "question": "<Insert Math Question>", "expected_answer": "<Number>" }}
        ]
    }}
    """
    
    payload = {
        "messages": [
            {"role": "system", "content": architect_prompt},
            {"role": "user", "content": f"ML DATA:\n{ml_data_string}\n\nTEACHER TEXT:\n{qwen_messy_text}"}
        ]
    }
    raw_output = call_llm_gateway(payload, rid, temp=0.0)
    if not raw_output: return None
    
    parsed_json = extract_json_robustly(raw_output)
    if not parsed_json:
         print(f"[REQ {rid}] [ERROR] Python rejected the AI's JSON syntax.")
    return parsed_json

def schema_validator(data, expected_count, allowed_domains, rid):
    """Orchestrates the validation pipeline and builds the metrics report."""
    if not isinstance(data, dict): return None, {"fatal": "Root data is not a dictionary."}
    
    validation_report = {
        "counts": {"expected": expected_count * len(allowed_domains), "returned": 0, "pruned": 0},
        "math_errors": [],
        "pedagogy_errors": [],
        "schema_errors": [],
        "warnings": []
    }

    required_top_keys = ["status", "decision_path_rationale", "overall_summary", "diagnostic_modules", "formative_assessment"]
    if not all(k in data for k in required_top_keys):
        validation_report["schema_errors"].append("Missing top-level schema keys.")
        return None, validation_report

    modules = data.get("diagnostic_modules", [])
    if len(modules) == 0:
        validation_report["schema_errors"].append("No diagnostic modules generated.")
        return None, validation_report
    elif len(modules) != len(allowed_domains): 
        validation_report["warnings"].append(f"Module count mismatch. Expected {len(allowed_domains)}, got {len(modules)}")

    allowed_norm = {normalize(d) for d in allowed_domains}
    valid_modules = []

    for m in modules:
        required_keys = [
            "domain_name", "clinical_explanation", "learning_objectives", 
            "conceptual_explanation", "worked_example", "practice_set", "teaching_strategy"
        ]
        
        m_cleaned = {k: v for k, v in m.items() if k in required_keys}
        if set(m_cleaned.keys()) != set(required_keys):
            validation_report["schema_errors"].append(f"Module keys mismatch in domain: {m.get('domain_name')}")
            continue

        if normalize(m_cleaned.get("domain_name", "")) not in allowed_norm:
            validation_report["warnings"].append(f"Forbidden domain skipped: {m_cleaned.get('domain_name')}")
            continue
            
        m_cleaned["conceptual_explanation"] = m_cleaned.get("conceptual_explanation", "")\
            .replace("Child-friendly 'Why':", "").replace("Child-friendly \"Why\":", "").replace("**Why**:", "").replace("Why:", "").strip()
            
        m_cleaned["teaching_strategy"] = m_cleaned.get("teaching_strategy", "").replace("High-level teacher approach:", "").strip()

        worked = m_cleaned.get("worked_example", {})
        if not pedagogy_validator(worked, rid, validation_report):
            continue 

        raw_practice = m_cleaned.get("practice_set", [])
        clean_practice = math_validator(raw_practice, rid, validation_report)
        
        if len(clean_practice) == 0:
             validation_report["schema_errors"].append(f"No valid practice problems survived in domain: {m_cleaned.get('domain_name')}")
             continue
        elif len(clean_practice) != expected_count:
             validation_report["warnings"].append(f"Practice count reduced to {len(clean_practice)} in domain: {m_cleaned.get('domain_name')}")
        
        m_cleaned["practice_set"] = clean_practice
        valid_modules.append(m_cleaned)
        
    if len(valid_modules) == 0:
         return None, validation_report
         
    data["diagnostic_modules"] = valid_modules
            
    assessment = data.get("formative_assessment", [])
    if not isinstance(assessment, list) or not (1 <= len(assessment) <= 3):
        validation_report["warnings"].append("Invalid formative_assessment count. Expected 1-3.")
        data["formative_assessment"] = [] 
    else:
        valid_assessments = []
        for item in assessment:
            if set(item.keys()) == {"question", "expected_answer"} and len(str(item.get("question", "")).strip()) >= 10:
                 valid_assessments.append(item)
            else:
                 validation_report["schema_errors"].append("Pruned invalid assessment item.")
        data["formative_assessment"] = valid_assessments
                
    return data, validation_report
    
@app.route('/generate_module', methods=['POST'])
def generate_module():
    rid = uuid.uuid4().hex[:6]
    try:
        data = request.json
        if not data: return jsonify({"error": "Invalid request"}), 400

        ml_data = data.get('diagnostic_data', '')
        if not ml_data: return jsonify({"error": "No ML data provided"}), 400

        top_domains = get_top_deficits(ml_data, rid)
        req_count = calculate_practice_tiers(ml_data)

        correction_prompt = "" 

        for attempt in range(3):
            print(f"\n[REQ {rid}] Pipeline Loop: Attempt {attempt + 1}")
            try:
                draft = get_lesson_from_qwen(ml_data, top_domains, req_count, rid, correction_prompt)
                if not draft: raise Exception("Pass 1 yielded empty draft")
                
                perfect_json = format_with_cloud_llm(draft, ml_data, req_count, rid)
                if not perfect_json: raise Exception("Pass 2 JSON extraction failed")

                validated_data, validation_report = schema_validator(perfect_json, req_count, top_domains, rid)
                
                if validation_report["math_errors"] or validation_report["pedagogy_errors"] or validation_report["schema_errors"]:
                     print(f"[REQ {rid}] [VALIDATION REPORT]: {json.dumps(validation_report, indent=2)}")

                if validated_data:
                    print(f"[REQ {rid}] [SUCCESS] VALIDATION PASSED")
                    validated_data["_meta_validation_report"] = validation_report
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