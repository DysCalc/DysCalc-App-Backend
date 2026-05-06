import re
import ast
import logging
import requests
import copy
import math
import json
import difflib

from .constants_old import (
    ML_INTERPRETATION_MAP, 
    ALLOWED_OPS, 
    MODELS_TO_TRY, 
    OPENROUTER_URL, 
    HEADERS,
    CLINICAL_COOCCURRENCE_MAP
)

logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

def interpret_decision_path(ml_data_string):
    """Synthesizes a root-cause clinical profile, strictly scaling tone by class and confidence."""
    try:
        data_str = str(ml_data_string)
        
        is_at_risk = "At-Risk (1)" in data_str or "Class  : 1" in data_str
        conf_match = re.search(r"Confidence\s*:\s*([\d\.]+)", data_str)
        confidence = float(conf_match.group(1)) if conf_match else 0.50

        if not is_at_risk:
            tone_desc = "mild, emerging variations"
            tone_impact = "suggesting"
        elif confidence > 0.75:
            tone_desc = "critical foundational gaps"
            tone_impact = "creating a severe bottleneck driven by"
        else:
            tone_desc = "significant cognitive difficulties"
            tone_impact = "indicating"

        match = re.search(r"Domain Severity\s*:\s*(\{.*?\})", data_str)
        if not match: 
            return "Profile indicates general baseline performance."
        
        severity_dict = ast.literal_eval(match.group(1))
        REVERSE_MAP = {v: k for k, v in ML_INTERPRETATION_MAP.items()}
        
        active_acronyms = set()
        top_domains = []
        
        for domain, score in sorted(severity_dict.items(), key=lambda item: item[1], reverse=True):
            if score >= 0.15:
                top_domains.append(domain)
                if domain in REVERSE_MAP: active_acronyms.add(REVERSE_MAP[domain])

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

def call_llm_gateway(payload, rid="SYSTEM", temp=0.2):
    for model_path in MODELS_TO_TRY:
        try:
            current_payload = copy.deepcopy(payload)  
            current_payload["model"] = model_path
            current_payload["temperature"] = temp
            current_payload["max_tokens"] = 3000

            current_payload["response_format"] = {"type": "json_object"}

            logger.info(f"[REQ {rid}] Calling model: {model_path}")
            response = requests.post(
                OPENROUTER_URL, headers=HEADERS, json=current_payload, timeout=900  
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.Timeout:
            logger.warning(f"[REQ {rid}] Timeout on {model_path}. Trying next...")
            continue
        except requests.exceptions.RequestException as e:
            logger.warning(f"[REQ {rid}] Model {model_path} failed: {e}")
            continue 
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

def get_top_deficits(ml_data_string, rid):
    try:
        match = re.search(r"Domain Severity\s*:\s*(\{.*?\})", ml_data_string)
        if not match: return ["General Arithmetic Fluency"]
        severity_dict = ast.literal_eval(match.group(1))
        valid_deficits = [d for d in severity_dict.items() if d[1] >= 0.10]
        sorted_deficits = sorted(valid_deficits, key=lambda x: x[1], reverse=True)
        top = [d[0] for d in sorted_deficits][:3]
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
        clean_eq = re.sub(r'[^0-9\+\-\*\/\.\(\)]', '', problem_str)
        
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

def get_lesson_from_qwen(ml_data, domains, count, rid, correction_prompt="", domain_explanations=""):
    print(f"[REQ {rid}] Pass 1: Drafting for {domains} (Target: {count} items)...")
    
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

    <philippine_context_and_tone>
    - Context: Use locally familiar physical objects (e.g., mangoes, candies, blocks, peso coins, jeepney toys) for concrete hints.
    - Tone: Use empathetic, encouraging language. Write short, simple, child-friendly sentences.
    </philippine_context_and_tone>

    <clinical_precision>
    - SEVERITY WEIGHTING: You MUST look at the numerical 'Domain Severity' scores in the ML DATA.
    - PEDAGOGY: NEVER suggest "timed drills" or focus on speed. Dyscalculic students suffer from math anxiety. Focus on derived facts, making 10s, and chunking.
    - CRA FRAMEWORK: All hints MUST use Concrete-Representational-Abstract language. NEVER use abstract procedural terms like "carry over" or "borrow". ALWAYS anchor the hint in physical objects (e.g., "Count 9 peso coins, add 1 to make 10, then add the rest").
    - If a domain has a HIGH severity (>0.20), provide "cognitive stretch" problems (e.g., crossing tens, 12+9 or 14-6). Avoid overly simple problems like 8-2 or 9-3.
    - If a domain is AS (Addition vs. Subtraction Asymmetry), the student likely relies on forward (addition-based) reasoning and has difficulty with mental inversion. To rehabilitate this, the practice set MUST contain BOTH addition (+) and subtraction (-) equations (e.g., at least 3 subtraction problems bridging tens, and at least 1 addition problem for scaffolding).
    - If a domain is BC (Basic vs. Complex Arithmetic Contrast), you MUST explicitly focus on "crossing tens". Do not use generic "break down problems" phrasing. Explicitly teach breaking numbers into tens and ones (e.g., 15 + 7 -> 10 + 5 + 7).
    - If a domain is SN (Symbolic vs. Non-Symbolic), focus on translating physical objects/visuals into digits.
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
    5. Worked Example: Provide "problem", "reasoning_steps" (ARRAY OF MAXIMUM 3 STEPS), and "final_answer" for a SINGLE simple math equation.
    6. Practice Set: A JSON array with EXACTLY {count} symbolic equations.
    </module_structure>

    <schema_typing>
    - CRITICAL: The "expected_answer" field MUST ALWAYS BE A STRICT INTEGER (e.g., 8). Do NOT wrap it in quotes (e.g., "8" is FATAL).
    </schema_typing>

    <strict_rules>
    1. ANTI-REPETITION: DO NOT USE the equations 14-6, 12-5, 15-7, or 8+5. You must generate completely unique number families to avoid redundancy.
    2. DIVERSITY RULE: Use at least 3 DIFFERENT fact families/number triplets in the practice set. Do NOT repeat the same base numbers.
    3. HINT VARIETY: Each hint MUST use entirely different wording and objects. Do NOT repeat phrasing. 
       - EXAMPLES: "Hold up 7 fingers, fold down 2", "Draw 9 dots, cross out 3", "Place 4 blocks, add 2 more".
    4. SYMBOLIC MATH ONLY: Practice problems MUST be pure math equations (e.g., "17 - 9"). NO text word problems.
    5. TARGETING: If the domain is 'Addition vs. Subtraction Asymmetry', your practice_set MUST contain a mix of BOTH '+' and '-' operators. Do not generate only subtraction.
    6. BOUNDS: Keep math simple but appropriate to severity. Operands <= 20.
    7. ASSESSMENT: Provide exactly 2 Formative Assessment questions at the very end.
    </strict_rules>
    
    {correction_prompt}
    """
    
    payload = {
        "messages": [
            {"role": "system", "content": inst},
            {"role": "user", "content": f"<ml_data>\n{ml_data}\n</ml_data>\n\nGenerate the modules based ONLY on the target domains."}
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
    - CRITICAL: The "diagnostic_modules" array MUST contain ALL modules present in the Teacher Text. DO NOT stop after formatting just one module. If the Teacher Text has 3 domains, you MUST output 3 objects in the array.
    - FINAL STEP - MANDATORY: You MUST include a "formative_assessment" array with EXACTLY 2 objects. This CANNOT be empty. If omitted, the output is rejected.
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
                    "reasoning_steps": ["<Step 1>", "<Step 2>"],
                    "final_answer": "<Insert SINGLE NUMBER ONLY, e.g., 8>"
                }},
                "teaching_strategy": "<Insert Strategy>",
                "practice_set": [ 
                    {{ "problem": "<String>", "expected_answer": <INTEGER ONLY, NO QUOTES>, "hint": "<String max 120 chars>" }} 
                ] 
            }}
        ],
        "formative_assessment": [
            {{ "question": "<String>", "expected_answer": <INTEGER ONLY, NO QUOTES> }},
            {{ "question": "<String>", "expected_answer": <INTEGER ONLY, NO QUOTES> }}
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
            
        m_cleaned["conceptual_explanation"] = m_cleaned.get("conceptual_explanation", "")\
            .replace("Child-friendly 'Why':", "").replace("Child-friendly \"Why\":", "").replace("**Why**:", "").replace("Why:", "").strip()
            
        if not validate_conceptual_steps(m_cleaned["conceptual_explanation"]):
            validation_report["pedagogy_errors"].append(f"Missing Step 1, 2, 3 format in domain: {m.get('domain_name')}")
            
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
        
        if not check_hint_diversity(clean_practice):
            validation_report["pedagogy_errors"].append(f"Low hint diversity in domain: {m.get('domain_name')}")

        as_ratio = difflib.SequenceMatcher(None, normalize(m_cleaned.get("domain_name", "")), normalize("Addition vs. Subtraction Asymmetry")).ratio()
        if as_ratio >= 0.70:
            problems = [str(p.get("problem", "")) for p in clean_practice]
            sub_count = sum(1 for p in problems if '-' in p)
            add_count = sum(1 for p in problems if '+' in p)
            if sub_count < 2 or add_count < 1:
                validation_report["schema_errors"].append(f"AS module failed subtraction/addition ratio (sub={sub_count}, add={add_count}).")
                continue

        m_cleaned["practice_set"] = clean_practice
        valid_modules.append(m_cleaned)
        
    validation_report["modules_passed"] = len(valid_modules)

    if len(valid_modules) == 0:
         validation_report["schema_errors"].append("Zero modules passed validation.")
         return None, validation_report
         
    data["diagnostic_modules"] = valid_modules
            
    assessment = data.get("formative_assessment", [])
    if not isinstance(assessment, list) or len(assessment) == 0:
        validation_report["warnings"].append("Formative assessment missing or empty. Modules preserved with degraded assessment.")
        data["formative_assessment"] = []
    else:
        valid_assessments = []
        for item in assessment:
            q_text = str(item.get("question", item.get("problem", ""))).strip()
            ans_text = str(item.get("expected_answer", item.get("answer", ""))).strip()
            
            if len(q_text) >= 3 and re.fullmatch(r"-?\d+", ans_text):
                 valid_assessments.append({"question": q_text, "expected_answer": int(ans_text)})
            else:
                 validation_report["warnings"].append(f"Pruned invalid assessment item: {item}")
                 
        if len(valid_assessments) != 2:
            validation_report["warnings"].append(f"Formative assessment has {len(valid_assessments)} items instead of 2. Keeping what passed.")
        data["formative_assessment"] = valid_assessments
                
    validation_report["success_rate"] = (
        validation_report["counts"]["returned"] / 
        max(1, validation_report["counts"]["expected"])
    )
                
    return data, validation_report