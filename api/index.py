from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_TOKEN}",
    "Content-Type": "application/json"
}

# Qwen2.5-Math-7B-Instruct 
# FALLBACK: Qwen2.5-7B-Instruct (for system stability)
MODELS_TO_TRY = ["qwen/qwen-2.5-math-7b-instruct", "qwen/qwen-2.5-7b-instruct"]

def call_llm_gateway(payload):
    """Iterates through models to ensure the Math series is prioritized."""
    for model_path in MODELS_TO_TRY:
        payload["model"] = model_path
        try:
            print(f"[INFO] Attempting API call with: {model_path}")
            response = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"[WARN] {model_path} failed or unavailable: {e}")
            continue 
    raise Exception("All configured models failed to respond.")

def get_lesson_from_qwen(ml_data_string):
    print("[INFO] Pass 1: Analyzing diagnostic data for pedagogical drafting...")
    
    qwen_system_instructions = """
    You are a Senior SPED Clinical Consultant. Your job is to translate multidimensional dyscalculia ML data into a Modular Diagnostic Report.

    FEATURE GLOSSARY (Use these to explain the 'Why' behind each deficit):
    - NC (Number Comparison): Difficulty distinguishing which number/group is larger.
    - DM (Digit-Dot Matching): Trouble mapping abstract numerals to concrete quantities.
    - NS (Number Series): Difficulty recognizing patterns and sequences.
    - ADD (Single-Digit Addition): Challenges with forward counting and combining.
    - SUB (Single-Digit Subtraction): Difficulty with the conceptual act of 'taking away'.
    - AS (Addition vs. Subtraction Asymmetry): Gaps in understanding inverse operations.
    - SN (Symbolic vs. Non-Symbolic): Gap between written numbers and physical quantities.
    - AF (Arithmetic Fluency): Issues with speed and accuracy of basic facts.
    - BC (Basic vs. Complex Contrast): Struggle with the transition to multi-digit math.
    - PF (Processing Fluency): Slower cognitive speed for numerical tasks.

    CORE RULES:
    1. MULTI-DOMAIN ANALYSIS: Identify EVERY feature in 'Domain Severity' with a score > 0.10.
    2. MODULAR INTERVENTION: For EACH identified feature, generate a dedicated section containing:
    - CLINICAL EXPLANATION: Use the Glossary to explain the deficit based on the student's score.
    - LEARNING OBJECTIVES: 2–3 clearly stated goals for this specific domain.
    - CONCEPTUAL EXPLANATION: A step-by-step 'Why' explanation for the student.
    - WORKED EXAMPLE: One fully solved problem with explicit reasoning steps.
    - SCAFFOLDED PRACTICE: 4 guided problems with hints.
        * IF Confidence >= 0.85: Hints must be microscopic physical actions.
        * IF Confidence < 0.85: Hints must be standard conceptual guides.
    - FORMATIVE ASSESSMENT: 2 monitoring questions to check progress.

    3. CONSTRAINTS: Use no jargon (use blocks/dots/fingers). Keep each domain strictly separated.
    """
    
    payload = {
        "messages": [
            {"role": "system", "content": qwen_system_instructions},
            {"role": "user", "content": f"Process this diagnostic data:\n{ml_data_string}"}
        ],
        "temperature": 0.2, 
        "max_tokens": 3000
    }
    return call_llm_gateway(payload)

def format_with_cloud_llm(qwen_messy_text, ml_data_string):
    print("[INFO] Pass 2: Converting draft to strict JSON for frontend integration...")
    
    architect_prompt = """
    You are a strict JSON formatting robot. Output RAW JSON ONLY. No markdown. No conversational filler.
    CRITICAL SYNTAX RULES: Your output MUST be perfectly valid JSON. Escape any quotation marks inside strings.

    REQUIRED SCHEMA:
    {
        "status": "Extract the Final Prediction (e.g., 'At-Risk (1)' or 'Typical (0)')",
        "decision_path_rationale": "EXTRACT THE EXACT, LITERAL DECISION PATH STRING. DO NOT SUMMARIZE.",
        "overall_summary": "Write a 2-sentence pedagogical summary combining all identified deficits.",
        "diagnostic_modules": [
            {
                "domain_name": "Name of the specific deficit (e.g., Addition vs. Subtraction Asymmetry)",
                "clinical_explanation": "The detailed explanation of what the score means for the student.",
                "learning_objectives": ["Goal 1 (e.g., The student will...)", "Goal 2"],
                "conceptual_explanation": "A step-by-step 'Why' explanation of the math concept for the student.",
                "worked_example": {
                    "problem": "One fully worked math problem",
                    "reasoning_steps": ["Step 1 explanation", "Step 2 explanation", "Final Result"]
                },
                "teaching_strategy": "The teacher's high-level strategy (1-2 sentences).",
                "practice_set": [
                    {
                        "problem": "Problem text",
                        "expected_answer": "Correct answer or action",
                        "hint": "Scaffolded hint (Microscopic if At-Risk, Conceptual if Typical)"
                    }
                ]
            }
        ],
        "formative_assessment": [
            {
                "question": "1 final assessment question covering the most severe domain.",
                "expected_answer": "The expected answer."
            },
            {
                "question": "1 final assessment question covering the second most severe domain.",
                "expected_answer": "The expected answer."
            }
        ]
    }
    """
    payload = {
        "messages": [
            {"role": "system", "content": architect_prompt},
            {"role": "user", "content": f"ML DATA:\n{ml_data_string}\n\nTEACHER TEXT:\n{qwen_messy_text}"}
        ],
        "temperature": 0.1, 
        "max_tokens": 3000,
        "response_format": {"type": "json_object"} 
    }
    raw_output = call_llm_gateway(payload)
    return json.loads(raw_output.replace('```json', '').replace('```', '').strip())

@app.route('/generate_module', methods=['POST'])
def generate_module():
    try:
        ml_data = request.json.get('diagnostic_data', '')
        if not ml_data:
            return jsonify({"error": "No ML data provided"}), 400

        raw_text = get_lesson_from_qwen(ml_data)
        perfect_json = format_with_cloud_llm(raw_text, ml_data)

        return jsonify(perfect_json), 200

    except Exception as e:
        print(f"[ERROR] Pipeline Failure: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/generate_retest', methods=['POST'])
def generate_retest():
    try:
        # get both the ML data AND the previous questions from the frontend
        data = request.json
        ml_data = data.get('diagnostic_data', '')
        previous_problems = data.get('previous_questions', []) # This is new!

        retest_prompt = f"""
        You are a SPED Teacher. Generate a FRESH set of 5 practice problems based on this ML data.
        
        DIAGNOSTIC PROFILE:
        {ml_data}
        
        PREVIOUS QUESTIONS (DO NOT REPEAT THESE):
        {json.dumps(previous_problems)}
        
        RULES:
        1. Ensure the new problems use DIFFERENT numbers and scenarios.
        2. Maintain the same pedagogical difficulty and scaffolding.
        3. Output in RAW JSON:
        {{
            "retest_questions": [
                {{"problem": "...", "hint": "...", "expected_answer": "..."}}
            ]
        }}
        """
        
        payload = {
            "model": "qwen/qwen-2.5-7b-instruct",
            "messages": [{"role": "user", "content": retest_prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.7 # Increased slightly to encourage more variety
        }
        
        raw_output = call_llm_gateway(payload)
        
        cleaned_output = raw_output.replace('```json', '').replace('```', '').strip()
        
        return jsonify(json.loads(cleaned_output)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)