from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

HF_URL = "https://router.huggingface.co/v1/chat/completions"

HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Qwen2.5-Math-7B-Instruct 
# FALLBACK: Qwen2.5-7B-Instruct (for system stability)
MODELS_TO_TRY = ["Qwen/Qwen2.5-Math-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]

def call_huggingface(payload):
    """Iterates through models to ensure the Math series is prioritized."""
    for model_path in MODELS_TO_TRY:
        payload["model"] = model_path
        try:
            print(f"[INFO] Attempting API call with: {model_path}")
            response = requests.post(HF_URL, headers=HEADERS, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"[WARN] {model_path} failed or unavailable: {e}")
            continue 
    raise Exception("All configured models failed to respond.")

def get_lesson_from_qwen(ml_data_string):
    print("[INFO] Pass 1: Analyzing diagnostic data for pedagogical drafting...")
    
    qwen_system_instructions = """
    You are a Senior SPED Mathematics Tutor. 
    GLOSSARY: NC: Comparison, DM: Digit-Dot, NS: Series, ADD: Addition, SUB: Subtraction, CA: Multi-Digit, NP: Processing.

    CORE RULES:
    1. SEVERITY SCAN: Scan data for the highest decimal score in "Domain Severity" or "Task Importance." Base the lesson ONLY on that skill.
    2. NO JARGON: Use "blocks" or "dots" only. You are FORBIDDEN from using terms like "minuend," "subtrahend," "sum," "difference," or "equation."
    3. PEDAGOGY MATCH: For "Digit-Dot Matching" or "Number Comparison," do NOT use addition/subtraction equations. Focus on counting and matching.
    4. DYNAMIC SCAFFOLDING (CRITICAL): Check the ML "Confidence" score in the data:
       - IF Confidence >= 0.85: The student is highly at-risk. Hints MUST be microscopic physical actions (e.g., "Touch the first block," "Move one dot away").
       - IF Confidence < 0.85: The student needs standard support. Hints can be conceptual (e.g., "Count how many blocks are left").
    5. EXPLAINABILITY: Include the "Decision Path" in the rationale for teacher review.
    6. SCHEMA RIGOR: Output strict plain text for Pass 1. 
    """
    
    payload = {
        "messages": [
            {"role": "system", "content": qwen_system_instructions},
            {"role": "user", "content": f"Process this diagnostic data:\n{ml_data_string}"}
        ],
        "temperature": 0.2, 
        "max_tokens": 1500
    }
    return call_huggingface(payload)

def format_with_cloud_llm(qwen_messy_text, ml_data_string):
    print("[INFO] Pass 2: Converting draft to strict JSON for frontend integration...")
    
    architect_prompt = """
    You are a strict JSON formatting robot. Output RAW JSON ONLY. No markdown. No conversational filler.
    CRITICAL SYNTAX RULES: Your output MUST be perfectly valid JSON. Double-check that all array items and object key-value pairs are separated by commas. Escape any quotation marks inside your strings.

    REQUIRED SCHEMA:
    {
        "primary_target_skill": "String",
        "decision_path_rationale": "EXTRACT THE EXACT, LITERAL DECISION PATH STRING (e.g., 'ADD <= 21.5'). DO NOT SUMMARIZE. DO NOT ADD CONVERSATIONAL TEXT.",
        "learning_objectives": ["Goal 1", "Goal 2"],
        "conceptual_explanation": {
            "title": "Title matched to proficiency",
            "step_by_step": "Explanation adapted to proficiency level"
        },
        "worked_example": {
            "problem": "One fully worked math problem",
            "reasoning_steps": ["Step 1", "Step 2", "Result"]
        },
        "scaffolded_practice": [
            {"problem": "Prob 1", "hint": "Extract the exact dynamic hint from the Teacher text. Match the requested scaffolding level."},
            {"problem": "Prob 2", "hint": "Ensure hints are microscopic if confidence is high."},
            "... GENERATE EXACTLY 8 PRACTICE PROBLEMS TOTAL FOLLOWING THIS FORMAT ..."
        ],
        "formative_assessment": [
            {"question": "Item 1"},
            {"question": "Item 2"},
            "... GENERATE EXACTLY 5 ASSESSMENT QUESTIONS TOTAL FOLLOWING THIS FORMAT ..."
        ]
    }
    """
    payload = {
        "messages": [
            {"role": "system", "content": architect_prompt},
            {"role": "user", "content": f"ML DATA:\n{ml_data_string}\n\nTEACHER TEXT:\n{qwen_messy_text}"}
        ],
        "temperature": 0.1, 
        "max_tokens": 1500,
        "response_format": {"type": "json_object"} 
    }
    raw_output = call_huggingface(payload)
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

if __name__ == '__main__':
    app.run(port=5000, debug=True)