import os
import ast
import operator
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_TOKEN}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:5000", 
    "X-Title": "Dyscalculia Thesis App"
}

MODELS_TO_TRY = [
    "qwen/qwen-2.5-7b-math-instruct",        
    "qwen/qwen-2.5-72b-instruct",      
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct"
]
ALLOWED_OPS = {
    ast.Add: operator.add, 
    ast.Sub: operator.sub,
    ast.USub: operator.neg
}

ML_INTERPRETATION_MAP = {
    # Raw Features
    "NC": "Number Comparison", "DM": "Digit-Dot Matching",
    "NS": "Number Series", "ADD": "Single-Digit Addition",
    "SUB": "Single-Digit Subtraction", "CA": "Multi-Digit Addition and Subtraction",
    # Derived Features
    "NP": "Number Processing", "SN": "Symbolic vs. Non-Symbolic Processing",
    "AF": "Overall Arithmetic Fluency", "BC": "Basic vs. Complex Arithmetic Contrast",
    "AS": "Addition vs. Subtraction Asymmetry", "PF": "Processing-Fluency Integration",
}

CLINICAL_COOCCURRENCE_MAP = {
    ("NC", "DM"): "a foundational number sense deficit affecting both symbolic and non-symbolic processing",
    ("ADD", "SUB"): "asymmetric arithmetic fluency where subtraction is procedurally learned rather than conceptually grounded",
    ("NS", "BC"): "difficulty extending basic patterns into multi-digit arithmetic",
    ("PF", "AF"): "a notable difficulty integrating processing speed with basic arithmetic fact retrieval",
    ("AS", "BC"): "foundational inverse operation gaps that hinder efficient transition to complex arithmetic tasks"
}