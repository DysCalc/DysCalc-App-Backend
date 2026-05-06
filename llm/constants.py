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
    "NC": "Number Comparison",
    "DM": "Digit-Dot Matching",
    "NS": "Number Series",
    "ADD": "Single-Digit Addition",
    "SUB": "Single-Digit Subtraction",
    "CA": "Multi-Digit Addition and Subtraction",

    # Derived Features
    "NP": "Overall Processing Efficiency",
    "SN": "Symbolic vs. Non-Symbolic Processing Difference",
    "AF": "Overall Arithmetic Fluency",
    "BC": "Basic vs. Complex Arithmetic Contrast",
    "AS": "Addition vs. Subtraction Asymmetry",
    "PF": "Processing-Fluency Integration",
}

CLINICAL_COOCCURRENCE_MAP = {
    ("NC", "DM"): "a foundational number sense deficit affecting both symbolic and non-symbolic processing",
    ("ADD", "SUB"): "asymmetric arithmetic fluency where subtraction is procedurally learned rather than conceptually grounded",
    ("NS", "BC"): "difficulty extending basic patterns into multi-digit arithmetic",
    ("PF", "AF"): "a notable difficulty integrating processing speed with basic arithmetic fact retrieval",
    ("AS", "BC"): "foundational inverse operation gaps that hinder efficient transition to complex arithmetic tasks"
}

DOMAIN_EXPLANATIONS = {
    "NC": (
        f"{ML_INTERPRETATION_MAP['NC']}: Measures how efficiently the learner compares numerical magnitudes. "
        "Difficulty in this area may suggest weakness in understanding which numbers are larger or smaller."
    ),

    "DM": (
        f"{ML_INTERPRETATION_MAP['DM']}: Measures how efficiently the learner connects symbolic numbers with non-symbolic quantities, "
        "such as matching a digit to a group of dots. Difficulty in this area may suggest weakness in linking number symbols "
        "to actual quantities."
    ),

    "NS": (
        f"{ML_INTERPRETATION_MAP['NS']}: Measures the learner's ability to recognize and continue numerical patterns or sequences. "
        "Difficulty in this area may suggest weakness in sequencing, number order, or pattern-based reasoning."
    ),

    "ADD": (
        f"{ML_INTERPRETATION_MAP['ADD']}: Measures the learner's speed and accuracy in solving basic addition problems. "
        "Difficulty in this area may suggest weak arithmetic fact retrieval or reliance on slow counting strategies."
    ),

    "SUB": (
        f"{ML_INTERPRETATION_MAP['SUB']}: Measures the learner's speed and accuracy in solving basic subtraction problems. "
        "Difficulty in this area may suggest weak subtraction fluency, limited fact retrieval, or difficulty reasoning backward from a quantity."
    ),

    "CA": (
        f"{ML_INTERPRETATION_MAP['CA']}: Measures the learner's ability to solve larger addition and subtraction problems. "
        "Difficulty in this area may suggest challenges with carrying, borrowing, place value, or applying basic arithmetic facts to more complex calculations."
    ),

    "NP": (
        f"{ML_INTERPRETATION_MAP['NP']}: A derived indicator showing how efficiently the learner processes numerical tasks overall. "
        "Lower efficiency may suggest that the learner needs more time, concrete supports, or reduced cognitive load when working with numbers."
    ),

    "AF": (
        f"{ML_INTERPRETATION_MAP['AF']}: A broader domain covering tasks such as Number Series, Single-Digit Addition, "
        "Single-Digit Subtraction, and Multi-Digit Addition/Subtraction. It reflects how quickly and accurately the learner performs arithmetic-related tasks."
    ),

    "SN": (
        f"{ML_INTERPRETATION_MAP['SN']}: A derived indicator comparing performance on symbolic number tasks "
        "and non-symbolic quantity tasks. A large gap may suggest difficulty connecting number symbols with actual quantities."
    ),

    "BC": (
        f"{ML_INTERPRETATION_MAP['BC']}: A derived indicator comparing performance on simpler arithmetic tasks "
        "against more complex arithmetic tasks. A large gap may suggest that the learner can handle basic facts but struggles when more steps, "
        "place value, or working memory demands are involved."
    ),

    "AS": (
        f"{ML_INTERPRETATION_MAP['AS']}: A derived indicator comparing addition and subtraction performance. "
        "A large gap may suggest that the learner is more fluent in one operation than the other, which can help guide targeted practice."
    ),

    "PF": (
        f"{ML_INTERPRETATION_MAP['PF']}: A derived indicator combining processing efficiency and arithmetic fluency. "
        "It helps distinguish whether a learner's difficulty is mainly related to slow number processing, weak arithmetic performance, or both."
    )
}