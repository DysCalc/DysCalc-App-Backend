import os
import ast
import operator
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

EXPERIMENT_MODE = os.getenv("EXPERIMENT_MODE", "3")
CLOUD_TOKEN = os.getenv("OPENROUTER_TOKEN")
OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN")

# --- BASE URLS & HEADERS ---
LOCAL_URL = "http://localhost:1234/v1/chat/completions"
LOCAL_HEADERS = {"Content-Type": "application/json"}

CLOUD_URL = "https://openrouter.ai/api/v1/chat/completions"
CLOUD_HEADERS = {
    "Authorization": f"Bearer {CLOUD_TOKEN}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:5000", 
    "X-Title": "Dyscalculia Thesis App"
}

if EXPERIMENT_MODE == "1":
    print("\n[SYSTEM] BOOTING IN EXPERIMENT 1 MODE: Single-Pass Local Baseline")
    DRAFT_URL = LOCAL_URL
    DRAFT_HEADERS = LOCAL_HEADERS
    DRAFT_MODELS_TO_TRY = ["qwen2.5-7b-math-instruct"]
    FORMAT_URL = None
    FORMAT_HEADERS = None
    FORMAT_MODELS_TO_TRY = []

elif EXPERIMENT_MODE == "2":
    print("\n[SYSTEM] BOOTING IN EXPERIMENT 2 MODE: Two-Pass Local")
    DRAFT_URL = LOCAL_URL
    DRAFT_HEADERS = LOCAL_HEADERS
    DRAFT_MODELS_TO_TRY = ["qwen2.5-math-7b-instruct"]
    FORMAT_URL = CLOUD_URL
    FORMAT_HEADERS = CLOUD_HEADERS
    FORMAT_MODELS_TO_TRY = ["qwen/qwen-2.5-7b-instruct"]

elif EXPERIMENT_MODE == "3": 
    print("\n[SYSTEM] BOOTING IN EXPERIMENT 3 MODE: Hybrid Architecture (Final)")
    DRAFT_URL = LOCAL_URL
    DRAFT_HEADERS = LOCAL_HEADERS
    DRAFT_MODELS_TO_TRY = ["qwen2.5-math-7b-instruct"]
    FORMAT_URL = CLOUD_URL
    FORMAT_HEADERS = CLOUD_HEADERS
    FORMAT_MODELS_TO_TRY = ["qwen/qwen-2.5-72b-instruct"]

elif EXPERIMENT_MODE == "4":
    print("\n[SYSTEM] BOOTING IN EXPERIMENT 4 MODE: Two-Pass Cloud (14B + 72B)")
    DRAFT_URL            = CLOUD_URL
    DRAFT_HEADERS        = CLOUD_HEADERS
    DRAFT_MODELS_TO_TRY  = ["qwen/qwen-2.5-7b-instruct"]
    FORMAT_URL           = CLOUD_URL
    FORMAT_HEADERS       = CLOUD_HEADERS
    FORMAT_MODELS_TO_TRY = ["qwen/qwen-2.5-72b-instruct"]

else:
    print("\n[SYSTEM] BOOTING IN EXPERIMENT 5 MODE: Two-Pass Cloud (72B + 72B)")
    DRAFT_URL            = CLOUD_URL
    DRAFT_HEADERS        = CLOUD_HEADERS
    DRAFT_MODELS_TO_TRY  = ["qwen/qwen-2.5-72b-instruct"]
    FORMAT_URL           = CLOUD_URL
    FORMAT_HEADERS       = CLOUD_HEADERS
    FORMAT_MODELS_TO_TRY = ["qwen/qwen-2.5-72b-instruct"]

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

BAD_HINT_PATTERNS = [
    r"\b-\d+\b",                       # negative numbers like -2
    r"\band\s*-\s*\d+\b",              # "and -2"
    r"\bborrow\b",
    r"\bcarry\b",
    r"\bcarry over\b",
    r"\btimed drill\b",
    r"\bspeed\b",
]

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
        "Difficulty in this area may suggest challenges with place value, crossing tens, regrouping quantities, or applying basic arithmetic facts to more complex calculations."
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

DOMAIN_GENERATION_RULES = {
    "Number Comparison": {
        "required_focus": "comparing numerical magnitude, identifying larger/smaller numbers, and explaining number size using concrete quantities",
        "problem_field_rule": "practice_set.problem must be a symbolic comparison-style equation or expression when possible, such as '7 + 2' or '13 - 5'. Do not use vague text-only prompts.",
        "hint_field_rule": "hints should use objects, number lines, or place-value language to show which quantity is larger or smaller",
        "avoid": [
            "speed drills",
            "guessing larger numbers without quantity support",
            "object-only problem fields",
            "advanced inequality notation if the student has not been introduced to it"
        ],
    },

    "Digit-Dot Matching": {
        "required_focus": "linking digits to exact visual quantities and checking one-to-one correspondence",
        "problem_field_rule": "practice_set.problem must remain a symbolic equation such as '4 + 3' or '9 - 2'. Do not write '7 dots' or 'match 7 dots' in the problem field.",
        "hint_field_rule": "hints should explicitly connect each digit to a countable set of dots, blocks, coins, or fingers",
        "avoid": [
            "object-only practice problems",
            "dot-count labels in the problem field",
            "pure arithmetic without quantity matching language",
            "speed pressure"
        ],
    },

    "Number Series": {
        "required_focus": "recognizing number order, counting patterns, missing numbers, and simple forward/backward sequences",
        "problem_field_rule": "practice_set.problem must be a symbolic equation if the validator requires equations, such as '6 + 2' for counting forward or '10 - 2' for counting backward.",
        "hint_field_rule": "hints should use number lines, counting steps, or grouped objects to show the sequence pattern",
        "avoid": [
            "complex multi-rule patterns",
            "large skip-counting jumps",
            "object-only problem fields",
            "timed drills"
        ],
    },

    "Single-Digit Addition": {
        "required_focus": "basic addition facts, making ten, counting on, and derived addition facts",
        "problem_field_rule": "practice_set.problem must be a symbolic addition equation like '8 + 4'.",
        "hint_field_rule": "hints should show concrete addition using objects, fingers, dots, blocks, or number lines",
        "avoid": [
            "subtraction-only practice",
            "multi-step complex arithmetic",
            "carrying",
            "speed drills"
        ],
    },

    "Single-Digit Subtraction": {
        "required_focus": "basic subtraction facts, taking away, counting back, and using related addition facts",
        "problem_field_rule": "practice_set.problem must be a symbolic subtraction equation like '13 - 6'.",
        "hint_field_rule": "hints should show taking away objects, counting back on a number line, or using a related addition fact",
        "avoid": [
            "addition-only practice",
            "negative answers",
            "negative decomposition",
            "borrowing",
            "speed drills"
        ],
    },

    "Multi-Digit Addition and Subtraction": {
        "required_focus": "place value, tens-and-ones decomposition, regrouping quantities, and crossing tens",
        "problem_field_rule": "practice_set.problem must be a symbolic addition or subtraction equation with operands appropriate to the learner, such as '15 + 7' or '23 - 8'.",
        "hint_field_rule": "hints should explain tens and ones using objects, bundles, coins, base-ten blocks, or number lines",
        "avoid": [
            "carry over",
            "borrow",
            "abstract column procedures without concrete representation",
            "large numbers beyond the student's support level",
            "timed drills"
        ],
    },

    "Overall Processing Efficiency": {
        "required_focus": "reduced cognitive load, visual supports, step-by-step number processing",
        "problem_field_rule": "practice_set.problem must be a symbolic equation such as '9 + 4' or '12 - 5'.",
        "hint_field_rule": "hints should break the task into short visible steps using objects, number lines, or chunking",
        "avoid": [
            "timed drills",
            "speed pressure",
            "long multi-step problems",
            "mental-only strategies without visual support"
        ],
    },

    "Symbolic vs. Non-Symbolic Processing Difference": {
        "required_focus": "connecting visual quantities to symbolic equations",
        "problem_field_rule": "practice_set.problem must be a symbolic equation like '8 + 4' or '12 - 5', never '8 dots' or object-only text",
        "hint_field_rule": "dots, blocks, coins, and other objects may appear only in hints or explanations",
        "avoid": [
            "object-only practice problems",
            "dot-count labels in the problem field",
            "pure symbol manipulation without quantity language",
            "speed pressure"
        ],
    },

    "Overall Arithmetic Fluency": {
        "required_focus": "accurate arithmetic fact retrieval, derived facts, and flexible use of addition/subtraction relationships",
        "problem_field_rule": "practice_set.problem must be a symbolic addition or subtraction equation.",
        "hint_field_rule": "hints should encourage derived facts, making ten, counting on/back, or fact-family reasoning",
        "avoid": [
            "timed drills",
            "speed pressure",
            "random mixed practice without strategy",
            "multiplication or division"
        ],
    },

    "Basic vs. Complex Arithmetic Contrast": {
        "required_focus": "crossing tens using tens-and-ones decomposition",
        "problem_field_rule": "practice_set.problem must be a symbolic equation involving crossing tens when possible, such as '13 + 8' or '16 - 9'.",
        "hint_field_rule": "hints must explicitly show breaking numbers into tens and ones, reaching 10 or 20, then recombining",
        "avoid": [
            "generic break down phrasing",
            "carrying",
            "borrowing",
            "overly simple non-crossing-ten problems",
            "speed drills"
        ],
    },

    "Addition vs. Subtraction Asymmetry": {
        "min_subtraction": 3,
        "min_addition": 1,
        "required_focus": "inverse operations, fact families, backward reasoning, and subtraction crossing ten",
        "problem_field_rule": "practice_set.problem must include both symbolic addition and symbolic subtraction equations.",
        "hint_field_rule": "hints should connect subtraction to related addition facts or show taking away to reach 10 first",
        "avoid": [
            "subtraction-only practice",
            "addition-only practice",
            "negative decomposition",
            "speed drills"
        ],
    },

    "Processing-Fluency Integration": {
        "required_focus": "chunking, reduced cognitive load, flexible switching between addition and subtraction",
        "problem_field_rule": "practice_set.problem must be a symbolic addition or subtraction equation.",
        "hint_field_rule": "hints should show a short chunking strategy using objects, number lines, or make-ten reasoning",
        "avoid": [
            "timed drills",
            "speed pressure",
            "long mental-only problems",
            "repetitive hint wording"
        ],
    },
}
