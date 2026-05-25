# DysCalc App Backend

Flask backend for the DysCalc project. It serves two main jobs:

1. Convert raw diagnostic test scores into structured ML diagnostic output.
2. Convert that ML diagnostic output into validated SPED intervention modules through a multi-pass LLM pipeline.

The current backend emphasizes deterministic validation and repair around LLM output so generated lessons are not accepted just because they look plausible.

## Project Layout

```text
api/index.py                     Flask routes
llm/constants.py                 OpenRouter config, model routing, clinical/domain rules
llm/helpers.py                   LLM gateway, prompting, validation, repair helpers
ml/helpers.py                    Diagnostic preprocessing and prediction wrapper
ml/C45DecisionTree.py            C4.5 model utilities
ml/models/v1.pkl                 Serialized model artifact
tests/generated_test_payloads.json
                                 Raw diagnostic endpoint test inputs
tests/select_and_request.py      Interactive endpoint test runner
tests/convert_ml_response_payload.py
                                 Converts ML responses into /generate_module payloads
```

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create `.env`:

```env
OPENROUTER_TOKEN=your_openrouter_api_key_here
FLASK_ENV=development
```

Run the API:

```bash
.venv/bin/python api/index.py
```

Server default:

```text
http://localhost:5000
```

## Endpoints

### `POST /generate-diagnostic`

Runs the ML diagnostic pipeline from raw test results.

Request:

```json
{
  "test_id": "test-0001-0000-0000-0000",
  "number_comparison": 1274.5098,
  "dot_matching": 2384.9189,
  "number_series": 16,
  "single_addition": 34,
  "single_subtraction": 37,
  "complex_arithmetic": 16
}
```

Response:

```json
{
  "predicted_class": "0",
  "confidence": 0.6818,
  "decision_path": [["NC", 1508.9295, "<="]],
  "decision_path_readable": "NC <= 1508.9295",
  "domain_severity_scores": {
    "Addition vs. Subtraction Asymmetry": 0.4406
  },
  "task_importance_scores": {
    "AS": 0.4168
  },
  "leaf_distribution": {
    "0": 14,
    "1": 6
  }
}
```

### `POST /generate_module`

Generates intervention modules from an ML diagnostic response.

Request:

```json
{
  "test_id": "test-0001-0000-0000-0000",
  "diagnostic_data": {
    "predicted_class": "0",
    "confidence": 0.6818,
    "decision_path": [["NC", 1508.9295, "<="]],
    "domain_severity_scores": {
      "Addition vs. Subtraction Asymmetry": 0.4406
    },
    "task_importance_scores": {
      "AS": 0.4168
    }
  }
}
```

Success response includes:

```json
{
  "status": "Typical (0)",
  "decision_path_rationale": "...",
  "overall_summary": "...",
  "decision_path_interpretation": "...",
  "diagnostic_modules": [
    {
      "domain_name": "Addition vs. Subtraction Asymmetry",
      "clinical_explanation": "...",
      "learning_objectives": ["...", "...", "..."],
      "conceptual_explanation": "Step 1: ... Step 2: ... Step 3: ...",
      "worked_example": {
        "problem": "13 - 8",
        "reasoning_steps": [
          "Step 1: ...",
          "Step 2: ...",
          "Step 3: ..."
        ],
        "final_answer": 5
      },
      "teaching_strategy": "...",
      "practice_set": [
        {
          "problem": "13 - 8",
          "expected_answer": 5,
          "hint": "Use 13 blocks. Take away 3 to reach 10, then take away 5 more. Count 5 blocks left."
        }
      ]
    }
  ],
  "formative_assessment": [
    {
      "question": "What is 9 + 4?",
      "expected_answer": 13
    }
  ],
  "_meta_validation_report": {
    "counts": {"expected": 10, "returned": 10, "pruned": 0},
    "modules_expected": 3,
    "modules_passed": 3,
    "math_errors": [],
    "pedagogy_errors": [],
    "schema_errors": [],
    "warnings": [],
    "success_rate": 1.0
  }
}
```

If all retries fail, the response now preserves the strongest near-valid attempt:

```json
{
  "error": "Failed to generate fully valid module after retries",
  "best_validation_report": {...},
  "best_candidate": {...}
}
```

### `POST /generate_retest`

Generates a fresh, fully validated set of retest questions targeting the top 3 deficit tasks from a student's longitudinal session history. Always uses the **latest session's** diagnostic profile as the generation basis. Pools all previous questions across all sessions for strict deduplication.

Questions are dynamically grouped by their specific task name and validated through the same math, pedagogy, and hint-quality checks used by `/generate_module` (excluding answer diversity checks due to the targeted 3-question limit). Non-symbolic string tasks (like number series or dot matching) safely bypass the mathematical AST parser.

Request:

```json
{
  "student_history": [
    {
      "session_id": 1,
      "date": "2025-11-01",
      "diagnostic_data": {
        "predicted_class": "At-Risk (1)",
        "domain_severity_scores": {
          "Addition vs. Subtraction Asymmetry": 0.55
        },
        "task_importance_scores": {
          "SUB": 0.55,
          "AS": 0.45,
          "ADD": 0.30,
          "NS": 0.20
        }
      },
      "questions_asked": [
        {
          "question": "5 + 3",
          "correct": 8
        },
        {
          "question": "8 - 3",
          "correct": 5
        }
      ]
    },
    {
      "session_id": 2,
      "date": "2026-05-08",
      "diagnostic_data": {
        "predicted_class": "At-Risk (1)",
        "domain_severity_scores": {
          "Addition vs. Subtraction Asymmetry": 0.35
        },
        "task_importance_scores": {
          "SUB": 0.35,
          "AS": 0.25,
          "ADD": 0.15,
          "NS": 0.18
        }
      },
      "questions_asked": []
    }
  ]
}
```

Notes:
- **Longitudinal Deduplication:** The endpoint automatically aggregates all items from the `questions_asked` arrays across *all* provided sessions. Any question matching a historical item is discarded and regenerated.
- **Dynamic Task Selection:** The endpoint automatically translates raw ML task acronyms (e.g., ADD, NC, SUB) provided in `task_importance_scores` dictionary into standard schema keys (single_addition, number_comparison, etc.). Overarching domain acronyms (e.g., AS or BC) are safely ignored. It then ranks the valid tasks in descending order to target the top 3 acute deficit areas.
- **Data Circularity:** To allow seamless data recycling between the database and the frontend, both the input payload (`questions_asked`) and the generated output arrays strictly use matching `"question"` and `"correct"` keys.

Success response (HTTP 200) — produces exactly 3 targeted questions per task with an individual SPED rationale:

```json
{
  "_meta_validation_report": {
    "counts": {
      "pruned": 0,
      "returned": 6
    },
    "math_errors": [],
    "pedagogy_errors": [],
    "schema_errors": []
  },
  "based_on_session": 2,
  "based_on_session_date": "2026-05-08",
  "retest_data": {
    "single_addition": {
      "rationale": "This task is being retested to reinforce the student's understanding of basic addition, which is essential for building a strong foundation in arithmetic...",
      "tests": [
        {
          "correct": 13,
          "hint": "Start with 6 blocks. Add 4 more blocks to reach 10, then add 3 more blocks.",
          "question": "6 + 7"
        }
      ]
    },
    "single_subtraction": {
      "rationale": "This task is being retested to address the student's difficulty with basic subtraction, which is a critical foundational skill...",
      "tests": [
        {
          "correct": 8,
          "hint": "Start with 15 blocks. Take away 5 blocks to reach 10, then take away 2 more blocks. Count the remaining blocks.",
          "question": "15 - 7"
        }
      ]
    }
  },
  "total_sessions_in_history": 2
}
```

Partial response (HTTP 207) — retries exhausted but at least one domain passed validation safely:

```json
{
  "retest_data": { ... },
  "_meta_validation_report": { ... },
  "based_on_session": 2,
  "based_on_session_date": "2026-05-08",
  "total_sessions_in_history": 2,
  "warning": "Partial validation failure. Review _meta_validation_report."
}
```

Failure response (HTTP 500) — no valid items could be generated or repaired after all retry loops:

```json
{
  "error": "Failed to generate valid retest questions after retries.",
  "best_validation_report": { ... }
}
```

## LLM Pipeline

The module generator uses two LLM passes:

1. Draft pass: produces SPED-informed intervention content.
2. Format pass: converts the draft into strict JSON.

Model routing is split by task:

```python
DRAFT_MODELS_TO_TRY = [
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct"
]

FORMAT_MODELS_TO_TRY = [
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct"
]
```

Formatting avoids the weaker 7B fallback because malformed JSON and dropped `expected_answer` fields were observed there.

## Domain Selection

`get_top_deficits()` combines:

- domain severity scores
- task importance scores
- broad-domain downweighting

This favors clinically actionable domains over broad summaries when possible.

Practice counts are severity-tiered per selected domain:

| Severity score | Practice items |
| --- | ---: |
| `< 0.20` | 2 |
| `0.20 - 0.39` | 4 |
| `>= 0.40` | 6 |

## Validation And Repair

Generated modules are not returned unless validation passes.

Current validation covers:

- required top-level and module schema keys
- exact module count
- exact practice count per domain
- integer `expected_answer`
- symbolic `+` / `-` practice problems only
- AST-based arithmetic correctness
- no multiplication/division
- operands `<= 20`
- subtraction subtrahend `<= 9` unless domain is `Multi-Digit Addition and Subtraction`
- non-empty hints
- hint diversity
- answer diversity
- duplicate equation detection within each module practice set
- duplicate equation detection within formative assessment
- conceptual explanation must contain `Step 1:`, `Step 2:`, `Step 3:`
- worked-example reasoning steps must be labeled `Step 1:`, `Step 2:`, etc.
- worked-example arithmetic correctness
- symbolic formative assessment questions with matching integer answers
- rejection of open-ended formative questions pretending to have integer answers
- tone guard for Typical profiles
- domain-specific Addition vs. Subtraction Asymmetry rules

### Semantic Reasoning Checks

The validator also checks reasoning text, not only final arithmetic:

- make-10 subtraction must take away the correct amount to reach 10
- final target stated in hints/reasoning must match the actual answer
- addition hints cannot say to add more than the addend unless they explicitly take back/remove the extra

Example rejected:

```text
12 - 5: Take away 2 to reach 10, then 3 more to reach 5.
```

Correct:

```text
12 - 5: Take away 2 to reach 10, then 3 more. Count 7 blocks left.
```

### Deterministic Hint Repair

Some repairable crossing-ten hints are fixed locally before pruning.

Example input:

```text
14 - 8: Hold up 14 fingers, then fold down 8 fingers.
```

Auto-repaired hint:

```text
Use 14 blocks. Take away 4 to reach 10, then take away 4 more. Count 6 blocks left.
```

## Retry Behavior

The generator makes up to 5 attempts with exponential backoff.

When validation fails, the correction prompt includes specific validation errors plus targeted repair instructions for:

- missing `expected_answer`
- practice count mismatch
- repeated equations
- AS addition/subtraction ratio failures
- answer diversity
- tone too strong for Typical profiles
- operand bounds
- crossing-ten subtraction reasoning

The best candidate/report is retained across attempts and returned in the final 500 response if no fully valid module is produced.

## Testing Utilities

### Interactive endpoint runner

Use this to select payloads from `tests/generated_test_payloads.json` and post them to one or both endpoints.

```bash
.venv/bin/python tests/select_and_request.py
```

Examples:

```bash
.venv/bin/python tests/select_and_request.py --select 5 --endpoint both
.venv/bin/python tests/select_and_request.py --select 1,4,7 --save tests/selected_results.json
.venv/bin/python tests/select_and_request.py --select all --endpoint diagnostic
```

Selections support:

```text
1
1,3,5
2-6
all
```

### ML response wrapper

Convert existing ML diagnostic responses into `/generate_module` payloads:

```bash
.venv/bin/python tests/convert_ml_response_payload.py \
  --input path/to/ml_response.json \
  --output tests/module_payloads.json
```

Run raw diagnostic inputs through `/generate-diagnostic`, wrap the result, then optionally post to `/generate_module`:

```bash
.venv/bin/python tests/convert_ml_response_payload.py \
  --input tests/generated_test_payloads.json \
  --from-diagnostic-input \
  --output tests/module_payloads.json \
  --post-module
```

## Notes For Frontend Integration

- `/generate_module` can take a long time because it may call multiple LLMs and retry.
- Use a generous timeout. A full retry cycle can exceed 90 seconds.
- Always inspect `_meta_validation_report`.
- `expected_answer` values are normalized to integers in valid outputs.
- A 500 response may include `best_candidate` and `best_validation_report`; these are useful for debugging, not for direct student display.