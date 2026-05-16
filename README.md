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

Generates a fresh, fully validated set of retest questions from a student's longitudinal
session history. Always uses the **latest session's** diagnostic profile as the generation
basis. Pools all previous questions across all sessions for deduplication. Questions are
validated through the same math, pedagogy, and hint-quality checks used by `/generate_module`.

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
          "AS": 0.55
        }
      },
      "questions_asked": [
        {"problem": "5 + 3", "expected_answer": 8},
        {"problem": "8 - 3", "expected_answer": 5}
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
          "AS": 0.35
        }
      },
      "questions_asked": []
    }
  ]
}
```

Notes:
- `questions_asked` must be an empty array `[]` for the current (latest) session.
- `task_importance_scores` is used by the domain selection algorithm and should be included when available.
- All problems in `questions_asked` must be symbolic equations (e.g. `"13 - 5"`), not word problems.

Success response (HTTP 200) — all 5 questions passed full validation:

```json
{
  "retest_questions": [
    {
      "problem": "13 - 6",
      "hint": "Use 13 blocks. Take away 3 to reach 10, then take away 3 more. Count 7 blocks left.",
      "expected_answer": 7
    }
  ],
  "_meta_validation_report": {
    "counts": {"returned": 5, "pruned": 0},
    "math_errors": [],
    "pedagogy_errors": [],
    "schema_errors": []
  },
  "based_on_session": 2,
  "based_on_session_date": "2026-05-08",
  "total_sessions_in_history": 2
}
```

Partial response (HTTP 207) — retries exhausted but at least one valid question was produced:

```json
{
  "retest_questions": [...],
  "_meta_validation_report": {...},
  "based_on_session": 2,
  "based_on_session_date": "2026-05-08",
  "total_sessions_in_history": 2,
  "warning": "Only 3 of 5 questions passed full validation. Review _meta_validation_report for details."
}
```

Failure response (HTTP 500) — no valid questions produced after all retries:

```json
{
  "error": "Failed to generate valid retest questions after retries.",
  "best_validation_report": {...}
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