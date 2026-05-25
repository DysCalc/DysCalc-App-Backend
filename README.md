# DysCalc App Backend

Flask backend for the DysCalc project. It serves two main jobs:

1. Convert raw diagnostic test scores into structured ML diagnostic output.
2. Convert that ML diagnostic output into validated SPED intervention modules through a multi-pass LLM pipeline.
3. Generate longitudinal retest question sets targeting a student's top deficit tasks across sessions.


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

Generates a fresh, validated set of retest questions targeting the top 3 deficit tasks from a student's longitudinal session history. Always uses the **latest session's** diagnostic profile as the generation basis. Pools all previous question IDs across all sessions for strict deduplication.

Questions are sourced from the validated item bank (`dyscalc_simulation.json`) and supplemented algorithmically when the bank is exhausted. The LLM is used only to generate one clinical rationale string per task — it does not generate questions for any task except `complex_arithmetic` gap-fill.

---

## Question Source Strategy

| Task | Bank Size | Retest Sample | Gap-fill Method |
|---|---|---|---|
| `number_comparison` | 42 | 10 | Algorithmic (all pairs from range 2–15) |
| `dot_matching` | 42 | 10 | Algorithmic (match / non-match pairs) |
| `number_series` | 20 | 10 | Algorithmic (arithmetic sequences, varied steps) |
| `single_addition` | 81 | 15 | Algorithmic (all a+b pairs, 1–9) |
| `single_subtraction` | 81 | 15 | Algorithmic (all a-b pairs, positive result) |
| `complex_arithmetic` | 80 | 15 | LLM gap-fill only |

Gap-fill is only triggered when the item bank for a task is exhausted due to prior session history. In early sessions this will not occur for any task. Teacher observation hints are pre-written per task type and are never shown to the student.

---

## Request

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
          "ADD": 0.30,
          "NS": 0.20,
          "NC": 0.15,
          "DM": 0.10,
          "CA": 0.05
        }
      },
      "questions_asked": [
        { "id": "add_39", "question": "5 + 3", "correct": 8 },
        { "id": "sub_10", "question": "8 - 3", "correct": 5 }
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
          "ADD": 0.15,
          "NS": 0.18,
          "NC": 0.12,
          "DM": 0.08,
          "CA": 0.04
        }
      },
      "questions_asked": []
    }
  ]
}
```

### Notes

- **`id` field required for precise deduplication.** Each item in `questions_asked` must include the `id` field matching the item bank (e.g., `"nc_07"`, `"add_39"`). Without it, the system falls back to question-string matching which is less reliable across sessions.

- **Longitudinal deduplication.** The endpoint aggregates all items from `questions_asked` across every session in `student_history`. Any question whose `id` matches a historical item is excluded from the pool before sampling. The remaining bank items are shuffled and sampled randomly.

- **Dynamic top 3 task selection.** On every call, the endpoint reads `task_importance_scores` from the latest session's `diagnostic_data`, maps the 6 raw task acronyms (`NC`, `DM`, `NS`, `ADD`, `SUB`, `CA`) to their schema keys, sorts them descending, and selects the top 3. Overarching derived domain acronyms (`AS`, `BC`, `PF`, `NP`, `SN`, `AF`) are not mappable to item bank tasks and are safely ignored. The top 3 therefore change dynamically as the student's profile evolves across sessions.

- **Retest sample size.** Each task returns a fixed number of questions from the bank, not the full bank count. This is intentional to prevent student fatigue — the retest is a targeted probe, not a full re-screening. The remaining unused bank items are preserved for future sessions.

- **Data circularity.** Both the input `questions_asked` array and the output `tests` array use matching `"id"`, `"question"`, and `"correct"` keys. Store the output `tests` as the next session's `questions_asked` to maintain deduplication continuity across sessions.

---

## Success Response (HTTP 200)

Returns exactly the `RETEST_SAMPLE_SIZE` questions per task across all 3 tasks.

```json
{
  "_meta_validation_report": {
    "counts": {
      "returned": 40,
      "pruned": 0
    },
    "tasks": {
      "single_subtraction": {
        "from_bank": 15,
        "from_gap_fill": 0,
        "excluded_from_pool": 2,
        "target": 15
      },
      "single_addition": {
        "from_bank": 15,
        "from_gap_fill": 0,
        "excluded_from_pool": 2,
        "target": 15
      },
      "number_series": {
        "from_bank": 10,
        "from_gap_fill": 0,
        "excluded_from_pool": 0,
        "target": 10
      }
    },
    "math_errors": [],
    "pedagogy_errors": [],
    "schema_errors": []
  },
  "based_on_session": 2,
  "based_on_session_date": "2026-05-08",
  "total_sessions_in_history": 2,
  "retest_data": {
    "single_subtraction": {
      "rationale": "Retesting single subtraction is necessary due to the student's At-Risk classification and the relatively high task importance score of 0.35, indicating a need to monitor and address foundational arithmetic skills.",
      "tests": [
        {
          "id": "sub_70",
          "question": "15 - 7",
          "correct": 8,
          "hint": "Watch for counting-down-from strategies versus fact retrieval. For crossing-ten items, note whether the student bridges through 10. Prompt: 'How many do you need to take away to reach 10 first?'"
        }
      ]
    },
    "single_addition": {
      "rationale": "Retesting single addition is warranted due to the task importance score of 0.15 and the student's At-Risk status, highlighting the need to reinforce basic arithmetic skills.",
      "tests": [
        {
          "id": "add_52",
          "question": "6 + 7",
          "correct": 13,
          "hint": "Watch for finger counting or counting-all strategies. Note whether the student uses make-10 or counts on from the larger number. Prompt: 'Can you start from the bigger number and count on?'"
        }
      ]
    },
    "number_series": {
      "rationale": "Number series completion is being retested because of its task importance score of 0.18, which highlights the need to assess the student's ability to recognise patterns and sequences.",
      "tests": [
        {
          "id": "ns_16",
          "question": "15, 25, 35, _",
          "correct": 45,
          "hint": "Observe whether the student identifies the rule (counting by 2s, 5s, etc.) or guesses. Ask the student to explain the pattern aloud. Watch for errors at decade boundaries."
        }
      ]
    }
  }
}
```

### `_meta_validation_report` — `tasks` field breakdown

| Field | Description |
|---|---|
| `from_bank` | Questions taken directly from the item bank |
| `from_gap_fill` | Questions filled algorithmically or via LLM (CA only) when the bank was exhausted |
| `excluded_from_pool` | Questions removed from sampling because they appear in the student's session history |
| `target` | The configured sample size for this task (`RETEST_SAMPLE_SIZE`) |

Gap-fill question IDs follow the pattern `<task_prefix>_gen_XX` (algorithmic) or `<task_prefix>_llm_XX` (LLM, CA only).

---

## Partial Response (HTTP 207)

Returned when at least one task has fewer questions than its target — typically only when both the bank and the gap-fill generator cannot produce enough unique items (very unlikely in practice).

```json
{
  "retest_data": { "...": "..." },
  "_meta_validation_report": { "...": "..." },
  "based_on_session": 2,
  "based_on_session_date": "2026-05-08",
  "total_sessions_in_history": 2,
  "warning": "Some tasks returned fewer questions than the target. Short: ['number_series']. Review _meta_validation_report."
}
```

---

## Failure Response (HTTP 500)

Returned only when a critical server error occurs before any questions can be assembled.

```json
{
  "error": "Failed to generate valid retest questions after retries.",
  "best_validation_report": { "...": "..." }
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