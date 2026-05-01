# DysCalc App Backend

This repository contains the Flask-based backend for the DysCalc project.  
It serves as an **AI Gateway**, transforming machine learning diagnostic outputs into structured, clinically grounded, and mathematically validated intervention modules using a multi-pass LLM pipeline (via OpenRouter).

---

## System Architecture & Behavior

### 1. Multi-Pass Generation Pipeline

The system uses a **2-pass LLM architecture**:

- **Pass 1 (Pedagogical Drafting)**  
  Generates SPED-informed intervention content using Qwen models.

- **Pass 2 (Strict JSON Formatting)**  
  Converts output into a strictly structured JSON schema with validation and repair handling.

---

### 2. Clinical Routing Engine

The backend parses ML diagnostic strings to extract:

- Confidence level  
- Domain severity scores  
- Dominant learning deficits  

#### Intervention Scaling Logic

| Severity Level | Practice Items per Module |
|---------------|--------------------------|
| Low           | 2                        |
| Moderate      | 4                        |
| High          | 6                        |

---

### 3. Deterministic math validation with rule-based schema and pedagogy checks

All generated content is validated and filtered before being returned

#### Math Validation (AST-based)

- Only allows: `+`, `-`
- Rejects:
  - Incorrect answers
  - Invalid expressions
  - Unsafe syntax

**Example:**
```
LLM Output: 5 + 3 = 9 
→ Rejected → Repair Loop Triggered
```

---

### 4. Schema + Pedagogy Enforcement

Each module is validated for:

- Required JSON structure  
- Step-based explanations (`Step 1`, `Step 2`, `Step 3`)  
- Worked example correctness  
- Hint diversity  
- Domain alignment  
- Prevents most invalid subtraction cases (e.g., 3 - 5), especially in practice sets and hints

---

### 5. Retry & Repair Loop

- Maximum: **5 attempts**
- Uses exponential backoff:

```python
time.sleep(2 ** attempt)
```

- Injects correction prompts based on:
  - Math errors  
  - Pedagogy violations  
  - Schema failures  

---

### LLM Resilience

- Uses multiple fallback models via OpenRouter
- Automatically retries across models if one fails or times out

## Frontend Requirements

Due to multi-pass LLM + retry loop:

- Response time: **10–45+ seconds**
- Frontend must:
  - Implement loading states  
  - Use timeout of **60–90 seconds minimum**

---

## Environment Setup

### 1. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file:

```env
OPENROUTER_TOKEN=your_openrouter_api_key_here
FLASK_ENV=development
```

### 3. Run the Server

```bash
python -m index
```

### 4. Access the Server
Server runs at:

```
http://localhost:5000
```

## File Structure Overview

- **index.py**  
  Main Flask application and API route definitions.

- **helpers.py**  
  Core logic layer (AST math validation, schema enforcement, LLM gateway, and ML parsing).

- **constants.py**  
  Stores API configuration, model fallbacks, allowed mathematical operations, and clinical interpretation mappings.

---

## API Endpoints

---

### 1. Generate Full Intervention Module

**POST** `/generate_module`

#### Request Body

```json
{
  "diagnostic_data": "Student Profile - At-Risk (1) | Confidence: 0.85 | Domain Severity: {'ADD': 0.30, 'SUB': 0.15}"
}
```

---

#### Success Response (200)

```json
{
  "status": "At-Risk (1)",
  "decision_path_rationale": "...",
  "overall_summary": "...",
  "decision_path_interpretation": "...",
  "diagnostic_modules": [...],
  "formative_assessment": [...],
  "_meta_validation_report": {...}
}
```

---

### Important Notes

- `expected_answer` in **practice_set**  
   Integer (strictly validated)

- `expected_answer` in **formative_assessment**  
   Returned as **string** (must be handled in frontend)

---

#### Validation Report Structure

```json
"_meta_validation_report": {
  "counts": {
    "expected": 18,
    "returned": 18,
    "pruned": 0
  },
  "modules_expected": 3,
  "modules_passed": 3,
  "math_errors": [],
  "pedagogy_errors": [],
  "schema_errors": [],
  "warnings": [],
  "success_rate": 1.0
}
```

---

#### Error Responses

**400**
```json
{"error": "Invalid request"}
```

**500**
```json
{"error": "Failed to generate valid module after retries"}
```

---

### 2. Generate Retest Questions

**POST** `/generate_retest`

---

#### Request Body

```json
{
  "diagnostic_data": "Student Profile - At-Risk (1)",
  "previous_questions": [
    {"problem": "5+3", "expected_answer": 8}
  ]
}
```

---

### Behavior

- Generates **5 new questions**
- Avoids duplicates using:

```python
difflib.SequenceMatcher > 0.85
```

---

### Constraints

- Problems must be:
  - Pure symbolic math (`"5+3"`)
  - No word problems  
  - Operands ≤ 20  

- Hints must reference **physical actions**

- Only `+` and `-` allowed

---

#### Success Response

```json
{
  "retest_questions": [
    {
      "problem": "6+4",
      "hint": "Use blocks",
      "expected_answer": 10
    }
  ],
  "_meta_validation_report": {
    "counts": {"returned": 5, "pruned": 0},
    "math_errors": []
  }
}
```

---

#### Error Responses

**400**
```json
{"error": "Invalid request"}
```

**500**
```json
{"error": "All generated retest questions were invalid"}
```

---

## Data Constraints (Frontend Checklist)

### Valid
- `"5+3"`
- `8`

### Invalid
- `"5 + 3 apples"`
- `"8"` (for practice_set)
- `3 - 5`

---

## System Guarantees

- Deterministic math validation (AST-based)  
- Strict schema enforcement  
- Retry + repair loop  
- Duplicate retest prevention  
- Clinically grounded outputs  
- Domain-aware difficulty scaling  

---

## Known Edge Cases

1. **Modules may be partially pruned**
   - Example: 3 expected → only 1 valid  
   - Check: `modules_passed`

2. **Conceptual explanation may fail format**
   - Missing Step 1–3 → appears in `pedagogy_errors`

3. **Formative answers returned as strings**
   - Returned as string due to validation normalization (frontend should cast to integer if needed)

---

## Suggested Frontend Safeguards

- Do not assume:
  - All modules exist  
  - Counts match expected  

- Always check:
  - `_meta_validation_report`

- Gracefully handle:
  - Partial outputs  
  - Reduced practice sets  

---