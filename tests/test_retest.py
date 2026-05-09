import requests
import json

RETEST_API_URL = "http://localhost:5000/generate_retest"

def test_retest_functionality():
    print("[SYSTEM] Commencing Longitudinal Retest History Test...\n")
    print("--- EXECUTING MULTI-SESSION SCENARIO ---")

    # Simulates a student's history across 3 sessions.
    # Session 1: Initial assessment (high severity AS)
    # Session 2: Follow-up (severity dropped — some improvement)
    # Session 3: Latest session (further improvement, generating new questions now)
    mock_student_history = [
        {
            "session_id": 1,
            "date": "2025-11-01",
            "diagnostic_data": {
                "predicted_class": "At-Risk (1)",
                "domain_severity_scores": {
                    "Addition vs. Subtraction Asymmetry": 0.55
                }
            },
            "questions_asked": [
                {"problem": "5 + 3", "expected_answer": 8},
                {"problem": "8 - 3", "expected_answer": 5}
            ]
        },
        {
            "session_id": 2,
            "date": "2026-02-15",
            "diagnostic_data": {
                "predicted_class": "At-Risk (1)",
                "domain_severity_scores": {
                    "Addition vs. Subtraction Asymmetry": 0.40
                }
            },
            "questions_asked": [
                {"problem": "12 - 5", "expected_answer": 7},
                {"problem": "9 + 4", "expected_answer": 13}
            ]
        },
        {
            "session_id": 3,
            "date": "2026-05-08",
            "diagnostic_data": {
                "predicted_class": "At-Risk (1)",
                "confidence": 0.6316,
                "domain_severity_scores": {
                    "Addition vs. Subtraction Asymmetry": 0.25,
                    "Basic vs. Complex Arithmetic Contrast": 0.15
                }
            },
            "questions_asked": []  # Empty — this is the current session, generating now
        }
    ]

    payload = {
        "student_history": mock_student_history
    }

    try:
        response = requests.post(RETEST_API_URL, json=payload, timeout=90)

        if response.status_code == 200:
            result = response.json()
            print("[STATUS] 200 OK - SUCCESS")
            print(f"\n[INFO] Based on session: {result.get('based_on_session')} ({result.get('based_on_session_date')})")
            print(f"[INFO] Total sessions in history: {result.get('total_sessions_in_history')}")
            print("\n=== NEW RETEST QUESTIONS ===")
            print(json.dumps(result, indent=2))
            print("============================\n")
            print("[VERIFICATION] Check: did it avoid 5+3, 8-3, 12-5, and 9+4 from previous sessions?")
            print("[VERIFICATION] Check: are new questions based on the LATEST session profile (session 3)?")
        else:
            print(f"[STATUS] {response.status_code} - FAILED")
            print(f"[ERROR] Response Text: {response.text}\n")

    except requests.exceptions.ConnectionError:
        print("[CRITICAL] Connection refused. Ensure Flask is running on port 5000.\n")
    except Exception as e:
        print(f"[ERROR] Unexpected exception occurred: {e}\n")

if __name__ == "__main__":
    test_retest_functionality()