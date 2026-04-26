import requests
import json

# The new endpoint you added to api/index.py
RETEST_API_URL = "http://localhost:5000/generate_retest"

def test_retest_functionality():
    print("[SYSTEM] Commencing Retest Generation Test...\n")
    print("--- EXECUTING AT-RISK RETEST SCENARIO ---")
    
    # snippet of Test Case #5 to simulate the student's profile
    sample_ml_data = """Predicted Class  : At-Risk (1)
Confidence       : 0.6316
Domain Severity  : {'Addition vs. Subtraction Asymmetry': 0.45082946493547227, 'Basic vs. Complex Arithmetic Contrast': 0.15953196920975835}"""
    
    # We pass "fake" previous questions so the AI knows what NOT to repeat
    mock_previous_questions = [
        {"problem": "5 + 3 = ?", "expected_answer": "8", "hint": "Count 5 blocks, then add 3 more."},
        {"problem": "8 - 3 = ?", "expected_answer": "5", "hint": "Start with 8 blocks, take away 3."}
    ]
    
    payload = {
        "diagnostic_data": sample_ml_data,
        "previous_questions": mock_previous_questions
    }
    
    try:
        response = requests.post(RETEST_API_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            print("[STATUS] 200 OK - SUCCESS")
            print("\n=== NEW RETEST QUESTIONS ===")
            print(json.dumps(response.json(), indent=2))
            print("============================\n")
        else:
            print(f"[STATUS] {response.status_code} - FAILED")
            print(f"[ERROR] Response Text: {response.text}\n")
            
    except requests.exceptions.ConnectionError:
        print("[CRITICAL] Connection refused. Ensure Flask is running on port 5000.\n")
    except Exception as e:
        print(f"[ERROR] Unexpected exception occurred: {e}\n")

if __name__ == "__main__":
    test_retest_functionality()