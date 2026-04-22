import requests
import json
import time

FLASK_API_URL = "http://localhost:5000/generate_module"

# ALL 10 UNSEEN TEST CASES FROM FUNA-DB C4.5 DEPLOYMENT LOGS
test_cases = [
    {
        "test_name": "Test Case #1: Typical Profile (Minor Asymmetry focus)",
        "diagnostic_data": """Predicted Class  : Typical (0)
Confidence       : 0.3095
Decision Path    : NC <= 1508.9295 AND ADD > 15.7500 AND DM <= 3564.2007 AND NC <= 1498.5457 AND ADD > 29.5000 AND ADD <= 63.0000 AND DM > 1056.7697 AND DM > 1142.8244 AND SUB <= 61.0000 AND ADD > 32.5000
Domain Severity  : {'Number Series': 0.0, 'Addition vs. Subtraction Asymmetry': 0.2913721987343177, 'Overall Arithmetic Fluency': 0.009475617710696434, 'Multi-Digit Addition and Subtraction': 0.0, 'Digit-Dot Matching': 0.005061122811804657, 'Symbolic vs. Non-Symbolic Processing Difference': 0.0648289846321958, 'Basic vs. Complex Arithmetic Contrast': 0.24025729316546665, 'Overall Processing Efficiency': 0.1607702537877139, 'Processing-Fluency Integration': 0.17444638401774817, 'Single-Digit Subtraction': 0.0011159536460000753, 'Single-Digit Addition': 0.01570704413343638, 'Number Comparison': 0.03696514736062019}
Task Importance  : NC: 0.10, DM: 0.03, ADD: 0.03, SUB: 0.01, NP: 0.14, SN: 0.06, AF: 0.01, BC: 0.21, AS: 0.26, PF: 0.15"""
    },
    {
        "test_name": "Test Case #2: Typical Profile (Zero Confidence / Noisy Data)",
        "diagnostic_data": """Predicted Class  : Typical (0)
Confidence       : 0.0000
Decision Path    : NC <= 1508.9295 AND ADD > 15.7500 AND DM <= 3564.2007 AND NC <= 1498.5457 AND ADD > 29.5000 AND ADD <= 63.0000 AND DM <= 1056.7697
Domain Severity  : {'Number Series': 0.0, 'Addition vs. Subtraction Asymmetry': 0.011459913846871356, 'Overall Arithmetic Fluency': 0.13290454919157232, 'Multi-Digit Addition and Subtraction': 0.0, 'Digit-Dot Matching': 0.006598422981702609, 'Symbolic vs. Non-Symbolic Processing Difference': 0.1962787577933142, 'Basic vs. Complex Arithmetic Contrast': 0.14077209873708713, 'Overall Processing Efficiency': 0.20494782176003845, 'Processing-Fluency Integration': 0.2658944863415979, 'Single-Digit Subtraction': 0.0, 'Single-Digit Addition': 0.024630062391666478, 'Number Comparison': 0.016513886956149547}
Task Importance  : NC: 0.04, DM: 0.05, ADD: 0.06, NP: 0.18, SN: 0.18, AF: 0.12, BC: 0.13, AS: 0.01, PF: 0.24"""
    }
]

def run_tests():
    print("[SYSTEM] Commencing Full Unseen Data Test Suite (10 Cases)...\n")
    
    for idx, test in enumerate(test_cases):
        print(f"--- EXECUTING {test['test_name']} ---")
        
        try:
            response = requests.post(FLASK_API_URL, json={"diagnostic_data": test['diagnostic_data']}, timeout=900)
            
            if response.status_code == 200:
                result = response.json()
                print("[STATUS] 200 OK - SUCCESS")
                
                status = result.get('status', 'Unknown')
                modules = result.get('diagnostic_modules', [])
                print(f"[OUTPUT] Final Status: {status}")
                print(f"[OUTPUT] Generated {len(modules)} distinct intervention modules.")
                
                print("\n=== COMPLETE JSON RESPONSE ===")
                print(json.dumps(result, indent=2))
                print("==============================\n")
                
            else:
                print(f"[STATUS] {response.status_code} - FAILED")
                print(f"[ERROR] Response Text: {response.text}\n")
                
        except requests.exceptions.ConnectionError:
            print("[CRITICAL] Connection refused. Ensure your Flask server is running on port 5000.\n")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected exception occurred: {e}\n")
        
        # 30-second throttle to avoid HF Rate Limits
        if idx < len(test_cases) - 1:
            print("[SYSTEM] Throttling for 30 seconds to give the server a breather...\n")
            time.sleep(30)

    print("[SYSTEM] All 10 Automated Tests Concluded.")

if __name__ == "__main__":
    run_tests()