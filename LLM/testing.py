import requests
import json
import time

FLASK_API_URL = "http://localhost:5000/generate_module"

test_cases = [
    {
        "test_name": "Test 1: Baseline Raw Output (High Confidence, AF Highest)",
        "diagnostic_data": """Predicted Class  : At-Risk (1)
Confidence       : 0.9773
Decision Path    : ADD <= 21.5000 AND SUB <= 34.5000 AND SUB <= 29.5000 AND DM <= 4017.4167
Domain Severity  : {'Single-Digit Subtraction': np.float64(0.04653998430421901), 'Digit-Dot Matching': np.float64(0.010378171462071097), 'Number Series': np.float64(0.0), 'Basic vs. Complex Arithmetic Contrast': np.float64(0.12246328580359726), 'Addition vs. Subtraction Asymmetry': np.float64(0.03259141660804523), 'Processing-Fluency Integration': np.float64(0.18285964425405743), 'Overall Processing Efficiency': np.float64(0.1562597094742844), 'Number Comparison': np.float64(0.0), 'Single-Digit Addition': np.float64(0.03167143035728462), 'Multi-Digit Addition and Subtraction': np.float64(0.0), 'Overall Arithmetic Fluency': np.float64(0.27016133187038566), 'Symbolic vs. Non-Symbolic Processing Difference': np.float64(0.14707502586605542)}
Task Importance  : {'NC': np.float64(0.0), 'DM': np.float64(0.016375073029549936), 'NS': np.float64(0.0), 'ADD': np.float64(0.04100502451499952), 'SUB': np.float64(0.16094273054459185), 'CA': np.float64(0.0), 'NP': np.float64(0.13401717374060218), 'SN': np.float64(0.12613986907251018), 'AF': np.float64(0.23170565383154262), 'BC': np.float64(0.10503144736159044), 'AS': np.float64(0.027952244098671856), 'PF': np.float64(0.15683078380594148)}"""
    },
    {
        "test_name": "Test 2: Low Confidence Validation (Rule 6 Test)",
        "diagnostic_data": """Predicted Class  : At-Risk (1)
Confidence       : 0.4500
Decision Path    : ADD <= 21.5000 AND SUB <= 34.5000 AND SUB <= 29.5000 AND DM <= 4017.4167
Domain Severity  : {'Single-Digit Subtraction': np.float64(0.04653998430421901), 'Digit-Dot Matching': np.float64(0.010378171462071097), 'Number Series': np.float64(0.0), 'Basic vs. Complex Arithmetic Contrast': np.float64(0.12246328580359726), 'Addition vs. Subtraction Asymmetry': np.float64(0.03259141660804523), 'Processing-Fluency Integration': np.float64(0.18285964425405743), 'Overall Processing Efficiency': np.float64(0.1562597094742844), 'Number Comparison': np.float64(0.0), 'Single-Digit Addition': np.float64(0.03167143035728462), 'Multi-Digit Addition and Subtraction': np.float64(0.0), 'Overall Arithmetic Fluency': np.float64(0.27016133187038566), 'Symbolic vs. Non-Symbolic Processing Difference': np.float64(0.14707502586605542)}
Task Importance  : {'NC': np.float64(0.0), 'DM': np.float64(0.016375073029549936), 'NS': np.float64(0.0), 'ADD': np.float64(0.04100502451499952), 'SUB': np.float64(0.16094273054459185), 'CA': np.float64(0.0), 'NP': np.float64(0.13401717374060218), 'SN': np.float64(0.12613986907251018), 'AF': np.float64(0.23170565383154262), 'BC': np.float64(0.10503144736159044), 'AS': np.float64(0.027952244098671856), 'PF': np.float64(0.15683078380594148)}"""
    },
    {
        "test_name": "Test 3: Pedagogy Match (Rule 5 Test - Digit-Dot Focus)",
        "diagnostic_data": """Predicted Class  : At-Risk (1)
Confidence       : 0.9500
Decision Path    : ADD <= 21.5000 AND SUB <= 34.5000 AND SUB <= 29.5000 AND DM <= 4017.4167
Domain Severity  : {'Single-Digit Subtraction': np.float64(0.04653998430421901), 'Digit-Dot Matching': np.float64(0.8500000000000000), 'Number Series': np.float64(0.0), 'Basic vs. Complex Arithmetic Contrast': np.float64(0.12246328580359726), 'Addition vs. Subtraction Asymmetry': np.float64(0.03259141660804523), 'Processing-Fluency Integration': np.float64(0.18285964425405743), 'Overall Processing Efficiency': np.float64(0.1562597094742844), 'Number Comparison': np.float64(0.0), 'Single-Digit Addition': np.float64(0.03167143035728462), 'Multi-Digit Addition and Subtraction': np.float64(0.0), 'Overall Arithmetic Fluency': np.float64(0.27016133187038566), 'Symbolic vs. Non-Symbolic Processing Difference': np.float64(0.14707502586605542)}
Task Importance  : {'NC': np.float64(0.0), 'DM': np.float64(0.016375073029549936), 'NS': np.float64(0.0), 'ADD': np.float64(0.04100502451499952), 'SUB': np.float64(0.16094273054459185), 'CA': np.float64(0.0), 'NP': np.float64(0.13401717374060218), 'SN': np.float64(0.12613986907251018), 'AF': np.float64(0.23170565383154262), 'BC': np.float64(0.10503144736159044), 'AS': np.float64(0.027952244098671856), 'PF': np.float64(0.15683078380594148)}"""
    },
    {
        "test_name": "Test 4: Task Importance Override (Testing Dictionary Scanning)",
        "diagnostic_data": """Predicted Class  : At-Risk (1)
Confidence       : 0.9200
Decision Path    : ADD <= 21.5000 AND SUB <= 34.5000 AND SUB <= 29.5000 AND DM <= 4017.4167
Domain Severity  : {'Single-Digit Subtraction': np.float64(0.04653998430421901), 'Digit-Dot Matching': np.float64(0.010378171462071097), 'Number Series': np.float64(0.0), 'Basic vs. Complex Arithmetic Contrast': np.float64(0.12246328580359726), 'Addition vs. Subtraction Asymmetry': np.float64(0.03259141660804523), 'Processing-Fluency Integration': np.float64(0.18285964425405743), 'Overall Processing Efficiency': np.float64(0.1562597094742844), 'Number Comparison': np.float64(0.0), 'Single-Digit Addition': np.float64(0.03167143035728462), 'Multi-Digit Addition and Subtraction': np.float64(0.0), 'Overall Arithmetic Fluency': np.float64(0.27016133187038566), 'Symbolic vs. Non-Symbolic Processing Difference': np.float64(0.14707502586605542)}
Task Importance  : {'NC': np.float64(0.0), 'DM': np.float64(0.016375073029549936), 'NS': np.float64(0.0), 'ADD': np.float64(0.04100502451499952), 'SUB': np.float64(0.9999999999999999), 'CA': np.float64(0.0), 'NP': np.float64(0.13401717374060218), 'SN': np.float64(0.12613986907251018), 'AF': np.float64(0.23170565383154262), 'BC': np.float64(0.10503144736159044), 'AS': np.float64(0.027952244098671856), 'PF': np.float64(0.15683078380594148)}"""
    }
]

def run_tests():
    print("[SYSTEM] Commencing Automated Backend API Tests...\n")
    
    for idx, test in enumerate(test_cases):
        print(f"--- EXECUTING {test['test_name']} ---")
        
        try:
            response = requests.post(FLASK_API_URL, json={"diagnostic_data": test['diagnostic_data']})
            
            if response.status_code == 200:
                result = response.json()
                print("[STATUS] 200 OK - SUCCESS")
                
                target_skill = result.get('primary_target_skill', 'Refer to Objectives in JSON')
                print(f"[OUTPUT] Target Skill Identified: {target_skill}")
                
                print("\n=== COMPLETE JSON RESPONSE ===")
                print(json.dumps(result, indent=2))
                print("==============================\n")
                
                try:
                    hint = result['scaffolded_practice'][0]['hint'] 
                    print(f"[VERIFICATION] Hint Snippet Parsed: {hint[:150]}...\n")
                except Exception as e:
                    print(f"[WARNING] Could not parse hint array. Error: {e}\n")
            else:
                print(f"[STATUS] {response.status_code} - FAILED")
                print(f"[ERROR] Response Text: {response.text}\n")
                
        except requests.exceptions.ConnectionError:
            print("[CRITICAL] Connection refused. Ensure app.py is running on port 5000.\n")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected exception occurred: {e}\n")
        
        # Rate limiting throttle to prevent Hugging Face endpoint rejection
        if idx < len(test_cases) - 1:
            print("[SYSTEM] Throttling for 15 seconds to comply with API rate limits...\n")
            time.sleep(15)

    print("[SYSTEM] Automated Backend API Tests Concluded.")

if __name__ == "__main__":
    run_tests()