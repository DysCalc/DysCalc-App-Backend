import requests
import json
import time

FLASK_API_URL = "http://localhost:5000/generate_module"

# ALL 10 UNSEEN TEST CASES FROM FUNA-DB C4.5 DEPLOYMENT LOGS
# Extracted directly from train.txt, formatted with Raw Confidence and Raw Class Label.
test_cases = [
    {
        "test_name": "Test Case #1: Typical Profile",
        "diagnostic_data": {
            "predicted_class": "Typical (0)",
            "confidence": 0.6882,
            "decision_path": [
                [
                    "NC",
                    1508.9295,
                    "<="
                ],
                [
                    "ADD",
                    15.75,
                    ">"
                ],
                [
                    "DM",
                    3564.2007,
                    "<="
                ],
                [
                    "NC",
                    1498.5457,
                    "<="
                ],
                [
                    "ADD",
                    29.5,
                    ">"
                ],
                [
                    "ADD",
                    63.0,
                    "<="
                ],
                [
                    "DM",
                    1056.7697,
                    ">"
                ],
                [
                    "DM",
                    1142.8244,
                    ">"
                ],
                [
                    "SUB",
                    61.0,
                    "<="
                ],
                [
                    "ADD",
                    32.5,
                    ">"
                ]
            ],
            "domain_severity": {
                "Number Series": 0.0,
                "Addition vs. Subtraction Asymmetry": 0.2913721987343177,
                "Overall Arithmetic Fluency": 0.009475617710696434,
                "Multi-Digit Addition and Subtraction": 0.0,
                "Digit-Dot Matching": 0.005061122811804657,
                "Symbolic vs. Non-Symbolic Processing Difference": 0.0648289846321958,
                "Basic vs. Complex Arithmetic Contrast": 0.24025729316546665,
                "Overall Processing Efficiency": 0.1607702537877139,
                "Processing-Fluency Integration": 0.17444638401774817,
                "Single-Digit Subtraction": 0.0011159536460000753,
                "Single-Digit Addition": 0.01570704413343638,
                "Number Comparison": 0.03696514736062019
            },
            "task_importance": {
                "NC": 0.1,
                "DM": 0.03,
                "ADD": 0.03,
                "SUB": 0.01,
                "NP": 0.14,
                "SN": 0.06,
                "AF": 0.01,
                "BC": 0.21,
                "AS": 0.26,
                "PF": 0.15
            }
        },
        "test_id": "test_case_#1"
    },
    {
        "test_name": "Test Case #2: Typical Profile",
        "diagnostic_data": {
            "predicted_class": "Typical (0)",
            "confidence": 0.8,
            "decision_path": [
                [
                    "NC",
                    1508.9295,
                    "<="
                ],
                [
                    "ADD",
                    15.75,
                    ">"
                ],
                [
                    "DM",
                    3564.2007,
                    "<="
                ],
                [
                    "NC",
                    1498.5457,
                    "<="
                ],
                [
                    "ADD",
                    29.5,
                    ">"
                ],
                [
                    "ADD",
                    63.0,
                    "<="
                ],
                [
                    "DM",
                    1056.7697,
                    "<="
                ]
            ],
            "domain_severity": {
                "Number Series": 0.0,
                "Addition vs. Subtraction Asymmetry": 0.011459913846871356,
                "Overall Arithmetic Fluency": 0.13290454919157232,
                "Multi-Digit Addition and Subtraction": 0.0,
                "Digit-Dot Matching": 0.006598422981702609,
                "Symbolic vs. Non-Symbolic Processing Difference": 0.1962787577933142,
                "Basic vs. Complex Arithmetic Contrast": 0.14077209873708713,
                "Overall Processing Efficiency": 0.20494782176003845,
                "Processing-Fluency Integration": 0.2658944863415979,
                "Single-Digit Subtraction": 0.0,
                "Single-Digit Addition": 0.024630062391666478,
                "Number Comparison": 0.016513886956149547
            },
            "task_importance": {
                "NC": 0.04,
                "DM": 0.05,
                "ADD": 0.06,
                "NP": 0.18,
                "SN": 0.18,
                "AF": 0.12,
                "BC": 0.13,
                "AS": 0.01,
                "PF": 0.24
            }
        },
        "test_id": "test_case_#2"
    },
    {
        "test_name": "Test Case #3: Typical Profile",
        "diagnostic_data": {
            "predicted_class": "Typical (0)",
            "confidence": 0.6882,
            "decision_path": [
                [
                    "NC",
                    1508.9295,
                    "<="
                ],
                [
                    "ADD",
                    15.75,
                    ">"
                ],
                [
                    "DM",
                    3564.2007,
                    "<="
                ],
                [
                    "NC",
                    1498.5457,
                    "<="
                ],
                [
                    "ADD",
                    29.5,
                    ">"
                ],
                [
                    "ADD",
                    63.0,
                    "<="
                ],
                [
                    "DM",
                    1056.7697,
                    ">"
                ],
                [
                    "DM",
                    1142.8244,
                    ">"
                ],
                [
                    "SUB",
                    61.0,
                    "<="
                ],
                [
                    "ADD",
                    32.5,
                    ">"
                ]
            ],
            "domain_severity": {
                "Number Series": 0.0,
                "Addition vs. Subtraction Asymmetry": 0.3241026368154854,
                "Overall Arithmetic Fluency": 0.16942767951343135,
                "Multi-Digit Addition and Subtraction": 0.0,
                "Digit-Dot Matching": 0.00012029451405441278,
                "Symbolic vs. Non-Symbolic Processing Difference": 0.10849941981967068,
                "Basic vs. Complex Arithmetic Contrast": 0.15041702811759447,
                "Overall Processing Efficiency": 0.061216380096170656,
                "Processing-Fluency Integration": 0.15665510726088827,
                "Single-Digit Subtraction": 0.0022480529098280523,
                "Single-Digit Addition": 0.0040404269857775635,
                "Number Comparison": 0.023272973967099087
            },
            "task_importance": {
                "NC": 0.07,
                "DM": 0.0,
                "ADD": 0.01,
                "SUB": 0.02,
                "NP": 0.06,
                "SN": 0.1,
                "AF": 0.16,
                "BC": 0.14,
                "AS": 0.3,
                "PF": 0.15
            }
        },
        "test_id": "test_case_#3"
    },
    {
        "test_name": "Test Case #4: Typical Profile",
        "diagnostic_data": {
            "predicted_class": "Typical (0)",
            "confidence": 0.6882,
            "decision_path": [
                [
                    "NC",
                    1508.9295,
                    "<="
                ],
                [
                    "ADD",
                    15.75,
                    ">"
                ],
                [
                    "DM",
                    3564.2007,
                    "<="
                ],
                [
                    "NC",
                    1498.5457,
                    "<="
                ],
                [
                    "ADD",
                    29.5,
                    ">"
                ],
                [
                    "ADD",
                    63.0,
                    "<="
                ],
                [
                    "DM",
                    1056.7697,
                    ">"
                ],
                [
                    "DM",
                    1142.8244,
                    ">"
                ],
                [
                    "SUB",
                    61.0,
                    "<="
                ],
                [
                    "ADD",
                    32.5,
                    ">"
                ]
            ],
            "domain_severity": {
                "Number Series": 0.0,
                "Addition vs. Subtraction Asymmetry": 0.2861261480432768,
                "Overall Arithmetic Fluency": 0.04509929898830997,
                "Multi-Digit Addition and Subtraction": 0.0,
                "Digit-Dot Matching": 0.01692270118313578,
                "Symbolic vs. Non-Symbolic Processing Difference": 0.33283838405770705,
                "Basic vs. Complex Arithmetic Contrast": 0.03318741165344397,
                "Overall Processing Efficiency": 0.21272647184570134,
                "Processing-Fluency Integration": 0.03333564915898247,
                "Single-Digit Subtraction": 9.002217511862247e-06,
                "Single-Digit Addition": 0.03678276368899844,
                "Number Comparison": 0.0029721691629321736
            },
            "task_importance": {
                "NC": 0.01,
                "DM": 0.11,
                "ADD": 0.08,
                "SUB": 0.0,
                "NP": 0.18,
                "SN": 0.28,
                "AF": 0.04,
                "BC": 0.03,
                "AS": 0.24,
                "PF": 0.03
            }
        },
        "test_id": "test_case_#4"
    },
    {
        "test_name": "Test Case #5: At-Risk Profile",
        "diagnostic_data": {
            "predicted_class": "At-Risk (1)",
            "confidence": 0.619,
            "decision_path": [
                [
                    "NC",
                    1508.9295,
                    "<="
                ],
                [
                    "ADD",
                    15.75,
                    ">"
                ],
                [
                    "DM",
                    3564.2007,
                    "<="
                ],
                [
                    "NC",
                    1498.5457,
                    "<="
                ],
                [
                    "ADD",
                    29.5,
                    ">"
                ],
                [
                    "ADD",
                    63.0,
                    "<="
                ],
                [
                    "DM",
                    1056.7697,
                    ">"
                ],
                [
                    "DM",
                    1142.8244,
                    ">"
                ],
                [
                    "SUB",
                    61.0,
                    "<="
                ],
                [
                    "ADD",
                    32.5,
                    "<="
                ]
            ],
            "domain_severity": {
                "Number Series": 0.0,
                "Addition vs. Subtraction Asymmetry": 0.45082946493547227,
                "Overall Arithmetic Fluency": 0.019422325216776917,
                "Multi-Digit Addition and Subtraction": 0.0,
                "Digit-Dot Matching": 0.0032984052876817013,
                "Symbolic vs. Non-Symbolic Processing Difference": 0.20261266286686855,
                "Basic vs. Complex Arithmetic Contrast": 0.15953196920975835,
                "Overall Processing Efficiency": 0.03994603799603673,
                "Processing-Fluency Integration": 0.0643081303580098,
                "Single-Digit Subtraction": 0.0014822579073659468,
                "Single-Digit Addition": 0.028917161700197,
                "Number Comparison": 0.02965158452183282
            },
            "task_importance": {
                "NC": 0.08,
                "DM": 0.02,
                "ADD": 0.06,
                "SUB": 0.01,
                "NP": 0.04,
                "SN": 0.18,
                "AF": 0.02,
                "BC": 0.14,
                "AS": 0.4,
                "PF": 0.06
            }
        },
        "test_id": "test_case_#5"
    },
    {
        "test_name": "Test Case #6: Typical Profile",
        "diagnostic_data": {
            "predicted_class": "Typical (0)",
            "confidence": 0.6882,
            "decision_path": [
                [
                    "NC",
                    1508.9295,
                    "<="
                ],
                [
                    "ADD",
                    15.75,
                    ">"
                ],
                [
                    "DM",
                    3564.2007,
                    "<="
                ],
                [
                    "NC",
                    1498.5457,
                    "<="
                ],
                [
                    "ADD",
                    29.5,
                    ">"
                ],
                [
                    "ADD",
                    63.0,
                    "<="
                ],
                [
                    "DM",
                    1056.7697,
                    ">"
                ],
                [
                    "DM",
                    1142.8244,
                    ">"
                ],
                [
                    "SUB",
                    61.0,
                    "<="
                ],
                [
                    "ADD",
                    32.5,
                    ">"
                ]
            ],
            "domain_severity": {
                "Number Series": 0.0,
                "Addition vs. Subtraction Asymmetry": 0.18913876977011831,
                "Overall Arithmetic Fluency": 0.12026453643786465,
                "Multi-Digit Addition and Subtraction": 0.0,
                "Digit-Dot Matching": 0.013080543445585109,
                "Symbolic vs. Non-Symbolic Processing Difference": 0.1596452948484811,
                "Basic vs. Complex Arithmetic Contrast": 0.12667159369585482,
                "Overall Processing Efficiency": 0.22212841038639794,
                "Processing-Fluency Integration": 0.1341954466161378,
                "Single-Digit Subtraction": 0.0016951119568962255,
                "Single-Digit Addition": 0.00945627495488461,
                "Number Comparison": 0.02372401788777943
            },
            "task_importance": {
                "NC": 0.06,
                "DM": 0.08,
                "ADD": 0.02,
                "SUB": 0.01,
                "NP": 0.19,
                "SN": 0.14,
                "AF": 0.1,
                "BC": 0.11,
                "AS": 0.16,
                "PF": 0.12
            }
        },
        "test_id": "test_case_#6"
    },
    {
        "test_name": "Test Case #7: At-Risk Profile",
        "diagnostic_data": {
            "predicted_class": "At-Risk (1)",
            "confidence": 0.619,
            "decision_path": [
                [
                    "NC",
                    1508.9295,
                    "<="
                ],
                [
                    "ADD",
                    15.75,
                    ">"
                ],
                [
                    "DM",
                    3564.2007,
                    "<="
                ],
                [
                    "NC",
                    1498.5457,
                    "<="
                ],
                [
                    "ADD",
                    29.5,
                    ">"
                ],
                [
                    "ADD",
                    63.0,
                    "<="
                ],
                [
                    "DM",
                    1056.7697,
                    ">"
                ],
                [
                    "DM",
                    1142.8244,
                    ">"
                ],
                [
                    "SUB",
                    61.0,
                    "<="
                ],
                [
                    "ADD",
                    32.5,
                    "<="
                ]
            ],
            "domain_severity": {
                "Number Series": 0.0,
                "Addition vs. Subtraction Asymmetry": 0.3035525511719146,
                "Overall Arithmetic Fluency": 0.01985053852131092,
                "Multi-Digit Addition and Subtraction": 0.0,
                "Digit-Dot Matching": 0.010733366518229428,
                "Symbolic vs. Non-Symbolic Processing Difference": 0.25392896205421633,
                "Basic vs. Complex Arithmetic Contrast": 0.25385175733227305,
                "Overall Processing Efficiency": 0.10961345582170148,
                "Processing-Fluency Integration": 0.022253327625528556,
                "Single-Digit Subtraction": 0.0010910430024330385,
                "Single-Digit Addition": 0.017611269171381862,
                "Number Comparison": 0.007513728781010536
            },
            "task_importance": {
                "NC": 0.02,
                "DM": 0.07,
                "ADD": 0.04,
                "SUB": 0.01,
                "NP": 0.1,
                "SN": 0.23,
                "AF": 0.02,
                "BC": 0.23,
                "AS": 0.27,
                "PF": 0.02
            }
        },
        "test_id": "test_case_#7"
    },
    {
        "test_name": "Test Case #8: At-Risk Profile",
        "diagnostic_data": {
            "predicted_class": "At-Risk (1)",
            "confidence": 0.8,
            "decision_path": [
                [
                    "NC",
                    1508.9295,
                    "<="
                ],
                [
                    "ADD",
                    15.75,
                    ">"
                ],
                [
                    "DM",
                    3564.2007,
                    "<="
                ],
                [
                    "NC",
                    1498.5457,
                    "<="
                ],
                [
                    "ADD",
                    29.5,
                    "<="
                ],
                [
                    "NS",
                    6.5,
                    ">"
                ],
                [
                    "SUB",
                    40.5,
                    "<="
                ],
                [
                    "DM",
                    1909.336,
                    ">"
                ]
            ],
            "domain_severity": {
                "Number Series": 0.0077676231762695645,
                "Addition vs. Subtraction Asymmetry": 0.029781937473266188,
                "Overall Arithmetic Fluency": 0.1284085704913605,
                "Multi-Digit Addition and Subtraction": 0.0,
                "Digit-Dot Matching": 0.03211595797114623,
                "Symbolic vs. Non-Symbolic Processing Difference": 0.20876574251616595,
                "Basic vs. Complex Arithmetic Contrast": 0.11072009242816631,
                "Overall Processing Efficiency": 0.2680830038802156,
                "Processing-Fluency Integration": 0.16214549899344982,
                "Single-Digit Subtraction": 0.0053501244226445265,
                "Single-Digit Addition": 0.01999532579819623,
                "Number Comparison": 0.02686612284911912
            },
            "task_importance": {
                "NC": 0.07,
                "DM": 0.07,
                "NS": 0.01,
                "ADD": 0.04,
                "SUB": 0.01,
                "NP": 0.24,
                "SN": 0.18,
                "AF": 0.11,
                "BC": 0.1,
                "AS": 0.03,
                "PF": 0.14
            }
        },
        "test_id": "test_case_#8"
    },
    {
        "test_name": "Test Case #9: Typical Profile",
        "diagnostic_data": {
            "predicted_class": "Typical (0)",
            "confidence": 0.8889,
            "decision_path": [
                [
                    "NC",
                    1508.9295,
                    "<="
                ],
                [
                    "ADD",
                    15.75,
                    ">"
                ],
                [
                    "DM",
                    3564.2007,
                    "<="
                ],
                [
                    "NC",
                    1498.5457,
                    ">"
                ]
            ],
            "domain_severity": {
                "Number Series": 0.0,
                "Addition vs. Subtraction Asymmetry": 0.10395392029333432,
                "Overall Arithmetic Fluency": 0.12209743566943619,
                "Multi-Digit Addition and Subtraction": 0.0,
                "Digit-Dot Matching": 0.0030171837242352175,
                "Symbolic vs. Non-Symbolic Processing Difference": 0.0528125871713574,
                "Basic vs. Complex Arithmetic Contrast": 0.21274407194947653,
                "Overall Processing Efficiency": 0.24047724782816923,
                "Processing-Fluency Integration": 0.21246260323810784,
                "Single-Digit Subtraction": 0.0,
                "Single-Digit Addition": 0.001988972319001752,
                "Number Comparison": 0.05044597780688156
            },
            "task_importance": {
                "NC": 0.14,
                "DM": 0.02,
                "ADD": 0.01,
                "NP": 0.21,
                "SN": 0.05,
                "AF": 0.11,
                "BC": 0.19,
                "AS": 0.09,
                "PF": 0.19
            }
        },
        "test_id": "test_case_#9"
    },
    {
        "test_name": "Test Case #10: Typical Profile",
        "diagnostic_data": {
            "predicted_class": "Typical (0)",
            "confidence": 0.6882,
            "decision_path": [
                [
                    "NC",
                    1508.9295,
                    "<="
                ],
                [
                    "ADD",
                    15.75,
                    ">"
                ],
                [
                    "DM",
                    3564.2007,
                    "<="
                ],
                [
                    "NC",
                    1498.5457,
                    "<="
                ],
                [
                    "ADD",
                    29.5,
                    ">"
                ],
                [
                    "ADD",
                    63.0,
                    "<="
                ],
                [
                    "DM",
                    1056.7697,
                    ">"
                ],
                [
                    "DM",
                    1142.8244,
                    ">"
                ],
                [
                    "SUB",
                    61.0,
                    "<="
                ],
                [
                    "ADD",
                    32.5,
                    ">"
                ]
            ],
            "domain_severity": {
                "Number Series": 0.0,
                "Addition vs. Subtraction Asymmetry": 0.192636945982545,
                "Overall Arithmetic Fluency": 0.15599732896697188,
                "Multi-Digit Addition and Subtraction": 0.0,
                "Digit-Dot Matching": 0.011963982287358196,
                "Symbolic vs. Non-Symbolic Processing Difference": 0.19634474512439462,
                "Basic vs. Complex Arithmetic Contrast": 0.044181922837824024,
                "Overall Processing Efficiency": 0.1734225210067074,
                "Processing-Fluency Integration": 0.1757903802790348,
                "Single-Digit Subtraction": 0.0006686143215775771,
                "Single-Digit Addition": 0.038340338269412896,
                "Number Comparison": 0.010653220924173508
            },
            "task_importance": {
                "NC": 0.03,
                "DM": 0.08,
                "ADD": 0.08,
                "SUB": 0.0,
                "NP": 0.15,
                "SN": 0.17,
                "AF": 0.13,
                "BC": 0.04,
                "AS": 0.17,
                "PF": 0.15
            }
        },
        "test_id": "test_case_#10"
    }
]

def run_tests():
    print("[SYSTEM] Commencing Full Unseen Data Test Suite (10 Cases)...\n")
    
    for idx, test in enumerate(test_cases):
        print(f"--- EXECUTING {test['test_name']} ---")
        
        try:
            payload = {
                "test_id": test.get("test_id", "test_" + str(idx)),
                "diagnostic_data": test["diagnostic_data"]
            }
            response = requests.post(FLASK_API_URL, json=payload, timeout=900)
            
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
        
        # 30-second throttle to avoid OpenRouter / Model limits
        if idx < len(test_cases) - 1:
            print("[SYSTEM] Throttling for 30 seconds to give the server a breather...\n")
            time.sleep(30)

    print("[SYSTEM] All 10 Automated Tests Concluded.")

if __name__ == "__main__":
    run_tests()
