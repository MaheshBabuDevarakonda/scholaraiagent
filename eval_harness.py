import os
import sys
import io
from agent_engine import evaluate_student_profile

# Force console output to be UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ground-truth test dataset for regression checks
TEST_SUITE = [
    {
        "id": "TC_001",
        "description": "Strong UK profile with good budget and no gaps",
        "profile": {
            "destination": "UK",
            "gpa": 8.5,
            "gpa_scale": "10",
            "english_test": "IELTS",
            "english_score": 6.5,
            "budget_lakhs": 25.0,
            "gap_years": 0,
            "work_exp_years": 0,
            "backlogs": 0
        },
        "expected_status": "Green",
        "expected_retrieved_region": "UNITED KINGDOM (UK)"
    },
    {
        "id": "TC_002",
        "description": "Weak Canadian profile with low GPA, low IELTS, and low budget",
        "profile": {
            "destination": "Canada",
            "gpa": 5.2,
            "gpa_scale": "10",
            "english_test": "IELTS",
            "english_score": 5.0,
            "budget_lakhs": 12.0,
            "gap_years": 0,
            "work_exp_years": 0,
            "backlogs": 2
        },
        "expected_status": "Red",
        "expected_retrieved_region": "CANADA"
    },
    {
        "id": "TC_003",
        "description": "Australian profile with standard academics but high study gap",
        "profile": {
            "destination": "Australia",
            "gpa": 7.2,
            "gpa_scale": "10",
            "english_test": "IELTS",
            "english_score": 6.0,
            "budget_lakhs": 15.0,
            "gap_years": 4,
            "work_exp_years": 0,
            "backlogs": 0
        },
        "expected_status": "Yellow",
        "expected_retrieved_region": "AUSTRALIA"
    },
    {
        "id": "TC_004",
        "description": "Strong US profile meeting all standard benchmarks",
        "profile": {
            "destination": "USA",
            "gpa": 3.6,
            "gpa_scale": "4",
            "english_test": "IELTS",
            "english_score": 7.0,
            "budget_lakhs": 42.0,
            "gap_years": 1,
            "work_exp_years": 1,
            "backlogs": 0
        },
        "expected_status": "Green",
        "expected_retrieved_region": "UNITED STATES (USA)"
    }
]

def run_evaluation_suite():
    print("="*80)
    print("SCHOLARSCAN AI - EVALUATION HARNESS & REGRESSION TESTS")
    print(f"Test cases to run: {len(TEST_SUITE)}")
    print("="*80 + "\n")
    
    passed_predictions = 0
    passed_retrievals = 0
    
    results = []
    
    for case in TEST_SUITE:
        print(f"Running Case {case['id']}: {case['description']}...")
        profile = case["profile"]
        
        # Execute the agent loop
        prediction = evaluate_student_profile(profile)
        
        # 1. Verify Bounding Bounding Box Status (Prediction Accuracy)
        predicted_status = prediction.get("status", "Unknown")
        status_correct = (predicted_status.lower() == case["expected_status"].lower())
        if status_correct:
            passed_predictions += 1
            
        # 2. Verify RAG Retrieval Quality (Check if target region rules are in the retrieved context)
        retrieved_context = prediction.get("retrieved_context", "").upper()
        expected_region = case["expected_retrieved_region"].upper()
        retrieval_correct = (expected_region in retrieved_context)
        if retrieval_correct:
            passed_retrievals += 1
            
        print(f"  -> Expected Status: {case['expected_status']} | Predicted: {predicted_status} [{'PASS' if status_correct else 'FAIL'}]")
        print(f"  -> Expected Context: '{case['expected_retrieved_region']}' in retrieved list? [{'PASS' if retrieval_correct else 'FAIL'}]")
        print(f"  -> Score: {prediction.get('score', 0)}/100")
        print(f"  -> Key Warnings: {prediction.get('recommendations', [])[:2]}")
        print("-" * 60 + "\n")
        
        results.append({
            "id": case["id"],
            "description": case["description"],
            "expected_status": case["expected_status"],
            "predicted_status": predicted_status,
            "status_pass": status_correct,
            "retrieval_pass": retrieval_correct
        })
        
    # Summarize Metrics
    total_cases = len(TEST_SUITE)
    accuracy = (passed_predictions / total_cases) * 100
    retrieval_rate = (passed_retrievals / total_cases) * 100
    
    print("="*80)
    print("EVALUATION HARNESS METRIC REPORT")
    print("="*80)
    print(f"Total Test Cases      : {total_cases}")
    print(f"Prediction Accuracy   : {accuracy:.1f}% ({passed_predictions}/{total_cases} passed)")
    print(f"Retrieval Success Rate: {retrieval_rate:.1f}% ({passed_retrievals}/{total_cases} passed)")
    print("="*80)
    
    if accuracy == 100.0 and retrieval_rate == 100.0:
        print("🎉 SUCCESS: All regression tests passed!")
    else:
        print("⚠️ WARNING: Some regression tests failed. Check prompts or vector similarities.")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_evaluation_suite()
