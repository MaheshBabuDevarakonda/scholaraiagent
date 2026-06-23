import os
import sys
import io
from agent_engine import evaluate_visa_profile

# Force console output to be UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ground-truth test dataset for regression checks
TEST_SUITE = [
    {
        "id": "TC_001",
        "description": "Canada Student Profile with IELTS 6.5 and ₹20L funds",
        "profile": {
            "citizenship": "India",
            "destination": "Canada",
            "age": 22,
            "education": "Bachelor's",
            "work_exp_years": 0,
            "english_test": "IELTS",
            "english_score": 6.5,
            "funds_lakhs": 20.0,
            "purpose": "Study"
        },
        "expected_status": "Eligible",
        "expected_retrieved_region": "CANADA"
    },
    {
        "id": "TC_002",
        "description": "Australia Work Profile with only 1 year experience (TSS requires 2)",
        "profile": {
            "citizenship": "India",
            "destination": "Australia",
            "age": 30,
            "education": "Bachelor's",
            "work_exp_years": 1,
            "english_test": "IELTS",
            "english_score": 5.5,
            "funds_lakhs": 6.0,
            "purpose": "Work"
        },
        "expected_status": "Not Possible",
        "expected_retrieved_region": "AUSTRALIA"
    },
    {
        "id": "TC_003",
        "description": "UK Tourist Profile with low funds (₹2L instead of ₹3-5L)",
        "profile": {
            "citizenship": "India",
            "destination": "UK",
            "age": 40,
            "education": "Master's",
            "work_exp_years": 10,
            "english_test": "None",
            "english_score": 0.0,
            "funds_lakhs": 2.0,
            "purpose": "Tourist"
        },
        "expected_status": "Not Possible",
        "expected_retrieved_region": "UNITED KINGDOM"
    },
    {
        "id": "TC_004",
        "description": "USA PR Profile with PhD and 5 years experience",
        "profile": {
            "citizenship": "India",
            "destination": "USA",
            "age": 28,
            "education": "PhD",
            "work_exp_years": 5,
            "english_test": "IELTS",
            "english_score": 8.0,
            "funds_lakhs": 15.0,
            "purpose": "PR"
        },
        "expected_status": "Eligible",
        "expected_retrieved_region": "UNITED STATES"
    }
]

def run_evaluation_suite():
    print("="*80)
    print("VISA ELIGIBILITY AGENT - EVALUATION HARNESS & REGRESSION TESTS")
    print(f"Test cases to run: {len(TEST_SUITE)}")
    print("="*80 + "\n")
    
    passed_predictions = 0
    passed_retrievals = 0
    
    results = []
    
    for case in TEST_SUITE:
        print(f"Running Case {case['id']}: {case['description']}...")
        profile = case["profile"]
        
        # Execute the agent loop
        prediction = evaluate_visa_profile(profile)
        
        # 1. Verify Status (Prediction Accuracy)
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
