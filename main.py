import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Initialize FastAPI App
app = FastAPI(
    title="ScholarScan AI",
    description="Intelligent Student Eligibility & Course Recommender Agent",
    version="1.0.0"
)

# Pydantic models for request and response validation
class StudentProfile(BaseModel):
    destination: str = Field(..., description="Target destination country (UK, Canada, USA, Australia)")
    gpa: float = Field(..., description="Student GPA/CGPA/Percentage score")
    gpa_scale: str = Field(..., description="GPA Scale: 4, 10, or 100 (percentage)")
    english_test: str = Field(..., description="English test type (IELTS, PTE, TOEFL, None)")
    english_score: float = Field(..., description="Overall English test score")
    budget_lakhs: float = Field(..., description="Annual tuition & living budget in Lakhs INR")
    gap_years: int = Field(..., description="Number of gap years since last study")
    work_exp_years: int = Field(..., description="Years of work experience")
    backlogs: int = Field(..., description="Number of active or cleared backlogs")

# Heuristic-based mock evaluation for offline testing
def get_mock_evaluation(profile: StudentProfile) -> Dict[str, Any]:
    # Calculate percentage for validation
    pct = profile.gpa
    if profile.gpa_scale == "4":
        pct = (profile.gpa / 4.0) * 100
    elif profile.gpa_scale == "10":
        pct = profile.gpa * 9.5  # Standard CBSE conversion

    # Heuristic scoring
    score = 100
    risks = []
    
    # Academic check
    if pct < 50:
        score -= 30
        risks.append("Low academic percentage (below 50%). Most universities require at least 55% for direct entry.")
    elif pct < 60:
        score -= 10
        risks.append("Academic score is between 50%-60%. May face restrictions or require a foundation program.")

    # English proficiency check
    if profile.english_test != "None":
        if profile.english_test == "IELTS" and profile.english_score < 6.0:
            score -= 15
            risks.append("IELTS score is below 6.0. Most destinations require a minimum of 6.0 overall with no band less than 5.5.")
        elif profile.english_test == "PTE" and profile.english_score < 50:
            score -= 15
            risks.append("PTE score is below 50. Equivalent English test score is low for direct degree entry.")
    else:
        score -= 25
        risks.append("No English proficiency test score provided. An English test is highly recommended for visa and admissions.")

    # Gap years check
    if profile.gap_years > 2 and profile.work_exp_years == 0:
        score -= 20
        risks.append(f"Significant study gap of {profile.gap_years} years with no documented work experience. Hard to justify for visa.")
    elif profile.gap_years > 5:
        score -= 10
        risks.append(f"Long study gap of {profile.gap_years} years. Will require strong work experience reference letters and tax documents.")

    # Budget check
    min_budget = {"UK": 15, "Canada": 18, "USA": 25, "Australia": 22}
    req_budget = min_budget.get(profile.destination, 15)
    if profile.budget_lakhs < req_budget:
        score -= 20
        risks.append(f"Annual budget (₹{profile.budget_lakhs} Lakhs) is below the recommended minimum for {profile.destination} (₹{req_budget} Lakhs/year including living expenses).")

    # Backlogs check
    if profile.backlogs > 5:
        score -= 15
        risks.append(f"High number of backlogs ({profile.backlogs}). Top-tier universities may reject the application.")

    # Final eligibility status
    status = "Green"
    if score < 50:
        status = "Red"
    elif score < 80:
        status = "Yellow"

    # Match mock universities
    unis = []
    if profile.destination == "UK":
        if status == "Green":
            unis = [
                {"name": "University of Greenwich", "course": "MSc Management", "rationale": "High acceptance rate, fits budget, strong student support."},
                {"name": "Coventry University", "course": "MSc Data Science", "rationale": "Excellent practical modules, matches GPA and budget."}
            ]
        elif status == "Yellow":
            unis = [
                {"name": "University of Hertfordshire", "course": "MSc Software Engineering", "rationale": "Accepts moderate CGPA/English scores, good location."},
                {"name": "BPP University", "course": "MSc Management with Project Management", "rationale": "Highly flexible entry criteria, fits budget."}
            ]
        else:
            unis = [
                {"name": "Study Group Pathway College", "course": "Pre-Masters Pathway", "rationale": "Required to bridge academic or gap requirements."}
            ]
    elif profile.destination == "Canada":
        if status == "Green":
            unis = [
                {"name": "Conestoga College", "course": "Post-Graduate Certificate in Mobile Solutions", "rationale": "Strong co-op options, fits budget."},
                {"name": "Seneca College", "course": "PG Diploma in Business Analytics", "rationale": "Located in Toronto, great post-study work prospects."}
            ]
        else:
            unis = [
                {"name": "Northern College", "course": "PG Diploma in Computer Applications", "rationale": "Accepts moderate academic grades and gap profiles."}
            ]
    else: # Default/USA/Australia
        unis = [
            {"name": f"State University of {profile.destination}", "course": "MS Information Technology", "rationale": "Appropriate for budget and academic standing."}
        ]

    # Generate write-ups
    academic_msg = f"Academic score evaluated at {pct:.1f}% equivalent. " + ("Meets direct entry thresholds for most universities." if pct >= 60 else "Requires careful selection of university partners.")
    english_msg = f"English level: {profile.english_test} {profile.english_score}. " + ("Satisfies direct admission requirements." if profile.english_score >= 6.0 or profile.english_score >= 55 else "Additional pre-sessional English courses may be required.")
    fin_msg = f"Annual budget of ₹{profile.budget_lakhs} Lakhs. " + ("Sufficient to cover tuition fees and basic living costs." if profile.budget_lakhs >= req_budget else f"Funding is tight. Recommend looking for scholarships or education loans of at least ₹{req_budget - profile.budget_lakhs} Lakhs.")
    gap_msg = f"Study gap of {profile.gap_years} years. " + ("No gap issues identified." if profile.gap_years <= 2 else "Ensure salary slips and experience letters are fully prepared to prove continuity.")

    return {
        "status": status,
        "score": max(10, score),
        "academic_analysis": academic_msg,
        "english_analysis": english_msg,
        "financial_analysis": fin_msg,
        "gap_analysis": gap_msg,
        "recommendations": risks if risks else ["Profile looks strong. Keep all transcripts and academic references ready for application."],
        "matched_universities": unis,
        "is_mock": True
    }

@app.post("/api/evaluate")
async def evaluate_student(profile: StudentProfile):
    # Check if Gemini API key is configured
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        # Fallback to local rule engine (Mock Mode)
        return get_mock_evaluation(profile)
    
    try:
        from google import genai
        from google.genai import types

        # Initialize the new Google GenAI client
        client = genai.Client(api_key=api_key)
        
        # Build prompt
        prompt = f"""
        You are 'ScholarScan Evaluator', an expert AI admission officer and visa compliance auditor.
        Analyze the following student profile for studying in {profile.destination}:
        
        - Academic Grade: {profile.gpa} on a scale of {profile.gpa_scale}
        - English Proficiency: {profile.english_test} - Score: {profile.english_score}
        - Annual Budget: {profile.budget_lakhs} Lakhs INR
        - Study Gaps: {profile.gap_years} years
        - Work Experience: {profile.work_exp_years} years
        - Number of Backlogs: {profile.backlogs}

        Provide a structured evaluation in JSON format. The response must follow this schema exactly, and contain NO formatting or extra text outside the JSON:
        {{
          "status": "Green" (for low risk, high eligibility), "Yellow" (for medium risk, conditional eligibility), or "Red" (for high risk, ineligible),
          "score": integer (out of 100),
          "academic_analysis": "string describing academic eligibility for the destination",
          "english_analysis": "string describing language requirement suitability",
          "financial_analysis": "string describing financial readiness and visa fund rules",
          "gap_analysis": "string evaluating gap risks and work proof need",
          "recommendations": ["list", "of", "risk", "mitigation", "steps", "or", "next", "steps"],
          "matched_universities": [
            {{
              "name": "University Name",
              "course": "Suggested Course/Program Name",
              "rationale": "Specific explanation matching this student's profile"
            }}
          ]
        }}
        
        Rules:
        - Analyze the guidelines for {profile.destination} specifically (e.g. UK has strict CAS checks, Canada has SDS rules, Australia has Genuine Student requirement).
        - Match realistic universities for their grade and budget.
        - Output ONLY a valid JSON object. Do not include markdown code block syntax (like ```json).
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        result_text = response.text.strip()
        
        # Parse JSON to verify correctness
        evaluation_data = json.loads(result_text)
        evaluation_data["is_mock"] = False
        return evaluation_data
        
    except Exception as e:
        # Fallback if API call fails
        print(f"Error calling Gemini API: {str(e)}. Falling back to mock engine.")
        mock_res = get_mock_evaluation(profile)
        mock_res["api_error"] = str(e)
        return mock_res

# Serve index.html at root
@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

# Mount remaining static directory for style.css, app.js
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    # Print status message
    has_key = "GEMINI_API_KEY" in os.environ
    print("\n" + "="*50)
    print("ScholarScan AI - Server Starting")
    print(f"Gemini API Key: {'CONNECTED' if has_key else 'NOT DETECTED (Running in Mock Mode)'}")
    print("Open http://localhost:8000 in your browser.")
    print("="*50 + "\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
