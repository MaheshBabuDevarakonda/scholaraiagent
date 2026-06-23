import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Initialize FastAPI App
app = FastAPI(
    title="Visa Eligibility Agent",
    description="Intelligent Visa Eligibility and Compliance Advisor Agent",
    version="1.0.0"
)

# Pydantic models for request and response validation
class UserProfile(BaseModel):
    citizenship: str = Field(..., description="Country of citizenship (e.g., India)")
    destination: str = Field(..., description="Target destination country (UK, Canada, USA, Australia)")
    age: int = Field(..., description="Applicant age in years")
    education: str = Field(..., description="Highest education level (High School, Bachelor's, Master's, PhD)")
    work_exp_years: int = Field(..., description="Years of relevant professional work experience")
    english_test: str = Field(..., description="English language proficiency test (IELTS, PTE, TOEFL, None)")
    english_score: float = Field(..., description="Overall test score")
    funds_lakhs: float = Field(..., description="Available income or funds in Lakhs INR")
    purpose: str = Field(..., description="Purpose of travel (Study, Work, Tourist, PR)")

from agent_engine import evaluate_visa_profile

@app.post("/api/evaluate")
async def evaluate_profile(profile: UserProfile):
    try:
        # Evaluate profile using the RAG agent
        result = evaluate_visa_profile(profile.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent evaluation failed: {str(e)}")

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
    print("Visa Eligibility Agent - Server Starting")
    print(f"Gemini API Key: {'CONNECTED' if has_key else 'NOT DETECTED (Running in Mock Mode)'}")
    print("Open http://localhost:8000 in your browser.")
    print("="*50 + "\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
