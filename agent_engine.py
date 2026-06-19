import os
import json
from typing import Dict, Any, List
from data_ingestion import query_vector_index

# Tool Definition for Agent
def query_visa_rules(query: str) -> str:
    """
    Search the university admission and visa rules database. 
    Use this tool to find GPA thresholds, IELTS requirements, backlog limits, and financial guidelines for specific regions (UK, Canada, USA, Australia).
    """
    print(f"  [Tool Execution] Searching Vector Database for: '{query}'")
    results = query_vector_index(query, k=3)
    if not results:
        return "No matching guidelines found."
    
    # Concatenate the retrieved text chunks
    context = "\n".join([f"- {item['chunk_text']} (Similarity: {item['similarity']:.2f})" for item in results])
    return context

# Fallback heuristic engine (uses retrieved text rules to score via Python)
def run_mock_agent_evaluation(profile: Dict[str, Any], context: str) -> Dict[str, Any]:
    # Simple rule matching based on retrieved context strings
    lower_context = context.lower()
    score = 100
    risks = []
    
    # Simple GPA percentage parsing
    gpa = profile["gpa"]
    scale = profile["gpa_scale"]
    pct = gpa
    if scale == "4":
        pct = (gpa / 4.0) * 100
    elif scale == "10":
        pct = gpa * 9.5
        
    # Check if context contains specific rules and evaluate against them
    dest = profile["destination"].upper()
    
    # Academic check
    if dest == "UK" and pct < 55:
        score -= 25
        risks.append("CGPA falls below the UK direct entry threshold (55%).")
    elif dest == "CANADA" and pct < 60:
        score -= 25
        risks.append("CGPA falls below Canada SDS academic threshold (60%).")
    elif dest == "USA" and pct < 70:
        score -= 25
        risks.append("CGPA is below USA MS direct entry benchmark (70%).")
    elif dest == "AUSTRALIA" and pct < 55:
        score -= 25
        risks.append("CGPA is below Australia direct entry benchmark (55%).")

    # English check
    eng_score = profile["english_score"]
    if profile["english_test"] == "IELTS":
        if dest == "CANADA" and eng_score < 6.0:
            score -= 30
            risks.append("IELTS score is below Canada SDS threshold (6.0 in all bands).")
        elif eng_score < 6.0:
            score -= 20
            risks.append(f"IELTS score is below standard {dest} university requirement (6.0).")
    elif profile["english_test"] == "None":
        score -= 25
        risks.append(f"No English test provided. Standard {dest} visa requires English proficiency proof.")

    # Financial check
    budget = profile["budget_lakhs"]
    if dest == "UK" and budget < 15:
        score -= 20
        risks.append("Annual budget (₹15 Lakhs) is below UK living and tuition guidelines.")
    elif dest == "CANADA" and budget < 18:
        score -= 20
        risks.append("Budget is below Canadian GIC ($20,635 CAD) + tuition requirements.")
    elif dest == "AUSTRALIA" and budget < 22:
        score -= 20
        risks.append("Budget is below Australia Genuine Student financial guidelines.")
    elif dest == "USA" and budget < 25:
        score -= 20
        risks.append("Budget is below standard US cost of attendance guidelines.")

    # Gaps check
    gaps = profile["gap_years"]
    if gaps > 2 and profile["work_exp_years"] == 0:
        score -= 20
        risks.append(f"Unjustified study gap of {gaps} years. High risk of visa rejection.")

    status = "Green"
    if score < 50:
        status = "Red"
    elif score < 80:
        status = "Yellow"

    # Mock universities matching the destination
    unis = [{"name": f"Mock University of {profile['destination']}", "course": "MS International Business", "rationale": "Matches credentials."}]

    return {
        "status": status,
        "score": max(10, score),
        "academic_analysis": f"GPA: {gpa}/{scale} ({pct:.1f}%). Evaluated against context rules: " + ("Meets academic criteria." if score > 70 else "Fails to satisfy key criteria."),
        "english_analysis": f"English Test: {profile['english_test']} ({eng_score}). Checked against retrieved guidelines.",
        "financial_analysis": f"Budget: ₹{budget} Lakhs/year. Checked against destination financial requirements.",
        "gap_analysis": f"Study Gaps: {gaps} years. Work Experience: {profile['work_exp_years']} years.",
        "recommendations": risks if risks else ["All criteria met. Proceed with application document collection."],
        "matched_universities": unis,
        "is_mock": True,
        "retrieved_context": context
    }

# Main Agent execution loop
def evaluate_student_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # 1. Build a target query for the vector database
    search_query = f"{profile['destination']} admissions requirements GPA IELTS living expenses financial guidelines"
    
    if not api_key:
        # Fallback to local RAG Mock Mode
        print("  [Agent Engine] GEMINI_API_KEY not found. Running local RAG Mock Engine...")
        context = query_visa_rules(search_query)
        return run_mock_agent_evaluation(profile, context)
        
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=api_key)
        
        # 2. Retrieve guidelines using the search tool first
        context = query_visa_rules(search_query)
        
        # 3. Model call with instructions to analyze profile using the retrieved context
        system_instruction = """
        You are 'ScholarScan Agent', an elite admissions and visa compliance officer.
        Your task is to analyze the student's profile against the retrieved university and visa guidelines.
        
        Rules:
        1. Rely ONLY on the facts provided in the 'Retrieved Guidelines' section. Do not hallucinate or make up rules.
        2. Evaluate all 4 areas: Academic Standing, Language Fit, Financial Feasibility, and Study Gaps/Continuity.
        3. Assign an overall status: 'Green' (Low risk), 'Yellow' (Medium risk/Conditional), or 'Red' (High risk/Ineligible).
        4. Match 1-2 realistic universities.
        """
        
        prompt = f"""
        Student Profile:
        - Target Destination: {profile['destination']}
        - GPA: {profile['gpa']} on scale {profile['gpa_scale']}
        - English Test: {profile['english_test']} - Score: {profile['english_score']}
        - Annual Budget: {profile['budget_lakhs']} Lakhs INR
        - Study Gaps: {profile['gap_years']} years
        - Work Experience: {profile['work_exp_years']} years
        - Active or History Backlogs: {profile['backlogs']}
        
        Retrieved Guidelines from Vector Store:
        {context}
        
        Return your response in strict JSON format matching this schema:
        {{
          "status": "Green" | "Yellow" | "Red",
          "score": integer (out of 100),
          "academic_analysis": "string summarizing academic fit",
          "english_analysis": "string summarizing language fit",
          "financial_analysis": "string detailing budget viability against living/tuition rule",
          "gap_analysis": "string evaluating gaps and backlogs",
          "recommendations": ["list", "of", "actionable", "tips", "or", "warnings"],
          "matched_universities": [
            {{
              "name": "University Name",
              "course": "Suggested Course",
              "rationale": "Why they fit"
            }}
          ]
        }}
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json"
            )
        )
        
        result_text = response.text.strip()
        evaluation_data = json.loads(result_text)
        evaluation_data["is_mock"] = False
        evaluation_data["retrieved_context"] = context
        return evaluation_data
        
    except Exception as e:
        print(f"  [Agent Engine] Error in Gemini Agent loop: {e}. Falling back to RAG Mock Mode.")
        context = query_visa_rules(search_query)
        res = run_mock_agent_evaluation(profile, context)
        res["api_error"] = str(e)
        return res
