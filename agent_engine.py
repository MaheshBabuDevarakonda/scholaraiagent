import os
import json
from typing import Dict, Any, List
from data_ingestion import query_vector_index

# Tool Definition for Agent
def query_visa_rules(query: str) -> str:
    """
    Search the immigration and visa rules database. 
    Use this tool to find age limits, work experience requirements, English thresholds, and financial guidelines for specific regions (UK, Canada, USA, Australia) and purposes (Study, Work, Tourist, PR).
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
    dest = profile["destination"].upper()
    purpose = profile["purpose"].upper()
    age = profile["age"]
    education = profile["education"]
    work_exp = profile["work_exp_years"]
    eng_test = profile["english_test"]
    eng_score = profile["english_score"]
    funds = profile["funds_lakhs"]
    citizenship = profile["citizenship"]
    
    score = 100
    risks = []
    
    age_analysis = f"Age is {age} years. "
    edu_analysis = f"Education level: {education}. "
    exp_analysis = f"Work experience: {work_exp} years. "
    eng_analysis = f"English Test: {eng_test} (Score: {eng_score}). "
    fin_analysis = f"Available funds: ₹{funds} Lakhs. "
    
    # 1. Evaluate STUDY Visa
    if purpose == "STUDY":
        # Age check
        if age < 18:
            age_analysis += "Minor applicant. Requires parental consent and guardianship proof."
        elif age > 35:
            score -= 20
            risks.append("Age is above 35. High study gap scrutiny applies for student visa.")
            age_analysis += "Mature student. Increased risk of study-gap visa rejection."
        else:
            age_analysis += "Within standard age range for international studies."
            
        # English check
        if eng_test == "None":
            score -= 25
            risks.append("No English test provided. SDS/direct entry student visas require IELTS/PTE scores.")
            eng_analysis += "Lacks language proficiency test."
        elif eng_test == "IELTS" and eng_score < 6.0:
            score -= 20
            risks.append("IELTS score is below standard direct entry student visa threshold (6.0).")
            eng_analysis += "Below standard direct entry IELTS requirement."
        else:
            eng_analysis += "Meets language requirements for student visa."
            
        # Funds check
        req_funds = {"UK": 15, "CANADA": 18, "USA": 25, "AUSTRALIA": 22}.get(dest, 20)
        if funds < req_funds:
            score -= 25
            risks.append(f"Funds (₹{funds} Lakhs) are below the recommended threshold (₹{req_funds} Lakhs) for first-year tuition + living costs.")
            fin_analysis += f"Insufficient proof of funds. Recommended: ₹{req_funds} Lakhs."
        else:
            fin_analysis += "Sufficient funds to cover first-year cost of attendance."
            
        # Education check
        if education in ["High School", "Bachelor's", "Master's", "PhD"]:
            edu_analysis += "Academic credentials satisfy entry levels."
            
        visa_options = [
            {"name": "UK Student Route", "status": "Eligible" if score >= 80 else "Potential" if score >= 50 else "Ineligible", "rationale": "Requires Confirmation of Acceptance for Studies (CAS) and 28-day funds rule."},
            {"name": "Canada Study Permit (SDS)", "status": "Eligible" if score >= 85 and eng_score >= 6.0 else "Ineligible", "rationale": "Requires GIC $20,635 CAD payment and IELTS 6.0 in all bands."},
            {"name": "US F-1 Student Visa", "status": "Eligible" if score >= 75 else "Potential" if score >= 50 else "Ineligible", "rationale": "Requires I-20 form and demonstration of strong ties to home country during F-1 interview."},
            {"name": "Australia Subclass 500", "status": "Eligible" if score >= 80 else "Potential" if score >= 50 else "Ineligible", "rationale": "Evaluated under the Genuine Student (GS) requirement."}
        ]

    # 2. Evaluate WORK Visa
    elif purpose == "WORK":
        # Experience check
        if work_exp < 2:
            score -= 30
            risks.append("Work experience is under 2 years. Most skilled worker visa routes require at least 2 years of relevant experience.")
            exp_analysis += "Insufficient professional experience for skilled visa routes."
        else:
            exp_analysis += "Meets basic work experience requirements."
            
        # Age check
        if age < 18:
            score -= 50
            risks.append("Applicant is under 18. Ineligible for standard work visas.")
            age_analysis += "Underage. Not eligible for work visa."
        elif age > 45:
            score -= 15
            risks.append("Age is above 45. Skilled visas face reduced point allocations or eligibility caps in Australia/Canada.")
            age_analysis += "Age is high for points-tested work visas."
        else:
            age_analysis += "Within prime working age range."
            
        # English check
        if eng_test == "None":
            score -= 20
            risks.append("No English test provided. Skilled worker routes in UK/Australia mandate English scores.")
            eng_analysis += "Missing language proficiency test."
        elif eng_test == "IELTS" and eng_score < 5.0:
            score -= 15
            risks.append("IELTS score is below standard work visa threshold (5.0).")
            eng_analysis += "Below minimum language score."
        else:
            eng_analysis += "Satisfies basic language requirements."
            
        # Funds check
        if funds < 5:
            score -= 15
            risks.append("Settlement funds are tight. Standard work pathways recommend at least ₹5 Lakhs for initial relocation costs.")
            fin_analysis += "Relocation funds are below recommendation (₹5 Lakhs)."
        else:
            fin_analysis += "Sufficient settlement funds."
            
        # Education check
        if education == "High School":
            score -= 15
            risks.append("Highest education is High School. Most skilled work visas require a university degree or equivalent trade diploma.")
            edu_analysis += "Academics below standard professional visa thresholds."
        else:
            edu_analysis += "Degree satisfies skilled worker eligibility benchmarks."
            
        visa_options = [
            {"name": "UK Skilled Worker Visa", "status": "Eligible" if score >= 75 else "Potential" if score >= 50 else "Ineligible", "rationale": "Requires job offer and Certificate of Sponsorship from a licensed UK employer."},
            {"name": "Canada LMIA Work Permit", "status": "Eligible" if score >= 70 else "Potential" if score >= 50 else "Ineligible", "rationale": "Requires employer to obtain a positive Labour Market Impact Assessment."},
            {"name": "US H-1B Specialty Occupation", "status": "Eligible" if score >= 80 else "Potential" if score >= 50 else "Ineligible", "rationale": "Requires a Bachelor's degree and employer sponsorship under the annual lottery."},
            {"name": "Australia Subclass 482 (TSS)", "status": "Eligible" if score >= 75 else "Potential" if score >= 50 else "Ineligible", "rationale": "Requires nomination by approved sponsor and 2 years relevant experience."}
        ]

    # 3. Evaluate TOURIST Visa
    elif purpose == "TOURIST":
        # Funds check (critical for tourists)
        if funds < 3:
            score -= 30
            risks.append("Liquid funds are under ₹3 Lakhs. High risk of refusal due to insufficient financial capacity.")
            fin_analysis += "Insufficient vacation/visit funds."
        elif funds < 5:
            score -= 10
            fin_analysis += "Moderate funds. Sufficient for short stays."
        else:
            fin_analysis += "Strong financial capacity for vacation/visits."
            
        # Age check
        age_analysis += "No specific age restrictions for tourist visa applications."
        
        # Employment / Ties check
        if work_exp == 0 and education == "High School":
            score -= 20
            risks.append("Unemployed with no higher education. High risk of visa refusal due to weak ties to home country.")
            exp_analysis += "Lack of professional ties increases immigration intent concerns."
        else:
            exp_analysis += "Professional profile demonstrates ties to home country."
            
        # English check
        eng_analysis += "English language testing is not officially required for tourist visas."
        
        # Education check
        edu_analysis += "Education does not impact tourist visa assessment."
        
        visa_options = [
            {"name": "UK Standard Visitor Visa", "status": "Eligible" if score >= 70 else "Potential" if score >= 45 else "Ineligible", "rationale": "Must demonstrate genuine intention to visit and leave the UK."},
            {"name": "Canada Visitor Visa (TRV)", "status": "Eligible" if score >= 70 else "Potential" if score >= 45 else "Ineligible", "rationale": "Requires proof of family ties, assets, and flight/hotel bookings."},
            {"name": "US B-2 Tourist Visa", "status": "Eligible" if score >= 75 else "Potential" if score >= 50 else "Ineligible", "rationale": "Evaluated in person. Presumption of immigrant intent must be refuted."},
            {"name": "Australia Visitor Subclass 600", "status": "Eligible" if score >= 70 else "Potential" if score >= 45 else "Ineligible", "rationale": "Assessed under standard Genuine Temporary Entrant conditions."}
        ]

    # 4. Evaluate PR Visa
    else:  # PR
        # Age check (Points-based)
        if age < 18:
            score -= 50
            risks.append("Applicant is under 18. Ineligible for independent permanent residency.")
            age_analysis += "Underage. Ineligible for PR."
        elif age > 45:
            score -= 35
            risks.append("Age is above 45. Points-based PR systems (Canada CRS, Australia Points) allocate 0 points for age after 45.")
            age_analysis += "Age is beyond points eligibility thresholds."
        elif age >= 25 and age <= 32:
            age_analysis += "Optimal age range. Receives maximum points in selection pools."
        else:
            score -= 10
            age_analysis += "Sub-optimal age points range."
            
        # Experience check
        if work_exp < 3:
            score -= 25
            risks.append("Work experience is under 3 years. Points-based PR highly rewards 3+ years of professional experience.")
            exp_analysis += "Limited professional experience points."
        else:
            exp_analysis += "Solid professional experience points."
            
        # English check (Critical points factor)
        if eng_test == "None":
            score -= 30
            risks.append("No English test provided. Language test is mandatory to log PR expressions of interest.")
            eng_analysis += "Lacks mandatory language score."
        elif eng_test == "IELTS" and eng_score < 7.0:
            score -= 20
            risks.append("IELTS score is below 7.0. A high English score (IELTS 7.0/8.0 equivalent) is vital to be competitive in PR pools.")
            eng_analysis += "Below competitive language thresholds."
        else:
            eng_analysis += "Excellent language scores. Attracts maximum PR points."
            
        # Education check
        if education in ["High School", "Bachelor's"]:
            score -= 15
            risks.append("Education is below Master's. Having a Master's or PhD significantly boosts PR competitiveness.")
            edu_analysis += "Lacks postgraduate points boost."
        else:
            edu_analysis += "Postgraduate credentials boost points standing."
            
        # Funds check
        if funds < 10:
            score -= 20
            risks.append("Settlement funds (₹10 Lakhs) are below typical PR requirements (approx. $14,000 CAD).")
            fin_analysis += "Settlement funds are below recommended threshold."
        else:
            fin_analysis += "Meets minimum settlement fund guidelines."
            
        visa_options = [
            {"name": "Canada Express Entry (FSW)", "status": "Eligible" if score >= 80 else "Potential" if score >= 60 else "Ineligible", "rationale": "Points-based selection based on age, education, and language skills."},
            {"name": "Australia Subclass 189/190", "status": "Eligible" if score >= 75 else "Potential" if score >= 60 else "Ineligible", "rationale": "Requires at least 65 points in points-test and positive skills assessment."},
            {"name": "US EB-2/EB-3 Green Card", "status": "Eligible" if score >= 80 else "Potential" if score >= 55 else "Ineligible", "rationale": "Requires employer sponsorship and labor certification (PERM)."},
            {"name": "UK Indefinite Leave (ILR)", "status": "Eligible" if score >= 70 and work_exp >= 5 else "Ineligible", "rationale": "Requires 5 years continuous residency on an eligible work visa."}
        ]
        
    status = "Green"
    if score < 50:
        status = "Red"
    elif score < 80:
        status = "Yellow"
        
    # Match the specific destination's option as the primary recommendation
    matched_option = next((v for v in visa_options if dest in v["name"].upper()), None)
    if not matched_option:
        matched_option = visa_options[0]
        
    # Format options for output
    options_list = [matched_option]
    for opt in visa_options:
        if opt != matched_option:
            options_list.append(opt)
            
    return {
        "status": status,
        "score": max(10, score),
        "eligibility_analysis": f"Evaluated eligibility for {purpose.capitalize()} visa to {dest.capitalize()}. Overall score is {score}/100.",
        "age_analysis": age_analysis,
        "education_analysis": edu_analysis,
        "experience_analysis": exp_analysis,
        "english_analysis": eng_analysis,
        "financial_analysis": fin_analysis,
        "recommendations": risks if risks else ["All criteria met. Proceed with visa application planning."],
        "visa_options": options_list,
        "is_mock": True,
        "retrieved_context": context
    }

# Main Agent execution loop
def evaluate_visa_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # 1. Build a target query for the vector database
    search_query = f"{profile['destination']} {profile['purpose']} visa requirements age education work experience financial guidelines"
    
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
        You are 'VisaEligibility Agent', an elite immigration compliance officer.
        Your task is to analyze the user's profile against the retrieved visa guidelines.
        
        Rules:
        1. Rely ONLY on the facts provided in the 'Retrieved Guidelines' section. Do not hallucinate or make up rules.
        2. Evaluate all categories: Age limits, Educational standing, Professional Work Experience, English proficiency, and Financial settlement capacity.
        3. Assign an overall status: 'Green' (Low risk / Highly Eligible), 'Yellow' (Medium risk / Conditional / Caution), or 'Red' (High risk / Ineligible).
        4. Recommend and evaluate at least 3-4 specific visa pathways (e.g., Canadian Express Entry, UK Skilled Worker Visa, Australian Subclass 189) appropriate for their destination and citizenship. Include the pathway name, eligibility status ('Eligible', 'Potential', 'Ineligible'), and the detailed rationale.
        """
        
        prompt = f"""
        User Profile:
        - Citizenship: {profile['citizenship']}
        - Target Destination: {profile['destination']}
        - Age: {profile['age']} years
        - Highest Education: {profile['education']}
        - Work Experience: {profile['work_exp_years']} years
        - English Test: {profile['english_test']} - Score: {profile['english_score']}
        - Available Funds/Income: {profile['funds_lakhs']} Lakhs INR
        - Purpose of Travel: {profile['purpose']}
        
        Retrieved Guidelines from Vector Store:
        {context}
        
        Return your response in strict JSON format matching this schema:
        {{
          "status": "Green" | "Yellow" | "Red",
          "score": integer (out of 100),
          "eligibility_analysis": "string summarizing overall visa eligibility fit",
          "age_analysis": "string evaluating age requirements",
          "education_analysis": "string evaluating academic requirements",
          "experience_analysis": "string evaluating work experience requirements",
          "english_analysis": "string evaluating language requirements",
          "financial_analysis": "string evaluating funds requirements",
          "recommendations": ["list", "of", "actionable", "tips", "or", "warnings"],
          "visa_options": [
            {{
              "name": "Visa Subclass/Route Name",
              "status": "Eligible" | "Potential" | "Ineligible",
              "rationale": "Detailed reason why they qualify or what is missing"
            }},
            ... (provide at least 3-4 visa route matches)
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

