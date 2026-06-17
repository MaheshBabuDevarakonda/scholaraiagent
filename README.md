# ScholarScan AI: Intelligent Student Eligibility & Course Recommender

ScholarScan AI is a lightweight, high-impact AI assistant designed for B2B/B2C international student recruitment. It automates the initial profile screening process for global study destinations (UK, Canada, USA, Australia) by analyzing academic credentials, English test scores, gap years, and budgets, generating a detailed visa and admission eligibility report.

---

## 🌟 Key Features

1. **Admissions Eligibility Check**: Validates student grades (supporting 10-point scales, 4-point scales, and percentages) against minimum university benchmarks.
2. **English Proficiency Evaluator**: Evaluates IELTS, PTE, or TOEFL scores for visa compliance and admission eligibility.
3. **Financial Feasibility Audit**: Compares student annual budgets against real-world living and tuition expectations for the target destination.
4. **Study Gap Assessment**: Flags potential visa refusal risks based on gap years and suggests necessary career documentation.
5. **Matched Universities**: Dynamically recommends target institutions and courses with a clear admission rationale.
6. **Dual Mode Architecture (Production Ready)**:
   - **AI Mode**: Uses `gemini-2.5-flash` to generate personalized, deep eligibility reports.
   - **Offline Mock Mode**: A built-in Python heuristics rule engine that runs automatically if no Gemini API Key is configured, enabling instant testing without setup.

---

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.12+)
- **WSGI/ASGI Server**: Uvicorn
- **AI Integration**: Google GenAI Python SDK (`google-genai`)
- **Frontend**: Vanilla HTML5, Custom CSS3 (featuring responsive dark-theme glassmorphism), and Vanilla JavaScript

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- An optional Gemini API Key (get one from Google AI Studio)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-github-repo-url>
   cd aiagent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API Key (Optional but Recommended)**:
   - On Windows (Command Prompt):
     ```cmd
     set GEMINI_API_KEY=your_actual_api_key_here
     ```
   - On Windows (PowerShell):
     ```powershell
     $env:GEMINI_API_KEY="your_actual_api_key_here"
     ```
   - On macOS/Linux:
     ```bash
     export GEMINI_API_KEY="your_actual_api_key_here"
     ```

4. **Run the server**:
   ```bash
   python main.py
   ```

5. **Open the App**:
   Navigate to `http://localhost:8000` in your web browser.

---

## 📐 Architecture & Workflow

1. The frontend collects student profile metrics and dispatches an asynchronous `POST` JSON payload to `/api/evaluate`.
2. The FastAPI backend validates fields, converts GPA scales, and checks if a `GEMINI_API_KEY` is present.
3. **If configured**, backend structures a prompt and queries `gemini-2.5-flash` using structured JSON output configurations.
4. **If offline**, backend runs a Python heuristics engine to compute an eligibility score (out of 100), assign risk colors (Green/Yellow/Red), and select mock universities.
5. The frontend handles CSS transitions, animates the progress score ring, and populates the dashboard reports.
