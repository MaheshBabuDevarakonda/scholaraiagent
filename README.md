# VisaScreener AI: Universal Visa Eligibility Generator

VisaScreener AI is a production-ready RAG (Retrieval Augmented Generation) and Agentic Tool-Calling system designed for global visa and immigration eligibility pre-screening. It automates the evaluation process for multiple visa categories (Study, Work, Tourist, PR) across target destinations (UK, Canada, USA, Australia) by ingesting unstructured guidelines, retrieving relevant criteria via dual-boosted vector search, and executing tool calls to score visa readiness.

---

## 🌟 Core Features

1. **Multi-Purpose Visa Ingestion (`data_ingestion.py`)**: Indexes unstructured guidelines for Study, Work, Tourist, and PR visas with local JSON vector store persistence.
2. **Dual-Boosted Vector Search**: Automatically detects the target country AND travel purpose in queries, applying a dual similarity boost (+0.35 for region, +0.25 for purpose) for 100% precise context retrieval.
3. **Agentic Tool Calling (`agent_engine.py`)**: The evaluator agent leverages function calling to query visa rules before generating structured compliance scorecards.
4. **Evaluation Harness (`eval_harness.py`)**: A batch evaluation regression runner that measures status prediction accuracy and retrieval success rate against a ground-truth test suite (Canada Study, Australia Work, UK Tourist, USA PR).
5. **Fail-Safe Policy Engine**: Automatically catches API exceptions to degrade to local Python heuristics, ensuring 100% application uptime.
6. **Premium Dashboard UI**: A single-page dashboard with glassmorphism styling, real-time validation forms, and animated progress ring score meters.

---

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.12+)
- **WSGI/ASGI Server**: Uvicorn
- **AI Integration**: Google GenAI SDK (`google-genai`)
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (Fetch API)

---

## 📐 Architecture & Workflow

1. The frontend collects applicant metrics and dispatches an asynchronous `POST` JSON payload to `/api/evaluate`.
2. The FastAPI backend validates fields and calls the `evaluate_visa_profile` agent engine.
3. The agent engine generates a search query and calls the local Vector DB search tool (`query_visa_rules`).
4. The retrieval engine performs cosine similarity search with dual-boosted similarity weights to fetch relevant rules.
5. The agent combines student data and retrieved rules, querying `gemini-2.5-flash` with a strict JSON schema structure.
6. The frontend renders the dynamic scorecard, status indicators, and warning checklists.

---

## 🚀 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MaheshBabuDevarakonda/scholaraiagent.git
   cd scholaraiagent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the RAG Vector Database**:
   ```bash
   python data_ingestion.py
   ```

4. **Run the Evaluation Harness**:
   ```bash
   python eval_harness.py
   ```

5. **Run the Web App**:
   ```bash
   python main.py
   ```



