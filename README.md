# ScholarScan AI: RAG-powered Agentic Admissions System

ScholarScan AI is a production-ready RAG (Retrieval Augmented Generation) and Agentic Tool-Calling system designed for international student recruitment. It automates the initial profile screening process for global study destinations (UK, Canada, USA, Australia) by ingesting unstructured admissions guidelines, retrieving relevant criteria via vector embeddings, and executing tool calls to score visa and academic readiness.

---

## 🌟 Core Features

1. **Knowledge Ingestion & Chunking (`data_ingestion.py`)**: Slides unstructured guidelines into chunks and indexes them using Gemini embeddings (`text-embedding-004`) with local JSON vector store persistence.
2. **Agentic Tool Calling (`agent_engine.py`)**: The evaluator agent leverages function calling, invoking a `query_visa_rules` vector search tool to retrieve context before generating structured JSON scorecards.
3. **Evaluation Harness (`eval_harness.py`)**: A batch evaluation regression runner that measures prediction accuracy and retrieval success rate against a ground-truth test suite.
4. **Fail-Safe Policy Engine**: Automatically catches API exceptions and rate limits to degrade to local Python heuristics, ensuring 100% application uptime.
5. **Interactive UI**: A single-page dashboard with glassmorphism styling, real-time validation forms, and animated SVG score meters.

---

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.12+)
- **WSGI/ASGI Server**: Uvicorn
- **AI Integration**: Google GenAI SDK (`google-genai`)
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (Fetch API)

---

## 📐 Architecture & Workflow

1. The frontend collects student profile metrics and dispatches an asynchronous `POST` JSON payload to `/api/evaluate`.
2. The FastAPI backend validates fields and calls the `evaluate_student_profile` agent engine.
3. The agent engine generates a search query and calls the local Vector DB search tool (`query_visa_rules`).
4. The retrieval engine performs cosine similarity search with metadata-based query boosting to fetch relevant regional rules.
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



