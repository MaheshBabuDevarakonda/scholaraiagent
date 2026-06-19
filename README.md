# AI Engineering & Multimodal Systems Portfolio

This repository contains two production-ready AI applications demonstrating end-to-end pipelines, API orchestrations, multi-model fallback loops, vector search (RAG), tool calling, and evaluation harnesses.

---

## 🚀 Project 1: Real-Time Multimodal OCR & Translation Pipeline (`realtime_ocr_translator.py`)

An assistive real-time computer vision and translation pipeline built for visually impaired users. It captures English text via camera frames, preprocesses the image, runs local OCR, translates the text to Telugu, and converts it to audio speech.

### 🌟 Key Features
- **Smart Reading Order**: Custom NLP sorting algorithm grouping bounding boxes into lines based on baseline Y-coordinate proximity and sorting horizontally by X-coordinates to handle multi-column text correctly.
- **Symbol & Logo Filtering**: Unicode-category filters and bounding box aspect-ratio checks to filter out UI symbols and non-alphanumeric noise before translation.
- **3-Tier Translation Fallback Loop**: 
  - *Primary (Online):* Google Translator (fast & natural).
  - *Secondary (Local Offline):* MarianMT (`Helsinki-NLP/opus-mt-en-mul` - ~300MB).
  - *Tertiary (Local Offline):* NLLB-Distilled (`facebook/nllb-200-distilled-600M` - ~600MB).
- **2-Tier TTS Speech Fallback Loop**: 
  - *Primary:* gTTS (online, natural).
  - *Secondary:* `espeak-ng` (local offline synthesized voice).

---

## 🎓 Project 2: ScholarScan AI — RAG & Tool-Calling Agent (`main.py`)

An automated admissions and visa eligibility evaluator that ingests unstructured PDF/text university rules, searches the database using vector embeddings (RAG), and calls tools to evaluate student profiles.

### 🌟 Key Features
- **Knowledge Ingestion & Chunking (`data_ingestion.py`)**: Slides unstructured guidelines into chunks and indexes them using Gemini embeddings (`text-embedding-004`) with local JSON vector store persistence.
- **Agentic Tool Calling (`agent_engine.py`)**: The evaluator agent leverages function calling, invoking a `query_visa_rules` vector search tool to retrieve context before generating structured JSON scorecards.
- **Evaluation Harness (`eval_harness.py`)**: A batch evaluation regression runner that measures prediction accuracy and retrieval success rate against a ground-truth test suite.
- **Fail-Safe Policy Engine**: Automatically catches API exceptions and rate limits to degrade to local Python heuristics, ensuring 100% application uptime.

---

## 📦 Setup & Installation

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

