# AI Engineering & Multimodal Systems Portfolio

This repository contains two production-ready AI applications demonstrating end-to-end pipelines, API orchestrations, multi-model fallback loops, and edge device optimization.

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

### 🛠️ Core Technologies
- **Computer Vision:** OpenCV (preprocessing, deskewing, sharpening)
- **OCR:** RapidOCR (multi-pass inference on normal & inverted frames)
- **NLP & Translation:** HuggingFace Transformers, PyTorch, SymSpell, spaCy
- **Interface:** Google Colab JS integration for camera streaming

---

## 🎓 Project 2: ScholarScan AI — Student Eligibility Screener (`main.py`)

A lightweight B2B/B2C SaaS application that automates the initial admissions and visa screening of international students applying to the UK, Canada, USA, and Australia.

### 🌟 Key Features
- **Structured JSON Audits**: Uses `gemini-2.5-flash` with JSON schema enforcement to parse grades, IELTS scores, budgets, and gap years into structured risk reports.
- **Fail-Safe Policy Engine (Mock Mode)**: Intercepts LLM rate limits and API credential failures to seamlessly degrade to a local Python heuristics validator, ensuring 100% application uptime.
- **Interactive UI**: A single-page dashboard with glassmorphism styling, real-time validation forms, and animated SVG score meters.

### 🛠️ Core Technologies
- **Backend:** FastAPI, Uvicorn, Pydantic, Google GenAI SDK
- **Frontend:** Vanilla HTML5, CSS3, JavaScript (Fetch API)

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

3. **Run Project 1 (OCR & Translation)**:
   ```bash
   python realtime_ocr_translator.py
   ```

4. **Run Project 2 (ScholarScan AI)**:
   ```bash
   python main.py
   ```
