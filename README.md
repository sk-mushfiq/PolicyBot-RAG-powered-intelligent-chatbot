# HR PolicyBot 🤖📋
### RAG-Powered Intelligent Chatbot for HR Policy Q&A
Built with LangChain · ChromaDB · OpenAI · FastAPI · Streamlit · Docker

---

## Table of Contents
1. [What This Project Does](#what-this-project-does)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Running the Project](#running-the-project)
6. [Running the Tests](#running-the-tests)
7. [Using the API](#using-the-api)
8. [Using the Streamlit UI](#using-the-streamlit-ui)
9. [Running with Docker](#running-with-docker)
10. [Project Structure](#project-structure)
11. [Troubleshooting](#troubleshooting)

---

## What This Project Does

HR PolicyBot is a production-ready conversational AI assistant that lets
employees ask natural language questions about company HR policies.

Instead of returning generic LLM answers, it uses **Retrieval-Augmented
Generation (RAG)** to:

1. Ingest real HR documents (PDF/DOCX) and store them as embeddings
2. When a question is asked, retrieve the most relevant document chunks
3. Feed those chunks as context to the LLM
4. Return a grounded, citation-backed answer

**Result:** Accurate, hallucination-resistant answers tied to your actual
HR documents, with multi-turn conversation memory per user session.

---

## Architecture Overview

```
User Question
     │
     ▼
FastAPI /chat endpoint
     │
     ▼
RAGChain.query()
  ├── Embed the question (OpenAI / HuggingFace)
  ├── Similarity search → ChromaDB → Top-K chunks
  ├── Build prompt: [System] + [Context] + [Question]
  ├── Call OpenAI LLM (gpt-3.5-turbo)
  └── Return answer + source citations
```

---

## Prerequisites

Before you begin, make sure you have:

| Tool | Version | Check Command |
|------|---------|---------------|
| Python | 3.10+ | `python --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |
| Docker (optional) | 20+ | `docker --version` |
| OpenAI API Key | — | platform.openai.com |

---

## Step-by-Step Setup

### Step 1 — Clone or Create the Project Folder

If you received this as a zip or created it manually:

```bash
cd hr-policybot
```

If cloning from GitHub:

```bash
git clone https://github.com/YOUR_USERNAME/hr-policybot.git
cd hr-policybot
```

---

### Step 2 — Create a Python Virtual Environment

A virtual environment isolates project dependencies from your system Python.

**On Windows (Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

> ⚠️ IMPORTANT: Always activate the virtual environment before running
> any command in this project.

---

### Step 3 — Upgrade pip

```bash
pip install --upgrade pip
```

---

### Step 4 — Install All Dependencies

```bash
pip install -r requirements.txt
```

This installs: LangChain, ChromaDB, OpenAI SDK, FastAPI, Uvicorn,
HuggingFace sentence-transformers, Streamlit, pytest, and more.

**Expected time:** 2–5 minutes depending on your connection.

> If you see errors about `Microsoft Visual C++` on Windows,
> install the Visual C++ Build Tools from:
> https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

### Step 5 — Set Up Your Environment Variables

```bash
# Copy the example file
cp .env.example .env
```

Now open `.env` in any text editor and fill in your values:

```
OPENAI_API_KEY=sk-your-actual-openai-key-here
```

**Everything else has sensible defaults and does not need to be changed
for local development.**

> 🔑 Get your OpenAI API key at: https://platform.openai.com/api-keys
>
> ⚠️ NEVER commit your .env file to Git. It is already in .gitignore.

---

### Step 6 — Verify the Installation

Run a quick check to make sure all imports work:

```bash
python -c "
import langchain
import chromadb
import fastapi
import streamlit
print('✅ All dependencies installed correctly')
print(f'   LangChain: {langchain.__version__}')
print(f'   ChromaDB:  {chromadb.__version__}')
print(f'   FastAPI:   {fastapi.__version__}')
print(f'   Streamlit:   {streamlit.__version__}')
"
```

You should see all three version numbers printed.

---

### Step 7 — Generate the Sample HR Document (Optional but Recommended)

This creates a sample HR policy PDF you can use to test the bot:

```bash
# Install reportlab for PDF generation (optional)
pip install reportlab

# Generate the sample PDF
python data/generate_sample_pdf.py
```

You should see: `✅ Sample PDF created: data/sample_docs/hr_policy_handbook.pdf`

---

## Running the Project

### Option A — Run Backend Only (API)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     HR PolicyBot v1.0.0 starting up...
```

Open your browser: **http://localhost:8000/docs**
You will see the interactive Swagger API documentation.

---

### Option B — Run Backend + Frontend Together

**Terminal 1 — Start the API:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Start the Streamlit UI:**
```bash
streamlit run frontend/streamlit_app.py
```

Open your browser: **http://localhost:8501**

---

## Running the Tests

### Run the Full Test Suite

```bash
pytest
```

### Run Tests with Detailed Output

```bash
pytest -v
```

### Run a Specific Test File

```bash
pytest tests/test_document_processor.py -v
pytest tests/test_rag_chain.py -v
pytest tests/test_api.py -v
```

### Run a Specific Test Class

```bash
pytest tests/test_api.py::TestChatEndpoint -v
```

### Run a Single Test

```bash
pytest tests/test_api.py::TestChatEndpoint::test_chat_valid_request_returns_200 -v
```

### Run Tests with Coverage Report

```bash
pip install pytest-cov
pytest --cov=app --cov-report=term-missing
```

**Expected output — all tests should pass:**
```
tests/test_document_processor.py .................. PASSED
tests/test_vector_store.py .............. PASSED
tests/test_rag_chain.py ................. PASSED
tests/test_api.py ........................ PASSED
```

> Note: Tests that require reportlab (PDF generation) will be skipped
> if reportlab is not installed. Install it with:
> `pip install reportlab`

---

## Using the API

### 1. Check API Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "app_name": "HR PolicyBot",
  "version": "1.0.0",
  "vector_store_ready": false
}
```

---

### 2. Ingest an HR Document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/sample_docs/hr_policy_handbook.pdf"
```

Expected response:
```json
{
  "message": "Document ingested successfully.",
  "filename": "hr_policy_handbook.pdf",
  "chunks_created": 24
}
```

---

### 3. Ask a Question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many annual leave days do employees get?",
    "session_id": "user_123"
  }'
```

Expected response:
```json
{
  "answer": "Employees are entitled to 20 working days of annual leave per calendar year...\n\n📄 Source: hr_policy_handbook.pdf, Page 1",
  "sources": [
    {"source": "hr_policy_handbook.pdf", "page": 1}
  ],
  "session_id": "user_123"
}
```

---

### 4. Ask a Follow-Up Question (Same Session)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can I carry forward unused leave?",
    "session_id": "user_123"
  }'
```

The bot remembers previous turns in the same session.

---

### 5. Clear Session Memory

```bash
curl -X DELETE http://localhost:8000/chat/session/user_123
```

---

### 6. Explore All Endpoints

Visit **http://localhost:8000/docs** for the full interactive
Swagger UI where you can test every endpoint from the browser.

---

## Using the Streamlit UI

1. Start both backend and frontend (see Option B above)
2. Open **http://localhost:8501**
3. In the **sidebar**, upload one or more PDF/DOCX files
4. Click **"Ingest Documents"** and wait for confirmation
5. Type your question in the chat input at the bottom
6. The bot will respond with a grounded answer and source citations

---

## Running with Docker

### Prerequisites
- Docker Desktop installed and running

### Build and Start Everything

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Build and start all services
docker compose up --build
```

Services started:
- **Backend API:** http://localhost:8000
- **Streamlit UI:** http://localhost:8501
- **API Docs:**     http://localhost:8000/docs

### Stop All Services

```bash
docker compose down
```

### Rebuild After Code Changes

```bash
docker compose up --build --force-recreate
```

---

## Project Structure

```
hr-policybot/
│
├── app/                          # Application source code
│   ├── main.py                   # FastAPI app entry point
│   ├── core/
│   │   ├── config.py             # Settings (Pydantic BaseSettings)
│   │   └── logger.py             # Centralised logging
│   ├── services/                 # Business logic (OOP classes)
│   │   ├── document_processor.py # Load & chunk PDF/DOCX
│   │   ├── vector_store.py       # ChromaDB wrapper
│   │   ├── rag_chain.py          # RAG pipeline
│   │   └── chat_memory.py        # Per-session conversation memory
│   ├── api/
│   │   ├── schemas.py            # Pydantic request/response models
│   │   └── routes/
│   │       ├── ingest.py         # POST /ingest
│   │       └── chat.py           # POST /chat
│   └── prompts/
│       └── templates.py          # All LLM prompt templates
│
├── tests/                        # Test suite (TDD)
│   ├── test_document_processor.py
│   ├── test_vector_store.py
│   ├── test_rag_chain.py
│   └── test_api.py
│
├── frontend/
│   └── streamlit_app.py          # Streamlit chat UI
│
├── data/
│   ├── generate_sample_pdf.py    # Script to create test documents
│   └── sample_docs/              # Place HR documents here
│
├── .env.example                  # Environment variable template
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'app'"

You are running commands from the wrong directory.
Make sure you are in the `hr-policybot/` root folder:

```bash
cd hr-policybot
pytest          # correct
```

---

### ❌ "openai.AuthenticationError: Incorrect API key"

Your API key is missing or invalid. Check your `.env` file:

```bash
cat .env | grep OPENAI
# Should show: OPENAI_API_KEY=sk-...
```

---

### ❌ "chromadb.errors.InvalidCollectionException"

The ChromaDB collection is corrupted. Delete it and re-ingest:

```bash
rm -rf chroma_db/
```

Then re-ingest your documents.

---

### ❌ Tests fail with "FileNotFoundError"

Some tests need reportlab to create test PDFs.
Install it and re-run:

```bash
pip install reportlab
pytest
```

---

### ❌ "RuntimeError: No documents found in the vector store"

You tried to chat before ingesting documents.
Ingest a document first:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/sample_docs/hr_policy_handbook.pdf"
```

---

### ❌ Port 8000 already in use

Something else is using port 8000. Run on a different port:

```bash
uvicorn app.main:app --reload --port 8001
```

Then update the Streamlit `API_BASE_URL` in `frontend/streamlit_app.py`
to `http://localhost:8001`.

---

### ❌ "sentence_transformers not found" when using HuggingFace embeddings

```bash
pip install sentence-transformers
```

---

### Slow first run (HuggingFace model download)

The first time you run with `EMBEDDING_PROVIDER=huggingface`, it downloads
the `all-MiniLM-L6-v2` model (~90MB). This is normal. Subsequent runs
use the cached model.

---

## Tech Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM | OpenAI gpt-3.5-turbo | Answer generation |
| Embeddings | OpenAI ada-002 / HuggingFace | Vector representations |
| Vector Store | ChromaDB | Similarity search |
| RAG Orchestration | LangChain | Chain + memory management |
| Backend | FastAPI + Uvicorn | REST API |
| Validation | Pydantic v2 | Schema enforcement |
| Frontend | Streamlit | Chat UI |
| Containerisation | Docker + Compose | Deployment |
| Testing | pytest | TDD test suite |

---

*Built as a portfolio project demonstrating RAG architecture,
LLM integration, OOP design, TDD, and production-ready AI engineering.*
