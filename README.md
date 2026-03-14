# RAG-Based Quiz & Learning System

A retrieval-augmented generation (RAG) system that ingests PDF documents, builds a RAPTOR tree index, enables conversational Q&A, generates targeted quizzes from specific pages or chapters, and tracks student performance with an analytics dashboard.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Configure Environment Variables](#2-configure-environment-variables)
  - [3. Start Infrastructure (Docker)](#3-start-infrastructure-docker)
  - [4. Pull an LLM Model](#4-pull-an-llm-model)
  - [5. Install Python Dependencies](#5-install-python-dependencies)
- [Running the Application](#running-the-application)
  - [Option A: Full Web UI (Recommended)](#option-a-full-web-ui-recommended)
  - [Option B: CLI Only](#option-b-cli-only)
- [CLI Reference](#cli-reference)
- [API Endpoints](#api-endpoints)
- [How It Works](#how-it-works)
  - [Phase 1: PDF Ingestion and RAPTOR Indexing](#phase-1-pdf-ingestion-and-raptor-indexing)
  - [Phase 2: RAG Query Pipeline](#phase-2-rag-query-pipeline)
  - [Phase 3: Quiz Generation](#phase-3-quiz-generation)
  - [Phase 4: Performance Dashboard](#phase-4-performance-dashboard)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

**Document Ingestion** -- Upload PDF files, extract text page-by-page, detect chapters automatically from bookmarks and heading patterns, chunk text with token-aware splitting, and build a RAPTOR (Recursive Abstractive Processing for Tree-Organised Retrieval) index with multi-level summaries stored in PostgreSQL + pgvector.

**Conversational Q&A** -- Ask questions about your documents. The system retrieves relevant passages across all RAPTOR tree levels using vector similarity, re-ranks them with a cross-encoder model, and generates streaming answers through a local Ollama LLM with cited source passages.

**Targeted Quiz Generation** -- Select specific pages or chapters and generate quizzes using OpenAI (GPT-4o-mini) or Anthropic (Claude). Supports multiple choice, true/false, and fill-in-the-blank questions with configurable difficulty and question count.

**Performance Dashboard** -- View quiz scores over time, per-topic accuracy breakdowns, difficulty-level analysis, question-type performance, weak topic identification, and AI-powered personalised study recommendations.

---

## Architecture

```
PDF --> Text Extraction --> Token Chunking --> RAPTOR Tree --> pgvector
            (pypdf)          (tiktoken)       (GMM + LLM)    (PostgreSQL)
                                                  |
                                                  v
User Question --> Embed --> Vector Search --> Cross-Encoder --> Ollama LLM
                (OpenAI)    (pgvector)      (MiniLM reranker)  (streaming)
                                                  |
                                                  v
Quiz Request --> Page/Chapter Filter --> LLM Generation --> Quiz Storage
                 (page_texts table)    (OpenAI/Claude)    (PostgreSQL)
                                                  |
                                                  v
Dashboard --> Analytics Engine --> Topic Performance --> AI Study Coach
             (quiz history)       (weak topics)        (LLM feedback)
```

---

## Project Structure

```
.
├── multiagent_rag_complete.py     # Complete system (single-file, all phases)
├── multiagent_app.py              # Streamlit UI (Documents, Chat, Quiz, Dashboard)
├── multiagent_api.py              # FastAPI server (REST + SSE streaming)
├── multiagent_analytics.py        # Analytics engine (performance metrics, feedback)
├── multiagent_requirements.txt    # Python dependencies
├── docker-compose-multiagent.yml  # Docker services (PostgreSQL + Ollama)
├── init.sql                       # Database schema (auto-run by Docker)
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
└── src/                           # Modular source (if using the multi-file layout)
    ├── config.py
    ├── database.py
    ├── pdf_loader.py
    ├── chunker.py
    ├── embeddings.py
    ├── raptor_indexer.py
    ├── page_extractor.py
    ├── ollama_client.py
    ├── reranker.py
    ├── query_pipeline.py
    ├── quiz_generator.py
    └── analytics.py
```

---

## Prerequisites

- **Docker** and **Docker Compose** (for PostgreSQL + pgvector and Ollama)
- **Python 3.11+**
- **OpenAI API key** (required for embeddings and quiz generation)
- **Anthropic API key** (optional, only needed if using Claude for quiz generation)
- **NVIDIA GPU** (optional, for faster Ollama inference; CPU works too)

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` in a text editor and set the required values:

```env
# Required
OPENAI_API_KEY=sk-your-actual-openai-key

# Required for database
PG_USER=raguser
PG_PASSWORD=your-secure-password
PG_DATABASE=ragdb

# Optional (only if using Claude for quiz generation)
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key
```

All other settings have sensible defaults. See [Configuration Reference](#configuration-reference) for the full list.

### 3. Start Infrastructure (Docker)

Start PostgreSQL (with pgvector) and Ollama:

```bash
docker compose -f docker-compose-multiagent.yml up -d
```

Verify both containers are running:

```bash
docker ps
```

You should see `rag_pgvector` and `rag_ollama` containers.

**Note on GPU support:** The docker-compose file includes NVIDIA GPU configuration for Ollama. If you do not have an NVIDIA GPU, edit `docker-compose-multiagent.yml` and remove or comment out the `deploy.resources` section under the `ollama` service. Ollama will fall back to CPU inference.

### 4. Pull an LLM Model

Pull a model into Ollama (this downloads the model weights, may take a few minutes):

```bash
docker exec rag_ollama ollama pull llama3.1:8b
```

You can use any Ollama-compatible model. Update `OLLAMA_MODEL` in your `.env` if you choose a different one. Some options:

```bash
docker exec rag_ollama ollama pull llama3.1:8b      # 4.7 GB, good balance
docker exec rag_ollama ollama pull mistral:7b        # 4.1 GB, fast
docker exec rag_ollama ollama pull llama3.1:70b      # 40 GB, best quality (needs GPU)
```

### 5. Install Python Dependencies

Create a virtual environment (recommended) and install:

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r multiagent_requirements.txt
```

---

## Running the Application

### Option A: Full Web UI (Recommended)

This requires two terminal windows.

**Terminal 1 -- Start the API server:**

```bash
python multiagent_rag_complete.py serve
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Terminal 2 -- Start the Streamlit UI:**

```bash
streamlit run multiagent_app.py --server.port 8501
```

Open `http://localhost:8501` in your browser. The UI has four pages:

| Page | What it does |
|------|-------------|
| **Documents** | Upload PDFs, view ingested documents and detected chapters, delete documents |
| **Chat** | Ask questions about your documents with streaming answers and source citations |
| **Quiz** | Select pages or chapters, configure difficulty, generate and take quizzes interactively |
| **Dashboard** | View performance analytics, topic breakdowns, weak areas, and AI study recommendations |

### Option B: CLI Only

You can use the system entirely from the command line without starting the API server or UI.

```bash
# Check system health
python multiagent_rag_complete.py health

# Ingest a PDF
python multiagent_rag_complete.py ingest path/to/document.pdf

# Ask a question
python multiagent_rag_complete.py ask "What are the main topics?" --stream

# List chapters in a document
python multiagent_rag_complete.py chapters 1

# Generate a quiz
python multiagent_rag_complete.py quiz --doc-id 1 --pages "1-10" --num 10 --difficulty medium

# View performance dashboard
python multiagent_rag_complete.py dashboard
```

---

## CLI Reference

| Command | Description | Example |
|---------|-------------|---------|
| `ingest <pdf>` | Ingest a PDF document | `python multiagent_rag_complete.py ingest textbook.pdf` |
| `serve` | Start the FastAPI server | `python multiagent_rag_complete.py serve` |
| `ui` | Launch the Streamlit UI | `python multiagent_rag_complete.py ui` |
| `health` | Check database and Ollama status | `python multiagent_rag_complete.py health` |
| `ask <question>` | Query your documents | `python multiagent_rag_complete.py ask "What is X?" --stream` |
| `chapters <doc_id>` | List detected chapters | `python multiagent_rag_complete.py chapters 1` |
| `quiz` | Generate a quiz | `python multiagent_rag_complete.py quiz --doc-id 1 --pages "1-5"` |
| `dashboard` | Show performance analytics | `python multiagent_rag_complete.py dashboard --doc-id 1` |

**Quiz command options:**

```
--doc-id       (required) Document ID
--pages        Page range, e.g. "1-10" or "1,3,5-8"
--chapters     Chapter titles, comma-separated
--num          Number of questions (default: 10)
--difficulty   easy, medium, or hard (default: medium)
--provider     openai or anthropic (default: openai)
```

---

## API Endpoints

Start the server with `python multiagent_rag_complete.py serve`, then use these endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| POST | `/ingest` | Upload and ingest a PDF (multipart form) |
| GET | `/documents` | List all ingested documents |
| GET | `/documents/{id}` | Document details with RAPTOR stats |
| DELETE | `/documents/{id}` | Delete a document and all related data |
| GET | `/documents/{id}/chapters` | List detected chapters |
| POST | `/query` | RAG query (blocking, returns full answer) |
| POST | `/query/stream` | RAG query (SSE streaming, token by token) |
| POST | `/quiz/generate` | Generate a quiz from pages or chapters |
| GET | `/quiz/{id}` | Retrieve a generated quiz |
| GET | `/quizzes` | List all quizzes (optional `?doc_id=` filter) |
| POST | `/quiz/{id}/score` | Save quiz score |
| DELETE | `/quiz/{id}` | Delete a quiz |
| GET | `/dashboard` | Performance analytics (optional `?doc_id=` filter) |
| POST | `/dashboard/feedback` | AI-generated study recommendations |

Full interactive API documentation is available at `http://localhost:8000/docs` when the server is running.

---

## How It Works

### Phase 1: PDF Ingestion and RAPTOR Indexing

1. **Text extraction** -- pypdf extracts text from each page of the PDF.
2. **Chapter detection** -- PDF bookmarks are read first. If none exist, text patterns are matched (e.g. "Chapter 1:", "1. Introduction", "CHAPTER ONE") to detect chapter boundaries.
3. **Chunking** -- The full text is split into overlapping chunks of ~512 tokens using tiktoken.
4. **RAPTOR tree construction** -- Leaf chunks are embedded with OpenAI `text-embedding-3-small`. Embeddings are soft-clustered using a Gaussian Mixture Model. Each cluster is summarised by an LLM. The summaries are embedded and the process repeats for up to 3 levels, creating a tree where higher levels capture broader themes.
5. **Storage** -- All nodes (leaves + summaries) with their embeddings are stored in PostgreSQL using pgvector's HNSW index for fast approximate nearest-neighbour search. Page-level text is stored separately for quiz generation.

### Phase 2: RAG Query Pipeline

1. **Query embedding** -- The user's question is embedded with the same OpenAI model.
2. **Vector search** -- The top 20 candidates are retrieved from pgvector across all RAPTOR tree levels (both leaf chunks and summary nodes).
3. **Cross-encoder re-ranking** -- A local `ms-marco-MiniLM-L-6-v2` cross-encoder scores each (query, passage) pair and keeps the top 3-5 most relevant results.
4. **Context assembly** -- Retrieved passages are formatted into a numbered context block.
5. **Answer generation** -- The context + question are sent to a local Ollama LLM which streams the answer token by token.

### Phase 3: Quiz Generation

1. **Content selection** -- The user specifies pages (e.g. "1-10,15") or chapters (e.g. "Chapter 3"). The corresponding page-level text is retrieved from the database.
2. **Prompt construction** -- A structured prompt instructs the LLM to generate questions in JSON format with specific question types (MCQ, true/false, fill-in-the-blank), difficulty levels, explanations, and source page references.
3. **LLM generation** -- Either OpenAI GPT-4o-mini or Anthropic Claude generates the quiz as structured JSON.
4. **Storage and scoring** -- Quizzes are stored in PostgreSQL. After the user takes a quiz, their score is saved for analytics.

### Phase 4: Performance Dashboard

1. **Analytics computation** -- All completed quizzes are aggregated by topic (chapter/page range), difficulty, and question type.
2. **Weak topic identification** -- Topics where accuracy falls below 70% are flagged with the gap size.
3. **Study recommendations** -- Rule-based suggestions are generated from the analytics (e.g. "Review Chapter 3, you scored 45%").
4. **AI study coach** -- Optionally, an LLM generates personalised, encouraging feedback based on the student's weak areas and overall performance.

---

## Configuration Reference

All configuration is done through environment variables in the `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (none) | Required. OpenAI API key for embeddings and quiz generation |
| `ANTHROPIC_API_KEY` | (none) | Optional. Anthropic API key for Claude-based quiz generation |
| `PG_HOST` | `localhost` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_USER` | (none) | PostgreSQL username |
| `PG_PASSWORD` | (none) | PostgreSQL password |
| `PG_DATABASE` | `ragdb` | PostgreSQL database name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `SUMMARISATION_MODEL` | `gpt-4o-mini` | Model for RAPTOR cluster summarisation |
| `RAPTOR_CHUNK_SIZE` | `512` | Tokens per chunk |
| `RAPTOR_CHUNK_OVERLAP` | `64` | Overlap tokens between chunks |
| `RAPTOR_MAX_TREE_LEVELS` | `3` | Maximum RAPTOR tree depth |
| `RAPTOR_NUM_CLUSTERS` | `10` | Target clusters per RAPTOR level |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `RERANKER_TOP_K` | `3` | Results to keep after re-ranking |
| `RETRIEVAL_TOP_K` | `20` | Initial vector search candidates |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model for answer generation |
| `OLLAMA_TEMPERATURE` | `0.3` | LLM temperature |
| `OLLAMA_CONTEXT_WINDOW` | `8192` | LLM context window size |
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `API_PORT` | `8000` | FastAPI port |
| `QUIZ_LLM_PROVIDER` | `openai` | Quiz generation provider (`openai` or `anthropic`) |
| `QUIZ_OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for quiz generation |
| `QUIZ_ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic model for quiz generation |
| `QUIZ_NUM_QUESTIONS` | `10` | Default questions per quiz |
| `QUIZ_TEMPERATURE` | `0.0` | Quiz generation temperature |

---

## Troubleshooting

**"API not reachable" in the Streamlit UI**
The FastAPI server must be running before you start the UI. Start it with `python multiagent_rag_complete.py serve` in a separate terminal.

**"No pages for doc X. Run reindex."**
The document was ingested before page-level text storage was added. Re-ingest the PDF: `python multiagent_rag_complete.py ingest path/to/document.pdf`

**Ollama model not found**
Pull the model first: `docker exec rag_ollama ollama pull llama3.1:8b`. Check available models with `docker exec rag_ollama ollama list`.

**GPU not available for Ollama**
If you do not have an NVIDIA GPU, remove the `deploy.resources` block from `docker-compose-multiagent.yml` under the `ollama` service. Ollama runs on CPU by default.

**PostgreSQL connection refused**
Make sure the Docker container is running: `docker ps`. If the container keeps restarting, check logs: `docker logs rag_pgvector`.

**Cross-encoder model download is slow on first run**
The `ms-marco-MiniLM-L-6-v2` model (~80 MB) is downloaded automatically on first query. This is a one-time download cached locally.

**Quiz generation returns empty or malformed results**
Try reducing `QUIZ_NUM_QUESTIONS` or selecting fewer pages. Very short pages may not contain enough content for meaningful questions.

---
