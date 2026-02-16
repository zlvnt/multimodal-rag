# Multimodal RAG

A Retrieval-Augmented Generation (RAG) system that supports **images** and **PDFs** as knowledge sources.

**Flow:**
- **Image**: Gemini describe image → Jina embed text → store in ChromaDB
- **PDF**: PyMuPDF extract text → chunk → Jina embed → store in ChromaDB
- **Query**: Jina embed query → ChromaDB search → Gemini generate answer

## Prerequisites

- Python 3.11+
- [Jina API key](https://jina.ai/) — for text embeddings
- [Google AI API key](https://aistudio.google.com/apikey) — for Gemini (image description & answer generation)

## Setup

```bash
# 1. Clone repo
git clone <repo-url>
cd multimodal-rag

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup API keys
cp .env.example .env
# Edit .env, fill in JINA_API_KEY and GOOGLE_API_KEY
```

## Usage

Open the notebook:

```bash
jupyter notebook notebook.ipynb
```

Follow the step-by-step guide in the notebook:

1. **Setup** — load config & engine
2. **Ingest images** — place images in `data/images/`, run the ingest cell
3. **Ingest PDFs** — place PDFs in `data/pdfs/`, run the ingest cell
4. **Query** — ask questions based on ingested documents
5. **Utils** — list documents, delete, or reset DB

## Project Structure

```
multimodal-rag/
├── app/
│   ├── config.py          # Settings (API keys, chunk size, etc.)
│   └── engine.py          # Core: embed, ingest, query, describe
├── data/
│   ├── images/            # Place images here
│   └── pdfs/              # Place PDFs here
├── storage/               # ChromaDB data (auto-generated)
├── notebook.ipynb         # Main notebook
├── requirements.txt
├── .env.example           # API keys template
└── .gitignore
```
