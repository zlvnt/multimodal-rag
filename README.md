# Multimodal RAG

Sistem Retrieval-Augmented Generation (RAG) yang mendukung **gambar** dan **PDF** sebagai sumber pengetahuan.

**Flow:**
- **Image**: Gemini describe image → Jina embed teks → simpan di ChromaDB
- **PDF**: PyMuPDF extract teks → chunk → Jina embed → simpan di ChromaDB
- **Query**: Jina embed query → ChromaDB search → Gemini generate jawaban

## Prerequisites

- Python 3.11+
- [Jina API key](https://jina.ai/) — untuk text embeddings
- [Google AI API key](https://aistudio.google.com/apikey) — untuk Gemini (describe image & generate jawaban)

## Setup

```bash
# 1. Clone repo
git clone <repo-url>
cd multimodal-rag

# 2. Buat virtual environment
python -m venv .venv
source .mrag/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup API keys
cp .env.example .env
# Edit .env, isi JINA_API_KEY dan GOOGLE_API_KEY
```

## Penggunaan

Buka notebook:

```bash
jupyter notebook notebook.ipynb
```

Ikuti step-by-step di notebook:

1. **Setup** — load config & engine
2. **Ingest gambar** — taruh gambar di `data/images/`, jalankan cell ingest
3. **Ingest PDF** — taruh PDF di `data/pdfs/`, jalankan cell ingest
4. **Query** — tanya apa saja berdasarkan dokumen yang sudah di-ingest
5. **Utils** — list dokumen, hapus, atau reset DB

## Struktur Project

```
multimodal-rag/
├── app/
│   ├── config.py          # Settings (API keys, chunk size, dll)
│   └── engine.py          # Core: embed, ingest, query, describe
├── data/
│   ├── images/            # Taruh gambar di sini
│   └── pdfs/              # Taruh PDF di sini
├── storage/               # ChromaDB data (auto-generated)
├── notebook.ipynb         # Notebook utama
├── requirements.txt
├── .env.example           # Template API keys
└── .gitignore
```
