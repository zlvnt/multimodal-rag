import base64
import logging
import os
import uuid
from typing import List

import chromadb
import fitz
import requests
from google import genai

from app.config import settings

logger = logging.getLogger(__name__)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    try:
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.jina_api_key}"},
            json={"model": "jina-embeddings-v4", "input": texts},
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in data]
    except requests.ConnectionError:
        raise RuntimeError("Gagal koneksi ke Jina API. Cek koneksi internet.")
    except requests.Timeout:
        raise RuntimeError("Request ke Jina API timeout. Coba lagi nanti.")
    except requests.HTTPError as e:
        raise RuntimeError(f"Jina API error (HTTP {e.response.status_code}): {e.response.text}")
    except requests.RequestException as e:
        raise RuntimeError(f"Gagal mendapatkan embeddings dari Jina API: {e}")


def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=settings.google_api_key)


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    return client.get_or_create_collection(
        settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def describe_image(file_path: str) -> str:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File gambar tidak ditemukan: {file_path}")

    client = get_gemini_client()
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
    mime_type = mime_map.get(ext, "image/jpeg")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "parts": [
                        {"text": "Describe this image for use in a RAG (Retrieval-Augmented Generation) system. Focus on the key information, concepts, and facts presented. Avoid describing visual styling like colors, fonts, or layout. Be concise and informative."},
                        {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}},
                    ]
                }
            ],
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gagal describe image via Gemini: {e}")


def extract_pdf_text(file_path: str) -> str:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File PDF tidak ditemukan: {file_path}")

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise RuntimeError(f"Gagal membuka PDF (file mungkin corrupt): {e}")

    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap

    separators = ["\n\n", "\n", " ", ""]
    chunks = _recursive_split(text, separators, chunk_size)

    # Tambah overlap antar chunks (sekali aja, di sini)
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-overlap:] if len(prev) >= overlap else prev
            overlapped.append(overlap_text + chunks[i])
        chunks = overlapped

    return chunks


def _recursive_split(text: str, separators: List[str], chunk_size: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    # Cari separator yang cocok
    sep = separators[-1]
    for s in separators:
        if s in text:
            sep = s
            break

    parts = text.split(sep) if sep else list(text)

    chunks = []
    current = ""

    for part in parts:
        # Tambah separator kecuali kosong
        candidate = current + sep + part if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Kalau part sendiri masih kegedean, split lagi pakai separator berikutnya
            if len(part) > chunk_size:
                remaining_seps = separators[separators.index(sep) + 1:]
                if remaining_seps:
                    chunks.extend(_recursive_split(part, remaining_seps, chunk_size))
                else:
                    chunks.append(part)
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks


def ingest_texts(texts: List[str], metadatas: List[dict] = None, force: bool = False) -> List[str]:
    collection = get_chroma_collection()
    metas = metadatas if metadatas else [{} for _ in texts]

    # Duplikasi check berdasarkan metadata "source"
    if not force:
        filtered_texts = []
        filtered_metas = []
        for text, meta in zip(texts, metas):
            source = meta.get("source")
            if source:
                existing = collection.get(where={"source": source})
                if existing and existing["ids"]:
                    logger.warning(f"Duplikat: '{source}' sudah ada di collection, skip. Pakai force=True untuk override.")
                    continue
            filtered_texts.append(text)
            filtered_metas.append(meta)

        if not filtered_texts:
            logger.warning("Semua dokumen sudah ada di collection. Tidak ada yang di-ingest.")
            return []

        texts = filtered_texts
        metas = filtered_metas

    embeddings = get_embeddings(texts)
    ids = [str(uuid.uuid4()) for _ in texts]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metas,
    )
    return ids


def query_rag(query: str, top_k: int = 3) -> dict:
    query_embedding = get_embeddings([query])[0]

    collection = get_chroma_collection()

    if collection.count() == 0:
        return {
            "answer": "Belum ada dokumen di collection. Ingest dokumen dulu sebelum query.",
            "sources": [],
        }

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    contexts = result["documents"][0]
    sources = []
    for i in range(len(contexts)):
        score = result["distances"][0][i] if result["distances"] else None
        meta = result["metadatas"][0][i] if result["metadatas"] else {}
        sources.append({
            "id": result["ids"][0][i],
            "score": score,
            "metadata": meta,
            "text_preview": contexts[i][:200],
        })

    context_str = "\n\n---\n\n".join(contexts)
    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""Based on the following context, answer the question.

Context:
{context_str}

Question: {query}

Answer:""",
        )
    except Exception as e:
        raise RuntimeError(f"Gagal generate jawaban via Gemini: {e}")

    return {
        "answer": response.text,
        "sources": sources,
    }
