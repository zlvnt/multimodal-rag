import base64
import os
from typing import List

import chromadb
import fitz
from google import genai
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery

from app.config import settings


def get_embed_model() -> JinaEmbedding:
    return JinaEmbedding(
        api_key=settings.jina_api_key,
        model="jina-embeddings-v4",
    )


def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=settings.google_api_key)


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    return client.get_or_create_collection(settings.chroma_collection)


def get_vector_store() -> ChromaVectorStore:
    collection = get_chroma_collection()
    return ChromaVectorStore(chroma_collection=collection)


def get_index() -> VectorStoreIndex:
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=get_embed_model(),
    )


def describe_image(file_path: str) -> str:
    client = get_gemini_client()
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
    mime_type = mime_map.get(ext, "image/jpeg")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "parts": [
                    {"text": "Describe this image in detail. Include all visible objects, text, colors, layout, and any other relevant information."},
                    {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}},
                ]
            }
        ],
    )
    return response.text


def extract_pdf_text(file_path: str) -> str:
    doc = fitz.open(file_path)
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

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def ingest_texts(texts: List[str], metadatas: List[dict] = None) -> List[str]:
    embed_model = get_embed_model()
    vector_store = get_vector_store()

    import uuid
    nodes = []
    ids = []
    for i, text in enumerate(texts):
        embedding = embed_model.get_text_embedding(text)
        node_id = str(uuid.uuid4())
        meta = metadatas[i] if metadatas else {}
        node = TextNode(
            text=text,
            id_=node_id,
            embedding=embedding,
            metadata=meta,
        )
        nodes.append(node)
        ids.append(node_id)

    vector_store.add(nodes=nodes)
    return ids


def query_rag(query: str, top_k: int = 3) -> dict:
    embed_model = get_embed_model()
    query_embedding = embed_model.get_text_embedding(query)

    vector_store = get_vector_store()
    result = vector_store.query(
        VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
        )
    )

    contexts = []
    sources = []
    for i, node in enumerate(result.nodes):
        contexts.append(node.text)
        score = result.similarities[i] if result.similarities and i < len(result.similarities) else None
        sources.append({
            "id": node.id_,
            "score": score,
            "metadata": node.metadata,
            "text_preview": node.text[:200],
        })

    context_str = "\n\n---\n\n".join(contexts)
    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Based on the following context, answer the question.

Context:
{context_str}

Question: {query}

Answer:""",
    )

    return {
        "answer": response.text,
        "sources": sources,
    }
