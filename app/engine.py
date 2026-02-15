import chromadb
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

from app.config import settings


def get_embed_model() -> JinaEmbedding:
    return JinaEmbedding(
        api_key=settings.jina_api_key,
        model="jina-embeddings-v4",
    )


def get_llm() -> GoogleGenAI:
    return GoogleGenAI(
        api_key=settings.google_api_key,
        model="gemini-2.5-flash",
    )


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
