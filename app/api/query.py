import base64
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock

from app.engine import get_embed_model, get_index, get_llm

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@router.post("/query")
async def query_images(request: QueryRequest):
    embed_model = get_embed_model()
    query_embedding = embed_model.get_text_embedding(request.query)

    index = get_index()
    retriever = index.as_retriever(similarity_top_k=request.top_k)

    from llama_index.core.vector_stores import VectorStoreQuery

    vector_store = index.vector_store
    query_result = vector_store.query(
        VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=request.top_k,
        )
    )

    image_paths = []
    sources = []
    for node in query_result.nodes:
        file_path = node.metadata.get("file_path", "")
        if file_path and Path(file_path).exists():
            image_paths.append(file_path)
            sources.append({
                "id": node.id_,
                "file_path": file_path,
                "filename": node.metadata.get("original_filename", ""),
                "score": None,
            })

    if query_result.similarities:
        for i, score in enumerate(query_result.similarities):
            if i < len(sources):
                sources[i]["score"] = score

    if not image_paths:
        return {
            "answer": "No relevant images found.",
            "sources": [],
        }

    llm = get_llm()

    msg = ChatMessage(role="user")
    msg.blocks = [TextBlock(text=f"Based on these images, answer the following question: {request.query}")]
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        media_type = "image/jpeg"
        if img_path.endswith(".png"):
            media_type = "image/png"
        elif img_path.endswith(".webp"):
            media_type = "image/webp"
        msg.blocks.append(ImageBlock(image=img_data, image_mimetype=media_type))

    response = llm.chat(messages=[msg])

    return {
        "answer": str(response.message.content),
        "sources": sources,
    }
