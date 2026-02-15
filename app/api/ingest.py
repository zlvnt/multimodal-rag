import os
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from llama_index.core.schema import ImageDocument

from app.config import settings
from app.engine import get_embed_model, get_vector_store

router = APIRouter()


@router.post("/ingest")
async def ingest_image(file: UploadFile = File(...)):
    os.makedirs(settings.data_dir, exist_ok=True)

    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    filename = f"{file_id}{ext}"
    file_path = os.path.join(settings.data_dir, filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    embed_model = get_embed_model()
    image_embedding = embed_model.get_image_embedding(file_path)

    vector_store = get_vector_store()
    from llama_index.core.schema import TextNode

    node = TextNode(
        text=file.filename,
        id_=file_id,
        embedding=image_embedding,
        metadata={
            "file_path": file_path,
            "original_filename": file.filename,
            "type": "image",
        },
    )
    vector_store.add(nodes=[node])

    return {
        "id": file_id,
        "filename": filename,
        "status": "indexed",
    }
