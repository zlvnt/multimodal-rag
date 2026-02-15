from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    jina_api_key: str
    google_api_key: str

    data_dir: str = "data/images"
    chroma_dir: str = "storage/chroma"
    chroma_collection: str = "multimodal_rag"

    chunk_size: int = 512
    chunk_overlap: int = 50

    model_config = {"env_file": ".env"}


settings = Settings()
