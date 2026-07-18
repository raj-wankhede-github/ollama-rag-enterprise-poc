"""
Configuration management for Ollama RAG application.
Supports local and docker deployment.
"""

import os
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class Environment(str, Enum):
    """Application environment types"""
    LOCAL = "local"
    DOCKER = "docker"


@dataclass
class Config:
    """Application configuration"""
    
    # Environment
    environment: str = os.getenv("ENV", "local")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Ollama Configuration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    
    # Vector Database Configuration
    vector_db_type: str = os.getenv("VECTOR_DB_TYPE", "chroma")  # chroma, pinecone, weaviate
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    
    # File Storage
    storage_type: str = os.getenv("STORAGE_TYPE", "local")  # local
    local_upload_dir: str = os.getenv("LOCAL_UPLOAD_DIR", "./data/uploads")
    
    # API Configuration
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    
    # RAG Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    max_chunks_per_document: int = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "20000"))

    # Ingestion Throughput and Resilience
    vector_upsert_batch_size: int = int(os.getenv("VECTOR_UPSERT_BATCH_SIZE", "128"))
    ingestion_workers: int = int(os.getenv("INGESTION_WORKERS", "4"))
    embedding_workers: int = int(os.getenv("EMBEDDING_WORKERS", "4"))
    max_batch_files_per_request: int = int(os.getenv("MAX_BATCH_FILES_PER_REQUEST", "1000"))
    max_parallel_file_ingestions: int = int(os.getenv("MAX_PARALLEL_FILE_INGESTIONS", "4"))
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", os.getenv("MAX_UPLOAD_SIZE_MB", "512")))
    max_upload_stream_chunk_kb: int = int(os.getenv("MAX_UPLOAD_STREAM_CHUNK_KB", "1024"))
    embedding_max_retries: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
    embedding_retry_backoff_seconds: float = float(os.getenv("EMBEDDING_RETRY_BACKOFF_SECONDS", "1.0"))

    # PDF Processing
    pdf_extract_tables: bool = os.getenv("PDF_EXTRACT_TABLES", "true").lower() == "true"
    pdf_extract_images_text: bool = os.getenv("PDF_EXTRACT_IMAGES_TEXT", "true").lower() == "true"
    pdf_ocr_enabled: bool = os.getenv("PDF_OCR_ENABLED", "false").lower() == "true"
    
    # Database Configuration (for PostgreSQL with pgvector)
    database_url: Optional[str] = os.getenv("DATABASE_URL", None)
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    postgres_db: str = os.getenv("POSTGRES_DB", "rag_embeddings")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Request Configuration
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
    
    def is_local(self) -> bool:
        """Check if running locally"""
        return self.environment.lower() == "local"

    def is_docker(self) -> bool:
        """Check if running in docker environment"""
        return self.environment.lower() == "docker"
    
    def validate(self) -> None:
        """Validate configuration"""
        if self.vector_db_type == "pinecone":
            if not os.getenv("PINECONE_API_KEY"):
                raise ValueError("PINECONE_API_KEY must be set when using Pinecone")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        
        if self.vector_upsert_batch_size <= 0:
            raise ValueError("VECTOR_UPSERT_BATCH_SIZE must be greater than 0")
        
        # Ensure upload directory exists for local storage
        if self.storage_type == "local" and (self.is_local() or self.is_docker()):
            os.makedirs(self.local_upload_dir, exist_ok=True)
        
        if self.vector_db_type == "chroma":
            os.makedirs(self.chroma_persist_dir, exist_ok=True)


# Global config instance
config = Config()
