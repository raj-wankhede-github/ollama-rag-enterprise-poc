"""
Embedding models using Ollama or other providers.
"""

import requests
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils.logger import get_logger
from ..config import config

logger = get_logger(__name__)


class OllamaEmbeddings:
    """Ollama embeddings provider"""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or config.ollama_base_url
        self.model = model or config.ollama_embedding_model
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        last_error = None
        for attempt in range(1, config.embedding_max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=config.request_timeout_seconds
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding")
                if not embedding:
                    raise ValueError("Ollama returned empty embedding")
                return embedding
            except Exception as e:
                last_error = e
                if attempt < config.embedding_max_retries:
                    backoff = config.embedding_retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        f"Embedding attempt {attempt}/{config.embedding_max_retries} failed: {e}. "
                        f"Retrying in {backoff:.2f}s."
                    )
                    time.sleep(backoff)
        
        logger.error(f"Embedding generation failed after retries: {last_error}")
        raise RuntimeError(f"Embedding generation failed: {last_error}") from last_error
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        max_workers = max(1, min(config.embedding_workers, len(texts)))
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        failures = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.embed_text, text): idx
                for idx, text in enumerate(texts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embeddings[idx] = future.result()
                except Exception as e:
                    failures.append((idx, str(e)))
        
        if failures:
            first_failure = failures[0]
            logger.error(
                f"Failed to generate embeddings for {len(failures)}/{len(texts)} chunks. "
                f"First failure at index {first_failure[0]}: {first_failure[1]}"
            )
            raise RuntimeError(
                f"Embedding batch failed for {len(failures)} chunks; first error: {first_failure[1]}"
            )
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        try:
            embedding = self.embed_text("test")
            return len(embedding)
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            return 384  # Default for nomic-embed-text


class EmbeddingModel:
    """Generic embedding model wrapper"""
    
    def __init__(self, model_type: str = "ollama", **kwargs):
        if model_type == "ollama":
            self.model = OllamaEmbeddings(**kwargs)
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")
    
    def embed(self, text: str) -> List[float]:
        """Embed a single text"""
        return self.model.embed_text(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        return self.model.embed_texts(texts)
