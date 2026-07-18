"""
Vector database abstraction supporting multiple providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import os

# Disable ChromaDB telemetry before importing chromadb.
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

import chromadb
from chromadb.config import Settings
from ..utils.logger import get_logger
from ..config import config
from .embeddings import OllamaEmbeddings

logger = get_logger(__name__)


class OllamaEmbeddingFunction:
    """Chromadb-compatible Ollama embedding function wrapper"""
    
    def __init__(self):
        self.ollama_embeddings = OllamaEmbeddings()
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for texts - Chromadb compatible interface"""
        return self.ollama_embeddings.embed_texts(input)


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """Add documents to the vector store"""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for documents similar to query"""
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the vector store"""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document"""
        pass


class ChromaVectorStore(VectorStore):
    """Chroma vector database implementation"""
    
    def __init__(self, persist_dir: str = None, collection_name: str = "documents"):
        self.persist_dir = persist_dir or config.chroma_persist_dir
        self.ollama_embeddings = OllamaEmbeddings()
        
        try:
            # Chromadb 0.6.1+ uses PersistentClient directly
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"Initialized Chroma PersistentClient with path: {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
            raise
        
        try:
            # Create collection WITHOUT embedding function - we'll provide embeddings directly
            # This avoids issues with ChromaDB calling our embedding function
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Chroma collection initialized (embeddings provided externally): {collection_name}")
            logger.info(f"Collection has {self.collection.count()} documents")
        except Exception as e:
            logger.warning(f"Collection creation failed: {e}")
            try:
                # Retry without metadata
                self.collection = self.client.get_or_create_collection(
                    name=collection_name
                )
                logger.info(f"Chroma collection initialized without metadata: {collection_name}")
            except Exception as e2:
                logger.error(f"Failed to create collection: {e2}")
                raise
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """Add documents to Chroma collection with embeddings"""
        try:
            if not documents:
                return
            
            if not (len(documents) == len(metadatas) == len(ids)):
                raise ValueError("documents, metadatas, and ids must have the same length")
            
            batch_size = max(1, config.vector_upsert_batch_size)
            total = len(documents)
            logger.info(f"Upserting {total} documents to Chroma in batches of {batch_size}")
            
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch_docs = documents[start:end]
                batch_metadatas = metadatas[start:end]
                batch_ids = ids[start:end]
                
                logger.info(f"Generating embeddings for batch {start}-{end - 1} ({len(batch_docs)} docs)")
                embeddings = self.ollama_embeddings.embed_texts(batch_docs)
                
                if not embeddings or len(embeddings) != len(batch_docs):
                    raise ValueError(
                        f"Embedding generation mismatch for batch {start}-{end - 1}: "
                        f"expected {len(batch_docs)} got {len(embeddings) if embeddings else 0}"
                    )
                
                if embeddings and embeddings[0]:
                    logger.debug(f"Embedding dimension: {len(embeddings[0])}")
                
                self.collection.upsert(
                    documents=batch_docs,
                    embeddings=embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            logger.info(f"Successfully upserted {total} documents to Chroma")
        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {e}", exc_info=True)
            raise
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar documents in Chroma"""
        try:
            # Check collection size first
            collection_count = self.collection.count()
            logger.info(f"Searching collection with {collection_count} documents")
            
            if collection_count == 0:
                logger.warning("Collection is empty - no documents to search")
                return []
            
            # Generate query embedding directly (bypasses ChromaDB 0.6.1 embedding function issues)
            query_embedding = self.ollama_embeddings.embed_text(query)
            logger.debug(f"Generated query embedding with {len(query_embedding)} dimensions")
            
            # Query using embeddings directly instead of query_texts
            # Note: ChromaDB 0.6.1 returns ids automatically, don't include "ids" in include parameter
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection_count),
                include=["documents", "metadatas", "distances"]
            )
            
            logger.debug(f"Chroma returned: {len(results.get('ids', [[]])[0]) if results.get('ids') else 0} results")
            
            # Format results
            formatted_results = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    document = results["documents"][0][i] if results.get("documents") else ""
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    
                    # Convert distance to similarity (Chroma uses distance metrics, lower = more similar)
                    similarity = 1 - distance
                    logger.debug(f"Result {i+1}: distance={distance:.4f}, similarity={similarity:.4f}")
                    formatted_results.append((document, similarity, metadata))
            else:
                logger.warning(f"No results returned from Chroma. Results structure: {list(results.keys()) if results else 'None'}")
            
            logger.info(f"Formatted {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching Chroma: {e}", exc_info=True)
            return []
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from Chroma"""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from Chroma")
        except Exception as e:
            logger.error(f"Error deleting documents from Chroma: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document from Chroma"""
        try:
            results = self.collection.get(ids=[doc_id])
            if results["documents"] and len(results["documents"]) > 0:
                return {
                    "id": doc_id,
                    "document": results["documents"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {}
                }
            return None
        except Exception as e:
            logger.error(f"Error getting document from Chroma: {e}")
            return None


def get_vector_store(store_type: str = None, **kwargs) -> VectorStore:
    """Factory function to get appropriate vector store"""
    store_type = store_type or config.vector_db_type
    
    if store_type.lower() == "chroma":
        return ChromaVectorStore(
            persist_dir=kwargs.get("persist_dir", config.chroma_persist_dir),
            collection_name=kwargs.get("collection_name", "documents")
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
