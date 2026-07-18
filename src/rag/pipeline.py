"""
Main RAG pipeline orchestrating document retrieval and generation.
"""

from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from ..utils.logger import get_logger
from ..config import config
from .embeddings import EmbeddingModel
from .vector_store import get_vector_store
from .document_processor import DocumentProcessor
from .llm import LLM

logger = get_logger(__name__)


class RAGPipeline:
    """Main RAG pipeline"""
    
    def __init__(self):
        """Initialize RAG pipeline components"""
        logger.info("Initializing RAG Pipeline...")
        
        # Initialize components
        self.embedding_model = EmbeddingModel(model_type="ollama")
        self.vector_store = get_vector_store()
        self.document_processor = DocumentProcessor()
        self.llm = LLM(provider="ollama")
        self._vector_write_lock = Lock()
        
        # Check if LLM is available
        if not self.llm.is_available():
            logger.warning("Ollama LLM not available. Ensure Ollama is running.")
        
        logger.info("RAG Pipeline initialized successfully")
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest a document and add to vector store"""
        try:
            logger.info(f"Ingesting document: {file_path}")
            
            # Process document
            documents = self.document_processor.process_file(file_path)
            
            if not documents:
                return {"success": False, "message": "Failed to process document"}
            
            # Generate embeddings and add to vector store
            doc_ids = []
            doc_contents = []
            metadatas = []
            
            for doc in documents:
                doc_ids.append(doc["id"])
                doc_contents.append(doc["content"])
                metadatas.append(doc["metadata"])
            
            # Add to vector store
            with self._vector_write_lock:
                self.vector_store.add_documents(doc_contents, metadatas, doc_ids)
            
            logger.info(f"Successfully ingested {len(documents)} chunks from {file_path}")
            return {
                "success": True,
                "message": f"Ingested {len(documents)} chunks",
                "document_count": len(documents),
                "doc_ids": doc_ids,
                "source": file_path
            }
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            return {"success": False, "error": str(e), "source": file_path}
    
    def ingest_documents(self, file_paths: List[str], workers: int = None) -> Dict[str, Any]:
        """Ingest multiple documents concurrently with bounded worker pool."""
        if not file_paths:
            return {"success": True, "message": "No files to ingest", "processed_files": 0, "results": []}
        
        max_workers = workers or config.max_parallel_file_ingestions
        max_workers = max(1, min(max_workers, len(file_paths)))
        
        results: List[Dict[str, Any]] = []
        succeeded = 0
        failed = 0
        
        logger.info(f"Ingesting {len(file_paths)} documents using {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.ingest_document, path) for path in file_paths]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.get("success"):
                    succeeded += 1
                else:
                    failed += 1
        
        return {
            "success": failed == 0,
            "message": f"Ingested {succeeded}/{len(file_paths)} files",
            "processed_files": len(file_paths),
            "successful_files": succeeded,
            "failed_files": failed,
            "results": results
        }
    
    def retrieve(self, query: str, k: int = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Retrieve relevant documents for a query"""
        try:
            k = k or config.top_k_results
            logger.info(f"Retrieving top {k} documents for query: '{query}'")
            
            # Search vector store
            results = self.vector_store.search(query, k=k)
            logger.info(f"Raw search returned {len(results)} results")
            
            # Log all results before filtering
            for i, (doc, score, metadata) in enumerate(results):
                logger.debug(f"  Result {i+1}: score={score:.4f}, file_type={metadata.get('file_type', 'unknown')}, source={metadata.get('source', 'unknown')}")
            
            # Filter by similarity threshold
            filtered_results = [
                (doc, score, metadata)
                for doc, score, metadata in results
                if score >= config.similarity_threshold
            ]
            
            logger.info(f"After filtering by threshold {config.similarity_threshold}: {len(filtered_results)} relevant documents")
            if len(results) > 0 and len(filtered_results) == 0:
                logger.warning(f"All {len(results)} results filtered out! Scores: {[score for _, score, _ in results]}")
            
            return filtered_results
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            return []
    
    def generate_response(self, query: str, context_docs: List[str]) -> str:
        """Generate response using LLM with context"""
        try:
            # Build context
            context = "\n---\n".join(context_docs)
            
            # Build prompt
            prompt = self._build_prompt(query, context)
            
            logger.info("Generating response from LLM...")
            
            # Generate response
            response = self.llm.generate(prompt)
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def stream_response(self, query: str, context_docs: List[str]):
        """Stream response from LLM"""
        try:
            # Build context
            context = "\n---\n".join(context_docs)
            
            # Build prompt
            prompt = self._build_prompt(query, context)
            
            logger.info("Streaming response from LLM...")
            
            # Stream response
            for chunk in self.llm.stream(prompt):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield f"Error generating response: {str(e)}"
    
    def query(self, query: str, stream: bool = False, k: int = None) -> Dict[str, Any]:
        """Complete RAG query: retrieve and generate"""
        try:
            # Retrieve relevant documents
            retrieved = self.retrieve(query, k=k)
            
            if not retrieved:
                return {
                    "query": query,
                    "response": "No relevant data found.",
                    "sources": []
                }
            
            # Extract context documents
            context_docs = [doc for doc, _, _ in retrieved]
            sources = [metadata for _, _, metadata in retrieved]
            
            # Generate response
            if stream:
                # Return generator for streaming
                return {
                    "query": query,
                    "stream": True,
                    "generator": self.stream_response(query, context_docs),
                    "sources": sources
                }
            else:
                response = self.generate_response(query, context_docs)
                return {
                    "query": query,
                    "response": response,
                    "sources": sources
                }
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "sources": []
            }
    
    def delete_documents(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Delete documents from vector store"""
        try:
            self.vector_store.delete_documents(doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
            return {"success": True, "deleted_count": len(doc_ids)}
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        """Build prompt for LLM"""
        return f"""Based on the following context, answer the question. If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question: {query}

Answer:"""
