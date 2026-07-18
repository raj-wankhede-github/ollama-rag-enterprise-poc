"""
FastAPI application for local development and testing.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..config import config
from ..utils.logger import get_logger
from ..utils.storage import get_storage_provider
from ..rag.pipeline import RAGPipeline

logger = get_logger(__name__, config.log_level)
config.validate()


# Initialize FastAPI
app = FastAPI(
    title="Ollama RAG Enterprise Demo",
    description="RAG application using Ollama and Chroma",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
try:
    rag_pipeline = RAGPipeline()
except Exception as e:
    logger.error(f"Failed to initialize RAG pipeline: {e}")
    raise

# Initialize storage
storage = get_storage_provider(
    config.storage_type,
    base_dir=config.local_upload_dir
)


# Pydantic models
class QueryRequest(BaseModel):
    """Query request model"""
    query: str
    top_k: Optional[int] = config.top_k_results
    stream: Optional[bool] = False


class QueryResponse(BaseModel):
    """Query response model"""
    query: str
    response: str
    sources: List[dict]


class DocumentResponse(BaseModel):
    """Document ingestion response"""
    success: bool
    message: str
    document_count: Optional[int] = None
    error: Optional[str] = None


class BatchDocumentResponse(BaseModel):
    """Batch document ingestion response"""
    success: bool
    message: str
    processed_files: int
    successful_files: int
    failed_files: int
    results: List[dict]


# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": config.environment,
        "ollama_available": rag_pipeline.llm.is_available()
    }


@app.get("/info")
async def info():
    """Application info endpoint"""
    return {
        "name": "Ollama RAG Enterprise Demo",
        "version": "1.0.0",
        "environment": config.environment,
        "ollama_model": config.ollama_model,
        "embedding_model": config.ollama_embedding_model,
        "vector_db": config.vector_db_type,
        "storage_type": config.storage_type,
        "supported_file_types": {
            "documents": [
                {"format": "PDF", "extension": ".pdf", "description": "PDF documents"},
                {"format": "CSV", "extension": ".csv", "description": "Comma-separated values"},
                {"format": "Excel 2007+", "extension": ".xlsx", "description": "Excel workbook"},
                {"format": "Excel 97-2003", "extension": ".xls", "description": "Legacy Excel workbook"},
                {"format": "Text", "extension": ".txt", "description": "Plain text files"},
                {"format": "Markdown", "extension": ".md, .markdown", "description": "Markdown files"}
            ]
        },
        "limits": {
            "max_file_size_mb": config.max_file_size_mb,
            "max_batch_files_per_request": config.max_batch_files_per_request,
            "vector_upsert_batch_size": config.vector_upsert_batch_size,
            "embedding_workers": config.embedding_workers,
            "parallel_file_ingestions": config.max_parallel_file_ingestions
        }
    }


@app.post("/ingest", response_model=DocumentResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a document into the RAG system
    
    Supported file types:
    - PDF (.pdf)
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - Text (.txt)
    - Markdown (.md, .markdown)
    """
    tmp_path = None
    try:
        filename = file.filename or "uploaded_file"

        # Supported file extensions
        SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.csv', '.xlsx', '.xls', '.md', '.markdown'}
        
        # Validate file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        
        max_bytes = config.max_file_size_mb * 1024 * 1024
        stream_chunk_bytes = max(1024, config.max_upload_stream_chunk_kb * 1024)

        # Save temporary file without reading full file into memory
        total_written = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(stream_chunk_bytes)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size exceeds {config.max_file_size_mb}MB limit"
                    )
                tmp.write(chunk)
        
        # Upload to storage
        storage_key = f"documents/{uuid.uuid4()}_{filename}"
        storage.upload_file(tmp_path, storage_key)
        
        # Ingest into RAG
        result = rag_pipeline.ingest_document(tmp_path)
        
        return DocumentResponse(
            success=result["success"],
            message=result.get("message", ""),
            document_count=result.get("document_count"),
            error=result.get("error")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        await file.close()


@app.post("/ingest/batch", response_model=BatchDocumentResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """Ingest multiple documents concurrently with bounded parallelism."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > config.max_batch_files_per_request:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files in one request. Maximum allowed: {config.max_batch_files_per_request}"
        )
    
    supported_extensions = {'.pdf', '.txt', '.csv', '.xlsx', '.xls', '.md', '.markdown'}
    max_bytes = config.max_file_size_mb * 1024 * 1024
    stream_chunk_bytes = max(1024, config.max_upload_stream_chunk_kb * 1024)
    temp_paths: List[str] = []
    
    try:
        for upload in files:
            filename = upload.filename or "uploaded_file"
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in supported_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(sorted(supported_extensions))}"
                )
            
            total_written = 0
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                temp_paths.append(tmp.name)
                while True:
                    chunk = await upload.read(stream_chunk_bytes)
                    if not chunk:
                        break
                    total_written += len(chunk)
                    if total_written > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"File '{filename}' exceeds {config.max_file_size_mb}MB limit"
                            )
                        )
                    tmp.write(chunk)
            
            storage_key = f"documents/{uuid.uuid4()}_{filename}"
            storage.upload_file(temp_paths[-1], storage_key)
            await upload.close()
        
        workers = max(1, min(config.max_parallel_file_ingestions, len(temp_paths)))
        paired_inputs = list(zip(temp_paths, files))
        results = []

        def _ingest_single(path_and_file):
            path, source_upload = path_and_file
            result = rag_pipeline.ingest_document(path)
            result["filename"] = source_upload.filename or "uploaded_file"
            return result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_ingest_single, pair) for pair in paired_inputs]
            for future in as_completed(futures):
                results.append(future.result())
        
        successful = sum(1 for r in results if r.get("success"))
        failed = len(results) - successful
        return BatchDocumentResponse(
            success=failed == 0,
            message=f"Ingested {successful}/{len(results)} files",
            processed_files=len(results),
            successful_files=successful,
            failed_files=failed,
            results=results
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting batch documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for upload in files:
            try:
                await upload.close()
            except Exception as e:
                logger.debug(f"Failed to close upload handle for {upload.filename}: {e}")
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)


@app.post("/query")
async def query(request: QueryRequest):
    """Query the RAG system"""
    try:
        if request.stream:
            # Stream response
            async def generate():
                retrieved = rag_pipeline.retrieve(request.query, k=request.top_k)
                if not retrieved:
                    yield b"No relevant data found."
                    return
                context_docs = [doc for doc, _, _ in retrieved]
                for chunk in rag_pipeline.stream_response(request.query, context_docs):
                    yield chunk.encode("utf-8")
            
            return StreamingResponse(generate(), media_type="text/plain")
        else:
            # Non-streaming response
            result = rag_pipeline.query(request.query, k=request.top_k)
            return QueryResponse(
                query=result["query"],
                response=result["response"],
                sources=result["sources"]
            )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the RAG system"""
    try:
        result = rag_pipeline.delete_documents([doc_id])
        return result
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Ollama RAG Enterprise Demo API",
        "docs": "/docs",
        "health": "/health",
        "info": "/info"
    }
