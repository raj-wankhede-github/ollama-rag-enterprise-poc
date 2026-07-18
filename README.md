# Ollama RAG Enterprise Demo

A production-ready Retrieval-Augmented Generation (RAG) application using Ollama for local LLM inference. It is designed for local development, Docker deployment, and document ingestion across multiple file formats.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Local Development](#local-development)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Supported File Formats](#supported-file-formats)
- [File Upload Examples](#file-upload-examples)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Configuration](#configuration)
- [Python 3.13 Notes](#python-313-notes)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Repository Access Control](#repository-access-control)
- [Troubleshooting](#troubleshooting)

## Overview

This project combines:

- Document ingestion from local files
- Text chunking and embedding generation
- Chroma-backed vector search
- Ollama-based answer generation
- FastAPI endpoints for querying and ingestion
- Local filesystem storage

It supports local development and Docker tooling included for repeatable setup.

## Features

- Local RAG pipeline using Ollama
- FastAPI REST API with automatic docs
- Docker and Docker Compose support
- Streaming responses for query generation
- Environment-based configuration
- Built-in logging and storage utilities
- Multi-format document ingestion
- Source attribution for retrieved answers
- Streaming upload path for large files (low-memory ingestion)
- Batched Chroma upserts and concurrent embedding generation
- Batch ingestion endpoint for high-volume (1000s) file indexing
- PDF extraction with text + table parsing and optional OCR

## Architecture

### End-to-End Flow

```text
Document Upload -> File Processing -> Chunking -> Embedding Generation -> Vector Store (Chroma)
                                                                          |
                                                                          v
Query Input -> Query Embedding -> Vector Search -> Context Retrieval -> LLM Generation (Ollama)
```

### Core Components

- `src/config.py` - configuration management
- `src/api/app.py` - FastAPI application
- `src/rag/document_processor.py` - file parsing and normalization
- `src/rag/embeddings.py` - embedding generation
- `src/rag/vector_store.py` - vector database interface
- `src/rag/pipeline.py` - RAG orchestration
- `src/rag/llm.py` - Ollama integration
- `src/utils/logger.py` - logging utilities
- `src/utils/storage.py` - storage abstraction

## Prerequisites

- Python 3.13+
- Ollama installed and running
- Git
- `pip` or `conda`

Recommended Ollama models:

- `mistral`
- `nomic-embed-text`

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/ollama-rag-enterprise-demo.git
cd ollama-rag-enterprise-demo
```

### Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install Ollama models

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### Configure environment

```bash
cp .env.example .env
```

Edit `.env` as needed, or keep the defaults for local development.

## Local Deployment

### Start the API server

```bash
python main.py
```

The server runs at:

- `http://localhost:8000`
- `http://localhost:8000/docs` for Swagger UI

### Local environment example

```bash
ENV=local
DEBUG=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
STORAGE_TYPE=local
VECTOR_DB_TYPE=chroma
```

## Quick Start

### Five-minute setup

1. Install dependencies with `pip install -r requirements.txt`
2. Pull models with `ollama pull mistral` and `ollama pull nomic-embed-text`
3. Start the server with `python main.py`
4. Open `http://localhost:8000/docs`

### Quick test commands

```bash
curl http://localhost:8000/health

echo "Python is a programming language" > test.txt
curl -X POST http://localhost:8000/ingest -F "file=@test.txt"

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'
```

### Quick action for Python 3.13.8 setups

```bash
python setup.py
python main.py
```

This project includes compatibility updates for newer Python environments, including newer versions of `chromadb`, `pypdf`, `numpy`, `pydantic`, and `fastapi`.

## API Endpoints

### GET `/health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "healthy",
  "environment": "local",
  "ollama_available": true
}
```

### GET `/info`

Returns application metadata, supported file types, and upload limits.

```bash
curl http://localhost:8000/info
```

### POST `/ingest`

Upload and ingest a document.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.txt"
```

Example response:

```json
{
  "success": true,
  "message": "Ingested 45 chunks",
  "document_count": 45
}
```

### POST `/ingest/batch`

Upload and ingest multiple documents in one request.

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.xlsx" \
  -F "files=@doc3.txt"
```

Example response:

```json
{
  "success": true,
  "message": "Ingested 3/3 files",
  "processed_files": 3,
  "successful_files": 3,
  "failed_files": 0
}
```

### POST `/query`

Query the RAG system.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "top_k": 5,
    "stream": false
  }'
```

### DELETE `/documents/{doc_id}`

Delete an indexed document.

## Supported File Formats

The `/ingest` endpoint accepts these formats:

| Format | Extensions | Status | Typical Use |
|--------|------------|--------|-------------|
| PDF | `.pdf` | Supported | Reports, papers, documents |
| CSV | `.csv` | Supported | Tabular data, exports |
| Excel 2007+ | `.xlsx` | Supported | Modern spreadsheets |
| Excel 97-2003 | `.xls` | Supported | Legacy spreadsheets |
| Plain Text | `.txt` | Supported | Notes and text files |
| Markdown | `.md`, `.markdown` | Supported | Docs and structured notes |

### Processing behavior by format

- PDF files are processed page by page with extracted text, table parsing (`pdfplumber`), and optional OCR for image-heavy PDFs.
- CSV files are converted row by row with header preservation.
- XLSX and XLS files are processed sheet by sheet.
- TXT files are chunked directly as text.
- Markdown files use header-aware chunking to preserve structure.

### Validation and limits

- Unsupported extensions are rejected with a clear error message.
- File size is enforced during streaming upload using `MAX_FILE_SIZE_MB`.
- Batch uploads are limited by `MAX_BATCH_FILES_PER_REQUEST`.
- Chunking, embedding concurrency, and vector upsert batch sizes are configurable for production tuning.
- OCR mode requires Tesseract to be installed on the host and `PDF_OCR_ENABLED=true`.

### Example unsupported-file response

```json
{
  "detail": "Unsupported file type: .doc. Supported types: .csv, .md, .markdown, .pdf, .txt, .xls, .xlsx"
}
```

## File Upload Examples

### cURL

```bash
curl -X POST "http://localhost:8000/ingest" -F "file=@document.pdf"
curl -X POST "http://localhost:8000/ingest" -F "file=@data.csv"
curl -X POST "http://localhost:8000/ingest" -F "file=@spreadsheet.xlsx"
```

### Python

```python
import requests

with open("document.pdf", "rb") as f:
    response = requests.post("http://localhost:8000/ingest", files={"file": f})
    print(response.json())
```

### JavaScript

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

fetch("http://localhost:8000/ingest", {
  method: "POST",
  body: formData
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

### FastAPI docs UI

1. Start the server with `python main.py`
2. Open `http://localhost:8000/docs`
3. Open the `/ingest` endpoint
4. Click Try it out
5. Select a file and execute the request

### Batch upload examples

```bash
for file in *.pdf; do
  curl -X POST "http://localhost:8000/ingest" -F "file=@$file"
done
```

```bash
curl -X POST "http://localhost:8000/ingest/batch" \
  -F "files=@file1.pdf" \
  -F "files=@file2.pdf" \
  -F "files=@file3.csv"
```

For filesystem-based bulk indexing, use CLI:

```bash
python cli.py ingest-dir ./data/uploads --recursive --workers 4
```

## Docker Deployment

### Docker Compose

```bash
cd docker
docker-compose up
```

### Docker build

```bash
docker build -f docker/Dockerfile -t ollama-rag .
docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 ollama-rag
```

Docker starts the RAG application with the expected supporting services and persistent storage volumes.

## Testing

### Automated testing

```bash
python test_rag.py
```

### Sample file generation

```bash
python create_sample_files.py
python create_sample_files.py test
```

Sample files covered by the repository include:

- `sample_data.csv`
- `sample_document.txt`
- `sample_guide.md`

### Manual smoke test

```bash
curl http://localhost:8000/health
curl http://localhost:8000/info
```

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `local` | Runtime environment |
| `DEBUG` | `false` | Debug logging toggle |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service URL |
| `OLLAMA_MODEL` | `mistral` | Generation model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `VECTOR_DB_TYPE` | `chroma` | Vector database type |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Chroma storage path |
| `STORAGE_TYPE` | `local` | Storage backend |
| `API_PORT` | `8000` | API server port |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `TOP_K_RESULTS` | `5` | Retrieval top-k |
| `SIMILARITY_THRESHOLD` | `0.5` | Minimum similarity score |
| `MAX_FILE_SIZE_MB` | `512` | Max single file upload size in MB |
| `MAX_BATCH_FILES_PER_REQUEST` | `1000` | Max files per `/ingest/batch` request |
| `VECTOR_UPSERT_BATCH_SIZE` | `128` | Chroma upsert batch size |
| `EMBEDDING_WORKERS` | `4` | Parallel embedding requests to Ollama |
| `MAX_PARALLEL_FILE_INGESTIONS` | `4` | Parallel file ingestion workers |
| `EMBEDDING_MAX_RETRIES` | `3` | Retry count per embedding request |
| `PDF_EXTRACT_TABLES` | `true` | Enable PDF table extraction |
| `PDF_OCR_ENABLED` | `false` | Enable OCR for PDF images (requires Tesseract) |

### Chroma notes

Chroma is used because it is embedded, easy to run locally, and works well with persistent storage.

## Python 3.13 Notes

The repository includes a quick-action setup path for Python 3.13.8 and related dependency updates.

### Recommended setup

```bash
python setup.py
python verify_installation.py
python main.py
```

### Common fixes

- Reinstall dependencies with `pip install --upgrade -r requirements.txt`
- Recreate the virtual environment if imports fail
- If Chroma import issues appear, reinstall `chromadb` and clear `data/chroma_db`

## Project Structure

```text
ollama-rag-enterprise-demo/
├── cli.py
├── main.py
├── setup.py
├── test_rag.py
├── verify_installation.py
├── requirements.txt
├── README.md
├── data/
├── docker/
├── src/
│   ├── config.py
│   ├── api/
│   ├── rag/
│   └── utils/
└── tests/
```

## Contributing

Thank you for contributing to the project.

### Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Commit with a clear message
6. Push to your fork
7. Open a pull request to this repository

Direct pushes to protected branches in this repository are restricted to the repository owner.

### Development expectations

- Follow Python PEP 8 style
- Use type hints where practical
- Add docstrings to public functions and classes
- Update documentation when behavior changes
- Run `python test_rag.py` and `pytest tests/` when relevant

### Areas for contribution

- Additional vector databases
- More document formats
- Better error handling
- More tests
- UI improvements
- Authentication support

## Repository Access Control

Use these settings in GitHub to ensure only you (the owner) can push to protected branches and everyone else contributes via forks + pull requests.

### 1. Remove direct write access for others

In GitHub:

1. Open your repository
2. Go to Settings -> Collaborators and teams (or Manage access)
3. Remove anyone with Write, Maintain, or Admin unless absolutely required

Note: Public users cannot push unless they have explicit write access. Keeping write access owner-only is the first control.

### 2. Protect main, dev, test, and prod

In GitHub:

1. Go to Settings -> Rules -> Rulesets (recommended) or Branches
2. Create a branch ruleset targeting:
  - `main`
  - `dev`
  - `test`
  - `prod`
3. Enable at minimum:
  - Require a pull request before merging
  - Require status checks to pass (optional but recommended)
  - Block force pushes
  - Block branch deletion
  - Restrict updates / restrict who can push: only your GitHub username
  - Apply restrictions to administrators (so protections are enforced consistently)

### 3. Protect all other branches too

Create a second ruleset with branch target `*` and restrict updates so only your GitHub username can push.

This ensures new branches are also owner-only by default.

### 4. Contribution model for others

Contributors should:

1. Fork this repository
2. Push changes to their fork
3. Open a pull request from fork -> this repository

This repository does not accept direct pushes from non-owners.

## Troubleshooting

### Ollama is not reachable

- Confirm the Ollama service is running
- Check `OLLAMA_BASE_URL`
- Verify the models have been pulled

### Upload returns unsupported file type

- Use one of the supported extensions: `.pdf`, `.csv`, `.xlsx`, `.xls`, `.txt`, `.md`, `.markdown`
- Confirm the filename extension matches the actual file type

### CSV files appear empty

- Make sure the first row contains headers
- Use UTF-8 encoding
- Check the delimiter is comma-separated

### Excel files fail to process

- Try `.xlsx` if possible
- Remove sheet protection or unusual formatting
- Confirm the workbook is not corrupted

### Import or dependency issues

- Reinstall dependencies with `pip install -r requirements.txt`
- Recreate the virtual environment
- Run `python verify_installation.py`

## Reference Notes

The following standalone markdown guides were consolidated into this README:

- Quick start instructions
- Quick reference commands
- File upload guide
- Supported file formats guide
- File format implementation summary
- Multi-format summary
- Python 3.13 setup notes
- Contribution guidelines
- Sample file usage notes

## Status

- Complete and tested
- Multi-format ingestion enabled
- README is now the single markdown documentation file in the workspace
