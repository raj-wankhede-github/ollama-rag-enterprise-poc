"""
Main entry point for testing the RAG system via CLI.
"""

import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import config
from src.utils.logger import get_logger
from src.rag.pipeline import RAGPipeline

logger = get_logger(__name__, config.log_level)


def ingest_command(args):
    """Handle ingest command"""
    pipeline = RAGPipeline()
    
    file_path = Path(args.file)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return 1
    
    logger.info(f"Ingesting file: {file_path}")
    result = pipeline.ingest_document(str(file_path))
    
    if result["success"]:
        logger.info(f"[OK] Success: {result['message']}")
        return 0
    else:
        logger.error(f"[FAIL] Failed: {result.get('error', 'Unknown error')}")
        return 1


def query_command(args):
    """Handle query command"""
    pipeline = RAGPipeline()
    
    query = args.query
    logger.info(f"Processing query: {query}")
    
    result = pipeline.query(query, stream=args.stream)
    
    if args.stream:
        print("\n" + "=" * 80)
        print(f"Query: {result['query']}")
        print("=" * 80)
        print("Response:")
        for chunk in result["generator"]:
            print(chunk, end="", flush=True)
        print("\n" + "=" * 80)
        if result["sources"]:
            print(f"\nSources ({len(result['sources'])} found):")
            for i, source in enumerate(result["sources"], 1):
                print(f"  {i}. {source.get('source', 'Unknown')}")
        else:
            print("No sources found.")
        print("=" * 80 + "\n")
        return 0
    
    print("\n" + "="*80)
    print(f"Query: {result['query']}")
    print("="*80)
    print(f"Response:\n{result['response']}")
    print("="*80)
    
    if result['sources']:
        print(f"\nSources ({len(result['sources'])} found):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source.get('source', 'Unknown')}")
    else:
        print("No sources found.")
    
    print("="*80 + "\n")
    return 0


def ingest_dir_command(args):
    """Handle ingest-dir command for large-scale ingestion."""
    pipeline = RAGPipeline()
    input_dir = Path(args.directory)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Directory not found: {input_dir}")
        return 1
    
    supported_ext = {".pdf", ".txt", ".csv", ".xlsx", ".xls", ".md", ".markdown"}
    pattern = "**/*" if args.recursive else "*"
    files = [p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() in supported_ext]
    
    if not files:
        logger.warning("No supported files found to ingest.")
        return 0
    
    logger.info(f"Found {len(files)} files for ingestion")
    workers = max(1, min(args.workers, len(files)))
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(pipeline.ingest_document, str(file_path)): file_path for file_path in files}
        for future in as_completed(futures):
            file_path = futures[future]
            result = future.result()
            if result.get("success"):
                successful += 1
                logger.info(f"[OK] {file_path.name}: {result.get('message', 'Ingested')}")
            else:
                failed += 1
                logger.error(f"[FAIL] {file_path.name}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"Ingestion summary: {successful} succeeded, {failed} failed, total {len(files)}")
    return 0 if failed == 0 else 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Ollama RAG Enterprise Demo - CLI Interface"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document")
    ingest_parser.add_argument("file", help="Path to file to ingest")
    ingest_parser.set_defaults(func=ingest_command)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("query", help="Query string")
    query_parser.add_argument("--stream", action="store_true", help="Stream response")
    query_parser.set_defaults(func=query_command)

    # Ingest directory command
    ingest_dir_parser = subparsers.add_parser("ingest-dir", help="Ingest all supported files from a directory")
    ingest_dir_parser.add_argument("directory", help="Directory containing files to ingest")
    ingest_dir_parser.add_argument("--recursive", action="store_true", help="Scan directories recursively")
    ingest_dir_parser.add_argument(
        "--workers",
        type=int,
        default=config.max_parallel_file_ingestions,
        help=f"Parallel ingestion workers (default: {config.max_parallel_file_ingestions})"
    )
    ingest_dir_parser.set_defaults(func=ingest_dir_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
