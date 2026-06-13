"""CLI wrapper for indexing data into ChromaDB."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.index import process_and_index_directory, process_and_index_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Process and index files (Docling -> ChromaDB)")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", metavar="FILE", help="Index a single file")
    source.add_argument("--dir", metavar="DIR", help="Index all files in a directory")

    parser.add_argument(
        "--drop", action="store_true", help="Drop existing collection before indexing"
    )
    parser.add_argument("--namespace", help="Namespace tag for indexed documents")
    parser.add_argument("--debug", metavar="DEBUG_DIR", help="Dump .md and .chunks debug artifacts")

    args = parser.parse_args()

    if args.file:
        process_and_index_file(
            file_path=args.file,
            drop_existing=args.drop,
            namespace=args.namespace,
            debug_dir=args.debug,
        )
    else:
        process_and_index_directory(
            directory_path=args.dir,
            drop_existing=args.drop,
            namespace=args.namespace,
            debug_dir=args.debug,
        )


if __name__ == "__main__":
    main()
