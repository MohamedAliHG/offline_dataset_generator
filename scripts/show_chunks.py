"""Inspect chunks stored in ChromaDB."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.infrastructure.chroma_store import ChromaStore


def _build_store() -> ChromaStore:
    return ChromaStore(
        persist_directory=config.get("database", "persist_directory", default="chroma_db"),
        collection_name=config.get("database", "collection_name", default="collection_demo"),
        embed_model=config.get("model", "embeddings", default="BAAI/bge-small-en-v1.5"),
        namespace=config.get("database", "namespace", default="CaseDoneDemo"),
        device=config.get("database", "chroma_device", default="cpu"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show first N chunks from ChromaDB collection")
    parser.add_argument("--limit", type=int, default=5, help="Number of chunks to display")
    parser.add_argument("--namespace", default=None, help="Namespace filter override")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Maximum chars per chunk to print (ignored with --full)",
    )
    parser.add_argument("--full", action="store_true", help="Print full chunk text")

    args = parser.parse_args()

    store = _build_store()
    namespace = (
        args.namespace
        if args.namespace is not None
        else config.get("database", "namespace", default=None)
    )

    result = store.collection.get(
        include=["documents", "metadatas"],
        where={"namespace": {"$eq": namespace}} if namespace else None,
    )

    ids = result.get("ids", [])
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])

    chunks = [
        {"id": chunk_id, "text": doc or "", "metadata": meta or {}}
        for chunk_id, doc, meta in zip(ids, docs, metas)
    ]
    chunks.sort(key=lambda item: (item["metadata"].get("page_no", 0), item["id"]))

    if not chunks:
        print("No chunks found in collection.")
        return

    limit = max(1, args.limit)
    selected = chunks[:limit]

    print(f"Collection: {store.collection_name}")
    print(f"Namespace: {namespace}")
    print(f"Total chunks matched: {len(chunks)}")
    print(f"Showing first {len(selected)} chunk(s)")

    for i, chunk in enumerate(selected, 1):
        page_no = chunk["metadata"].get("page_no", "?")
        source = chunk["metadata"].get("source", "?")
        text = chunk["text"]
        if not args.full and len(text) > args.max_chars:
            text = text[: args.max_chars].rstrip() + "\n...[truncated]"

        print("\n" + "=" * 80)
        print(f"Chunk {i}")
        print(f"id: {chunk['id']}")
        print(f"page: {page_no}")
        print(f"source: {source}")
        print("-" * 80)
        print(text)


if __name__ == "__main__":
    main()
