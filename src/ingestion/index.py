"""Document ingestion and indexing pipeline."""

import logging
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from src.config import ConfigLoader, config as default_config
from src.infrastructure.chroma_store import ChromaStore
from src.ingestion.chunker import get_chunker
from src.ingestion.converter import get_document_converter

logger = logging.getLogger(__name__)
DEFAULT_LOG_FILE = Path("logs") / "index.log"


def setup_logging(level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None) -> None:
    """Configure module-level logging handlers."""
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.setLevel(level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def set_log_level(level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None) -> None:
    setup_logging(level=level, log_file=log_file)
    logger.info("Log level set to: %s", logging.getLevelName(level))


def dump_debug(file_path: Path, docs: List[Document], debug_dir: Path) -> None:
    """Write markdown and chunk debug dumps for a processed file."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = file_path.stem

    md_path = debug_dir / f"{stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {file_path.name}\n\n")
        for doc in docs:
            f.write(doc.page_content)
            f.write("\n\n")

    chunks_path = debug_dir / f"{stem}.chunks"
    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write(f"# Chunks for {file_path.name} ({len(docs)} total)\n\n")
        for i, doc in enumerate(docs, 1):
            sep = "-" * 60
            f.write(f"{sep}\n")
            f.write(
                f"## Chunk {i} | page {doc.metadata.get('page_no', '?')} "
                f"| {len(doc.page_content)} chars\n"
            )
            f.write(f"{sep}\n")
            f.write(doc.page_content)
            f.write("\n\n")

    logger.info("[debug] markdown -> %s", md_path)
    logger.info("[debug] chunks   -> %s", chunks_path)


def process_file(
    file_path: Union[str, Path],
    converter,
    chunker,
    namespace: str,
    debug_dir: Optional[Path] = None,
) -> List[Document]:
    """Process one file into chunk-level ``Document`` objects ready for indexing."""
    file_path = Path(file_path)

    loader = DoclingLoader(
        file_path=file_path,
        converter=converter,
        chunker=chunker,
        export_type=ExportType.DOC_CHUNKS,
    )

    docs = loader.load()
    processed: List[Document] = []

    for doc in docs:
        meta = doc.metadata
        cleaned_meta = {
            "source": str(meta["source"]),
            "page_no": meta["dl_meta"]["doc_items"][0]["prov"][0]["page_no"],
            "namespace": namespace,
        }
        processed.append(Document(page_content=doc.page_content, metadata=cleaned_meta))

    if debug_dir is not None:
        dump_debug(file_path, processed, debug_dir)

    return processed


def _build_store(cfg: ConfigLoader, namespace: str, drop_existing: bool) -> ChromaStore:
    return ChromaStore(
        persist_directory=cfg.get("database", "persist_directory", default="chroma_db"),
        collection_name=cfg.get("database", "collection_name", default="collection_demo"),
        embed_model=cfg.get("model", "embeddings", default="BAAI/bge-small-en-v1.5"),
        drop_old=drop_existing,
        namespace=namespace,
        device=cfg.get("database", "chroma_device", default="cpu"),
    )


def process_and_index_file(
    file_path: Union[str, Path],
    drop_existing: bool = False,
    namespace: Optional[str] = None,
    cfg: Optional[ConfigLoader] = None,
    debug_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Process one file and index all chunks into ChromaDB."""
    file_path = Path(file_path)
    debug_dir = Path(debug_dir) if debug_dir else None

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    cfg = cfg or default_config
    namespace = namespace or cfg.get("database", "namespace", default="CaseDoneDemo")

    converter = get_document_converter(cfg=cfg)
    chunker = get_chunker(cfg=cfg)
    store = _build_store(cfg=cfg, namespace=namespace, drop_existing=drop_existing)

    docs = process_file(file_path, converter, chunker, namespace, debug_dir=debug_dir)
    logger.info("Extracted %s chunks from %s", len(docs), file_path.name)

    if docs:
        ids = store.add_documents(docs)
        logger.info("Indexing complete. Stored %s chunks", len(ids))
    else:
        logger.warning("No chunks extracted; nothing indexed")


def process_and_index_directory(
    directory_path: Union[str, Path],
    drop_existing: bool = False,
    namespace: Optional[str] = None,
    file_extensions: Optional[List[str]] = None,
    cfg: Optional[ConfigLoader] = None,
    debug_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Process every supported file in a directory and index all chunks."""
    directory_path = Path(directory_path)
    debug_dir = Path(debug_dir) if debug_dir else None
    cfg = cfg or default_config

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory_path} does not exist")

    namespace = namespace or cfg.get("database", "namespace", default="CaseDoneDemo")
    file_extensions = file_extensions or cfg.get(
        "document", "supported_file_types", default=[".pdf"]
    )

    files: List[Path] = []
    for ext in file_extensions:
        files.extend(directory_path.glob(f"*{ext}"))

    if not files:
        logger.warning("No files with extensions %s found in %s", file_extensions, directory_path)
        return

    logger.info("Found %s files to process", len(files))

    converter = get_document_converter(cfg=cfg)
    chunker = get_chunker(cfg=cfg)
    store = _build_store(cfg=cfg, namespace=namespace, drop_existing=drop_existing)

    all_docs: List[Document] = []
    for file in files:
        if file.name == ".DS_Store":
            continue
        try:
            docs = process_file(file, converter, chunker, namespace, debug_dir=debug_dir)
            all_docs.extend(docs)
            logger.info("Processed %s -> %s chunks", file.name, len(docs))
        except Exception as exc:
            logger.error("Error processing %s: %s", file, exc, exc_info=True)

    if all_docs:
        ids = store.add_documents(all_docs)
        logger.info("Indexing complete. Stored %s chunks", len(ids))
    else:
        logger.warning("No documents to index")


setup_logging(log_file=DEFAULT_LOG_FILE)
