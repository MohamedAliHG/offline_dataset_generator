"""Top-level pipeline orchestration (Agent 1 -> Agent 2)."""

import logging
import sys
from pathlib import Path
from typing import Optional, Type

from src.config import ConfigLoader, config as default_config

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    has_console = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    )
    if not has_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    pipeline_log_path = str((Path("logs") / "pipeline.log").resolve())
    has_pipeline_file = any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == Path(pipeline_log_path)
        for handler in root_logger.handlers
    )
    if not has_pipeline_file:
        file_handler = logging.FileHandler(pipeline_log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)


def run_pipeline(
    run_agent1: bool = True,
    run_agent2: bool = True,
    namespace: Optional[str] = None,
    questions_path: Optional[str] = None,
    output_jsonl: Optional[str] = None,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
    doc_id: Optional[str] = None,
    doc_ref: Optional[str] = None,
    top_k: Optional[int] = None,
    thread_id: Optional[str] = None,
    cfg: Optional[ConfigLoader] = None,
    agent1_graph_cls: Optional[Type] = None,
    agent2_graph_cls: Optional[Type] = None,
) -> None:
    """Run Agent 1 and/or Agent 2 in sequence."""
    _setup_logging()
    cfg = cfg or default_config

    from src.infrastructure.chroma_store import ChromaStore

    store = ChromaStore(
        persist_directory=cfg.get("database", "persist_directory", default="chroma_db"),
        collection_name=cfg.get("database", "collection_name", default="collection_demo"),
        embed_model=cfg.get("model", "embeddings", default="BAAI/bge-small-en-v1.5"),
        namespace=cfg.get("database", "namespace", default="CaseDoneDemo"),
        device=cfg.get("database", "chroma_device", default="cpu"),
    )

    if store.count() == 0:
        logger.error(
            "ChromaDB collection is empty. Index documents first, for example:\n"
            "python scripts/index.py --dir data/raw"
        )
        sys.exit(1)

    if agent1_graph_cls is None:
        from src.generation.agent1.graph import QuestionGenerationGraph

        agent1_graph_cls = QuestionGenerationGraph

    if agent2_graph_cls is None:
        from src.generation.agent2.graph import AnswerGenerationGraph

        agent2_graph_cls = AnswerGenerationGraph

    a1_thread = f"{thread_id}-agent1" if thread_id else "agent1-run-1"
    a2_thread = f"{thread_id}-agent2" if thread_id else "agent2-run-1"

    if run_agent1:
        logger.info("=" * 60)
        logger.info("AGENT 1 - Question Generation")
        logger.info("=" * 60)

        agent1 = agent1_graph_cls(
            store=store,
            output_path=questions_path,
            namespace=namespace,
            doc_id=doc_id,
            doc_ref=doc_ref,
            cfg=cfg,
        )
        accepted_questions = agent1.run(thread_id=a1_thread)
        logger.info("Agent 1 complete - %s accepted questions", len(accepted_questions))

    if run_agent2:
        logger.info("=" * 60)
        logger.info("AGENT 2 - Answer Generation")
        logger.info("=" * 60)

        agent2 = agent2_graph_cls(
            store=store,
            questions_path=questions_path,
            output_jsonl=output_jsonl,
            output_json=output_json,
            output_csv=output_csv,
            doc_id=doc_id,
            doc_ref=doc_ref,
            top_k=top_k,
            cfg=cfg,
        )
        accepted_pairs = agent2.run(thread_id=a2_thread)
        logger.info("Pipeline complete - %s accepted QA pairs", len(accepted_pairs))
