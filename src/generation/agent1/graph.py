"""Agent 1 graph wiring and execution entrypoint."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.config import ConfigLoader, config as default_config
from src.generation.agent1.nodes import make_nodes
from src.generation.agent1.state import Agent1State
from src.generation.agent1.validators import SCENARIO_FLOOR
from src.infrastructure.chroma_store import ChromaStore
from src.infrastructure.llm_client import get_llm

logger = logging.getLogger(__name__)


def route_next(state: Agent1State) -> str:
    return "fetch_chunk" if state["chunk_index"] < len(state["chunks"]) else END


class QuestionGenerationGraph:
    """LangGraph-powered question generation over indexed document chunks."""

    def __init__(
        self,
        store: Optional[ChromaStore] = None,
        output_path: Optional[str] = None,
        namespace: Optional[str] = None,
        doc_id: Optional[str] = None,
        doc_ref: Optional[str] = None,
        checkpoint_db: str = "checkpoints/agent1.sqlite",
        cfg: Optional[ConfigLoader] = None,
    ) -> None:
        self.cfg = cfg or default_config

        self.store = store or ChromaStore(
            persist_directory=self.cfg.get("database", "persist_directory", default="chroma_db"),
            collection_name=self.cfg.get("database", "collection_name", default="collection_demo"),
            embed_model=self.cfg.get("model", "embeddings", default="BAAI/bge-small-en-v1.5"),
            namespace=self.cfg.get("database", "namespace", default="CaseDoneDemo"),
            device=self.cfg.get("database", "chroma_device", default="cpu"),
        )

        self.namespace = namespace or self.cfg.get("database", "namespace", default=None)
        self.doc_id = doc_id or self.cfg.get("dataset_gen", "doc_id", default="doc_001")
        self.doc_ref = self.cfg.resolve_doc_ref(self.doc_id, override=doc_ref)
        self.output_path = output_path or self.cfg.get(
            "dataset_gen",
            "accepted_questions_path",
            default="output/accepted_questions.jsonl",
        )
        self._db_path = checkpoint_db

        llm = get_llm(self.cfg, timeout_default=120)
        fetch_chunk, gen_questions, save_accepted = make_nodes(llm=llm)

        builder = StateGraph(Agent1State)
        builder.add_node("fetch_chunk", fetch_chunk)
        builder.add_node("gen_questions", gen_questions)
        builder.add_node("save_accepted", save_accepted)

        builder.add_edge(START, "fetch_chunk")
        builder.add_edge("fetch_chunk", "gen_questions")
        builder.add_edge("gen_questions", "save_accepted")
        builder.add_conditional_edges("save_accepted", route_next)

        self._builder = builder

    def run(self, thread_id: Optional[str] = None) -> List[Dict]:
        """Run Agent 1 across all indexed chunks and return accepted questions."""
        thread_id = thread_id or f"agent1-{int(time.time())}"

        chunks = self._fetch_all_chunks()
        if not chunks:
            logger.warning("[A1] No chunks found in ChromaDB. Aborting")
            return []

        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

        initial_state: Agent1State = {
            "chunks": chunks,
            "chunk_index": 0,
            "current_chunk": None,
            "raw_questions": [],
            "accepted_questions": [],
            "output_path": self.output_path,
            "namespace": self.namespace,
            "doc_id": self.doc_id,
            "doc_ref": self.doc_ref,
        }

        cfg = {"configurable": {"thread_id": thread_id}}

        with SqliteSaver.from_conn_string(self._db_path) as checkpointer:
            graph = self._builder.compile(checkpointer=checkpointer)
            final = graph.invoke(initial_state, cfg)

        accepted = final.get("accepted_questions", [])
        self._log_summary(accepted)
        return accepted

    def _fetch_all_chunks(self) -> List[Dict]:
        result = self.store.collection.get(
            include=["documents", "metadatas"],
            where={"namespace": {"$eq": self.namespace}} if self.namespace else None,
        )
        chunks = [
            {"id": cid, "text": doc, "metadata": meta}
            for doc, meta, cid in zip(result["documents"], result["metadatas"], result["ids"])
        ]
        chunks.sort(key=lambda item: item["metadata"].get("page_no", 0))
        return chunks

    def _log_summary(self, accepted: List[Dict]) -> None:
        if not accepted:
            logger.warning("[A1] Run complete with 0 accepted questions")
            return

        total = len(accepted)
        by_difficulty: Dict[str, int] = {}
        by_domain: Dict[str, int] = {}
        scenario_count = 0

        for question in accepted:
            difficulty = question.get("difficulty", "unknown")
            by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1

            domain = question.get("domain", "unknown")
            by_domain[domain] = by_domain.get(domain, 0) + 1

            if question.get("is_scenario"):
                scenario_count += 1

        logger.info("[A1] Accepted questions: %s", total)
        logger.info("[A1] Difficulty breakdown: %s", by_difficulty)
        logger.info("[A1] Domain breakdown: %s", by_domain)
        logger.info(
            "[A1] Scenario coverage: %s/%s (%.1f%%)",
            scenario_count,
            total,
            scenario_count / total * 100,
        )

        if scenario_count / total < SCENARIO_FLOOR:
            logger.warning("[A1] Scenario coverage below floor of %.0f%%", SCENARIO_FLOOR * 100)
