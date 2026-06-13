"""Agent 2 graph wiring and execution entrypoint."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.config import ConfigLoader, config as default_config
from src.export.exporters import _export_csv, _export_json
from src.generation.agent2.nodes import make_nodes
from src.generation.agent2.state import Agent2State
from src.generation.agent2.validators import (
    CHOSEN_REJECTED_MAX_SIMILARITY,
    DEFAULT_JUDGE_MARGIN,
    DEFAULT_MIN_REJECTED_SCORE,
    DEFAULT_TOP_K,
    SCENARIO_FLOOR,
)
from src.infrastructure.chroma_store import ChromaStore
from src.infrastructure.llm_client import get_llm

logger = logging.getLogger(__name__)


def route_next(state: Agent2State) -> str:
    return "fetch_question" if state["question_index"] < len(state["questions"]) else END


class AnswerGenerationGraph:
    """LangGraph-powered chosen/rejected answer pair generation."""

    def __init__(
        self,
        store: Optional[ChromaStore] = None,
        questions_path: Optional[str] = None,
        output_jsonl: Optional[str] = None,
        output_json: Optional[str] = None,
        output_csv: Optional[str] = None,
        doc_id: Optional[str] = None,
        doc_ref: Optional[str] = None,
        top_k: Optional[int] = None,
        num_candidates: Optional[int] = None,
        checkpoint_db: str = "checkpoints/agent2.sqlite",
        cfg: Optional[ConfigLoader] = None,
    ) -> None:
        self.cfg = cfg or default_config

        self.questions_path = questions_path or self.cfg.get(
            "dataset_gen", "accepted_questions_path", default="output/accepted_questions.jsonl"
        )
        self.output_jsonl = output_jsonl or self.cfg.get(
            "dataset_gen", "output_jsonl", default="output/dataset_pairs.jsonl"
        )
        self.output_json = output_json or self.cfg.get(
            "dataset_gen", "output_json", default="output/dataset.json"
        )
        self.output_csv = output_csv or self.cfg.get(
            "dataset_gen", "output_csv", default="output/dataset.csv"
        )
        self.doc_id = doc_id or self.cfg.get("dataset_gen", "doc_id", default="doc_001")
        self.doc_ref = self.cfg.resolve_doc_ref(self.doc_id, override=doc_ref)
        self.top_k = top_k or int(
            self.cfg.get("dataset_gen", "top_k_retrieval", default=DEFAULT_TOP_K)
        )
        self.num_candidates = num_candidates or int(
            self.cfg.get("dataset_gen", "num_answer_candidates", default=4)
        )
        self.judge_selection_margin = int(
            self.cfg.get("dataset_gen", "judge_selection_margin", default=DEFAULT_JUDGE_MARGIN)
        )
        self.judge_min_rejected_score = int(
            self.cfg.get(
                "dataset_gen", "judge_min_rejected_score", default=DEFAULT_MIN_REJECTED_SCORE
            )
        )
        self._db_path = checkpoint_db

        self.store = store or ChromaStore(
            persist_directory=self.cfg.get("database", "persist_directory", default="chroma_db"),
            collection_name=self.cfg.get("database", "collection_name", default="collection_demo"),
            embed_model=self.cfg.get("model", "embeddings", default="BAAI/bge-small-en-v1.5"),
            namespace=self.cfg.get("database", "namespace", default="CaseDoneDemo"),
            device=self.cfg.get("database", "chroma_device", default="cpu"),
        )

        gen_llm = get_llm(self.cfg, timeout_default=180, section="llm")
        judge_llm = get_llm(self.cfg, timeout_default=180, section="judge_llm")
        (
            fetch_question,
            retrieve_context,
            generate_candidates,
            judge_candidates,
            select_pair,
            validate_pair,
            save_pair,
        ) = make_nodes(gen_llm=gen_llm, judge_llm=judge_llm, store=self.store)

        builder = StateGraph(Agent2State)
        builder.add_node("fetch_question", fetch_question)
        builder.add_node("retrieve_context", retrieve_context)
        builder.add_node("generate_candidates", generate_candidates)
        builder.add_node("judge_candidates", judge_candidates)
        builder.add_node("select_pair", select_pair)
        builder.add_node("validate_pair", validate_pair)
        builder.add_node("save_pair", save_pair)

        builder.add_edge(START, "fetch_question")
        builder.add_edge("fetch_question", "retrieve_context")
        builder.add_edge("retrieve_context", "generate_candidates")
        builder.add_edge("generate_candidates", "judge_candidates")
        builder.add_edge("judge_candidates", "select_pair")
        builder.add_edge("select_pair", "validate_pair")
        builder.add_edge("validate_pair", "save_pair")
        builder.add_conditional_edges("save_pair", route_next)

        self._builder = builder

    def run(self, thread_id: Optional[str] = None) -> List[Dict]:
        """Run Agent 2 over Agent 1 output and export JSON/CSV datasets."""
        thread_id = thread_id or f"agent2-{int(time.time())}"

        questions = self._load_questions()
        if not questions:
            logger.warning("[A2] No questions loaded. Aborting")
            return []

        out_path = Path(self.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")

        initial_state: Agent2State = {
            "questions": questions,
            "question_index": 0,
            "current_question": None,
            "retrieved_chunks": [],
            "candidate_answers": [],
            "judged_candidates": [],
            "chosen": "",
            "rejected": "",
            "selection_method": "",
            "validated_pair": None,
            "accepted_pairs": [],
            "output_jsonl": self.output_jsonl,
            "output_json": self.output_json,
            "output_csv": self.output_csv,
            "doc_id": self.doc_id,
            "doc_ref": self.doc_ref,
            "top_k": self.top_k,
            "num_candidates": self.num_candidates,
            "judge_selection_margin": self.judge_selection_margin,
            "judge_min_rejected_score": self.judge_min_rejected_score,
        }

        cfg = {"configurable": {"thread_id": thread_id}}

        with SqliteSaver.from_conn_string(self._db_path) as checkpointer:
            graph = self._builder.compile(checkpointer=checkpointer)
            final = graph.invoke(initial_state, cfg)

        pairs = final.get("accepted_pairs", [])
        _export_json(pairs, self.output_json)
        _export_csv(pairs, self.output_csv)
        self._log_summary(pairs, len(questions))
        return pairs

    def _load_questions(self) -> List[Dict]:
        path = Path(self.questions_path)
        if not path.exists():
            logger.error("[A2] Questions file not found: %s", path)
            return []

        questions: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("[A2] Skipping malformed line %s: %s", line_no, exc)

        questions.sort(key=lambda item: (item.get("source_page") or 0, item.get("id", "")))
        return questions

    def _log_summary(self, pairs: List[Dict], total_questions: int) -> None:
        count = len(pairs)
        drop_rate = 1 - (count / total_questions) if total_questions else 0

        by_diff: Dict[str, int] = {}
        by_domain: Dict[str, int] = {}
        by_page: Dict[str, int] = {}
        scenario_count = 0
        similarities: List[float] = []
        chosen_scores: List[int] = []
        rejected_scores: List[int] = []

        for pair in pairs:
            difficulty = pair.get("difficulty", "unknown")
            by_diff[difficulty] = by_diff.get(difficulty, 0) + 1

            domain = pair.get("domain", "unknown")
            by_domain[domain] = by_domain.get(domain, 0) + 1

            page = str(pair.get("_source_page", "?"))
            by_page[page] = by_page.get(page, 0) + 1

            if pair.get("_is_scenario"):
                scenario_count += 1

            sim = pair.get("_similarity")
            if isinstance(sim, (float, int)):
                similarities.append(float(sim))

            judge_scores = pair.get("_judge_scores", {})
            chosen_score = judge_scores.get("chosen", {}).get("total_score")
            rejected_score = judge_scores.get("rejected", {}).get("total_score")
            if isinstance(chosen_score, int):
                chosen_scores.append(chosen_score)
            if isinstance(rejected_score, int):
                rejected_scores.append(rejected_score)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        avg_chosen_score = sum(chosen_scores) / len(chosen_scores) if chosen_scores else 0.0
        avg_rejected_score = sum(rejected_scores) / len(rejected_scores) if rejected_scores else 0.0

        logger.info("[A2] Questions processed: %s", total_questions)
        logger.info("[A2] Pairs accepted: %s", count)
        logger.info("[A2] Drop rate: %.1f%%", drop_rate * 100)
        logger.info(
            "[A2] Avg chosen/rejected similarity: %.3f (target < %.2f)",
            avg_similarity,
            CHOSEN_REJECTED_MAX_SIMILARITY,
        )
        logger.info(
            "[A2] Avg judge scores chosen=%.2f rejected=%.2f", avg_chosen_score, avg_rejected_score
        )

        if count:
            scenario_ratio = scenario_count / count
            logger.info(
                "[A2] Scenario coverage: %s/%s (%.1f%%)",
                scenario_count,
                count,
                scenario_ratio * 100,
            )
            if scenario_ratio < SCENARIO_FLOOR:
                logger.warning("[A2] Scenario coverage below floor of %.0f%%", SCENARIO_FLOOR * 100)

        logger.info("[A2] Difficulty breakdown: %s", by_diff)
        logger.info("[A2] Domain breakdown: %s", by_domain)
        logger.info("[A2] Page coverage: %s", by_page)
