"""LangGraph node functions for Agent 1."""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.generation.agent1.state import Agent1State
from src.generation.agent1.validators import (
    CEILING_MAP,
    DEFAULT_CEILING,
    MAX_CANDIDATES,
    VALID_DIFFICULTIES,
    VALID_DOMAINS,
    _enforce_ceiling,
    _is_near_duplicate,
    _is_structural,
    _parse_classification,
    _parse_json_array,
)
from src.generation.prompts.agent1_prompts import GEN_QUESTIONS_HUMAN, GEN_QUESTIONS_SYSTEM

logger = logging.getLogger(__name__)


def make_nodes(llm: ChatOpenAI) -> Tuple:
    """Create all Agent 1 graph nodes bound to the same LLM instance."""

    def fetch_chunk(state: Agent1State) -> Dict:
        idx = state["chunk_index"]
        chunk = state["chunks"][idx]
        logger.info(
            "[A1] fetch_chunk %s/%s page=%s doc=%s",
            idx + 1,
            len(state["chunks"]),
            chunk["metadata"].get("page_no", "?"),
            state.get("doc_ref", state.get("doc_id", "doc_001")),
        )
        return {"current_chunk": chunk, "raw_questions": []}

    def gen_questions(state: Agent1State) -> Dict:
        chunk = state["current_chunk"]
        doc_id = state.get("doc_id", "doc_001")
        doc_ref = state.get("doc_ref", doc_id)
        page_no = chunk["metadata"].get("page_no", "?")

        system_msg = GEN_QUESTIONS_SYSTEM.format(
            chunk_id=chunk["id"],
            max_candidates=MAX_CANDIDATES,
        )
        human_msg = GEN_QUESTIONS_HUMAN.format(
            chunk_id=chunk["id"],
            page_no=page_no,
            doc_id=doc_id,
            doc_ref=doc_ref,
            chunk_text=chunk["text"],
            max_candidates=MAX_CANDIDATES,
        )

        response = llm.invoke(
            [
                SystemMessage(content=system_msg),
                HumanMessage(content=human_msg),
            ]
        )

        raw_text = getattr(response, "content", "") or ""
        questions = _parse_json_array(raw_text)
        classification = _parse_classification(raw_text)

        for question in questions:
            if question.get("chunk_type", "").upper() not in CEILING_MAP:
                question["chunk_type"] = classification["chunk_type"]
            if question.get("domain", "").lower() not in VALID_DOMAINS:
                question["domain"] = classification["domain"]

        logger.info("[A1] gen_questions -> %s parsed for chunk %s", len(questions), chunk["id"])
        return {"raw_questions": questions}

    def save_accepted(state: Agent1State) -> Dict:
        chunk = state["current_chunk"]
        chunk_id = chunk["id"]
        page_no = chunk["metadata"].get("page_no")
        doc_id = state.get("doc_id", "doc_001")
        doc_ref = state.get("doc_ref", doc_id)

        newly: List[Dict] = []
        seen_texts = [q["question"] for q in state["accepted_questions"]]
        scenario_count = 0
        rejected_counts = {
            "structural": 0,
            "near_dup": 0,
            "bad_difficulty": 0,
            "ceiling_violation": 0,
            "empty": 0,
        }

        for question in state["raw_questions"]:
            q_text = question.get("question", "").strip()
            difficulty = question.get("difficulty", "basic").strip().lower()
            domain = question.get("domain", "aircraft_systems").strip().lower()
            chunk_type = question.get("chunk_type", "MIXED").upper()
            is_scenario = bool(question.get("is_scenario", False))

            if not q_text:
                rejected_counts["empty"] += 1
                continue
            if _is_structural(q_text):
                rejected_counts["structural"] += 1
                continue
            if _is_near_duplicate(q_text, seen_texts):
                rejected_counts["near_dup"] += 1
                continue

            if difficulty not in VALID_DIFFICULTIES:
                rejected_counts["bad_difficulty"] += 1
                difficulty = "basic"

            enforced = _enforce_ceiling(difficulty, chunk_type)
            if enforced is None:
                rejected_counts["ceiling_violation"] += 1
                difficulty = max(
                    CEILING_MAP.get(chunk_type, DEFAULT_CEILING),
                    key=lambda level: ["basic", "intermediate", "advanced"].index(level),
                )

            if domain not in VALID_DOMAINS:
                domain = "aircraft_systems"

            record = {
                "id": str(uuid.uuid4()),
                "question": q_text,
                "difficulty": difficulty,
                "domain": domain,
                "chunk_type": chunk_type,
                "is_scenario": is_scenario,
                "chunk_ids": question.get("chunk_ids", [chunk_id]),
                "source_page": page_no,
                "doc_id": doc_id,
                "doc_ref": doc_ref,
            }
            newly.append(record)
            seen_texts.append(q_text)
            if is_scenario:
                scenario_count += 1

        if newly:
            path = Path(state["output_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for record in newly:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_new = len(newly)
        if total_new:
            ratio = scenario_count / total_new
            logger.info(
                "[A1] chunk %s scenario coverage %.1f%% (%s/%s)",
                chunk_id,
                ratio * 100,
                scenario_count,
                total_new,
            )

        updated = state["accepted_questions"] + newly
        logger.info(
            "[A1] chunk %s: +%s accepted, %s rejected, total=%s",
            chunk_id,
            total_new,
            sum(rejected_counts.values()),
            len(updated),
        )

        return {
            "accepted_questions": updated,
            "chunk_index": state["chunk_index"] + 1,
        }

    return fetch_chunk, gen_questions, save_accepted
