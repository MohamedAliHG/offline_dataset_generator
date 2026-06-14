"""LangGraph node functions for Agent 2."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.generation.agent2.state import Agent2State
from src.generation.agent2.strategies import (
  
    _parse_judge_payload,
    _strip_meta_commentary,
)
from src.generation.agent2.validators import (
    CHUNK_TYPE_TOP_K,
    DEFAULT_TOP_K,
    DISTANCE_THRESHOLD,
    _build_source_reference,
    _validate_pair,
)
from src.generation.prompts.agent2_prompts import (
    GEN_CANDIDATE_HUMAN,
    GEN_CANDIDATE_SYSTEM,
    JUDGE_CANDIDATES_HUMAN,
    JUDGE_CANDIDATES_SYSTEM,
)
from src.infrastructure.chroma_store import ChromaStore

logger = logging.getLogger(__name__)
CANDIDATE_LOG_PREVIEW_CHARS = 100


def _format_context(chunks: List[Dict]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, 1):
        page = chunk["metadata"].get("page_no", "?")
        source = chunk["metadata"].get("source", "?")
        chunk_id = chunk.get("id", f"chunk_{index}")
        parts.append(
            f"[{index}] chunk_id={chunk_id} page=p{page} source={source}\n{chunk['text'].strip()}"
        )
    return "\n\n".join(parts)


def _extract_page_ref(chunks: List[Dict]) -> str:
    if not chunks:
        return "?"
    return str(chunks[0]["metadata"].get("page_no", "?"))


def _format_candidates(candidates: List[Dict]) -> str:
    parts = []
    for candidate in candidates:
        parts.append(
            "\n".join(
                [
                    f"candidate_id: {candidate['candidate_id']}",
                    candidate["text"],
                ]
            )
        )
    return "\n\n".join(parts)


def _preview_text(text: str, limit: int = CANDIDATE_LOG_PREVIEW_CHARS) -> str:
    """Return a single-line truncated preview for logging."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def make_nodes(gen_llm: ChatOpenAI, judge_llm: ChatOpenAI, store: ChromaStore) -> Tuple:
    """Create all Agent 2 graph nodes bound to shared dependencies."""

    def fetch_question(state: Agent2State) -> Dict:
        idx = state["question_index"]
        question = state["questions"][idx]
        logger.info(
            "[A2] fetch_question %s/%s id=%s difficulty=%s chunk_type=%s",
            idx + 1,
            len(state["questions"]),
            question.get("id", "?"),
            question.get("difficulty", "?"),
            question.get("chunk_type", "?"),
        )
        return {
            "current_question": question,
            "retrieved_chunks": [],
            "candidate_answers": [],
            "judged_candidates": [],
            "chosen": "",
            "rejected": "",
            "selection_method": "",
            "validated_pair": None,
        }

    def retrieve_context(state: Agent2State) -> Dict:
        question = state["current_question"]
        query_text = question["question"]
        chunk_type = question.get("chunk_type", "MIXED").upper()
        top_k = CHUNK_TYPE_TOP_K.get(chunk_type, state.get("top_k", DEFAULT_TOP_K))

        anchor_ids = question.get("chunk_ids", [])
        anchor_chunks = []

        if anchor_ids:
            try:
                result = store.collection.get(ids=anchor_ids, include=["documents", "metadatas"])
                anchor_chunks = [
                    {"id": chunk_id, "text": doc, "metadata": meta, "distance": 0.0}
                    for chunk_id, doc, meta in zip(
                        result["ids"],
                        result["documents"],
                        result["metadatas"],
                    )
                ]
            except Exception as exc:
                logger.warning("[A2] Anchor retrieval failed: %s", exc)

        aug_needed = max(0, top_k - len(anchor_chunks))    # working correctly

        aug_chunks = []
        if aug_needed > 0:
            try:
                retrieved = store.query_chunks(
                    query=query_text,
                    k=aug_needed + len(anchor_ids) + 2,
                )
                seen_ids = set(anchor_ids)
                candidates = []
                logger.info(
                    "[A2] similarity query: %s anchor_chunk_id=%s",
                    query_text,
                    anchor_chunks[0]["id"] if anchor_chunks else "none",
                )

                for item in retrieved:
                    chunk_id = item["id"]
                    doc = item["text"]
                    meta = item["metadata"]
                    distance = item["distance"]
                    logger.info(
                        "[A2] Retrieved chunk_id=%s text=%s distance=%.4f metadata=%s",
                        chunk_id,
                        doc,
                        distance,
                        meta,
                    )

                    if chunk_id in seen_ids or distance > DISTANCE_THRESHOLD:
                        continue
                    candidates.append(
                        {"id": chunk_id, "text": doc, "metadata": meta, "distance": distance}
                    )
                    seen_ids.add(chunk_id)

                aug_chunks = sorted(candidates, key=lambda item: item["distance"])[:aug_needed]
            except Exception as exc:
                logger.error("[A2] Similarity retrieval failed: %s", exc)

        chunks = anchor_chunks + aug_chunks

        if not chunks:
            try:
                chunks = store.query_chunks(query=query_text, k=1)
            except Exception as exc:
                logger.error("[A2] Fallback retrieval failed: %s", exc)
        
        logger.info(
            "[A2] retrieve_context: %s total chunks (anchor=%s aug=%s) aug_needed=%s",
            len(chunks),
            len(anchor_chunks),
            len(aug_chunks),
            aug_needed
        )
        """
        logger.info(
            "[A2] retrieve_context: %s aug_needed",
            aug_needed,
        )"""
        return {"retrieved_chunks": chunks}

    def generate_candidates(state: Agent2State) -> Dict:
        question = state["current_question"]
        chunks = state["retrieved_chunks"]
        doc_ref = question.get("doc_ref") or state.get("doc_ref", state.get("doc_id", "doc_001"))

        if not chunks:
            fallback = (
                "The provided documentation does not contain sufficient information "
                f"to answer this question. Source: {doc_ref} p?"
            )
            return {
                "candidate_answers": [
                    {
                        "candidate_id": "cand_1",
                        "variant_name": "fallback",
                        "text": fallback,
                    }
                ]
            }

        context = _format_context(chunks)
        page_ref = _extract_page_ref(chunks)

        candidates: List[Dict] = []
        seen_texts = set()
        

        for i in  range(state.get("num_candidates", 4)):
            response = gen_llm.invoke(
                [
                    SystemMessage(
                        content=GEN_CANDIDATE_SYSTEM.format(
                            doc_ref=doc_ref,
                            page_ref=page_ref,
                        )
                    ),
                    HumanMessage(
                        content=GEN_CANDIDATE_HUMAN.format(
                            question=question["question"],
                            difficulty=question.get("difficulty", "basic"),
                            domain=question.get("domain", "aircraft_systems"),
                            context=context,
                            doc_ref=doc_ref,
                            page_ref=page_ref,
                        )
                    ),
                ]
            )
            text = _strip_meta_commentary((getattr(response, "content", "") or "").strip())
            if not text:
                continue
            normalized = text.lower()
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            candidates.append(
                {
                    "candidate_id": f"cand_{i + 1}",
                    "text": text,
                }
            )


        logger.info("[A2] generate_candidates: %s candidate(s)", len(candidates))
        return {"candidate_answers": candidates}


    def judge_candidates(state: Agent2State) -> Dict:
        question = state["current_question"]
        chunks = state["retrieved_chunks"]
        candidates = state["candidate_answers"]

        if len(candidates) < 2:
            logger.warning("[A2] judge_candidates: fewer than 2 candidates available")
            return {"judged_candidates": []}

        context = _format_context(chunks) if chunks else "(no context available)"
        response = judge_llm.invoke(
            [
                SystemMessage(content=JUDGE_CANDIDATES_SYSTEM),
                HumanMessage(
                    content=JUDGE_CANDIDATES_HUMAN.format(
                        question=question["question"],
                        difficulty=question.get("difficulty", "basic"),
                        domain=question.get("domain", "aircraft_systems"),
                        context=context,
                        candidates=_format_candidates(candidates),
                    )
                ),
            ]
        )

        ranked = _parse_judge_payload((getattr(response, "content", "") or "").strip())
        if not ranked:
            logger.warning("[A2] judge_candidates: failed to parse judge output")
            return {"judged_candidates": []}

        candidate_map = {candidate["candidate_id"]: candidate for candidate in candidates}
        judged: List[Dict] = []
        for item in ranked:
            candidate = candidate_map.get(item["candidate_id"])
            if candidate is None:
                continue
            judged.append({**candidate, **item})

        logger.info("[A2] judge_candidates: ranked %s candidate(s)", len(judged))
        for item in judged:
            logger.info(
                "[A2] candidate %s total=%s fact=%s comp=%s trace=%s fit=%s",
                item["candidate_id"],
                item.get("total_score", 0),
                item.get("factuality", 0),
                item.get("completeness", 0),
                item.get("traceability", 0),
                item.get("document_fit", 0),
            )
        return {"judged_candidates": judged}

    def select_pair(state: Agent2State) -> Dict:
        judged = state["judged_candidates"]
        if len(judged) < 2:
            logger.warning("[A2] select_pair: insufficient judged candidates")
            return {"chosen": "", "rejected": "", "selection_method": "judge_failed"}

        ranked = sorted(
            judged,
            key=lambda item: (
                item.get("total_score", 0),
                item.get("factuality", 0),
                item.get("traceability", 0),
            ),
            reverse=True,
        )

        chosen = ranked[0]
        min_margin = state.get("judge_selection_margin", 3)
        min_rejected_score = state.get("judge_min_rejected_score", 8)

        rejected = None
        for candidate in sorted(ranked[1:], key=lambda item: item.get("total_score", 0)):
            score = candidate.get("total_score", 0)
            margin = chosen.get("total_score", 0) - score
            if score >= min_rejected_score and margin >= min_margin:
                rejected = candidate
                break

        if rejected is None:
            logger.warning("[A2] select_pair: no rejected candidate met score/margin thresholds")
            return {"chosen": "", "rejected": "", "selection_method": "judge_margin_failed"}

        logger.info(
            "[A2] select_pair: chosen=%s (%s) rejected=%s (%s)",
            chosen["candidate_id"],
            chosen.get("total_score", 0),
            rejected["candidate_id"],
            rejected.get("total_score", 0),
        )
        logger.info(
            "[A2] chosen preview: %s",
            _preview_text(chosen.get("text", "")),
        )
        logger.info(
            "[A2] rejected preview: %s",
            _preview_text(rejected.get("text", "")),
        )
        return {
            "chosen": chosen["text"],
            "rejected": rejected["text"],
            "selection_method": "judge_ranked_candidates",
        }

    def validate_pair(state: Agent2State) -> Dict:
        question = state["current_question"]
        chunks = state["retrieved_chunks"]
        source_ref = _build_source_reference(chunks) if chunks else "p?"
        doc_ref = question.get("doc_ref") or state.get("doc_ref", state["doc_id"])

        judged_map = {
            item["candidate_id"]: {
                "total_score": item.get("total_score", 0),
                "factuality": item.get("factuality", 0),
                "completeness": item.get("completeness", 0),
                "traceability": item.get("traceability", 0),
                "document_fit": item.get("document_fit", 0),
                "rationale": item.get("rationale", ""),
            }
            for item in state["judged_candidates"]
        }

        chosen_meta = next(
            (item for item in state["judged_candidates"] if item.get("text") == state["chosen"]),
            None,
        )
        rejected_meta = next(
            (item for item in state["judged_candidates"] if item.get("text") == state["rejected"]),
            None,
        )

        judge_scores = {
            "chosen": judged_map.get(chosen_meta["candidate_id"], {}) if chosen_meta else {},
            "rejected": judged_map.get(rejected_meta["candidate_id"], {}) if rejected_meta else {},
            "candidate_count": len(state["candidate_answers"]),
        }

        pair = _validate_pair(
            question=question,
            chosen=state["chosen"],
            rejected=state["rejected"],
            doc_id=state["doc_id"],
            doc_ref=doc_ref,
            source_ref=source_ref,
            selection_method=state["selection_method"],
            judge_scores=judge_scores,
        )

        return {"validated_pair": pair}

    def save_pair(state: Agent2State) -> Dict:
        pair = state["validated_pair"]

        if pair is not None:
            path = Path(state["output_jsonl"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            updated = state["accepted_pairs"] + [pair]
        else:
            updated = state["accepted_pairs"]

        return {
            "accepted_pairs": updated,
            "question_index": state["question_index"] + 1,
        }

    return (
        fetch_question,
        retrieve_context,
        generate_candidates,
        judge_candidates,
        select_pair,
        validate_pair,
        save_pair,
    )
