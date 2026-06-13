"""Validation helpers for Agent 2 chosen/rejected pairs."""

import logging
import re
import uuid
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from src.generation.agent2.strategies import _strip_meta_commentary

logger = logging.getLogger(__name__)

VALID_DIFFICULTIES = {"basic", "intermediate", "advanced", "expert"}
VALID_DOMAINS = {
    "flight_rules",
    "meteorology",
    "aerodynamics",
    "human_factors",
    "aircraft_systems",
    "maintenance_procedures",
    "troubleshooting",
    "regulatory_compliance",
    "safety_management",
    "dispatch_and_ops",
    "navigation",
    "emergency_procedures",
}

CHUNK_TYPE_TOP_K: Dict[str, int] = {
    "DEFINITIONAL": 1, #2
    "TABULAR": 1, #2
    "PROCEDURAL": 1, #4
    "CONDITIONAL": 1, #4
    "MIXED": 1, #3
}
DEFAULT_TOP_K = 3
DISTANCE_THRESHOLD = 0.35
SCENARIO_FLOOR = 0.30

CHOSEN_MIN_LEN = 80
CHOSEN_MAX_LEN = 2000
REJECTED_MIN_LEN = 40
REJECTED_MAX_LEN = 2000
CHOSEN_REJECTED_MAX_SIMILARITY = 0.92
DEFAULT_JUDGE_MARGIN = 3
DEFAULT_MIN_REJECTED_SCORE = 8


def _build_source_doc_ids(doc_id: str, source_ref: str) -> List[str]:
    """Build spec-compliant ``source_doc_ids`` list."""
    return [doc_id, source_ref]


def _build_source_reference(chunks: List[Dict]) -> str:
    """Summarize all retrieved pages into a compact traceability label."""
    pages: List[int] = []
    for chunk in chunks:
        page = chunk.get("metadata", {}).get("page_no")
        if isinstance(page, int):
            pages.append(page)

    if not pages:
        return "p?"

    unique_pages = sorted(set(pages))
    if len(unique_pages) == 1:
        return f"p{unique_pages[0]}"

    contiguous = all(
        unique_pages[index + 1] == unique_pages[index] + 1 for index in range(len(unique_pages) - 1)
    )
    if contiguous:
        return f"p{unique_pages[0]}-{unique_pages[-1]}"

    return ", ".join(f"p{page}" for page in unique_pages)


def _content_similarity(text_a: str, text_b: str) -> float:
    """Compute normalized text similarity after removing citations."""

    def strip_citation(value: str) -> str:
        return re.sub(r"\s*[Ss]ource\s*:.*$", "", value, flags=re.MULTILINE).strip()

    a = strip_citation(text_a).lower()
    b = strip_citation(text_b).lower()
    return SequenceMatcher(None, a, b).ratio()


def _validate_pair(
    question: Dict,
    chosen: str,
    rejected: str,
    doc_id: str,
    doc_ref: str,
    source_ref: str,
    selection_method: str,
    judge_scores: Optional[Dict] = None,
) -> Optional[Dict]:
    """Validate and normalize a chosen/rejected candidate pair."""
    qid = question.get("id", "unknown")
    chosen = _strip_meta_commentary(chosen)
    rejected = _strip_meta_commentary(rejected)

    if len(chosen) < CHOSEN_MIN_LEN:
        logger.warning("[A2] %s: chosen too short (%s chars)", qid, len(chosen))
        return None
    if len(chosen) > CHOSEN_MAX_LEN:
        chosen = chosen[:CHOSEN_MAX_LEN].rsplit(" ", 1)[0] + "..."

    if len(rejected) < REJECTED_MIN_LEN:
        logger.warning("[A2] %s: rejected too short (%s chars)", qid, len(rejected))
        return None
    if len(rejected) > REJECTED_MAX_LEN:
        rejected = rejected[:REJECTED_MAX_LEN].rsplit(" ", 1)[0] + "..."

    similarity = _content_similarity(chosen, rejected)
    if similarity > CHOSEN_REJECTED_MAX_SIMILARITY:
        logger.warning("[A2] %s: chosen/rejected too similar (%s)", qid, round(similarity, 4))
        return None

    if chosen.strip().lower() == rejected.strip().lower():
        logger.warning("[A2] %s: chosen == rejected", qid)
        return None

    source_doc_ids = _build_source_doc_ids(doc_id,source_ref)
    if len(source_doc_ids) != 2 or not all(source_doc_ids):
        logger.warning("[A2] %s: malformed source_doc_ids=%s", qid, source_doc_ids)
        return None

    difficulty = question.get("difficulty", "basic")
    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "basic"

    domain = question.get("domain", "aircraft_systems")
    if domain not in VALID_DOMAINS:
        domain = "aircraft_systems"

    return {
        "prompt": question["question"],
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
        "source_doc_ids": source_doc_ids,
        "difficulty": difficulty,
        "domain": domain,
        "_id": str(uuid.uuid4()),
        "_question_id": question.get("id", ""),
        "_selection_method": selection_method,
        "_similarity": round(similarity, 4),
        "_is_scenario": question.get("is_scenario", False),
        "_chunk_type": question.get("chunk_type", ""),
        "_source_page": source_ref,
        "_judge_scores": judge_scores or {},
    }
