"""Candidate-generation and judge parsing helpers for Agent 2."""

import json
import re
from typing import Dict, List

"""
CANDIDATE_PROMPT_VARIANTS: List[Dict[str, str]] = [
    {
        "name": "strict_grounded",
        "instruction": (
            "Prioritize exact factual grounding and concise coverage. Include only details that are "
            "explicitly supported by the retrieved chunks."
        ),
    },
    {
        "name": "procedural_focus",
        "instruction": (
            "If the question involves a process, make the sequence of steps and conditions explicit. "
            "Keep the answer operational and source-grounded."
        ),
    },
    {
        "name": "cross_reference_focus",
        "instruction": (
            "Pay close attention to relationships across the retrieved chunks, including tables, notes, "
            "and any linked textual or visual references."
        ),
    },
    {
        "name": "constraint_focus",
        "instruction": (
            "Highlight thresholds, exceptions, prerequisites, and edge conditions whenever they are "
            "present in the retrieved chunks."
        ),
    },
]"""

"""
def _get_candidate_variants(num_candidates: int) -> List[Dict[str, str]]:
    Return a stable list of prompt variants sized for the requested candidate count
    total = max(2, num_candidates)
    return [
        {
            "candidate_id": f"cand_{index + 1}",
            "name": CANDIDATE_PROMPT_VARIANTS[index % len(CANDIDATE_PROMPT_VARIANTS)]["name"],
            "instruction": CANDIDATE_PROMPT_VARIANTS[index % len(CANDIDATE_PROMPT_VARIANTS)][
                "instruction"
            ],
        }
        for index in range(total)
    ]
"""

def _parse_judge_payload(text: str) -> List[Dict]:
    """Parse judge JSON output into ranked candidate metadata."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return []

    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return []

    ranked = payload.get("ranked_candidates", [])
    if not isinstance(ranked, list):
        return []

    normalized: List[Dict] = []
    for item in ranked:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id", "")).strip()
        if not candidate_id:
            continue
        normalized.append(
            {
                "candidate_id": candidate_id,
                "factuality": int(item.get("factuality", 0) or 0),
                "completeness": int(item.get("completeness", 0) or 0),
                "traceability": int(item.get("traceability", 0) or 0),
                "document_fit": int(item.get("document_fit", 0) or 0),
                "total_score": int(item.get("total_score", 0) or 0),
                "rationale": str(item.get("rationale", "")).strip(),
            }
        )
    return normalized


def _strip_meta_commentary(text: str) -> str:
    """Remove self-referential model commentary from generated answers."""
    patterns = [
        r"\n+Note that (the rejected|this) answer.*$",
        r"\n+Note:.*$",
        r"\n+The rejected answer.*$",
        r"\n+This omission.*$",
        r"\n+This (change|error|modification).*$",
        r"\n+\(Note:.*?\)$",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()
