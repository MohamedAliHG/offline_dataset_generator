"""Candidate-generation and judge parsing helpers for Agent 2."""

import json
import re
from typing import Dict, List



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
