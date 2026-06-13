"""Validation and parsing helpers for Agent 1."""

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

VALID_DIFFICULTIES = {"basic", "intermediate", "advanced"}
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

CEILING_MAP: Dict[str, set] = {
    "DEFINITIONAL": {"basic"},
    "TABULAR": {"basic", "intermediate"},
    "PROCEDURAL": {"basic", "intermediate"},
    "CONDITIONAL": {"basic", "intermediate", "advanced"},
    "MIXED": {"basic", "intermediate", "advanced"},
}
DEFAULT_CEILING = {"basic", "intermediate", "advanced"}

DEDUP_THRESHOLD = 0.85
SCENARIO_FLOOR = 0.30
MAX_CANDIDATES = 7

_STRUCTURAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bsection\b",
        r"\bpage\b",
        r"\bchapter\b",
        r"\bparagraph\b",
        r"\bnote\b",
        r"\btable of contents\b",
        r"\bheading\b",
        r"\btitle\b",
        r"\bappendix\b",
        r"\bdocument structure\b",
        r"\blayout\b",
        r"\bformat\b",
        r"what is (the purpose|significance|meaning) of (section|chapter|note|page)",
        r"what (does|is) (section|chapter|page|note|figure|table)\b",
        r"how many (steps|sections|chapters|items|parts)",
        r"what (type|kind|sort) of information",
        r"separator",
        r"according to (the table|this section|the note|this figure)",
        r"as mentioned in",
        r"what does the note say",
    ]
]


def _parse_json_array(text: str) -> List[Dict]:
    """Extract and parse the first JSON array from an LLM response."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        logger.warning("[A1] No JSON array found in model response")
        return []

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("[A1] JSON parse error: %s", exc)
        return []


def _is_structural(question: str) -> bool:
    """Return ``True`` when the question targets document structure only."""
    q = question.lower()
    return any(pattern.search(q) for pattern in _STRUCTURAL_PATTERNS)


def _is_near_duplicate(new_q: str, seen: List[str], threshold: float = DEDUP_THRESHOLD) -> bool:
    """Return ``True`` if ``new_q`` is semantically near-duplicate of prior text."""
    new_lower = new_q.lower()
    for existing in seen:
        ratio = SequenceMatcher(None, new_lower, existing.lower()).ratio()
        if ratio >= threshold:
            return True
    return False


def _enforce_ceiling(difficulty: str, chunk_type: str) -> Optional[str]:
    """Enforce chunk-type difficulty ceilings."""
    allowed = CEILING_MAP.get(chunk_type.upper(), DEFAULT_CEILING)
    return difficulty if difficulty in allowed else None


def _parse_classification(text: str) -> Dict[str, str]:
    """Extract ``chunk_type`` and ``domain`` from an XML-style classification block."""
    defaults = {"chunk_type": "MIXED", "domain": "aircraft_systems"}
    match = re.search(r"<classification>(.*?)</classification>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return defaults

    block = match.group(1)
    result: Dict[str, str] = {}

    chunk_type_match = re.search(r"chunk_type\s*:\s*(\w+)", block, re.IGNORECASE)
    if chunk_type_match:
        result["chunk_type"] = chunk_type_match.group(1).upper()

    domain_match = re.search(r"domain\s*:\s*(\S+)", block, re.IGNORECASE)
    if domain_match:
        domain = domain_match.group(1).strip().lower().rstrip(".,;")
        result["domain"] = domain if domain in VALID_DOMAINS else defaults["domain"]

    return {**defaults, **result}
