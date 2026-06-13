"""Typed state for Agent 1 LangGraph execution."""

from typing import Dict, List, Optional, TypedDict


class Agent1State(TypedDict):
    chunks: List[Dict]
    chunk_index: int
    current_chunk: Optional[Dict]
    raw_questions: List[Dict]
    accepted_questions: List[Dict]
    output_path: str
    namespace: Optional[str]
    doc_id: str
    doc_ref: str
