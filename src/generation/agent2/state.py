"""Typed state for Agent 2 LangGraph execution."""

from typing import Dict, List, Optional, TypedDict


class Agent2State(TypedDict):
    questions: List[Dict]
    question_index: int
    current_question: Optional[Dict]
    retrieved_chunks: List[Dict]
    candidate_answers: List[Dict]
    judged_candidates: List[Dict]
    chosen: str
    rejected: str
    selection_method: str
    validated_pair: Optional[Dict]
    accepted_pairs: List[Dict]
    output_jsonl: str
    output_json: str
    output_csv: str
    doc_id: str
    doc_ref: str
    top_k: int
    num_candidates: int
    judge_selection_margin: int
    judge_min_rejected_score: int
