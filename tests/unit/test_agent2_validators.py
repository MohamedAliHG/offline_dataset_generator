from src.generation.agent2.validators import (
    _build_source_reference,
    _content_similarity,
    _validate_pair,
)


def test_content_similarity_ignores_source_suffix() -> None:
    chosen = "The maximum pressure is 90 psi. Source: c130_reference_manual p12"
    rejected = "The maximum pressure is 90 psi."
    assert _content_similarity(chosen, rejected) > 0.95


def test_build_source_reference_compacts_contiguous_pages() -> None:
    chunks = [
        {"metadata": {"page_no": 12}},
        {"metadata": {"page_no": 13}},
        {"metadata": {"page_no": 12}},
    ]
    assert _build_source_reference(chunks) == "p12-13"


def test_validate_pair_returns_schema_for_valid_input() -> None:
    question = {
        "id": "q1",
        "question": "What is the max pressure?",
        "difficulty": "basic",
        "domain": "aircraft_systems",
        "is_scenario": False,
        "chunk_type": "DEFINITIONAL",
    }
    chosen = (
        "The maximum continuous oil pressure is 90 psi during normal operation. "
        "Always verify against the maintenance procedure before release. "
        "Source: c130_reference_manual p12"
    )
    rejected = (
        "The maximum continuous oil pressure is 110 psi during normal operation, "
        "and this can be used for all engine states without additional checks."
    )

    pair = _validate_pair(
        question=question,
        chosen=chosen,
        rejected=rejected,
        doc_id="doc_001",
        doc_ref="c130_reference_manual",
        source_ref="p12-13",
        selection_method="judge_ranked_candidates",
        judge_scores={"chosen": {"total_score": 18}, "rejected": {"total_score": 12}},
    )

    assert pair is not None
    assert pair["source_doc_ids"] == ["doc_001", "c130_reference_manual p12-13"]
    assert pair["difficulty"] == "basic"
    assert pair["_selection_method"] == "judge_ranked_candidates"
