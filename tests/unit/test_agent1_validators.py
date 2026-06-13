from src.generation.agent1.validators import (
    _enforce_ceiling,
    _is_near_duplicate,
    _is_structural,
)


def test_structural_detection_matches_document_language() -> None:
    assert _is_structural("What does this section say about pressure limits?")
    assert not _is_structural("What is the maximum oil pressure limit?")


def test_enforce_ceiling_rejects_disallowed_difficulty() -> None:
    assert _enforce_ceiling("advanced", "DEFINITIONAL") is None
    assert _enforce_ceiling("intermediate", "PROCEDURAL") == "intermediate"


def test_near_duplicate_detection() -> None:
    seen = ["What is the maximum oil pressure limit?"]
    assert _is_near_duplicate("What is the maximum oil pressure limit", seen)
    assert not _is_near_duplicate("How many steps are in the shutdown procedure?", seen)
