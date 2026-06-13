import csv
import json

from src.export.exporters import _export_csv, _export_json, _to_public_schema


def test_to_public_schema_strips_internal_fields() -> None:
    pair = {
        "prompt": "Q?",
        "chosen": "A",
        "rejected": "B",
        "source_doc_ids": ["doc_001", "c130_reference_manual p1"],
        "difficulty": "basic",
        "domain": "aircraft_systems",
        "_internal": "ignore",
    }
    public = _to_public_schema(pair)
    assert set(public.keys()) == {
        "prompt",
        "chosen",
        "rejected",
        "source_doc_ids",
        "difficulty",
        "domain",
    }


def test_export_json_and_csv(tmp_path) -> None:
    pairs = [
        {
            "prompt": "What is max pressure?",
            "chosen": "90 psi. Source: c130_reference_manual p1",
            "rejected": "110 psi.",
            "source_doc_ids": ["doc_001", "c130_reference_manual p1"],
            "difficulty": "basic",
            "domain": "aircraft_systems",
        }
    ]

    json_path = tmp_path / "dataset.json"
    csv_path = tmp_path / "dataset.csv"

    _export_json(pairs, str(json_path))
    _export_csv(pairs, str(csv_path))

    exported_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert exported_json[0]["prompt"] == "What is max pressure?"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows[0]["difficulty"] == "basic"
