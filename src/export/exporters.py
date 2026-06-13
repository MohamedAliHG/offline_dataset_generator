"""Dataset export helpers."""

import csv
import json
from pathlib import Path
from typing import Dict, List

SCHEMA_FIELDS = ["prompt", "chosen", "rejected", "source_doc_ids", "difficulty", "domain"]


def _to_public_schema(pair: Dict) -> Dict:
    """Return only externally published schema fields."""
    return {
        "prompt": pair["prompt"],
        "chosen": pair["chosen"],
        "rejected": pair["rejected"],
        "source_doc_ids": pair["source_doc_ids"],
        "difficulty": pair["difficulty"],
        "domain": pair["domain"],
    }


def _export_json(pairs: List[Dict], path: str) -> None:
    """Export final dataset to JSON array format."""
    public = [_to_public_schema(pair) for pair in pairs]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(public, f, ensure_ascii=False, indent=2)


def _export_csv(pairs: List[Dict], path: str) -> None:
    """Export final dataset to CSV format."""
    public = [_to_public_schema(pair) for pair in pairs]
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA_FIELDS)
        writer.writeheader()
        for row in public:
            payload = dict(row)
            payload["source_doc_ids"] = json.dumps(payload["source_doc_ids"])
            writer.writerow(payload)
