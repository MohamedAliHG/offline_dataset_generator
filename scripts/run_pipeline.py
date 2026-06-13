"""CLI wrapper for dataset generation pipeline."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QA preference pairs from indexed C-130 chunks"
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--agent1-only", action="store_true", help="Run Agent 1 only")
    mode.add_argument("--agent2-only", action="store_true", help="Run Agent 2 only")

    parser.add_argument("--thread-id", default=None, help="Stable run ID for checkpoint resume")
    parser.add_argument("--namespace", help="Namespace to query in ChromaDB")
    parser.add_argument("--questions-path", help="Path to accepted_questions.jsonl")
    parser.add_argument("--output-jsonl", help="Output JSONL path for pair dumps")
    parser.add_argument("--output-json", help="Output JSON path for public dataset")
    parser.add_argument("--output-csv", help="Output CSV path for public dataset")
    parser.add_argument("--doc-id", help="Stable source document ID")
    parser.add_argument("--doc-ref", help="Readable source document reference used in citations")
    parser.add_argument("--top-k", type=int, help="Base retrieval K for Agent 2")

    args = parser.parse_args()

    run_pipeline(
        run_agent1=not args.agent2_only,
        run_agent2=not args.agent1_only,
        namespace=args.namespace,
        questions_path=args.questions_path,
        output_jsonl=args.output_jsonl,
        output_json=args.output_json,
        output_csv=args.output_csv,
        doc_id=args.doc_id,
        doc_ref=args.doc_ref,
        top_k=args.top_k,
        thread_id=args.thread_id,
    )


if __name__ == "__main__":
    main()
