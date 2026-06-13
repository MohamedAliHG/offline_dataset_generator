# Offline dataset generator Pipeline

Production-oriented pipeline for generating QA preference-pair datasets from Aviation technical manuals.

## Architecture

```
offline_dataset_generator/
├── config/
├── data/
├── output/
├── logs/
├── src/
│   ├── infrastructure/
│   ├── ingestion/
│   ├── generation/
│   │   ├── prompts/
│   │   ├── agent1/
│   │   └── agent2/
│   ├── export/
│   └── pipeline.py
├── scripts/
├── tests/
├── checkpoints/
├── Makefile
└── pyproject.toml
```

## Quick Start

1. Install dependencies:

```bash
python -m pip install -e .[dev]
```

2. Put source PDFs in `data/raw/`.

3. Index documents:

```bash
make index
```

4. Run full dataset generation pipeline:

```bash
make run
```

## CLI Usage

Indexing:

```bash
python scripts/index.py --dir data/raw --drop
```

Pipeline:

```bash
python scripts/run_pipeline.py
python scripts/run_pipeline.py --agent1-only
python scripts/run_pipeline.py --agent2-only
```

## Testing

```bash
make test
```

## Notes

- Runtime artifacts go to `output/`, `logs/`, and `checkpoints/`.
- Prompt templates are isolated in `src/generation/prompts/` for fast iteration.
- LLM creation is centralized in `src/infrastructure/llm_client.py`.
