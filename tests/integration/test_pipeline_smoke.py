import sys
import types

from src.pipeline import run_pipeline


class FakeConfig:
    def get(self, *keys, default=None):
        values = {
            ("database", "persist_directory"): "chroma_db",
            ("database", "collection_name"): "collection_demo",
            ("database", "namespace"): "CaseDoneDemo",
            ("database", "chroma_device"): "cpu",
            ("model", "embeddings"): "BAAI/bge-small-en-v1.5",
        }
        return values.get(tuple(keys), default)

    def resolve_doc_ref(self, doc_id, override=None):
        return override or {"doc_001": "c130_reference_manual"}.get(doc_id, doc_id)


class FakeAgent1:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, thread_id=None):
        assert thread_id is not None
        return [{"id": "q1"}, {"id": "q2"}]


class FakeAgent2:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, thread_id=None):
        assert thread_id is not None
        return [{"prompt": "Q", "chosen": "A", "rejected": "B"}]


def test_pipeline_smoke_runs_with_fakes(monkeypatch):
    fake_module = types.ModuleType("src.infrastructure.chroma_store")

    class FakeStore:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def count(self):
            return 2

    fake_module.ChromaStore = FakeStore
    monkeypatch.setitem(sys.modules, "src.infrastructure.chroma_store", fake_module)

    run_pipeline(
        run_agent1=True,
        run_agent2=True,
        cfg=FakeConfig(),
        thread_id="smoke",
        agent1_graph_cls=FakeAgent1,
        agent2_graph_cls=FakeAgent2,
    )
