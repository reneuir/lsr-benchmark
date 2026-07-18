import importlib.util
import gzip
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parent / "build-and-search-kannolo-index.py"


def load_module():
    module_spec = importlib.util.spec_from_file_location("kannolo_retrieval", MODULE_PATH)
    module = importlib.util.module_from_spec(module_spec)
    fake_kannolo = types.ModuleType("kannolo")
    fake_kannolo.SparsePlainHNSW = type("SparsePlainHNSW", (), {})
    with patch.dict(sys.modules, {"kannolo": fake_kannolo}):
        module_spec.loader.exec_module(module)
    return module


class FakeDataset:
    def doc_embeddings(self, model_name):
        return [("d1", ["1"], [1.0])]

    def query_embeddings(self, model_name):
        return [("q1", ["1"], [1.0])]


class FakeIndex:
    def __init__(self):
        self.calls = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [0.5], [0]


def test_search_uses_current_kannolo_signature(tmp_path):
    kannolo_retrieval = load_module()
    fake_index = FakeIndex()

    with (
        patch.object(kannolo_retrieval.lsr_benchmark, "register_to_ir_datasets"),
        patch.object(kannolo_retrieval.ir_datasets, "load", return_value=FakeDataset()),
        patch.object(kannolo_retrieval, "register_metadata"),
        patch.object(kannolo_retrieval, "tracking", side_effect=lambda **kwargs: nullcontext()),
        patch.object(kannolo_retrieval, "rmtree"),
        patch.object(
            kannolo_retrieval.SparsePlainHNSW,
            "build_from_arrays",
            return_value=fake_index,
            create=True,
        ),
    ):
        kannolo_retrieval.main.callback(
            dataset="tiny-example-20251002_0-training",
            embedding="lightning-ir/naver-splade-v3-doc",
            output=tmp_path,
            ef_search=200,
            k=10,
        )

    assert len(fake_index.calls) == 1
    args, kwargs = fake_index.calls[0]
    assert len(args) == 2
    assert kwargs == {"k": 10, "ef_search": 200}

    with gzip.open(tmp_path / "run.txt.gz", "rt") as run_file:
        assert run_file.read().strip() == "q1 Q0 d1 1 0.5 kannolo"
