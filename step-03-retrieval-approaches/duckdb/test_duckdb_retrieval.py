import gzip
from contextlib import nullcontext

import duckdb_retrieval
import numpy as np
import pytest


class Dataset:
    def __init__(self, documents, queries):
        self.documents = documents
        self.queries = queries

    def doc_embeddings(self, model_name):
        return self.documents

    def query_embeddings(self, model_name):
        return self.queries


def run_retrieval(monkeypatch, tmp_path, documents, queries):
    dataset = Dataset(documents, queries)
    monkeypatch.setattr(duckdb_retrieval.lsr_benchmark, "register_to_ir_datasets", lambda dataset: None)
    monkeypatch.setattr(duckdb_retrieval.ir_datasets, "load", lambda name: dataset)
    monkeypatch.setattr(duckdb_retrieval, "register_metadata", lambda metadata: None)
    monkeypatch.setattr(duckdb_retrieval, "tracking", lambda **kwargs: nullcontext())
    monkeypatch.setattr(duckdb_retrieval, "rmtree", lambda path: None)

    duckdb_retrieval.main.callback(
        dataset="tiny-example-20251002_0-training",
        embedding="test-embedding",
        output=tmp_path,
        quantize=False,
        k=10,
    )

    with gzip.open(tmp_path / "run.txt.gz", "rt") as run_file:
        return [line.split() for line in run_file.read().strip().splitlines()]


@pytest.mark.parametrize(
    ("dtype", "value", "expected_score"),
    [
        (np.int8, 100, 20000),
        (np.uint8, 200, 80000),
    ],
)
def test_retrieve_with_8_bit_integer_scores(monkeypatch, tmp_path, dtype, value, expected_score):
    documents = [
        ("d1", ["0", "1"], np.array([value, value], dtype=dtype)),
        ("d2", ["0", "1"], np.array([1, 1], dtype=dtype)),
    ]
    queries = [("q1", ["0", "1"], np.array([value, value], dtype=dtype))]

    results = run_retrieval(monkeypatch, tmp_path, documents, queries)

    assert [row[2] for row in results] == ["d1", "d2"]
    assert int(results[0][4]) == expected_score


def test_retrieve_with_float32_scores(monkeypatch, tmp_path):
    documents = [
        ("d1", ["0", "1"], np.array([0.4, 0.2], dtype=np.float32)),
        ("d2", ["0", "1"], np.array([0.2, 0.1], dtype=np.float32)),
    ]
    queries = [("q1", ["0", "1"], np.array([0.4, 0.4], dtype=np.float32))]

    results = run_retrieval(monkeypatch, tmp_path, documents, queries)

    assert [row[2] for row in results] == ["d1", "d2"]
    assert [float(row[4]) for row in results] == pytest.approx([0.24, 0.12])
