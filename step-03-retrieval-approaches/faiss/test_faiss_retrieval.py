import gzip
from contextlib import nullcontext

import numpy as np
import pytest

import faiss_retrieval
from faiss_retrieval import (
    build_index,
    determine_dimension,
    retrieve,
    to_dense_matrix,
)


def test_sparse_embeddings_are_converted_to_dense_float32_matrix():
    embeddings = [
        ("d1", ["0", "2"], [1.0, 0.5]),
        ("d2", ["1"], [0.25]),
    ]

    ids, matrix = to_dense_matrix(embeddings, dimension=3)

    assert ids == ["d1", "d2"]
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(
        matrix,
        np.array([[1.0, 0.0, 0.5], [0.0, 0.25, 0.0]], dtype=np.float32),
    )


def test_dimension_covers_document_and_query_components():
    documents = [("d1", ["1"], [1.0])]
    queries = [("q1", ["4"], [1.0])]

    assert determine_dimension(documents, queries) == 5


def test_retrieve_returns_inner_product_top_k_in_score_order():
    doc_ids = ["d1", "d2", "d3"]
    documents = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        dtype=np.float32,
    )
    queries = np.array([[1.0, 0.0]], dtype=np.float32)

    results = retrieve(build_index(documents), ["q1"], queries, doc_ids, k=2)

    assert [result[2] for result in results[0]] == ["d1", "d2"]
    assert [result[1] for result in results[0]] == pytest.approx([1.0, 0.8])


def test_retrieve_handles_multiple_queries_and_k_larger_than_corpus():
    doc_ids = ["d1", "d2"]
    documents = np.eye(2, dtype=np.float32)
    queries = np.eye(2, dtype=np.float32)

    results = retrieve(build_index(documents), ["q1", "q2"], queries, doc_ids, k=10)

    assert [[result[2] for result in ranking] for ranking in results] == [["d1"], ["d2"]]


def test_retrieve_omits_non_positive_scores():
    documents = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)

    results = retrieve(build_index(documents), ["q1"], queries, ["d1", "d2"], k=2)

    assert results == [[]]


def test_retrieve_rejects_non_positive_k():
    documents = np.array([[1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="k must be at least 1"):
        retrieve(build_index(documents), ["q1"], documents, ["d1"], k=0)


def test_main_writes_a_trec_run(monkeypatch, tmp_path):
    embeddings = {
        "doc": [
            ("d1", ["0"], [1.0]),
            ("d2", ["1"], [1.0]),
        ],
        "query": [("q1", ["0"], [1.0])],
    }
    monkeypatch.setattr(faiss_retrieval.lsr_benchmark, "register_to_ir_datasets", lambda dataset: None)
    monkeypatch.setattr(
        faiss_retrieval,
        "load_embeddings",
        lambda dataset, embedding, text_type: embeddings[text_type],
    )
    monkeypatch.setattr(faiss_retrieval, "register_metadata", lambda metadata: None)
    monkeypatch.setattr(faiss_retrieval, "tracking", lambda **kwargs: nullcontext())

    faiss_retrieval.main.callback(
        dataset="tiny-example-20251002_0-training",
        embedding="lightning-ir/naver-splade-v3-doc",
        output=tmp_path,
        k=10,
        batch_size=128,
    )

    with gzip.open(tmp_path / "run.txt.gz", "rt") as run_file:
        assert run_file.read().strip() == "q1 Q0 d1 1 1.0 faiss"
