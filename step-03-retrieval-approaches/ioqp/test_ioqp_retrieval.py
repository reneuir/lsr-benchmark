import gzip
from contextlib import nullcontext

import ioqp_retrieval


def test_quantize_preserves_relative_document_weights():
    assert ioqp_retrieval.quantize({"0": 1.0}, 100, reference_max=2.0) == {"0": 50}


def test_quantization_levels_prevent_u16_accumulator_overflow():
    queries = [("q1", [str(i) for i in range(100)], [1.0] * 100)]

    document_impact, query_weight = ioqp_retrieval.determine_quantization_levels(
        queries,
        max_document_impact=255,
        max_query_weight=32,
    )

    assert document_impact * query_weight * 100 <= ioqp_retrieval.MAX_SCORE


def test_ioqp_indexes_and_searches_small_corpus(tmp_path):
    documents = [
        ("d1", ["0"], [1.0]),
        ("d2", ["0"], [0.5]),
        ("d3", ["1"], [1.0]),
    ]
    queries = [("original-qid", ["0"], [1.0])]
    ciff_path = tmp_path / "index.ciff"
    index_path = tmp_path / "index.ioqp"
    query_path = tmp_path / "queries.txt"
    run_path = tmp_path / "run.txt"

    ioqp_retrieval.write_ciff(ciff_path, documents, max_document_impact=100)
    ioqp_retrieval.run_ioqp(
        ["ioqp-create", "--input", str(ciff_path), "--output", str(index_path)]
    )
    query_ids = ioqp_retrieval.write_queries(query_path, queries, max_query_weight=10)
    ioqp_retrieval.run_ioqp(
        [
            "ioqp-query",
            "--index",
            str(index_path),
            "--queries",
            str(query_path),
            "--output-file",
            str(run_path),
            "--k",
            "3",
            "--mode",
            "fraction-1",
            "--weighted",
        ]
    )

    results = ioqp_retrieval.parse_run(run_path, query_ids)

    assert results[0][0] == "original-qid"
    assert [doc_id for doc_id, _ in results[0][1]] == ["d1", "d2"]
    assert results[0][1][0][1] > results[0][1][1][1]


def test_main_writes_compressed_trec_run(monkeypatch, tmp_path):
    embeddings = {
        "doc": [
            ("d1", ["0"], [1.0]),
            ("d2", ["0"], [0.5]),
            ("d3", ["1"], [1.0]),
        ],
        "query": [("query-with-text-id", ["0"], [1.0])],
    }
    monkeypatch.setattr(ioqp_retrieval.lsr_benchmark, "register_to_ir_datasets", lambda dataset: None)
    monkeypatch.setattr(
        ioqp_retrieval,
        "load_embeddings",
        lambda dataset, embedding, text_type: embeddings[text_type],
    )
    monkeypatch.setattr(ioqp_retrieval, "register_metadata", lambda metadata: None)
    monkeypatch.setattr(ioqp_retrieval, "tracking", lambda **kwargs: nullcontext())

    ioqp_retrieval.main.callback(
        dataset="tiny-example-20251002_0-training",
        embedding="lightning-ir/naver-splade-v3-doc",
        output=tmp_path,
        k=10,
        rho=1.0,
        postings_budget=None,
        max_document_impact=255,
        max_query_weight=32,
    )

    with gzip.open(tmp_path / "run.txt.gz", "rt") as run_file:
        lines = run_file.read().strip().splitlines()

    assert lines[0].split()[:4] == ["query-with-text-id", "Q0", "d1", "1"]
    assert lines[0].endswith(" ioqp")
    assert all(" d3 " not in line for line in lines)
