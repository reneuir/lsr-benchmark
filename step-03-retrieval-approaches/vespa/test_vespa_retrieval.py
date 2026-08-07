import gzip
from contextlib import nullcontext

import pytest

import vespa_retrieval


def test_merge_embedding_combines_duplicate_tokens_and_ignores_non_positive_values():
    embedding = vespa_retrieval.merge_embedding(
        ["2", "0", "2", "1"],
        [0.25, 1.0, 0.75, 0.0],
    )

    assert embedding == {0: 1.0, 2: 1.0}


@pytest.mark.parametrize(
    ("tokens", "values", "message"),
    [
        (["0"], [], "same length"),
        (["not-an-index"], [1.0], "not an integer"),
        (["-1"], [1.0], "signed 32-bit"),
        (["0"], [float("inf")], "finite"),
    ],
)
def test_merge_embedding_rejects_invalid_vectors(tokens, values, message):
    with pytest.raises(ValueError, match=message):
        vespa_retrieval.merge_embedding(tokens, values)


def test_quantization_uses_global_document_scale():
    documents = [
        ("d1", ["0"], [2.0]),
        ("d2", ["1"], [1.0]),
    ]

    scale = vespa_retrieval.quantization_scale(documents, max_weight=100)

    assert scale == 50
    assert vespa_retrieval.quantize_embedding({0: 2.0, 1: 1.0}, scale, 100) == {
        0: 100,
        1: 50,
    }


def test_application_package_contains_weighted_set_and_raw_score(tmp_path):
    vespa_retrieval.write_application_package(tmp_path)

    schema = (tmp_path / "schemas" / "sparse.sd").read_text()
    assert "weightedset<int>" in schema
    assert "attribute: fast-search" in schema
    assert "rawScore(embedding)" in schema


def test_safe_tracking_falls_back_when_tracker_directory_is_inaccessible(monkeypatch, tmp_path):
    events = []

    class PermissionDeniedTracker:
        def __enter__(self):
            raise PermissionError(13, "Permission denied", "/tira-data/output/.tirex-tracker")

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")
            return False

    monkeypatch.setattr(
        vespa_retrieval,
        "tracking",
        lambda **kwargs: PermissionDeniedTracker(),
    )

    with vespa_retrieval.safe_tracking(
        export_file_path=tmp_path / "index-metadata.yml",
        export_format=vespa_retrieval.ExportFormat.IR_METADATA,
    ):
        events.append("body")

    assert events == ["body"]


def test_safe_tracking_uses_tracker_when_it_starts_successfully(monkeypatch, tmp_path):
    events = []

    class RecordingTracker:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")
            return False

    monkeypatch.setattr(vespa_retrieval, "tracking", lambda **kwargs: RecordingTracker())

    with vespa_retrieval.safe_tracking(
        export_file_path=tmp_path / "index-metadata.yml",
        export_format=vespa_retrieval.ExportFormat.IR_METADATA,
    ):
        events.append("body")

    assert events == ["enter", "body", "exit"]


class FakeClient:
    def __init__(self):
        self.documents = []
        self.queries = []

    def feed(self, internal_id, document_id, vector):
        self.documents.append((internal_id, document_id, vector))

    def query(self, vector, k):
        self.queries.append((vector, k))
        return {
            "root": {
                "children": [
                    {"relevance": 10000, "fields": {"doc_id": "lower"}},
                    {"relevance": 20000, "fields": {"doc_id": "higher"}},
                    {"relevance": 30000, "fields": {}},
                    {"relevance": 0, "fields": {"doc_id": "zero"}},
                ]
            }
        }


def test_build_index_preserves_ids_and_skips_empty_documents():
    client = FakeClient()
    documents = [
        ("document-with-text-id", ["0"], [1.0]),
        ("empty", ["1"], [0.0]),
    ]

    index = vespa_retrieval.build_index(
        client,
        documents,
        max_weight=100,
        feed_workers=1,
    )

    assert index.document_count == 2
    assert index.indexed_document_count == 1
    assert client.documents == [(0, "document-with-text-id", {0: 100})]


def test_build_index_rejects_duplicate_document_ids():
    with pytest.raises(ValueError, match="must be unique"):
        vespa_retrieval.build_index(
            FakeClient(),
            [
                ("duplicate", ["0"], [1.0]),
                ("duplicate", ["1"], [1.0]),
            ],
            max_weight=100,
            feed_workers=1,
        )


def test_retrieve_dequantizes_scores_and_sorts_results():
    client = FakeClient()
    index = vespa_retrieval.VespaIndex(
        document_count=2,
        indexed_document_count=2,
        document_scale=100.0,
        max_weight=100,
    )

    results = vespa_retrieval.retrieve(
        client,
        index,
        [
            ("q1", ["0"], [2.0]),
            ("empty", [], []),
        ],
        k=10,
    )

    assert results[0][0] == "q1"
    assert [document_id for document_id, _ in results[0][1]] == ["higher", "lower"]
    assert [score for _, score in results[0][1]] == pytest.approx([4.0, 2.0])
    assert results[1] == ("empty", [])
    assert client.queries == [({0: 100}, 2)]


def test_retrieve_rejects_inconsistent_index_metadata():
    index = vespa_retrieval.VespaIndex(1, 1, 0.0, 100)

    with pytest.raises(ValueError, match="invalid document scale"):
        vespa_retrieval.retrieve(
            FakeClient(),
            index,
            [("q1", ["0"], [1.0])],
            k=1,
        )


def test_main_writes_a_compressed_trec_run(monkeypatch, tmp_path):
    embeddings = {
        "doc": [("d1", ["0"], [1.0])],
        "query": [("q1", ["0"], [1.0])],
    }
    client = FakeClient()
    client.deploy = lambda application_path: None
    monkeypatch.setattr(
        vespa_retrieval.lsr_benchmark,
        "register_to_ir_datasets",
        lambda dataset: None,
    )
    monkeypatch.setattr(
        vespa_retrieval,
        "load_embeddings",
        lambda dataset, embedding, text_type: embeddings[text_type],
    )
    monkeypatch.setattr(vespa_retrieval, "register_metadata", lambda metadata: None)
    monkeypatch.setattr(vespa_retrieval, "tracking", lambda **kwargs: nullcontext())
    monkeypatch.setattr(
        vespa_retrieval,
        "vespa_server",
        lambda temporary_directory: nullcontext(client),
    )

    vespa_retrieval.main.callback(
        dataset="tiny-example-20251002_0-training",
        embedding="lightning-ir/naver-splade-v3-doc",
        output=tmp_path,
        k=10,
        max_weight=100,
        feed_workers=1,
    )

    with gzip.open(tmp_path / "run.txt.gz", "rt") as run_file:
        assert run_file.read().strip() == "q1 Q0 higher 1 2.0 vespa"


def test_main_continues_when_tracker_directory_is_inaccessible(monkeypatch, tmp_path):
    embeddings = {
        "doc": [("d1", ["0"], [1.0])],
        "query": [("q1", ["0"], [1.0])],
    }
    client = FakeClient()
    client.deploy = lambda application_path: None

    class PermissionDeniedTracker:
        def __enter__(self):
            raise PermissionError(13, "Permission denied", "/tira-data/output/.tirex-tracker")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        vespa_retrieval.lsr_benchmark,
        "register_to_ir_datasets",
        lambda dataset: None,
    )
    monkeypatch.setattr(
        vespa_retrieval,
        "load_embeddings",
        lambda dataset, embedding, text_type: embeddings[text_type],
    )
    monkeypatch.setattr(vespa_retrieval, "register_metadata", lambda metadata: None)
    monkeypatch.setattr(vespa_retrieval, "tracking", lambda **kwargs: PermissionDeniedTracker())
    monkeypatch.setattr(
        vespa_retrieval,
        "vespa_server",
        lambda temporary_directory: nullcontext(client),
    )

    vespa_retrieval.main.callback(
        dataset="tiny-example-20251002_0-training",
        embedding="lightning-ir/naver-splade-v3-doc",
        output=tmp_path,
        k=10,
        max_weight=100,
        feed_workers=1,
    )

    with gzip.open(tmp_path / "run.txt.gz", "rt") as run_file:
        assert run_file.read().strip() == "q1 Q0 higher 1 2.0 vespa"
