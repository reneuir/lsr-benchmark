import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from ir_datasets import load
from click.testing import CliRunner

from lsr_benchmark import main, register_to_ir_datasets
from lsr_benchmark._commands import _download
from lsr_benchmark._commands import sisap_io as sisap_io_module
from lsr_benchmark._commands._modify_data import JOINT_TO_DATASETS
from lsr_benchmark._commands.sisap_io import (
    MissingSisapDependencyError,
    _decreasing_scores_for_ranking,
    _invert_id_mapping,
    _normalize_truth_doc_ids,
    _write_json_gz,
    convert_sisap_results_to_trec_run,
    convert_sisap_truths_to_qrels,
    export_embeddings_to_sisap,
)


TEST_RESOURCES = Path(__file__).resolve().parent / "resources"


def _metadata_fixture_text(tag: str) -> str:
    return (
        "tag: test-tag\n"
        "data:\n"
        "  embedding model:\n"
        "    name: naver/splade-v3\n"
        "    tira-embedding-software: lsr-benchmark/lightning-ir/naver-splade-v3\n"
        "  test collection:\n"
        "    ir-datasets-id: rteb/aila/casedocs\n"
        "    name: /tira-data/input\n"
        "implementation:\n"
        "  script:\n"
        "    path: /run.py\n"
        f"{tag}: true\n"
    )


def _write_embedding_dir(base_dir: Path, doc_count: int = 3) -> Path:
    doc_dir = base_dir / "doc"
    query_dir = base_dir / "query"
    doc_dir.mkdir(parents=True)
    query_dir.mkdir(parents=True)

    doc_indptr = [0, 1, 2, 3]
    while len(doc_indptr) < doc_count + 1:
        doc_indptr.append(doc_indptr[-1])

    np.savez_compressed(
        doc_dir / "doc-embeddings.npz",
        data=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        indices=np.array([0, 0, 1], dtype=np.int32),
        indptr=np.array(doc_indptr, dtype=np.int64),
    )
    (doc_dir / "doc-ids.txt").write_text("\n".join(f"d{i}" for i in range(doc_count)) + "\n")
    (doc_dir / "doc-ir-metadata.yml").write_text(_metadata_fixture_text("doc-metadata"))

    np.savez_compressed(
        query_dir / "query-embeddings.npz",
        data=np.array([1.0, 1.0], dtype=np.float32),
        indices=np.array([0, 1], dtype=np.int32),
        indptr=np.array([0, 1, 2], dtype=np.int64),
    )
    (query_dir / "query-ids.txt").write_text("q0\nq1\n")
    (query_dir / "query-ir-metadata.yml").write_text(_metadata_fixture_text("query-metadata"))

    return base_dir


def _write_embedding_dir_with_ids(base_dir: Path, doc_ids: list[str], query_ids: list[str]) -> Path:
    doc_dir = base_dir / "doc"
    query_dir = base_dir / "query"
    doc_dir.mkdir(parents=True)
    query_dir.mkdir(parents=True)

    doc_data = np.arange(1, len(doc_ids) + 1, dtype=np.float32)
    doc_indices = np.zeros(len(doc_ids), dtype=np.int32)
    doc_indptr = np.arange(len(doc_ids) + 1, dtype=np.int64)

    query_data = np.ones(len(query_ids), dtype=np.float32)
    query_indices = np.zeros(len(query_ids), dtype=np.int32)
    query_indptr = np.arange(len(query_ids) + 1, dtype=np.int64)

    np.savez_compressed(
        doc_dir / "doc-embeddings.npz",
        data=doc_data,
        indices=doc_indices,
        indptr=doc_indptr,
    )
    (doc_dir / "doc-ids.txt").write_text("\n".join(doc_ids) + "\n")
    (doc_dir / "doc-ir-metadata.yml").write_text(_metadata_fixture_text("doc-metadata"))

    np.savez_compressed(
        query_dir / "query-embeddings.npz",
        data=query_data,
        indices=query_indices,
        indptr=query_indptr,
    )
    (query_dir / "query-ids.txt").write_text("\n".join(query_ids) + "\n")
    (query_dir / "query-ir-metadata.yml").write_text(_metadata_fixture_text("query-metadata"))

    return base_dir


def _write_results_h5(path: Path, knns: list[list[int]], dists: list[list[float]]) -> Path:
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("knns", data=np.array(knns, dtype=np.int32), dtype=np.int32)
        h5_file.create_dataset("dists", data=np.array(dists, dtype=np.float32), dtype=np.float32)
    return path


def _write_sisap_forwarded_metadata(bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "document-embedding-metadata.yml").write_text(_metadata_fixture_text("doc-metadata"))
    (bundle_dir / "query-embedding-metadata.yml").write_text(_metadata_fixture_text("query-metadata"))
    return bundle_dir


def _assert_sisap_bundle_structure(bundle_dir: Path, *, expect_mappings: bool):
    config = json.loads((bundle_dir / "config.json").read_text())

    assert config["task"] == "task3"
    assert config["data"] == "train"
    assert config["queries"] == "otest/queries"
    assert config["gt_I"] == "otest/knns"
    assert config["k"] > 0
    assert config["sparse"] is True

    with h5py.File(bundle_dir / config["filename"], "r") as h5_file:
        train_group = h5_file[config["data"]]
        query_group = h5_file[config["queries"]]
        otest_group = h5_file["otest"]
        knns = h5_file[config["gt_I"]]
        dists = h5_file["otest/dists"]

        assert train_group["data"].ndim == 1
        assert train_group["indices"].ndim == 1
        assert train_group["indptr"].ndim == 1
        assert train_group["data"].shape == train_group["indices"].shape
        assert int(train_group["indptr"][-1]) == len(train_group["data"])
        assert train_group.attrs["shape"][0] == len(train_group["indptr"]) - 1
        assert train_group.attrs["shape"][1] >= 0

        assert query_group["data"].ndim == 1
        assert query_group["indices"].ndim == 1
        assert query_group["indptr"].ndim == 1
        assert query_group["data"].shape == query_group["indices"].shape
        assert int(query_group["indptr"][-1]) == len(query_group["data"])
        assert query_group.attrs["shape"][0] == len(query_group["indptr"]) - 1
        assert query_group.attrs["shape"][1] == train_group.attrs["shape"][1]

        assert otest_group.attrs["algo"]
        assert otest_group.attrs["querytime"] == 0.0

        assert knns.shape == dists.shape
        assert knns.shape[0] == len(query_group["indptr"]) - 1
        assert config["k"] <= knns.shape[1]

    query_mapping_path = bundle_dir / "query-id-to-index.json.gz"
    document_mapping_path = bundle_dir / "document-id-to-index.json.gz"
    if expect_mappings:
        with gzip.open(query_mapping_path, "rt", encoding="utf-8") as input_file:
            query_mapping = json.load(input_file)
        with gzip.open(document_mapping_path, "rt", encoding="utf-8") as input_file:
            document_mapping = json.load(input_file)

        assert sorted(query_mapping.values()) == list(range(len(query_mapping)))
        assert sorted(document_mapping.values()) == list(range(len(document_mapping)))
    else:
        assert not query_mapping_path.exists()
        assert not document_mapping_path.exists()


def _write_mock_run_file(
    path: Path,
    rows: list[tuple[str, str, int, float, str]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as output_file:
        for query_id, doc_id, rank, score, software in rows:
            output_file.write(f"{query_id} Q0 {doc_id} {rank} {score} {software}\n")
    return path


def _default_mock_run_rows(
    query_ids: list[str],
    doc_ids: list[str],
    software: str = "ground-truth",
) -> list[tuple[str, str, int, float, str]]:
    rows = []
    for query_id in query_ids:
        for rank, doc_id in enumerate(doc_ids, start=1):
            rows.append((query_id, doc_id, rank, float(len(doc_ids) - rank + 1), software))
    return rows


def _mock_ground_truth_run_writer(
    monkeypatch,
    rows: list[tuple[str, str, int, float, str]] | None = None,
):
    def fake_write_ground_truth_run_file(source_dir: Path, target_dir: Path, dataset: str) -> Path:
        assert source_dir.exists()
        assert dataset
        query_ids = (source_dir / "query" / "query-ids.txt").read_text().strip().splitlines()
        doc_ids = (source_dir / "doc" / "doc-ids.txt").read_text().strip().splitlines()
        run_rows = rows if rows is not None else _default_mock_run_rows(query_ids, doc_ids)
        target_dir.mkdir(parents=True, exist_ok=False)
        return _write_mock_run_file(target_dir / "run.txt.gz", run_rows)

    monkeypatch.setattr(sisap_io_module, "_write_ground_truth_run_file", fake_write_ground_truth_run_file)


def test_export_embeddings_to_sisap_writes_expected_bundle(tmp_path, monkeypatch):
    source_dir = _write_embedding_dir(tmp_path / "source")
    target_dir = tmp_path / "target"
    _mock_ground_truth_run_writer(
        monkeypatch,
        rows=[
            ("q0", "d1", 1, 2.0, "ground-truth"),
            ("q0", "d0", 2, 1.0, "ground-truth"),
            ("q0", "d2", 3, 0.0, "ground-truth"),
            ("q1", "d2", 1, 3.0, "ground-truth"),
            ("q1", "d0", 2, 0.0, "ground-truth"),
            ("q1", "d1", 3, 0.0, "ground-truth"),
        ],
    )

    export_embeddings_to_sisap(source_dir, target_dir, "rteb/aila/casedocs")

    assert json.loads((target_dir / "config.json").read_text()) == {
        "task": "task3",
        "data": "train",
        "queries": "otest/queries",
        "gt_I": "otest/knns",
        "k": 30,
        "sparse": True,
        "dataset_name": "rteb-aila-casedocs",
        "filename": "benchmark-dev-rteb-aila-casedocs.h5",
    }

    with gzip.open(target_dir / "query-id-to-index.json.gz", "rt", encoding="utf-8") as input_file:
        assert json.load(input_file) == {"q0": 0, "q1": 1}

    with gzip.open(target_dir / "document-id-to-index.json.gz", "rt", encoding="utf-8") as input_file:
        assert json.load(input_file) == {"d0": 0, "d1": 1, "d2": 2}

    with gzip.open(target_dir / "run.txt.gz", "rt", encoding="utf-8") as input_file:
        assert input_file.read() == (
            "q0 Q0 d1 1 2.0 ground-truth\n"
            "q0 Q0 d0 2 1.0 ground-truth\n"
            "q0 Q0 d2 3 0.0 ground-truth\n"
            "q1 Q0 d2 1 3.0 ground-truth\n"
            "q1 Q0 d0 2 0.0 ground-truth\n"
            "q1 Q0 d1 3 0.0 ground-truth\n"
        )

    with h5py.File(target_dir / "benchmark-dev-rteb-aila-casedocs.h5", "r") as h5_file:
        assert h5_file["train/data"][:].tolist() == [1.0, 2.0, 3.0]
        assert h5_file["train/indices"][:].tolist() == [0, 0, 1]
        assert h5_file["train/indptr"][:].tolist() == [0, 1, 2, 3]
        assert h5_file["train"].attrs["shape"].tolist() == [3, 2]
        assert h5_file["otest/queries/data"][:].tolist() == [1.0, 1.0]
        assert h5_file["otest/queries/indices"][:].tolist() == [0, 1]
        assert h5_file["otest/queries/indptr"][:].tolist() == [0, 1, 2]
        assert h5_file["otest/queries"].attrs["shape"].tolist() == [2, 2]
        assert h5_file["otest"].attrs["algo"] == "ground-truth"
        assert h5_file["otest"].attrs["querytime"] == 0.0
        assert h5_file["otest/knns"][:].tolist() == [[2, 1, 3], [3, 1, 2]]
        assert h5_file["otest/dists"][:].tolist() == [[2.0, 1.0, 0.0], [3.0, 0.0, 0.0]]


def test_download_embeddings_cli_supports_sisap_format(tmp_path, monkeypatch):
    class FakeClient:
        def get_run_output(self, system_name, dataset_name):
            assert system_name == "lsr-benchmark/lightning-ir/naver-splade-v3"
            assert dataset_name == "aila-casedocs-20260426-training"
            return _write_embedding_dir(tmp_path / "source")

        def public_system_details(self, team_name, system_name):
            assert team_name == "lightning-ir"
            assert system_name == "naver-splade-v3"
            return {
                "command": "/lightning-ir.py --dataset $inputDataset --save_dir $outputDir --model naver/splade-v3",
            }

    monkeypatch.setattr(_download, "Client", FakeClient)
    monkeypatch.setattr(sisap_io_module, "Client", FakeClient)
    _mock_ground_truth_run_writer(monkeypatch)

    runner = CliRunner()
    target_dir = tmp_path / "target"
    result = runner.invoke(
        main,
        [
            "download-embeddings",
            "--dataset",
            "rteb/aila/casedocs",
            "--embedding",
            "naver-splade-v3",
            "--format",
            "sisap",
            "--out",
            str(target_dir),
        ],
    )

    assert result.exit_code == 0
    assert str(target_dir) in result.output
    assert (target_dir / "config.json").exists()
    assert yaml.safe_load((target_dir / "document-embedding-metadata.yml").read_text()) == {
        "tag": "test-tag",
        "data": {
            "test collection": {
                "name": "aila-casedocs-20260426-training",
                "ir-datasets-id": "rteb/aila/casedocs",
            },
            "embedding model": {
                "name": "naver/splade-v3",
                "tira-embedding-software": "lsr-benchmark/lightning-ir/naver-splade-v3",
            },
        },
        "implementation": {"script": {"path": "/run.py"}},
        "doc-metadata": True,
    }
    assert yaml.safe_load((target_dir / "query-embedding-metadata.yml").read_text()) == {
        "tag": "test-tag",
        "data": {
            "test collection": {
                "name": "aila-casedocs-20260426-training",
                "ir-datasets-id": "rteb/aila/casedocs",
            },
            "embedding model": {
                "name": "naver/splade-v3",
                "tira-embedding-software": "lsr-benchmark/lightning-ir/naver-splade-v3",
            },
        },
        "implementation": {"script": {"path": "/run.py"}},
        "query-metadata": True,
    }


def test_download_embeddings_cli_supports_local_directory(tmp_path, monkeypatch):
    source_dir = _write_embedding_dir(tmp_path / "source")
    target_dir = tmp_path / "target"
    _mock_ground_truth_run_writer(monkeypatch)

    class UnexpectedClient:
        def __init__(self):
            raise AssertionError("Local directory mode must not access TIRA.")

    monkeypatch.setattr(_download, "Client", UnexpectedClient)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "download-embeddings",
            "--directory",
            str(source_dir),
            "--format",
            "sisap",
            "--out",
            str(target_dir),
        ],
    )

    assert result.exit_code == 0
    assert (target_dir / "benchmark-dev-rteb-aila-casedocs.h5").exists()
    assert yaml.safe_load((target_dir / "document-embedding-metadata.yml").read_text())["data"] == {
        "embedding model": {
            "name": "naver/splade-v3",
            "tira-embedding-software": "lsr-benchmark/lightning-ir/naver-splade-v3",
        },
        "test collection": {
            "ir-datasets-id": "rteb/aila/casedocs",
            "name": "/tira-data/input",
        },
    }


def test_download_embeddings_cli_supports_joint_dataset_metadata(tmp_path, monkeypatch):
    source_dir = _write_embedding_dir(tmp_path / "source")
    target_dir = tmp_path / "target"
    joint_dataset = "msmarco-passage-trec-dl-2019+2020-judged"
    dataset_names = JOINT_TO_DATASETS[joint_dataset]["datasets"]
    expected_metadata = {}
    for embedding_type in ("doc", "query"):
        (source_dir / embedding_type / f"{embedding_type}-ir-metadata.yml").unlink()
        for index, dataset_name in enumerate(dataset_names):
            metadata = yaml.safe_load(_metadata_fixture_text(f"{embedding_type}-metadata-{index}"))
            metadata["data"]["test collection"] = {
                "ir-datasets-id": f"irds/{index}",
                "name": dataset_name,
            }
            metadata_path = source_dir / embedding_type / f"d{index}-{embedding_type}-ir-metadata.yml"
            metadata_path.write_text(yaml.safe_dump(metadata))
            target_type = "document" if embedding_type == "doc" else "query"
            expected_metadata[f"d{index}-{target_type}-embedding-metadata.yml"] = metadata_path.read_bytes()
    _mock_ground_truth_run_writer(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "download-embeddings",
            "--directory",
            str(source_dir),
            "--format",
            "sisap",
            "--out",
            str(target_dir),
        ],
    )

    assert result.exit_code == 0
    assert (target_dir / f"benchmark-dev-{joint_dataset}.h5").exists()
    for metadata_file, expected in expected_metadata.items():
        assert (target_dir / metadata_file).read_bytes() == expected


def test_download_embeddings_cli_copies_local_directory(tmp_path):
    source_dir = _write_embedding_dir(tmp_path / "source")
    target_dir = tmp_path / "target"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "download-embeddings",
            "--directory",
            str(source_dir),
            "--out",
            str(target_dir),
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == str(target_dir)
    assert (target_dir / "doc" / "doc-embeddings.npz").read_bytes() == (
        source_dir / "doc" / "doc-embeddings.npz"
    ).read_bytes()
    assert (target_dir / "query" / "query-ir-metadata.yml").read_text() == (
        source_dir / "query" / "query-ir-metadata.yml"
    ).read_text()


def test_download_embeddings_cli_rejects_directory_with_remote_identifiers(tmp_path):
    source_dir = _write_embedding_dir(tmp_path / "source")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "download-embeddings",
            "--directory",
            str(source_dir),
            "--dataset",
            "rteb/aila/casedocs",
            "--embedding",
            "naver-splade-v3",
        ],
    )

    assert result.exit_code != 0
    assert "--directory cannot be combined with --dataset or --embedding" in result.output


def test_download_embeddings_cli_rejects_mismatched_directory_metadata(tmp_path):
    source_dir = _write_embedding_dir(tmp_path / "source")
    query_metadata_path = source_dir / "query" / "query-ir-metadata.yml"
    query_metadata = yaml.safe_load(query_metadata_path.read_text())
    query_metadata["data"]["embedding model"]["name"] = "different-model"
    query_metadata_path.write_text(yaml.safe_dump(query_metadata))

    runner = CliRunner()
    result = runner.invoke(main, ["download-embeddings", "--directory", str(source_dir)])

    assert result.exit_code != 0
    assert "Document and query embedding metadata do not match" in result.output


def test_download_embeddings_cli_reports_missing_sisap_dependency(tmp_path, monkeypatch):
    class FakeClient:
        def get_run_output(self, system_name, dataset_name):
            return _write_embedding_dir(tmp_path / "source")

        def public_system_details(self, team_name, system_name):
            return {"display_name": "Naver SPLADE v3"}

    def fail_to_export(*args, **kwargs):
        raise MissingSisapDependencyError(
            "SISAP export requires the optional dependency group. Install it with `pip install lsr-benchmark[sisap]`."
        )

    monkeypatch.setattr(_download, "Client", FakeClient)
    monkeypatch.setattr(_download, "export_embeddings_to_sisap", fail_to_export)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "download-embeddings",
            "--dataset",
            "rteb/aila/casedocs",
            "--embedding",
            "naver-splade-v3",
            "--format",
            "sisap",
            "--out",
            str(tmp_path / "target"),
        ],
    )

    assert result.exit_code != 0
    assert "pip install lsr-benchmark[sisap]" in result.output


def test_tiny_sisap_example_bundle_has_expected_structure():
    _assert_sisap_bundle_structure(TEST_RESOURCES / "example-sisap-task3", expect_mappings=True)


def test_exported_sisap_bundle_matches_reference_structure(tmp_path, monkeypatch):
    source_dir = _write_embedding_dir(tmp_path / "source", doc_count=30)
    target_dir = tmp_path / "target"
    _mock_ground_truth_run_writer(monkeypatch)

    export_embeddings_to_sisap(source_dir, target_dir, "rteb/aila/casedocs")

    _assert_sisap_bundle_structure(target_dir, expect_mappings=True)


def test_exported_sisap_bundle_matches_example_dataset_ids_and_sizes(tmp_path, monkeypatch):
    dataset_dir = TEST_RESOURCES / "example-dataset"
    register_to_ir_datasets(str(dataset_dir))
    dataset = load(str(dataset_dir))

    query_ids = [query.query_id for query in dataset.queries_iter()]
    doc_ids = [doc.doc_id for doc in dataset.docs_iter()]

    source_dir = _write_embedding_dir_with_ids(tmp_path / "source", doc_ids=doc_ids, query_ids=query_ids)
    target_dir = tmp_path / "target"
    _mock_ground_truth_run_writer(monkeypatch)

    export_embeddings_to_sisap(source_dir, target_dir, str(dataset_dir))

    with gzip.open(target_dir / "query-id-to-index.json.gz", "rt", encoding="utf-8") as input_file:
        query_mapping = json.load(input_file)
    with gzip.open(target_dir / "document-id-to-index.json.gz", "rt", encoding="utf-8") as input_file:
        document_mapping = json.load(input_file)

    assert query_mapping == {query_id: idx for idx, query_id in enumerate(query_ids)}
    assert document_mapping == {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    config = json.loads((target_dir / "config.json").read_text())
    with h5py.File(target_dir / config["filename"], "r") as h5_file:
        assert len(h5_file["otest/queries/indptr"]) - 1 == len(query_ids)
        assert len(h5_file["train/indptr"]) - 1 == len(doc_ids)
        assert h5_file["otest/queries"].attrs["shape"][0] == len(query_ids)
        assert h5_file["train"].attrs["shape"][0] == len(doc_ids)


def test_build_ground_truth_from_mock_run_file_variants(tmp_path):
    run_path = _write_mock_run_file(
        tmp_path / "run.txt.gz",
        [
            ("q1", "d0", 2, 0.25, "mock-software"),
            ("q0", "d0", 2, 1.0, "mock-software"),
            ("q0", "d1", 1, 2.0, "mock-software"),
            ("q1", "d2", 1, 3.0, "mock-software"),
        ],
    )

    knns, dists, algo = sisap_io_module._build_ground_truth_from_run_file(
        run_path,
        ["q0", "q1"],
        ["d0", "d1", "d2"],
        2,
    )

    assert algo == "mock-software"
    assert knns.tolist() == [[2, 1], [3, 1]]
    assert dists.tolist() == [[2.0, 1.0], [3.0, 0.25]]


def test_build_ground_truth_from_mock_run_file_rejects_multiple_software_names(tmp_path):
    run_path = _write_mock_run_file(
        tmp_path / "run.txt.gz",
        [
            ("q0", "d1", 1, 2.0, "first-software"),
            ("q0", "d0", 2, 1.0, "second-software"),
        ],
    )

    with pytest.raises(ValueError, match="Expected run.txt.gz to contain exactly one software"):
        sisap_io_module._build_ground_truth_from_run_file(run_path, ["q0"], ["d0", "d1"], 2)


def test_convert_sisap_results_to_trec_run_uses_one_based_id_mappings(tmp_path):
    embeddings_dir = _write_sisap_forwarded_metadata(tmp_path / "embeddings")
    _write_json_gz(embeddings_dir / "query-id-to-index.json.gz", {"q0": 0, "q1": 1})
    _write_json_gz(embeddings_dir / "document-id-to-index.json.gz", {"d0": 0, "d1": 1, "d2": 2})
    results_path = _write_results_h5(
        tmp_path / "results.h5",
        knns=[[2, 1, 3], [3, 1, 2]],
        dists=[[2.0, 1.0, 0.0], [3.0, 0.5, 0.25]],
    )
    output_dir = tmp_path / "run"

    convert_sisap_results_to_trec_run(results_path, embeddings_dir, output_dir, "sisap-test")

    with gzip.open(output_dir / "run.txt.gz", "rt", encoding="utf-8") as input_file:
        assert input_file.read() == (
            "q0 Q0 d1 1 3.0 sisap-test\n"
            "q0 Q0 d0 2 2.0 sisap-test\n"
            "q0 Q0 d2 3 1.0 sisap-test\n"
            "q1 Q0 d2 1 3.0 sisap-test\n"
            "q1 Q0 d0 2 2.0 sisap-test\n"
            "q1 Q0 d1 3 1.0 sisap-test\n"
        )
    assert (output_dir / "document-embedding-metadata.yml").read_text() == _metadata_fixture_text("doc-metadata")
    assert (output_dir / "query-embedding-metadata.yml").read_text() == _metadata_fixture_text("query-metadata")


def test_sisap_to_trec_run_cli_writes_gzipped_run(tmp_path):
    embeddings_dir = _write_sisap_forwarded_metadata(tmp_path / "embeddings")
    _write_json_gz(embeddings_dir / "query-id-to-index.json.gz", {"q0": 0, "q1": 1})
    _write_json_gz(embeddings_dir / "document-id-to-index.json.gz", {"d0": 0, "d1": 1, "d2": 2})
    results_path = _write_results_h5(
        tmp_path / "results.h5",
        knns=[[2, 1], [3, 1]],
        dists=[[2.0, 1.0], [3.0, 0.5]],
    )
    output_dir = tmp_path / "run"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "sisap-to-trec-run",
            "--results",
            str(results_path),
            "--embeddings",
            str(embeddings_dir),
            "--output",
            str(output_dir),
            "--system-name",
            "cli-test",
        ],
    )

    assert result.exit_code == 0
    assert str(output_dir) in result.output
    with gzip.open(output_dir / "run.txt.gz", "rt", encoding="utf-8") as input_file:
        assert input_file.read() == (
            "q0 Q0 d1 1 2.0 cli-test\n"
            "q0 Q0 d0 2 1.0 cli-test\n"
            "q1 Q0 d2 1 2.0 cli-test\n"
            "q1 Q0 d0 2 1.0 cli-test\n"
        )
    assert (output_dir / "document-embedding-metadata.yml").read_text() == _metadata_fixture_text("doc-metadata")
    assert (output_dir / "query-embedding-metadata.yml").read_text() == _metadata_fixture_text("query-metadata")


def test_sisap_to_trec_run_cli_processes_tira_style_results_directory(tmp_path):
    embeddings_dir = _write_sisap_forwarded_metadata(tmp_path / "embeddings")
    _write_json_gz(embeddings_dir / "query-id-to-index.json.gz", {"q0": 0, "q1": 1})
    _write_json_gz(embeddings_dir / "document-id-to-index.json.gz", {"d0": 0, "d1": 1, "d2": 2})

    batch_results_dir = tmp_path / "results-root"
    first_results = batch_results_dir / "2026-06-23-21-33-32" / "output" / "results.h5"
    first_results.parent.mkdir(parents=True, exist_ok=True)
    _write_results_h5(first_results, knns=[[2, 1], [3, 1]], dists=[[2.0, 1.0], [3.0, 0.5]])

    second_results = batch_results_dir / "2026-06-23-23-31-33" / "output" / "seismic_nP3000_qc5_hf1_nk20.h5"
    second_results.parent.mkdir(parents=True, exist_ok=True)
    _write_results_h5(second_results, knns=[[1, 2], [2, 3]], dists=[[5.0, 4.0], [3.0, 2.0]])

    output_dir = tmp_path / "converted-runs"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "sisap-to-trec-run",
            "--results",
            str(batch_results_dir),
            "--embeddings",
            str(embeddings_dir),
            "--output",
            str(output_dir),
            "--system-name",
            "cli-test",
        ],
    )

    assert result.exit_code == 0
    assert str(output_dir) in result.output

    first_output_dir = output_dir / "2026-06-23-21-33-32" / "results"
    with gzip.open(first_output_dir / "run.txt.gz", "rt", encoding="utf-8") as input_file:
        assert input_file.read() == (
            "q0 Q0 d1 1 2.0 cli-test\n"
            "q0 Q0 d0 2 1.0 cli-test\n"
            "q1 Q0 d2 1 2.0 cli-test\n"
            "q1 Q0 d0 2 1.0 cli-test\n"
        )

    second_output_dir = output_dir / "2026-06-23-23-31-33" / "seismic_nP3000_qc5_hf1_nk20"
    with gzip.open(second_output_dir / "run.txt.gz", "rt", encoding="utf-8") as input_file:
        assert input_file.read() == (
            "q0 Q0 d0 1 2.0 cli-test\n"
            "q0 Q0 d1 2 1.0 cli-test\n"
            "q1 Q0 d1 1 2.0 cli-test\n"
            "q1 Q0 d2 2 1.0 cli-test\n"
        )

    for converted_dir in (first_output_dir, second_output_dir):
        assert (converted_dir / "document-embedding-metadata.yml").read_text() == _metadata_fixture_text("doc-metadata")
        assert (converted_dir / "query-embedding-metadata.yml").read_text() == _metadata_fixture_text("query-metadata")


def test_convert_sisap_truths_to_qrels_normalizes_zero_based_doc_ids(tmp_path):
    output_path = tmp_path / "qrels.txt"

    convert_sisap_truths_to_qrels(TEST_RESOURCES / "example-sisap-task3", output_path)

    lines = output_path.read_text().splitlines()
    assert len(lines) == 60
    assert lines[:4] == [
        "1 0 2 1",
        "1 0 1 1",
        "1 0 3 1",
        "1 0 4 1",
    ]
    assert lines[30:34] == [
        "2 0 3 1",
        "2 0 1 1",
        "2 0 2 1",
        "2 0 4 1",
    ]


def test_convert_sisap_truths_to_qrels_preserves_one_based_doc_ids(tmp_path, monkeypatch):
    source_dir = _write_embedding_dir(tmp_path / "source", doc_count=30)
    truths_dir = tmp_path / "truths"
    output_path = tmp_path / "qrels.txt"
    _mock_ground_truth_run_writer(monkeypatch)

    export_embeddings_to_sisap(source_dir, truths_dir, "rteb/aila/casedocs")
    convert_sisap_truths_to_qrels(truths_dir, output_path)

    lines = output_path.read_text().splitlines()
    assert len(lines) == 60
    assert lines[:4] == [
        "1 0 1 1",
        "1 0 2 1",
        "1 0 3 1",
        "1 0 4 1",
    ]
    assert lines[30:34] == [
        "2 0 1 1",
        "2 0 2 1",
        "2 0 3 1",
        "2 0 4 1",
    ]


def test_sisap_to_qrels_cli_writes_gzipped_qrels(tmp_path):
    output_path = tmp_path / "qrels.txt.gz"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "sisap-to-qrels",
            "--truths",
            str(TEST_RESOURCES / "example-sisap-task3"),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert str(output_path) in result.output
    with gzip.open(output_path, "rt", encoding="utf-8") as input_file:
        assert input_file.read().splitlines()[:4] == [
            "1 0 2 1",
            "1 0 1 1",
            "1 0 3 1",
            "1 0 4 1",
        ]


def test_convert_sisap_results_to_trec_run_writes_decreasing_custom_scores(tmp_path):
    results_path = _write_results_h5(
        tmp_path / "results.h5",
        knns=[[2, 1, 3], [3, 1, 2]],
        dists=[[0.0, 99.0, -7.0], [42.0, 5.0, 1.0]],
    )
    output_dir = tmp_path / "run"

    convert_sisap_results_to_trec_run(results_path, TEST_RESOURCES / "example-sisap-task3", output_dir, "test")

    scores_by_query = {}
    with gzip.open(output_dir / "run.txt.gz", "rt", encoding="utf-8") as input_file:
        for line in input_file.read().splitlines():
            qid, _, _, _, score, _ = line.split()
            scores_by_query.setdefault(qid, []).append(float(score))

    assert scores_by_query == {"q0": [3.0, 2.0, 1.0], "q1": [3.0, 2.0, 1.0]}


def test_convert_sisap_results_to_trec_run_rejects_shape_mismatch(tmp_path):
    results_path = _write_results_h5(
        tmp_path / "results.h5",
        knns=[[2, 1, 3], [3, 1, 2]],
        dists=[[2.0, 1.0], [3.0, 0.5]],
    )

    with pytest.raises(ValueError, match="Expected knns and dists to share the same shape"):
        convert_sisap_results_to_trec_run(results_path, TEST_RESOURCES / "example-sisap-task3", tmp_path / "run", "test")


def test_convert_sisap_results_to_trec_run_rejects_out_of_range_one_based_doc_index(tmp_path):
    results_path = _write_results_h5(
        tmp_path / "results.h5",
        knns=[[31, 1, 3], [3, 1, 2]],
        dists=[[2.0, 1.0, 0.0], [3.0, 0.5, 0.25]],
    )

    with pytest.raises(ValueError, match="Document index 31 .* one-based SISAP document IDs"):
        convert_sisap_results_to_trec_run(results_path, TEST_RESOURCES / "example-sisap-task3", tmp_path / "run", "test")


def test_invert_id_mapping_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="Duplicate query index 0"):
        _invert_id_mapping({"q0": 0, "q1": 0}, "query")


def test_invert_id_mapping_rejects_missing_indices():
    with pytest.raises(ValueError, match=r"Missing document identifiers for indices \[1\]"):
        _invert_id_mapping({"d0": 0, "d2": 2}, "document")


def test_decreasing_scores_for_ranking_returns_descending_values():
    assert _decreasing_scores_for_ranking(4) == [4.0, 3.0, 2.0, 1.0]


def test_normalize_truth_doc_ids_converts_zero_based_indices():
    normalized = _normalize_truth_doc_ids(np.array([[0, 2], [1, 3]], dtype=np.int32))
    assert normalized.tolist() == [[1, 3], [2, 4]]


def test_normalize_truth_doc_ids_preserves_one_based_indices():
    normalized = _normalize_truth_doc_ids(np.array([[1, 3], [2, 4]], dtype=np.int32))
    assert normalized.tolist() == [[1, 3], [2, 4]]
