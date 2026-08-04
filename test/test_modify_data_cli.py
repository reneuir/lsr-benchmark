import io
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from lsr_benchmark._commands._modify_data import DuplicateBehaviour, load_and_merge_embeddings, perform_quantization, prefix_json


def test_load_and_merge_embeddings(tmp_path):
    path1 = tmp_path / "run1"
    path1.mkdir()
    np.savez_compressed(
        path1 / "test.npz", data=np.array([0.1, 0.2]), indices=np.array([0, 1]), indptr=np.array([0, 2])
    )
    mask1 = np.array([True])

    path2 = tmp_path / "run2"
    path2.mkdir()
    np.savez_compressed(
        path2 / "test.npz", data=np.array([0.3, 0.4, 0.5]), indices=np.array([0, 1, 2]), indptr=np.array([0, 3])
    )
    mask2 = np.array([True])

    merged = load_and_merge_embeddings([path1, path2], "test.npz", keep_masks=[mask1, mask2])

    assert np.array_equal(merged["data"], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert np.array_equal(merged["indices"], np.array([0, 1, 0, 1, 2]))
    assert np.array_equal(merged["indptr"], np.array([0, 2, 5]))


def test_load_and_merge_embeddings_with_skips(tmp_path):
    path1 = tmp_path / "run1"
    path1.mkdir()
    np.savez_compressed(
        path1 / "test.npz",
        data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        indices=np.array([0, 0, 1, 0, 1, 2]),
        indptr=np.array([0, 1, 3, 6]),
    )

    mask1 = np.array([True, False, True])

    merged = load_and_merge_embeddings([path1], "test.npz", keep_masks=[mask1])

    assert np.array_equal(merged["data"], np.array([0.1, 0.4, 0.5, 0.6]))
    assert np.array_equal(merged["indices"], np.array([0, 0, 1, 2]))
    assert np.array_equal(merged["indptr"], np.array([0, 1, 4]))


def test_prefix_json_prefix_behaviour():
    input_data = '{"qid": "1", "text": "hello"}\n\n{"qid": "2", "text": "world"}\n'
    infile = io.StringIO(input_data)
    outfile = io.StringIO()
    seen_ids = set()

    prefix_json(infile, outfile, DuplicateBehaviour.PREFIX, "qid", seen_ids, "d1")

    outfile.seek(0)
    lines = outfile.readlines()

    assert len(lines) == 2  # empty line should be ignored
    assert json.loads(lines[0]) == {"qid": "d1-1", "text": "hello"}
    assert json.loads(lines[1]) == {"qid": "d1-2", "text": "world"}


def test_prefix_json_skip_behaviour():
    input_data = '{"qid": "1", "text": "hello"}\n{"qid": "1", "text": "hello"}\n{"qid": "2", "text": "world"}\n'
    infile = io.StringIO(input_data)
    outfile = io.StringIO()
    seen_ids = set()

    prefix_json(infile, outfile, DuplicateBehaviour.SKIP, "qid", seen_ids, "d1")

    outfile.seek(0)
    lines = outfile.readlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {"qid": "1", "text": "hello"}
    assert json.loads(lines[1]) == {"qid": "2", "text": "world"}
    assert seen_ids == {"1", "2"}


def test_prefix_json_fail_behaviour():
    input_data = '{"qid": "1", "text": "hello"}\n{"qid": "1", "text": "hello"}\n'
    infile = io.StringIO(input_data)
    outfile = io.StringIO()
    seen_ids = set()

    with pytest.raises(ValueError, match="Duplicate qid '1' found while joining datasets."):
        prefix_json(infile, outfile, DuplicateBehaviour.FAIL, "qid", seen_ids, "d1")


def create_mock_embedding_dir(base_path: Path, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray) -> None:
    for emb_type in ["doc", "query"]:
        dir_path = base_path / emb_type
        dir_path.mkdir(parents=True)

        np.savez_compressed(dir_path / f"{emb_type}-embeddings.npz", data=data, indices=indices, indptr=indptr)

        with open(dir_path / f"{emb_type}-ids.txt", "w") as f:
            f.write("id1\nid2\n")

        with open(dir_path / f"{emb_type}-ir-metadata.yml", "w") as f:
            yaml.dump({"data": {"test collection": {}}}, f)


def test_perform_quantization_basic_precisions(tmp_path):
    emb_path = tmp_path / "input"
    tira_dir = tmp_path / "tira"

    data = np.array([-2.5, 0.0, 1.5, 3.0], dtype=np.float32)
    indices = np.array([0, 1, 0, 1])
    indptr = np.array([0, 2, 4])
    create_mock_embedding_dir(emb_path, data, indices, indptr)

    # Binary
    out_bin = perform_quantization(emb_path, "binary", None, "test-ds", "emb-model", str(tira_dir))
    with np.load(out_bin / "doc" / "doc-embeddings.npz") as npz:
        assert np.array_equal(npz["data"], np.array([0, 0, 1, 1], dtype=np.int8))

    # Ternary
    out_ter = perform_quantization(emb_path, "ternary", None, "test-ds", "emb-model", str(tira_dir))
    with np.load(out_ter / "doc" / "doc-embeddings.npz") as npz:
        assert np.array_equal(npz["data"], np.array([-1, 0, 1, 1], dtype=np.int8))

    # FP16
    out_fp = perform_quantization(emb_path, "fp16", None, "test-ds", "emb-model", str(tira_dir))
    with np.load(out_fp / "doc" / "doc-embeddings.npz") as npz:
        assert npz["data"].dtype == np.float16
        assert np.allclose(npz["data"], data.astype(np.float16))

    # Metadata
    with open(out_bin / "doc" / "doc-ir-metadata.yml", "r") as f:
        meta = yaml.safe_load(f)
        assert meta["data"]["test collection"]["quantization"] == "binary"


def test_perform_quantization_8bit_scaling(tmp_path):
    emb_path = tmp_path / "input_8bit"
    tira_dir = tmp_path / "tira"

    # Doc 1: [0.0, 100.0]
    # Doc 2: [25.5, 150.0]
    # Doc 3: [51.0, 200.0]
    data = np.array([0.0, 100.0, 25.5, 150.0, 51.0, 200.0])
    indices = np.array([0, 1, 0, 1, 0, 1])
    indptr = np.array([0, 2, 4, 6])
    create_mock_embedding_dir(emb_path, data, indices, indptr)

    # Col 0 min=0, max=51 -> scale=51/255=0.2
    #   Doc 1: 0 -> 0.
    #   Doc 2: 25.5 -> floor(25.5/0.2) = floor(127.5) = 127
    #   Doc 3: 51 -> 255
    # Col 1 min=100, max=200 -> scale=100/255=0.3921
    #   Doc 1: 100 -> 0.
    #   Doc 2: 150 -> floor((150 - 100) / 0.3921) = floor(127.5) = 127.
    #   Doc 3: 200 -> 255.
    out_uint8 = perform_quantization(emb_path, "uint8", None, "test-ds-2", "emb-model", str(tira_dir))
    with np.load(out_uint8 / "doc" / "doc-embeddings.npz") as npz:
        assert npz["data"].dtype == np.uint8
        assert np.array_equal(npz["data"], np.array([0, 0, 127, 127, 255, 255], dtype=np.uint8))

    out_int8 = perform_quantization(emb_path, "int8", None, "test-ds-3", "emb-model", str(tira_dir))
    with np.load(out_int8 / "doc" / "doc-embeddings.npz") as npz:
        assert npz["data"].dtype == np.int8
        assert np.array_equal(npz["data"], np.array([-128, -128, -1, -1, 127, 127], dtype=np.int8))
