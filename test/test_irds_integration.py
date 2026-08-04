from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory

from ir_datasets import load
from pytest import raises

from lsr_benchmark import register_to_ir_datasets
from lsr_benchmark import load as lsr_load
from lsr_benchmark.datasets import IR_DATASET_TO_TIRA_DATASET


def test_fails_for_non_existing_dataset() -> None:
    with raises(ValueError):
        register_to_ir_datasets("this-does-not-exist")


def test_works_for_none_as_dataset() -> None:
    register_to_ir_datasets()


def test_from_local_directory() -> None:
    resource_dir = str(Path(__file__).parent / "resources" / "example-dataset")
    register_to_ir_datasets(resource_dir)
    ds = load(resource_dir)

    assert 3 == sum(1 for _ in ds.queries_iter())
    assert 4 == sum(1 for _ in ds.docs_iter())


def test_from_local_directory_with_prefix() -> None:
    resource_dir = str(Path(__file__).parent / "resources" / "example-dataset")
    register_to_ir_datasets(resource_dir)
    ds = load("lsr-benchmark/" + resource_dir)

    assert 3 == sum(1 for _ in ds.queries_iter())
    assert 4 == sum(1 for _ in ds.docs_iter())


def test_ms_marco_dataset() -> None:
    register_to_ir_datasets("msmarco-passage/trec-dl-2019/judged")
    ds = load("lsr-benchmark/msmarco-passage/trec-dl-2019/judged")

    assert "lsr-benchmark/msmarco-passage/trec-dl-2019/judged" == ds.dataset_id()
    assert 43 == sum(1 for _ in ds.queries_iter())
    assert 32123 == sum(1 for _ in ds.docs_iter())
    assert 9260 == sum(1 for _ in ds.qrels_iter())


def test_private_dataset_error_message() -> None:
    with TemporaryDirectory() as tmp_dir:
        environ["TIRA_CACHE_DIR"] = str(tmp_dir)
        with raises(ValueError) as e:
            register_to_ir_datasets("disks45/nocr/trec-robust-2004/fold3")
            ds = load("lsr-benchmark/disks45/nocr/trec-robust-2004/fold3")
            list(ds.queries_iter())
        del environ["TIRA_CACHE_DIR"]
        assert e is not None


def test_all_datasets_can_be_loaded() -> None:
    for ds in IR_DATASET_TO_TIRA_DATASET.keys():
        register_to_ir_datasets(ds)
        assert load(f"lsr-benchmark/{ds}") is not None

    for ds in IR_DATASET_TO_TIRA_DATASET.values():
        register_to_ir_datasets(ds)
        assert load(ds) is not None

def test_private_joint_dataset_can_load_qrels() -> None:
    with TemporaryDirectory() as tmp_dir:
        environ["TIRA_CACHE_DIR"] = str(tmp_dir)
        register_to_ir_datasets("disks45-nocr-trec-robust-2004-fold1+2+3+4+5")
        ds = lsr_load("disks45-nocr-trec-robust-2004-fold1+2+3+4+5")
        del environ["TIRA_CACHE_DIR"]
        assert len(list(ds.qrels)) > 0
