from pathlib import Path

import pytest
from click.testing import CliRunner
import tira.io_utils

from lsr_benchmark import main
from lsr_benchmark.datasets import IR_DATASET_TO_TIRA_DATASET
from lsr_benchmark.retrieval_suites import RETRIEVAL_SUITES


DATASET = "tiny-example-20251002_0-training"
EMBEDDING = "naver-splade-v3"


@pytest.fixture
def mocked_retrieval(monkeypatch):
    retrieval_module = __import__(
        "lsr_benchmark._commands._retrieval", fromlist=["_retrieval"]
    )
    calls = []

    def unexpected_tira_client():
        raise AssertionError("The test must not contact TIRA.")

    monkeypatch.setattr(retrieval_module, "Client", unexpected_tira_client)
    monkeypatch.setattr(
        retrieval_module,
        "verify_docker_installation",
        lambda: (retrieval_module.FormatMsgType.OK, ""),
    )
    monkeypatch.setattr(
        retrieval_module,
        "docker_supported_target_platform",
        lambda: "linux/amd64",
    )
    monkeypatch.setattr(
        retrieval_module,
        "get_approach_to_execution",
        lambda approaches, platform, embedding, print_message: {
            approach: {
                "tag": f"image/{approach}",
                "command": f"/run-{approach}",
            }
            for approach in approaches
        },
    )
    monkeypatch.setattr(
        retrieval_module,
        "run_retrieval_engine",
        lambda image, command, dataset, embedding, output_dir, **kwargs: calls.append(
            {
                "image": image,
                "command": command,
                "dataset": dataset,
                "embedding": embedding,
                "output_dir": output_dir,
                **kwargs,
            }
        ),
    )
    monkeypatch.setattr(retrieval_module.os, "system", lambda command: 0)
    return retrieval_module, calls


def test_retrieval_command_runs_selected_approaches(mocked_retrieval, tmp_path):
    _, calls = mocked_retrieval

    result = CliRunner().invoke(
        main,
        [
            "retrieval",
            "--out",
            str(tmp_path),
            "--dataset",
            DATASET,
            "--embedding",
            EMBEDDING,
            "--cpus",
            "4",
            "--memory",
            "8g",
            "seismic",
            "kannolo",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "image": "image/seismic",
            "command": "/run-seismic",
            "dataset": DATASET,
            "embedding": EMBEDDING,
            "output_dir": Path(tmp_path) / DATASET / EMBEDDING / "seismic",
            "platform": "linux/amd64",
            "cpus": 4,
            "memory": "8g",
        },
        {
            "image": "image/kannolo",
            "command": "/run-kannolo",
            "dataset": DATASET,
            "embedding": EMBEDDING,
            "output_dir": Path(tmp_path) / DATASET / EMBEDDING / "kannolo",
            "platform": "linux/amd64",
            "cpus": 4,
            "memory": "8g",
        },
    ]


def test_retrieval_command_expands_suite(mocked_retrieval, tmp_path):
    _, calls = mocked_retrieval
    suite = RETRIEVAL_SUITES["reneuir-2026/small"]

    result = CliRunner().invoke(
        main,
        [
            "retrieval",
            "--suite",
            "reneuir-2026/small",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == (
        len(suite["retrieval_engines"])
        * len(suite["datasets"])
        * len(suite["embeddings"])
    )
    assert [call["image"] for call in calls] == [
        f"image/{approach}" for approach in suite["retrieval_engines"]
    ]
    assert {call["dataset"] for call in calls} == {
        IR_DATASET_TO_TIRA_DATASET[suite["datasets"][0]]
    }
    assert {call["embedding"] for call in calls} == set(suite["embeddings"])


def test_retrieval_command_rejects_suite_with_manual_approach(
    mocked_retrieval, tmp_path
):
    _, calls = mocked_retrieval

    result = CliRunner().invoke(
        main,
        [
            "retrieval",
            "--suite",
            "reneuir-2026/small",
            "--out",
            str(tmp_path),
            "seismic",
        ],
    )

    assert result.exit_code == 2
    assert "--suite cannot be combined" in result.output
    assert calls == []


def test_retrieval_command_rejects_local_dataset_with_remote_embedding(
    mocked_retrieval, tmp_path
):
    _, calls = mocked_retrieval
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "retrieval",
            "--out",
            str(tmp_path / "output"),
            "--dataset",
            str(dataset_dir),
            "--embedding",
            EMBEDDING,
            "seismic",
        ],
    )

    assert result.exit_code == 2
    assert "local dataset has to be used in combination with local embeddings" in (
        result.output
    )
    assert calls == []


def test_retrieval_command_calls_mocked_tira_local_execution(monkeypatch, tmp_path):
    retrieval_module = __import__(
        "lsr_benchmark._commands._retrieval", fromlist=["_retrieval"]
    )
    dataset_dir = tmp_path / "dataset"
    embedding_dir = tmp_path / "embeddings"
    execution_dir = tmp_path / "execution"
    output_root = tmp_path / "output"
    dataset_dir.mkdir()
    embedding_dir.mkdir()
    execution_dir.mkdir()
    (execution_dir / "retrieval-metadata.yml").write_text("tag: test\n")
    execution_arguments = {}

    class LocalExecution:
        def run(self, **kwargs):
            execution_arguments.update(kwargs)

    class TiraClient:
        local_execution = LocalExecution()

    monkeypatch.setattr(retrieval_module, "Client", TiraClient)
    monkeypatch.setattr(
        retrieval_module,
        "verify_docker_installation",
        lambda: (retrieval_module.FormatMsgType.OK, ""),
    )
    monkeypatch.setattr(
        retrieval_module,
        "docker_supported_target_platform",
        lambda: "linux/amd64",
    )
    monkeypatch.setattr(
        retrieval_module,
        "get_approach_to_execution",
        lambda approaches, platform, embedding, print_message: {
            "seismic": {"tag": "image/seismic", "command": "/run-seismic"}
        },
    )
    monkeypatch.setattr(
        retrieval_module, "temporary_directory", lambda: execution_dir
    )
    monkeypatch.setattr(
        retrieval_module,
        "check_format",
        lambda directory, expected_files, context: (
            retrieval_module.FormatMsgType.OK,
            "",
        ),
    )
    monkeypatch.setattr(tira.io_utils, "patch_ir_metadata", lambda *args: None)
    monkeypatch.setattr(retrieval_module.os, "system", lambda command: 0)

    result = CliRunner().invoke(
        main,
        [
            "retrieval",
            "--out",
            str(output_root),
            "--dataset",
            str(dataset_dir),
            "--embedding",
            str(embedding_dir),
            "--cpus",
            "2",
            "--memory",
            "4g",
            "seismic",
        ],
    )

    assert result.exit_code == 0, result.output
    assert execution_arguments == {
        "image": "image/seismic",
        "command": "/run-seismic",
        "input_dir": dataset_dir.resolve(),
        "output_dir": execution_dir,
        "allow_network": False,
        "input_run": embedding_dir.resolve(),
        "mount_directory": {"embeddings": embedding_dir.resolve()},
        "platform": "linux/amd64",
        "cpu_count": 2,
        "mem_limit": "4g",
    }
    assert (
        output_root
        / dataset_dir.stem
        / embedding_dir.stem
        / "seismic"
        / "retrieval-metadata.yml"
    ).is_file()


def test_retrieval_command_reports_mocked_execution_failure(
    mocked_retrieval, monkeypatch, tmp_path
):
    retrieval_module, _ = mocked_retrieval
    attempted_approaches = []

    def execute_with_one_failure(image, *args, **kwargs):
        attempted_approaches.append(image)
        if image == "image/seismic":
            raise ValueError("mocked retrieval failure")

    monkeypatch.setattr(
        retrieval_module, "run_retrieval_engine", execute_with_one_failure
    )

    result = CliRunner().invoke(
        main,
        [
            "retrieval",
            "--out",
            str(tmp_path),
            "--dataset",
            DATASET,
            "--embedding",
            EMBEDDING,
            "seismic",
            "kannolo",
        ],
    )

    assert result.exit_code == 1
    assert "mocked retrieval failure" in result.output
    assert "1 retrieval configuration(s) failed" in result.output
    assert attempted_approaches == ["image/seismic", "image/kannolo"]
