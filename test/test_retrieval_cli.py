from pathlib import Path

import click
import pytest
from click.testing import CliRunner
import tira.io_utils

from lsr_benchmark import main
from lsr_benchmark._commands._retrieval import (
    RetrievalJob,
    build_retrieval_jobs,
    execute_retrieval_jobs,
    normalize_retrieval_inputs,
    report_retrieval_stats,
    resolve_execution_platform,
    validate_retrieval_selection,
)
from lsr_benchmark.datasets import IR_DATASET_TO_TIRA_DATASET
from lsr_benchmark.retrieval_hyperparameters import load_retrieval_hyperparameters
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


def test_normalize_retrieval_inputs_expands_defaults(monkeypatch):
    retrieval_module = __import__(
        "lsr_benchmark._commands._retrieval", fromlist=["_retrieval"]
    )
    monkeypatch.setattr(retrieval_module, "all_datasets", lambda: ["dataset"])
    monkeypatch.setattr(retrieval_module, "all_embeddings", lambda: ["embedding"])

    datasets, embeddings = normalize_retrieval_inputs((), ())

    assert datasets == ["dataset"]
    assert embeddings == ["embedding"]


def test_normalize_retrieval_inputs_prioritizes_none_embedding():
    datasets, embeddings = normalize_retrieval_inputs(
        [DATASET], [EMBEDDING, "none"]
    )

    assert datasets == [DATASET]
    assert embeddings == ["none"]


def test_validate_retrieval_selection_reports_first_missing_input():
    retrieval_module = __import__(
        "lsr_benchmark._commands._retrieval", fromlist=["_retrieval"]
    )
    messages = []

    valid = validate_retrieval_selection(
        ["seismic"],
        [],
        [EMBEDDING],
        lambda message, level: messages.append((message, level)),
    )

    assert valid is False
    assert messages == [
        ("No datasets are passed.", retrieval_module.FormatMsgType.ERROR)
    ]


def test_resolve_execution_platform_returns_supported_platform(monkeypatch):
    retrieval_module = __import__(
        "lsr_benchmark._commands._retrieval", fromlist=["_retrieval"]
    )
    messages = []
    monkeypatch.setattr(
        retrieval_module,
        "verify_docker_installation",
        lambda: (retrieval_module.FormatMsgType.OK, ""),
    )
    monkeypatch.setattr(
        retrieval_module,
        "docker_supported_target_platform",
        lambda: "linux/arm64",
    )

    platform = resolve_execution_platform(
        lambda message, level: messages.append((message, level))
    )

    assert platform == "linux/arm64"
    assert messages == [
        ("Your TIRA installation is valid.", retrieval_module.FormatMsgType.OK)
    ]


def test_resolve_execution_platform_rejects_missing_docker(monkeypatch):
    retrieval_module = __import__(
        "lsr_benchmark._commands._retrieval", fromlist=["_retrieval"]
    )
    messages = []
    monkeypatch.setattr(
        retrieval_module,
        "verify_docker_installation",
        lambda: (retrieval_module.FormatMsgType.ERROR, "missing"),
    )

    platform = resolve_execution_platform(
        lambda message, level: messages.append((message, level))
    )

    assert platform is None
    assert len(messages) == 1
    assert messages[0][1] == retrieval_module.FormatMsgType.ERROR


def test_resolve_execution_platform_rejects_unsupported_platform(monkeypatch):
    retrieval_module = __import__(
        "lsr_benchmark._commands._retrieval", fromlist=["_retrieval"]
    )
    messages = []
    monkeypatch.setattr(
        retrieval_module,
        "verify_docker_installation",
        lambda: (retrieval_module.FormatMsgType.OK, ""),
    )
    monkeypatch.setattr(
        retrieval_module,
        "docker_supported_target_platform",
        lambda: "windows/amd64",
    )

    platform = resolve_execution_platform(
        lambda message, level: messages.append((message, level))
    )

    assert platform is None
    assert messages[-1] == (
        "The platform windows/amd64 is not supported.",
        retrieval_module.FormatMsgType.ERROR,
    )


def test_build_retrieval_jobs_creates_deterministic_product(tmp_path):
    jobs = build_retrieval_jobs(
        ["seismic", "kannolo"],
        ["dataset"],
        ["embedding"],
        {
            "seismic": {"tag": "image/seismic", "command": "/run-seismic"},
            "kannolo": {"tag": "image/kannolo", "command": "/run-kannolo"},
        },
        tmp_path,
    )

    assert [job.approach for job in jobs] == ["seismic", "kannolo"]
    assert [job.output_dir for job in jobs] == [
        tmp_path / "dataset" / "embedding" / "seismic",
        tmp_path / "dataset" / "embedding" / "kannolo",
    ]
    assert [job.command for job in jobs] == ["/run-seismic", "/run-kannolo"]


def test_build_retrieval_jobs_expands_grid_with_unique_outputs(tmp_path):
    jobs = build_retrieval_jobs(
        ["kannolo"],
        ["dataset"],
        ["embedding"],
        {"kannolo": {"tag": "image/kannolo", "command": "/run-kannolo"}},
        tmp_path,
        load_retrieval_hyperparameters(),
        3,
    )

    assert [job.configuration_id for job in jobs] == [
        "default",
        "ef-search-50",
        "ef-search-100",
    ]
    assert [dict(job.parameters) for job in jobs] == [
        {},
        {"ef-search": 50},
        {"ef-search": 100},
    ]
    assert [job.command for job in jobs] == [
        "/run-kannolo",
        "/run-kannolo --ef-search 50",
        "/run-kannolo --ef-search 100",
    ]
    assert [job.output_dir for job in jobs] == [
        tmp_path / "dataset" / "embedding" / "kannolo" / "default",
        tmp_path / "dataset" / "embedding" / "kannolo" / "ef-search-50",
        tmp_path / "dataset" / "embedding" / "kannolo" / "ef-search-100",
    ]


def test_build_retrieval_jobs_uses_default_for_unknown_grid_approach(tmp_path):
    jobs = build_retrieval_jobs(
        ["custom"],
        ["dataset"],
        ["embedding"],
        {"custom": {"tag": "image/custom", "command": "/run-custom"}},
        tmp_path,
        load_retrieval_hyperparameters(),
        10,
    )

    assert len(jobs) == 1
    assert jobs[0].configuration_id == "default"
    assert dict(jobs[0].parameters) == {}
    assert jobs[0].command == "/run-custom"
    assert jobs[0].output_dir == (
        tmp_path / "dataset" / "embedding" / "custom" / "default"
    )


def test_build_retrieval_jobs_requires_catalog_and_size_together(tmp_path):
    with pytest.raises(ValueError, match="catalog and grid_size"):
        build_retrieval_jobs(
            ["kannolo"],
            ["dataset"],
            ["embedding"],
            {"kannolo": {"tag": "image/kannolo", "command": "/run-kannolo"}},
            tmp_path,
            load_retrieval_hyperparameters(),
        )


def test_execute_retrieval_jobs_aggregates_success_and_failure(
    monkeypatch, tmp_path
):
    retrieval_module = __import__(
        "lsr_benchmark._commands._retrieval", fromlist=["_retrieval"]
    )
    jobs = (
        RetrievalJob(
            "seismic",
            DATASET,
            EMBEDDING,
            DATASET,
            EMBEDDING,
            "image/seismic",
            "/run-seismic",
            tmp_path / "seismic",
        ),
        RetrievalJob(
            "kannolo",
            DATASET,
            EMBEDDING,
            DATASET,
            EMBEDDING,
            "image/kannolo",
            "/run-kannolo",
            tmp_path / "kannolo",
        ),
    )
    messages = []

    def execute(image, *args, **kwargs):
        if image == "image/kannolo":
            raise ValueError("failed")

    monkeypatch.setattr(retrieval_module, "run_retrieval_engine", execute)

    stats, failures = execute_retrieval_jobs(
        jobs,
        "linux/amd64",
        2,
        "4g",
        lambda message, level: messages.append((message, level)),
    )

    assert failures == 1
    assert stats == {
        "seismic": {"datasets": {DATASET}, "embeddings": {EMBEDDING}}
    }
    assert "Approach kannolo failed" in messages[0][0]


def test_report_retrieval_stats_reports_coverage():
    messages = []

    report_retrieval_stats(
        {
            "seismic": {
                "datasets": {"dataset-1", "dataset-2"},
                "embeddings": {"embedding"},
            }
        },
        lambda message, level: messages.append((message, level)),
    )

    assert messages[0][0] == (
        "Approach seismic produced valid outputs on 2 datasets for 1 embeddings."
    )


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


def test_retrieval_command_expands_grid(mocked_retrieval, tmp_path):
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
            "--grid-size",
            "3",
            "kannolo",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [call["command"] for call in calls] == [
        "/run-kannolo",
        "/run-kannolo --ef-search 50",
        "/run-kannolo --ef-search 100",
    ]
    assert [call["output_dir"] for call in calls] == [
        Path(tmp_path) / DATASET / EMBEDDING / "kannolo" / "default",
        Path(tmp_path) / DATASET / EMBEDDING / "kannolo" / "ef-search-50",
        Path(tmp_path) / DATASET / EMBEDDING / "kannolo" / "ef-search-100",
    ]


def test_retrieval_command_rejects_invalid_grid_size(mocked_retrieval, tmp_path):
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
            "--grid-size",
            "0",
            "kannolo",
        ],
    )

    assert result.exit_code == 2
    assert "0 is not in the range x>=1" in result.output
    assert calls == []


def test_retrieval_command_loads_grid_before_docker(
    mocked_retrieval, monkeypatch, tmp_path
):
    retrieval_module, calls = mocked_retrieval

    def invalid_catalog():
        raise click.UsageError("invalid test catalog")

    def unexpected_docker_check():
        raise AssertionError("Docker must not be checked for an invalid catalog.")

    monkeypatch.setattr(
        retrieval_module, "load_retrieval_hyperparameters", invalid_catalog
    )
    monkeypatch.setattr(
        retrieval_module, "verify_docker_installation", unexpected_docker_check
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
            "--grid-size",
            "1",
            "kannolo",
        ],
    )

    assert result.exit_code == 2
    assert "invalid test catalog" in result.output
    assert calls == []


def test_retrieval_command_runs_dataset_embedding_product(
    mocked_retrieval, tmp_path
):
    _, calls = mocked_retrieval
    dataset_dirs = [tmp_path / "dataset-1", tmp_path / "dataset-2"]
    embedding_dirs = [tmp_path / "embedding-1", tmp_path / "embedding-2"]
    for directory in dataset_dirs + embedding_dirs:
        directory.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "retrieval",
            "--out",
            str(tmp_path / "output"),
            "--dataset",
            str(dataset_dirs[0]),
            "--dataset",
            str(dataset_dirs[1]),
            "--embedding",
            str(embedding_dirs[0]),
            "--embedding",
            str(embedding_dirs[1]),
            "seismic",
        ],
    )

    assert result.exit_code == 0, result.output
    assert {
        (
            call["dataset"],
            call["embedding"],
            call["output_dir"].relative_to(tmp_path / "output"),
        )
        for call in calls
    } == {
        (
            dataset,
            embedding,
            Path(dataset.stem) / embedding.stem / "seismic",
        )
        for dataset in dataset_dirs
        for embedding in embedding_dirs
    }


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


def test_retrieval_command_skips_existing_output_without_tira(monkeypatch, tmp_path):
    retrieval_module = __import__(
        "lsr_benchmark._commands._retrieval", fromlist=["_retrieval"]
    )
    output_root = tmp_path / "output"
    existing_output = output_root / DATASET / EMBEDDING / "seismic"
    existing_output.mkdir(parents=True)

    def unexpected_tira_client():
        raise AssertionError("Existing output must not contact TIRA.")

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
            "seismic": {"tag": "image/seismic", "command": "/run-seismic"}
        },
    )
    monkeypatch.setattr(retrieval_module.os, "system", lambda command: 0)

    result = CliRunner().invoke(
        main,
        [
            "retrieval",
            "--out",
            str(output_root),
            "--dataset",
            DATASET,
            "--embedding",
            EMBEDDING,
            "seismic",
        ],
    )

    assert result.exit_code == 0, result.output
    assert existing_output.is_dir()


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
