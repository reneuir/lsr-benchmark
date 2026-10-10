import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import click
from tira.io_utils import (
    FormatMsgType,
    MonitoredExecution,
    docker_supported_target_platform,
    huggingface_model_mounts,
    log_message,
)
from tira.rest_api_client import Client

from lsr_benchmark._commands._retrieval import ChoiceOrPath, resolve_execution_platform
from lsr_benchmark.datasets import all_datasets

DEFAULT_PREFIX = "lsr-benchmark/lightning-ir/"


def normalize_embedding_system(system: str) -> str:
    """Expand a short name like 'naver-splade-v3' to 'lsr-benchmark/lightning-ir/naver-splade-v3'."""
    parts = system.split("/")
    if len(parts) == 1:
        return DEFAULT_PREFIX + system
    if len(parts) != 3:
        raise click.UsageError(
            f"Invalid embedding system {system!r}. Expected <task>/<team>/<system>, "
            f"e.g., '{DEFAULT_PREFIX}naver-splade-v3'."
        )
    return system


def parse_hf_models(mount_hf_model) -> list:
    if not mount_hf_model:
        return []
    if isinstance(mount_hf_model, str):
        return [m for m in mount_hf_model.replace(",", " ").split() if m]
    return list(mount_hf_model)


def run_embedding_system(
    system: str,
    dataset,
    output_dir: Path,
    platform,
    cpus=None,
    memory=None,
    gpus=0,
    new_model: Optional[str] = None,
):
    """Run the TIRA embedding system on the dataset and copy the embeddings to output_dir."""
    if output_dir.exists():
        return
    tira = Client()
    software = tira.docker_software(system)
    image = software.get("public_image_name") or software["tira_image_name"]

    if isinstance(dataset, Path):
        dataset_path = dataset.resolve()
    else:
        dataset_path = tira.download_dataset("lsr-benchmark", dataset)

    command = software["command"]
    additional_volumes = None
    hf_models = parse_hf_models(software.get("mount_hf_model"))
    if new_model:
        if len(hf_models) != 1 or hf_models[0] not in command:
            raise ValueError(
                f"Can not replace the model of {system}: expected exactly one mounted model "
                f"that is used in the command, got {hf_models}."
            )
        command = command.replace(hf_models[0], new_model)
        hf_models = [new_model]
    if hf_models:
        from huggingface_hub import snapshot_download

        for model in hf_models:
            snapshot_download(model)
        mounts = huggingface_model_mounts(hf_models)
        additional_volumes = [k + ":" + v["bind"] + ":" + v["mode"] for k, v in mounts.items()]

    execution_dir = MonitoredExecution().run(
        lambda tmp_dir: tira.local_execution.run(
            image=image,
            command=command,
            input_dir=dataset_path,
            output_dir=tmp_dir,
            allow_network=False,
            additional_volumes=additional_volumes,
            platform=platform if platform else docker_supported_target_platform(),
            cpu_count=cpus,
            mem_limit=memory,
            gpu_count=gpus,
        )
    )
    execution_dir = Path(execution_dir)
    result_dir = execution_dir / "output"
    missing = [d for d in ("doc", "query") if not (result_dir / d).is_dir() or not any((result_dir / d).iterdir())]
    if missing:
        logs = []
        for name in ("stdout.txt", "stderr.txt"):
            log_file = execution_dir / name
            content = log_file.read_text() if log_file.exists() else "<not available>"
            logs.append(f"--- {name} ---\n{content}")
        raise ValueError(
            f"The embedding system {system} produced invalid output (missing or empty: {missing}). "
            f"Execution directory: {execution_dir}\n\n" + "\n\n".join(logs)
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(result_dir, output_dir)


@click.argument("system", type=str)
@click.option(
    "-o", "--out",
    type=str,
    required=True,
    help="The output directory to write the embeddings to.",
)
@click.option(
    "--dataset",
    type=ChoiceOrPath(["all"] + all_datasets()),
    multiple=True,
    help="The datasets to embed (default: all).",
)
@click.option(
    "--mount-new-model",
    type=str,
    default=None,
    metavar="HF_MODEL",
    help="Use this Hugging Face model (e.g., opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini) "
    "instead of the model of the embedding system.",
)
@click.option("--cpus", type=click.IntRange(min=1), metavar="CPUS", help="The number of CPUs used for execution.")
@click.option("--memory", type=str, metavar="MEMORY", help="The memory limit.")
@click.option("--gpus", type=str, default="0", help="Number of GPUs to use ('all' for all GPUs).")
def embedd(
    system: str,
    dataset: list,
    cpus: Optional[int],
    memory: Optional[str],
    gpus: str,
    mount_new_model: Optional[str],
    out: str,
) -> int:
    """Run an embedding system, e.g., lsr-benchmark/lightning-ir/naver-splade-v3, to generate embeddings."""
    system = normalize_embedding_system(system)
    system_name = mount_new_model.replace("/", "-") if mount_new_model else system.split("/")[-1]
    datasets = all_datasets() if not dataset or "all" in dataset else list(dataset)
    gpu_count = -1 if gpus == "all" else int(gpus)

    all_messages = []

    def print_message(message, level):
        all_messages.append((message, level))
        os.system("cls" if os.name == "nt" else "clear")  # noqa: S605
        print(" ".join([sys.argv[0].split("/")[-1]] + sys.argv[1:]))
        for msg, lvl in all_messages:
            log_message(msg, lvl)

    platform = resolve_execution_platform(print_message)
    if platform is None:
        return 1

    failures = 0
    for ds in datasets:
        ds_name = ds.stem if isinstance(ds, Path) else ds
        try:
            run_embedding_system(
                system, ds, Path(out) / ds_name / system_name, platform, cpus, memory, gpu_count,
                mount_new_model,
            )
            print_message(f"Embedded dataset {ds_name} with {system}.", FormatMsgType.OK)
        except Exception as error:
            failures += 1
            print_message(f"{system} failed on dataset {ds_name}: {error}", FormatMsgType.ERROR)

    if failures:
        raise click.ClickException(f"{failures} embedding execution(s) failed.")
    return 0
