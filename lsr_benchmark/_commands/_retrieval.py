import click
import sys
from tira.io_utils import log_message, verify_docker_installation, FormatMsgType
from tira.third_party_integrations import temporary_directory
from tira.check_format import check_format
from tira.rest_api_client import Client
from pathlib import Path
from lsr_benchmark.datasets import all_embeddings, all_dense_embeddings, all_datasets
from lsr_benchmark._commands._verify_installation import EXAMPLE_RETRIEVAL_ENGINE
import shutil
import yaml
from tira.io_utils import docker_supported_target_platform
import os
from platform import system


def run_foo(docker_image, command, dataset_id, embedding, output_dir=None):
    if output_dir is not None and Path(output_dir).exists():
        return
    tira = Client()

    import tempfile, os
    meta = tira.get_dataset(f"lsr-benchmark/{dataset_id}")
    if meta and meta.get("is_confidential", False):
        os.environ["TIRA_INPUT_DATASET"] = tempfile.mkdtemp()
    dataset_path = tira.download_dataset("lsr-benchmark", dataset_id)
    
    if isinstance(embedding, Path):
        embeddings_dir = embedding.resolve()
    elif embedding.lower() != "none" and embedding not in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f'lsr-benchmark/lightning-ir/{embedding}', dataset_id)
    elif embedding.lower() != "none" and embedding in all_dense_embeddings():
        embeddings_dir = tira.get_run_output(f'lsr-benchmark/sentence-transformers/{embedding}', dataset_id)
    else:
        embeddings_dir = None
    mount_directory = None
    if embeddings_dir:
        mount_directory = {"embeddings": embeddings_dir}
    tmp_dir = temporary_directory()
    tira.local_execution.run(
        image=docker_image,
        command=command,
        input_dir=dataset_path,
        output_dir=tmp_dir,
        allow_network=False,
        input_run=embeddings_dir,
        mount_directory=mount_directory,
    )

    result, msg = check_format(Path(tmp_dir), ["run.txt"], {})
    if result != FormatMsgType.OK:
        print(msg)
        raise ValueError(msg)

    tag = yaml.safe_load((Path(tmp_dir) / "retrieval-metadata.yml").read_text())["tag"]
    
    if output_dir is not None:
        from tira.io_utils import patch_ir_metadata
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_dir, output_dir)
        patch_ir_metadata(output_dir, {"data": {"test collection": {"name": "/tira-data/input"}}}, {"data": {"test collection": {"name": dataset_id}}})
    
    return tag


def get_approach_to_execution(approaches, platform):
    tira = Client()
    approach_to_execution = {}
    for approach in approaches:
        if approach in EXAMPLE_RETRIEVAL_ENGINE[platform]:
            approach_to_execution[approach] = {"tag": EXAMPLE_RETRIEVAL_ENGINE[platform][approach]["image"], "command": EXAMPLE_RETRIEVAL_ENGINE[platform][approach]["command"]}
        else:
            docker_tag, zipped_code, remotes, commit, active_branch = tira.build_docker_image_from_code(
                Path(approach), log_message, False
            )
            if docker_tag in approach_to_execution.values():
                raise ValueError(f"Approach {approach} produces a docker tag that is already used by another approach.")
            cmd = (Path(approach) / "README.md").read_text().split("tira-cli code-submission")[1].split('--command')[1].split("'")[1]

            log_message(f"Approach {approach} is compiled.", FormatMsgType.OK)
            system_tag = run_foo(docker_tag, cmd, 'tiny-example-20251002_0-training', embedding[0])
            print_message(f"Approach {approach} compiled and produced valid outputs on example dataset (tag={system_tag}).", FormatMsgType.OK)
            approach_to_execution[approach] = {"tag": docker_tag, "command": cmd}

    return approach_to_execution

class ChoiceOrPath(click.ParamType):
    def __init__(self, choices):
        self.choices = tuple(choices)

    def get_metavar(self, param, ctx):
        return f"[{'|'.join(self.choices)}|PATH]"

    def convert(self, value, param, ctx):
        if value in self.choices:
            return value

        path = Path(value)
        if path.exists() and path.is_dir():
            return path

        choices_str = ", ".join([f"'{choice}'" for choice in self.choices])
        self.fail(
            f"{value!r} is not one of "
            f"{choices_str} or an existing path to a directory.",
            param,
            ctx
        )

@click.argument(
    "approaches",
    type=str,
    nargs=-1,
)
@click.option(
    "-o", "--out",
    type=str,
    required=True,
    multiple=False,
    help="The output directory to write to.",
)
@click.option(
    "--dataset",
    type=click.Choice(["all"] + all_datasets()),
    multiple=True,
    help="The datasets to run on.",
)
@click.option(
    "--embedding",
    type=ChoiceOrPath(["all", "none", ] + all_embeddings() + list(all_dense_embeddings())),
    multiple=True,
    help="The datasets to run on.",
)
def retrieval(approaches: list[str], dataset: list[str], embedding: list[str], out: str) -> int:
    all_messages = []

    def print_message(message, level):
        all_messages.append((message, level))
        os.system("cls" if os.name == "nt" else "clear") # noqa: S605
        print(' '.join([sys.argv[0].split('/')[-1]] + sys.argv[1:]))
        for message, level in all_messages:
            log_message(message, level)

    if dataset is None or not dataset or "all" in dataset:
        dataset = all_datasets()

    if embedding is None or not embedding or "all" in embedding:
        embedding = all_embeddings()
    if embedding and "none" in embedding:
        embedding = ["none"]

    status = verify_docker_installation()

    if status[0] != FormatMsgType.OK:
        print_message("You do not have Docker installed. Please install Docker and then run 'tira-cli verify-installation' to ensure that everyting is correctly installed.", status)

        return 1

    print_message("Your TIRA installation is valid.", FormatMsgType.OK)

    platform = docker_supported_target_platform()

    if platform not in ("linux/amd64", "linux/arm64"):
        print_message(f"The platform {docker_supported_target_platform()} is not supported.", _fmt.ERROR)
        return 1

    approach_to_execution = get_approach_to_execution(approaches, platform)

    if len(dataset) == 0:
        print_message(f"No datasets are passed.", _fmt.ERROR)
        return 1

    if len(embedding) == 0:
        print_message(f"No embedding are passed.", _fmt.ERROR)
        return 1

    if len(approaches) == 0:
        print_message(f"No approaches are passed.", _fmt.ERROR)
        return 1

    stats = {}
    for d in dataset:
        for e in embedding:
            for approach in approaches:
                out_dir = Path(out) / d / e / approach
                try:
                    run_foo(approach_to_execution[approach]["tag"], approach_to_execution[approach]["command"], d, e, out_dir)
                    if approach not in stats:
                        stats[approach] = {"embeddings": set(), "datasets": set()}
                    stats[approach]["embeddings"].add(e)
                    stats[approach]["datasets"].add(d)
                except Exception:  # noqa: S112
                    continue

    for approach in stats:
        print_message(f"Approach {approach} produced valid outputs on {len(stats[approach]['datasets'])} datasets for {len(stats[approach]['embeddings'])} embeddings.", FormatMsgType.OK)
    
    return 0
