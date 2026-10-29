import contextlib
import click
from dataclasses import dataclass, field
import io
import sys
from typing import Mapping, Optional, Union
from tira.io_utils import (
    FormatMsgType,
    MonitoredExecution,
    log_message,
    verify_docker_installation,
)
from tira.third_party_integrations import temporary_directory
from tira.check_format import check_format
from tira.rest_api_client import Client
from pathlib import Path
from lsr_benchmark.datasets import (
    IR_DATASET_TO_TIRA_DATASET,
    all_datasets,
    all_dense_embeddings,
    all_embeddings,
)
from lsr_benchmark._commands._verify_installation import EXAMPLE_RETRIEVAL_ENGINE
from lsr_benchmark._commands._modify_data import JOINT_TO_DATASETS
from lsr_benchmark._commands.download_embeddings import download_embeddings
from lsr_benchmark.retrieval_suites import RETRIEVAL_SUITES
import shutil
import yaml
from tira.io_utils import docker_supported_target_platform
import os
import itertools
from tempfile import TemporaryDirectory


def is_rank_fusion_approach(approach):
    return "rank-fusion" in approach


def run_retrieval_engine(
    docker_image,
    command,
    dataset_id,
    embedding,
    is_rank_fusion,
    additional_mounts,
    output_dir=None,
    platform=None,
    cpus=None,
    memory=None,
):
    if output_dir is not None and Path(output_dir).exists():
        return
    tira = Client()
    mount_directory = dict()
    tmp = TemporaryDirectory()
    is_joint_dataset = dataset_id  in JOINT_TO_DATASETS
    temporary_dset_path = False

    if is_rank_fusion:
        dataset_path = tmp.name
        embeddings_dir = None
    else:
        is_lexical_index = "pyterrier-naive" in docker_image
        is_lexical_joint_index = is_joint_dataset and is_lexical_index

        if isinstance(dataset_id, Path):
            dataset_path = dataset_id.resolve()
            dataset_id = dataset_id.stem
        elif is_lexical_index:
            dataset_path = tira.download_dataset("lsr-benchmark", dataset_id)
        else:
            dataset_path = temporary_directory()
            temporary_dset_path = True

        if is_lexical_joint_index:
            for dset in JOINT_TO_DATASETS[dataset_id]["datasets"]:
                emb_dir = download_embeddings(embedding, dataset_id, tira)
                shutil.copytree(emb_dir / dset, tmp)
            embeddings_dir = Path(tmp)
        else:
            embeddings_dir = download_embeddings(embedding, dataset_id, tira)

        if embeddings_dir:
            mount_directory["embeddings"] =  embeddings_dir

    mount_directory.update(additional_mounts)

    execution_dir = MonitoredExecution().run(
        lambda tmp_dir: tira.local_execution.run(
            image=docker_image,
            command=command,
            input_dir=dataset_path,
            output_dir=tmp_dir,
            allow_network=False,
            input_run=embeddings_dir,
            mount_directory=mount_directory,
            platform=platform if platform else docker_supported_target_platform(),
            cpu_count=cpus,
            mem_limit=memory,
        )
    )
    tmp_dir = execution_dir / "output"

    result, msg = check_format(tmp_dir, ["run.txt"], {})
    if result != FormatMsgType.OK:
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True)
            shutil.copytree(execution_dir, output_dir / ".failed")
        print(msg)
        raise ValueError(msg)

    tag = yaml.safe_load((Path(tmp_dir) / "retrieval-metadata.yml").read_text())["tag"]

    if output_dir is not None:
        from tira.io_utils import patch_ir_metadata
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_dir, output_dir)
        patch_ir_metadata(output_dir, {"data": {"test collection": {"name": "/tira-data/input"}}}, {"data": {"test collection": {"name": dataset_id}}})
        if is_joint_dataset or is_rank_fusion:
            for meta_file in ["index-metadata.yml", "retrieval-metadata.yml"]:
                with open(output_dir/meta_file, "r") as f:
                    meta = yaml.safe_load(f)

                if is_joint_dataset:
                    meta["data"]["test collection"]["union of subsamples"] = JOINT_TO_DATASETS[dataset_id]["datasets"]
                if is_rank_fusion:
                    meta["data"]["fused runs"] = output_dir.parent.name.split("+")
                with open(output_dir / meta_file, "w") as f:
                    yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    if temporary_dset_path:
        shutil.rmtree(dataset_path)

    tmp.cleanup()

    return tag


def get_approach_to_execution(approaches, platform, embedding, print_message):
    tira = Client()
    approach_to_execution = {}
    for approach in approaches:
        if approach in EXAMPLE_RETRIEVAL_ENGINE[platform]:
            approach_to_execution[approach] = {"tag": EXAMPLE_RETRIEVAL_ENGINE[platform][approach]["image"], "command": EXAMPLE_RETRIEVAL_ENGINE[platform][approach]["command"]}
        else:
            software_def = None
            last_error = None
            candidates = [approach + "-" + platform.split("/")[1]]
            for c in candidates:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        software_def = tira.public_system_details("reneuir-baselines", c)
                    break
                except ValueError as error:
                    last_error = error

            if software_def is None and last_error is not None:
                raise last_error

            approach_to_execution[approach] = {"tag": software_def["public_image_name"], "command": software_def["command"]}

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


def resolve_retrieval_configuration(suite, approaches, datasets, embeddings):
    """Resolve a named suite or return the manually selected inputs."""
    if suite is None:
        return approaches, datasets, embeddings

    if approaches or datasets or embeddings:
        raise click.UsageError(
            "--suite cannot be combined with retrieval engines, --dataset, or --embedding."
        )

    configuration = RETRIEVAL_SUITES[suite]
    suite_datasets = []
    for dataset in configuration["datasets"]:
        if dataset == "all" or dataset in all_datasets():
            suite_datasets.append(dataset)
        elif dataset in IR_DATASET_TO_TIRA_DATASET:
            suite_datasets.append(IR_DATASET_TO_TIRA_DATASET[dataset])
        else:
            raise click.UsageError(
                f"Suite {suite!r} contains unsupported dataset {dataset!r}."
            )

    supported_embeddings = {"all", "none", *all_embeddings(), *all_dense_embeddings()}
    unsupported_embeddings = [
        embedding
        for embedding in configuration["embeddings"]
        if embedding not in supported_embeddings
    ]
    if unsupported_embeddings:
        raise click.UsageError(
            f"Suite {suite!r} contains unsupported embedding "
            f"{unsupported_embeddings[0]!r}."
        )

    return (
        tuple(configuration["retrieval_engines"]),
        tuple(suite_datasets),
        tuple(configuration["embeddings"]),
    )


def normalize_retrieval_inputs(datasets, embeddings, approaches):
    """Expand default/all inputs and validate them.

    For rank fusion, the embeddings are the directories of the runs to fuse
    and the dataset is read from their metadata.
    """
    is_rank_fusion = bool(approaches) and is_rank_fusion_approach(approaches[0])

    for approach in approaches[1:]:
        if is_rank_fusion != is_rank_fusion_approach(approach):
            raise click.UsageError("The approach 'rank-fusion' can not be used alongside other approaches.")

    if is_rank_fusion:
        is_paired = False
        datasets, embeddings = _normalize_rank_fusion_inputs(embeddings)
    else:
        datasets, embeddings, is_paired = _normalize_embedding_inputs(datasets, embeddings)

    return datasets, embeddings, is_paired, is_rank_fusion


def _normalize_embedding_inputs(datasets, embeddings):
    """Expand default/all inputs and validate local dataset/embedding pairs."""
    is_paired = False
    if not datasets and embeddings and all(isinstance(e, Path) for e in embeddings):
        is_paired = True
        datasets = []
        for e in embeddings:
            meta_file = e / "doc"
            if (meta_file / "doc-ir-metadata.yml").exists():
                meta_file /= "doc-ir-metadata.yml"
            elif (meta_file / "d0-doc-ir-metadata.yml").exists():
                meta_file /= "d0-doc-ir-metadata.yml"
            else:
                raise FileNotFoundError("No metadata file found to retrieve dataset from. Aborting!")
            with open(meta_file, "r") as f:
                meta = yaml.safe_load(f)
            test_collecton = meta["data"]["test collection"]
            if "subsample of" in test_collecton:
                datasets.append(test_collecton["subsample of"])
            else:
                datasets.append(test_collecton["name"])
    elif not datasets or "all" in datasets:
        if any(isinstance(e, Path) for e in embeddings):
            raise ValueError("Can't use both local and remote embeddings when running on all datasets. Aborting!")
        datasets = all_datasets()

    if not embeddings or "all" in embeddings:
        embeddings = all_embeddings()
    elif "none" in embeddings:
        embeddings = ["none"]

    for dataset in datasets:
        for embedding in embeddings:
            if isinstance(dataset, Path) and not isinstance(embedding, Path):
                raise click.UsageError(
                    "A local dataset has to be used in combination with local embeddings! Aborting."
                )

    return datasets, embeddings, is_paired


def _normalize_rank_fusion_inputs(runs):
    """Validate the run directories to fuse and return (datasets, runs)."""
    if len(runs) < 2:
        raise click.UsageError("The approach 'rank-fusion' requires at least two runs to be passed.")
    if any(not isinstance(run, Path) for run in runs):
        raise click.UsageError("The approach 'rank-fusion' currently doesn't support remote runs. Aborting!")
    # TODO: check only for "run.txt.gz"
    if not all((run / "run.txt.gz").exists() for run in runs):
        raise click.UsageError("Not all passed directories contain valid runs (txt.gz files are expected).")

    dataset_names = set()
    for run in runs:
        with open(run / "index-metadata.yml") as f:
            meta = yaml.safe_load(f)
        dataset_names.add(meta["data"]["test collection"]["name"])
    if len(dataset_names) > 1:
        raise click.UsageError(
            "The approach 'rank-fusion' requires all runs to be of the same dataset, "
            f"but found: {', '.join(sorted(dataset_names))}. Aborting!"
        )

    return (dataset_names.pop(),), tuple(run.resolve() for run in runs)


def validate_retrieval_selection(approaches, datasets, embeddings, print_message):
    """Report missing retrieval inputs and return whether the selection is usable."""
    missing = (
        (datasets, "No datasets are passed."),
        (embeddings, "No embedding are passed."),
        (approaches, "No approaches are passed."),
    )
    for values, message in missing:
        if not values:
            print_message(message, FormatMsgType.ERROR)
            return False
    if "lexical" in embeddings and any(a != "pyterrier-naive" for a in approaches):
        print_message("Lexical PyTerrier Indeces can only be used with lexical retrieval.", FormatMsgType.ERROR)
        return False
    return True


def resolve_execution_platform(print_message):
    """Validate Docker and return a supported execution platform, if available."""
    status, _ = verify_docker_installation()
    if status != FormatMsgType.OK:
        print_message(
            "You do not have Docker installed. Please install Docker and then run "
            "'tira-cli verify-installation' to ensure that everything is correctly "
            "installed.",
            status,
        )
        return None

    print_message("Your TIRA installation is valid.", FormatMsgType.OK)
    platform = docker_supported_target_platform()
    if platform not in ("linux/amd64", "linux/arm64"):
        print_message(
            f"The platform {platform} is not supported.", FormatMsgType.ERROR
        )
        return None
    return platform


@dataclass(frozen=True)
class RetrievalJob:
    """One retrieval-engine execution with resolved names and output location."""

    approach: str
    dataset: Union[str, Path]
    embedding: Optional[Union[str, Path]]
    dataset_name: str
    embedding_name: str
    image: str
    command: str
    output_dir: Path
    additional_mounts: dict[str, Path] = field(default_factory=dict)  # rank fusion: run1..runN


def read_embedding_metadata(embedding_dir, dataset):
    """Return (quantization level or None, embedding model name) of an embedding."""
    meta_path = embedding_dir / f"{'d0-' if dataset in JOINT_TO_DATASETS else ''}doc-ir-metadata.yml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    quant_level = meta["data"]["test collection"].get("quantization")
    embedding_model = meta["data"]["embedding model"]["name"]
    return quant_level, embedding_model


def approach_args_suffix(approach_args):
    """Output directory suffix encoding the extra approach arguments, e.g. '_k_100'."""
    if not approach_args:
        return ""
    arg_suffixes = [
        ("_".join(arg.split()) if " " in arg else arg).strip("-")
        for arg in approach_args
    ]
    return "_" + "-".join(sorted(arg_suffixes))


def fused_runs_name(runs, dataset):
    """Directory name for the fused runs, e.g. 'model-a_seismic+model-b_int8_seismic'."""
    run_tags = []
    for run in runs:
        quant_level, embedding_model = read_embedding_metadata(run / ".embedding", dataset)
        run_tags.append((embedding_model, quant_level or "", run.name))

    return "+".join(
        f"{model}_{quant}_{name}" if quant else f"{model}_{name}"
        for model, quant, name in sorted(run_tags)
    )


def build_retrieval_jobs(
    approaches,
    datasets,
    embeddings,
    approach_to_execution: Mapping[str, Mapping[str, str]],
    output_root: Path,
    is_paired: bool,
    is_rank_fusion: bool,
    approach_args: list[str]
):
    """Build deterministic jobs for the selected approaches.

    Regular approaches run on the dataset/embedding product (or pairs), while
    rank fusion becomes a single job fusing the passed runs.
    """
    if is_rank_fusion:
        approach = approaches[0]
        return (
            _build_rank_fusion_job(
                approach, datasets[0], embeddings, approach_to_execution[approach], output_root, approach_args
            ),
        )
    return _build_embedding_jobs(
        approaches, datasets, embeddings, approach_to_execution, output_root, is_paired, approach_args
    )


def _build_embedding_jobs(
    approaches, datasets, embeddings, approach_to_execution, output_root, is_paired, approach_args
):
    jobs = []
    approach_suffix = approach_args_suffix(approach_args)

    if is_paired:
        dataset_embedding_pairs = zip(datasets, embeddings)
    else:
        dataset_embedding_pairs = itertools.product(datasets, embeddings)

    for dataset, embedding in dataset_embedding_pairs:
        dataset_name = dataset.stem if isinstance(dataset, Path) else dataset
        embedding_name = embedding
        quant_suffix = ""

        if isinstance(embedding, Path):
            embedding_name = embedding.name
            quant_level, _ = read_embedding_metadata(embedding / "doc", dataset_name)
            quant_suffix = f"-{quant_level}" if quant_level else ""

        for approach in approaches:
            execution = approach_to_execution[approach]
            command = " ".join([execution["command"], *approach_args])
            jobs.append(
                RetrievalJob(
                    approach=approach,
                    dataset=dataset,
                    embedding=embedding,
                    dataset_name=dataset_name,
                    embedding_name=embedding_name,
                    image=execution["tag"],
                    command=command,
                    output_dir=(
                        output_root
                        / (dataset_name + quant_suffix)
                        / embedding_name
                        / (approach + approach_suffix)
                    ),
                )
            )
    return tuple(jobs)


def _build_rank_fusion_job(approach, dataset, runs, execution, output_root, approach_args):
    mounts = {f"run{i}": run for i, run in enumerate(runs, start=1)}
    command = " ".join([execution["command"], *(f"${name}" for name in mounts), *approach_args])
    fused_name = fused_runs_name(runs, dataset)

    return RetrievalJob(
        approach=approach,
        dataset=dataset,
        embedding=None,
        dataset_name=dataset,
        embedding_name=fused_name,
        image=execution["tag"],
        command=command,
        output_dir=output_root / dataset / fused_name / (approach + approach_args_suffix(approach_args)),
        additional_mounts=mounts,
    )


def execute_retrieval_jobs(
    jobs, platform, cpus, memory, print_message, rerun_failed=False
):
    """Execute jobs independently and aggregate successful inputs and failures."""
    stats = {}
    failures = 0
    for job in jobs:
        if (job.output_dir / ".failed").is_dir():
            if rerun_failed:
                shutil.rmtree(job.output_dir)
            else:
                failures += 1
                print_message(
                    f"Approach {job.approach} previously failed on dataset "
                    f"{job.dataset_name} with embedding {job.embedding_name}.",
                    FormatMsgType.ERROR,
                )
                continue
        try:
            run_retrieval_engine(
                job.image,
                job.command,
                job.dataset,
                job.embedding,
                is_rank_fusion_approach(job.approach),
                job.additional_mounts,
                job.output_dir,
                platform=platform,
                cpus=cpus,
                memory=memory,
            )
            if job.approach not in stats:
                stats[job.approach] = {"embeddings": set(), "datasets": set()}
            stats[job.approach]["embeddings"].add(job.embedding_name)
            stats[job.approach]["datasets"].add(job.dataset_name)
        except Exception as error:
            failures += 1
            print_message(
                f"Approach {job.approach} failed on dataset {job.dataset_name} "
                f"with embedding {job.embedding_name}: {error}",
                FormatMsgType.ERROR,
            )
    return stats, failures


def report_retrieval_stats(stats, print_message):
    """Report the successful dataset and embedding coverage per approach."""
    for approach, approach_stats in stats.items():
        print_message(
            f"Approach {approach} produced valid outputs on "
            f"{len(approach_stats['datasets'])} datasets for "
            f"{len(approach_stats['embeddings'])} embeddings.",
            FormatMsgType.OK,
        )


@click.argument(
    "approaches",
    type=str,
    nargs=-1,
)
@click.option(
    "--suite",
    type=click.Choice(sorted(RETRIEVAL_SUITES)),
    help="A predefined retrieval suite.",
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
    type=ChoiceOrPath(["all"] + all_datasets()),
    multiple=True,
    help="The datasets to run on.",
)
@click.option(
    "--embedding",
    type=ChoiceOrPath(["all", "none", "lexical"] + all_embeddings() + list(all_dense_embeddings())),
    multiple=True,
    help="The datasets to run on.",
)
@click.option(
    "--cpus",
    type=click.IntRange(min=1),
    metavar="CPUS",
    help="The number of CPUs used for execution.",
)
@click.option(
    "--memory",
    type=str,
    metavar="MEMORY",
    help="The memory limit.",
)
@click.option("--rerun-failed", is_flag=True, help="Re-run failed executions.")
@click.option(
    "--approach-arg",
    "approach_args",
    type=str,
    multiple=True,
    help="Extra Arguments to add to the command executed in the container"
)
def retrieval(
    approaches: list[str],
    suite: Optional[str],
    dataset: list[str],
    embedding: list[str],
    cpus: Optional[int],
    memory: Optional[str],
    rerun_failed: bool,
    out: str,
    approach_args: list[str]
) -> int:
    all_messages = []

    def print_message(message, level):
        all_messages.append((message, level))
        os.system("cls" if os.name == "nt" else "clear") # noqa: S605
        print(' '.join([sys.argv[0].split('/')[-1]] + sys.argv[1:]))
        for message, level in all_messages:
            log_message(message, level)

    approaches, dataset, embedding = resolve_retrieval_configuration(
        suite, approaches, dataset, embedding
    )
    dataset, embedding, is_paired, is_rank_fusion = normalize_retrieval_inputs(dataset, embedding, approaches)
    if not validate_retrieval_selection(
        approaches, dataset, embedding, print_message
    ):
        return 1

    platform = resolve_execution_platform(print_message)
    if platform is None:
        return 1

    approach_to_execution = get_approach_to_execution(
        approaches, platform, embedding, print_message
    )
    jobs = build_retrieval_jobs(
        approaches, dataset, embedding, approach_to_execution, Path(out), is_paired, is_rank_fusion, approach_args
    )
    stats, failures = execute_retrieval_jobs(
        jobs, platform, cpus, memory, print_message, rerun_failed
    )
    report_retrieval_stats(stats, print_message)

    if failures:
        raise click.ClickException(
            f"{failures} retrieval configuration(s) failed."
        )

    return 0
