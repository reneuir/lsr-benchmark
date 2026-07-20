import click
from pathlib import Path
from tira.rest_api_client import Client
import yaml
from lsr_benchmark.datasets import (
    all_embeddings, all_dense_embeddings, all_ir_datasets,
    IR_DATASET_TO_TIRA_DATASET, EMBEDDING_MODEL_TO_ENGINE
)
from shutil import copytree

from .sisap_io import MissingSisapDependencyError, export_embeddings_to_sisap


@click.option(
    "--dataset",
    type=click.Choice(all_ir_datasets()),
    required=False,
)
@click.option(
    "--embedding",
    type=click.Choice(all_embeddings() + sorted(list(all_dense_embeddings()))),
    required=False,
)
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=False,
    help="Use embeddings from this directory instead of downloading them.",
)
@click.option(
    "-o", "--out",
    type=str,
    required=False,
    multiple=False,
    default=None,
    help="The output directory to write to.",
)
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["reneuir", "sisap"]),
    required=False,
    multiple=False,
    default="reneuir",
    help="The output format to write.",
)
def download_embeddings(dataset, embedding, directory, out, export_format):
    embedding_id = None
    if directory is not None:
        if dataset is not None or embedding is not None:
            raise click.UsageError("--directory cannot be combined with --dataset or --embedding.")
        dataset = _dataset_from_embedding_directory(directory)
        source_dir = directory
    else:
        if dataset is None or embedding is None:
            raise click.UsageError("Either --directory or both --dataset and --embedding are required.")
        tira = Client()
        engine = EMBEDDING_MODEL_TO_ENGINE.get(embedding, "lightning-ir")
        tira_dataset = IR_DATASET_TO_TIRA_DATASET[dataset]
        embedding_id = f'lsr-benchmark/{engine}/{embedding}'
        source_dir = Path(tira.get_run_output(embedding_id, tira_dataset))

    ret = source_dir
    if export_format == "sisap":
        if out is None:
            raise click.UsageError("--out is required when --format sisap is used.")
        try:
            ret = export_embeddings_to_sisap(
                source_dir,
                Path(out),
                dataset,
                embedding_id,
                preserve_source_metadata=directory is not None,
            )
        except MissingSisapDependencyError as exc:
            raise click.ClickException(str(exc)) from exc
    elif out is not None:
        copytree(ret, out)
        ret = out
    print(ret)


def _dataset_from_embedding_directory(directory: Path) -> str:
    metadata = []
    for embedding_type in ("doc", "query"):
        metadata_path = directory / embedding_type / f"{embedding_type}-ir-metadata.yml"
        if not metadata_path.exists():
            raise click.ClickException(f"Expected embedding metadata file is missing: {metadata_path}")
        content = yaml.safe_load(metadata_path.read_text())
        try:
            dataset = content["data"]["test collection"]["ir-datasets-id"]
            embedding_model = content["data"]["embedding model"]
        except (KeyError, TypeError) as exc:
            raise click.ClickException(
                f"Expected dataset and embedding model metadata in: {metadata_path}"
            ) from exc
        if not dataset or not isinstance(embedding_model, dict) or not embedding_model.get("name"):
            raise click.ClickException(
                f"Expected dataset and embedding model metadata in: {metadata_path}"
            )
        metadata.append((dataset, embedding_model))

    if metadata[0] != metadata[1]:
        raise click.ClickException("Document and query embedding metadata do not match.")
    return metadata[0][0]


@click.option(
    "--dataset",
    type=click.Choice(all_ir_datasets()),
    required=True,
)
@click.option(
    "--embedding",
    type=click.Choice(all_embeddings() + sorted(list(all_dense_embeddings()))),
    required=True,
)
@click.option(
    "--retrieval",
    type=click.Choice(sorted(["seismic", "duckdb", "kannolo", "naive-search",
                               "pyterrier-splade-pisa", "pyterrier-splade",
                               "pytorch-naive", "numpy-exhaustive"])),
    required=True,
)
@click.option(
    "-o", "--out",
    type=str,
    required=False,
    multiple=False,
    default=None,
    help="The output directory to write to.",
)
def download_run(dataset, embedding, retrieval, out):
    tira = Client()
    system_name = f'lsr-benchmark/reneuir-baselines/{retrieval}-on-{embedding.replace("/", "-")}'
    tira_dataset = IR_DATASET_TO_TIRA_DATASET[dataset]
    ret = tira.get_run_output(system_name, tira_dataset)
    if out is not None:
        copytree(ret, out)
        ret = out
    print(ret)
