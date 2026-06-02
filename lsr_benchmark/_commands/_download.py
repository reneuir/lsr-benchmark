import click
from tira.rest_api_client import Client
from lsr_benchmark.datasets import (
    all_embeddings, all_dense_embeddings, all_ir_datasets,
    IR_DATASET_TO_TIRA_DATASET, EMBEDDING_MODEL_TO_ENGINE
)
from shutil import copytree


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
    "-o", "--out",
    type=str,
    required=False,
    multiple=False,
    default=None,
    help="The output directory to write to.",
)
def download_embeddings(dataset, embedding, out):
    tira = Client()
    engine = EMBEDDING_MODEL_TO_ENGINE.get(embedding, "lightning-ir")
    tira_dataset = IR_DATASET_TO_TIRA_DATASET[dataset]
    ret = tira.get_run_output(f'lsr-benchmark/{engine}/{embedding}', tira_dataset)
    if out is not None:
        copytree(ret, out)
        ret = out
    print(ret)


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
