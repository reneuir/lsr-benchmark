__version__ = "0.0.1rc5"
import json
from pathlib import Path
from ir_datasets import registry
from lsr_benchmark.irds import build_dataset, ir_datasets_from_tira
from lsr_benchmark.corpus import materialize_corpus, materialize_queries, materialize_qrels
from click import group, argument
from tira.third_party_integrations import default_tira_cache_dir

from ._commands._evaluate import evaluate
from ._commands._retrieval import retrieval
from ._commands._download import download_embeddings, download_run
from ._commands._verify_installation import verify_installation
from ._commands.sisap_io import sisap_to_qrels, sisap_to_trec_run
from ._commands._modify_data import modify_data, JOINT_TO_DATASETS
from .datasets import TIRA_DATASET_ID_TO_IR_DATASET_ID, IR_DATASET_TO_TIRA_DATASET, SUPPORTED_IR_DATASETS
import os


def register_to_ir_datasets(dataset=None):
    if dataset and os.path.isdir(dataset):
        if dataset not in registry:
            ds = build_dataset(dataset, False)
            registry.register(dataset, ds)
            registry.register("lsr-benchmark/" + dataset, ds)
    elif dataset and (dataset in ir_datasets_from_tira() or dataset in ir_datasets_from_tira(True)):
        if dataset not in registry:
            ds = build_dataset(dataset, False)

            registry.register(dataset, ds)
            registry.register("lsr-benchmark/" + dataset, ds)
    elif dataset and dataset in IR_DATASET_TO_TIRA_DATASET:
        if ("lsr-benchmark/" + dataset) not in registry:
            ds = build_dataset(IR_DATASET_TO_TIRA_DATASET[dataset], False)

            if IR_DATASET_TO_TIRA_DATASET[dataset] not in registry:
                registry.register(IR_DATASET_TO_TIRA_DATASET[dataset], ds)
            registry.register("lsr-benchmark/" + dataset, ds)
    elif dataset and dataset in JOINT_TO_DATASETS:
        if dataset not in registry:
            ds_path = Path(f"{default_tira_cache_dir()}/extracted_datasets/lsr-benchmark/{dataset}")
            ds = build_dataset(ds_path, False)
            registry.register(dataset, ds)
            registry.register("lsr-benchmark/" + dataset, ds)
    elif dataset and dataset not in SUPPORTED_IR_DATASETS:
        raise ValueError(f"Can not register {dataset}. Supported are: {sorted(IR_DATASET_TO_TIRA_DATASET.keys())}")
    else:
        for k in SUPPORTED_IR_DATASETS:
            irds_id = f"lsr-benchmark/{k}/segmented"
            if irds_id not in registry:
                registry.register(irds_id, build_dataset(k, True))

            irds_id = f"lsr-benchmark/{k}"
            if irds_id not in registry:
                registry.register(irds_id, build_dataset(k, False))


def load(ir_datasets_id: str):
    return build_dataset(ir_datasets_id, False)


@group()
def main():
    pass


def create_subsampled_corpus(directory, config):
    from tirex_tracker import tracking, ExportFormat
    target_directory = directory

    target_directory.mkdir(exist_ok=True)
    with tracking(export_file_path=Path(target_directory) / "dataset-metadata.yml", export_format=ExportFormat.IR_METADATA):
        materialize_corpus(target_directory, config)
        materialize_queries(target_directory, config)
        materialize_qrels(target_directory/"qrels.txt", config)


@main.command()
@argument('directory', type=Path)
def create_lsr_corpus(directory):
    config = json.loads((directory/"config.json").read_text())
    create_subsampled_corpus(directory, config)


@main.command()
def overview():
    overview = json.loads((Path(__file__).parent / "datasets" / "overview.json").read_text())
    df_dataset = []
    overall_size = 0
    overall_datasets = len(overview)
    overall_embeddings = set()
    def f(s):
        return str(int(s/(1024))) + " MB"

    import pandas as pd
    model_to_size = {}

    for dataset_id, stats in overview.items():
        overall_size += int(stats['dataset-size'])
        embeddings_for_dataset = 0
        for embedding, embedding_size in stats['embedding-sizes'].items():
            overall_size += int(embedding_size)
            embeddings_for_dataset += int(embedding_size)
            overall_embeddings.add(embedding)
            model_to_size[embedding] = int(embedding_size) + model_to_size.get(embedding, 0)

        df_dataset += [{"Dataset": TIRA_DATASET_ID_TO_IR_DATASET_ID.get(dataset_id, dataset_id), "Text": f(int(stats['dataset-size'])), "Avg. Embeddings": f(embeddings_for_dataset/len(overall_embeddings))}]
    df_dataset = pd.DataFrame(df_dataset)
    df_dataset.index = ['']*len(df_dataset)

    df_embeddings = []
    for k, v in model_to_size.items():
        df_embeddings += [{"Model": k, "Size (avg)": f(v/len(overall_embeddings))}]


    df_embeddings = pd.DataFrame(df_embeddings)
    df_embeddings.index = ['']*len(df_embeddings)

    print(f"Overview of the lsr-benchmark:\n\n\t- {overall_datasets} Datasets with {len(overall_embeddings)} pre-computed embeddings ({f(overall_size)})\n\nDatasets:\n{df_dataset.sort_values('Dataset')}\n\nEmbeddings:\n{df_embeddings}")


main.command()(download_embeddings)
main.command()(download_run)
main.command()(evaluate)
main.command()(retrieval)
main.command()(verify_installation)
main.add_command(sisap_to_trec_run)
main.add_command(sisap_to_qrels)
main.command()(modify_data)

if __name__ == '__main__':
    main()
