import gzip
import json
import shutil
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import click
import numpy as np
import yaml
from tira.rest_api_client import Client
from tira.third_party_integrations import default_tira_cache_dir
from tqdm import tqdm

from lsr_benchmark._commands.download_embeddings import download_embeddings
from lsr_benchmark.datasets import all_datasets, all_dense_embeddings, all_embeddings


class DuplicateBehaviour(Enum):
    FAIL = 1
    SKIP = 2
    PREFIX = 3


class DuplicateHandling:
    def __init__(
        self, docs: DuplicateBehaviour = DuplicateBehaviour.FAIL, queries: DuplicateBehaviour = DuplicateBehaviour.FAIL
    ):
        self.doc = docs
        self.query = queries


JOINT_TO_DATASETS = {
    "msmarco-passage-trec-dl-2019+2020-judged": {
        "settings": DuplicateHandling(docs=DuplicateBehaviour.SKIP),
        "datasets": [
            "trec-28-deep-learning-passages-20250926-training",
            "trec-29-deep-learning-passages-20250926-training",
        ],
    },
    "disks45-nocr-trec-robust-2004-fold1+2+3+4+5": {
        "settings": DuplicateHandling(docs=DuplicateBehaviour.SKIP),
        "datasets": [
            "trec-robust-2004-fold-1-20250927-test",
            "trec-robust-2004-fold-2-20250926-test",
            "trec-robust-2004-fold-3-20250926-test",
            "trec-robust-2004-fold-4-20250926-test",
            "trec-robust-2004-fold-5-20250926-test",
        ],
    },
    "clueweb12-trec-web-2013+2014+clueweb12-b13-trec-misinfo-2019": {
        "settings": DuplicateHandling(docs=DuplicateBehaviour.SKIP, queries=DuplicateBehaviour.PREFIX),
        "datasets": [
            "trec-22-web-20251008-test",
            "trec-23-web-20251008-test",
            "trec-28-misinfo-20251008_1-test",
        ],
    },
    "clueweb09-en-trec-web-2009+2010+2011+2012": {
        "settings": DuplicateHandling(docs=DuplicateBehaviour.SKIP),
        "datasets": [
            "trec-18-web-20251008-test",
            "trec-19-web-20251008-test",
            "trec-20-web-20251008-test",
            "trec-21-web-20251008-test",
        ],
    },
}


def prefix_json(
    file, out, behaviour: DuplicateBehaviour, field: str, seen_ids: set[str], prefix: str, desc: str = ""
) -> None:
    for line in tqdm(file, desc=desc, leave=False, unit=" lines"):
        if not line.strip():
            continue

        record = json.loads(line)
        record_field = record[field]

        if behaviour == DuplicateBehaviour.FAIL:
            if record_field in seen_ids:
                raise ValueError(f"Duplicate {field} '{record_field}' found while joining datasets.")
            seen_ids.add(record_field)
        elif behaviour == DuplicateBehaviour.PREFIX:
            record[field] = f"{prefix}-{record_field}"
        elif behaviour == DuplicateBehaviour.SKIP:
            if record_field in seen_ids:
                continue
            seen_ids.add(record_field)

        out.write(json.dumps(record) + "\n")


def load_and_merge_embeddings(
    embedding_paths: list[Path],
    data_dir: str,
    keep_masks: list[np.ndarray],
) -> dict[str, np.ndarray]:
    all_data = []
    all_indices = []
    all_row_lengths = []

    for i, emb_path in enumerate(tqdm(embedding_paths, desc=f"Loading {data_dir}", leave=False)):
        with np.load(emb_path / data_dir) as npz:
            data = npz["data"]
            indices = npz["indices"]
            indptr = npz["indptr"]

        mask = keep_masks[i]
        row_lengths = np.diff(indptr)

        if not mask.all():
            element_mask = np.repeat(mask, row_lengths)
            data = data[element_mask]
            indices = indices[element_mask]
            row_lengths = row_lengths[mask]

        all_data.append(data)
        all_indices.append(indices)
        all_row_lengths.append(row_lengths)

    merged_row_lengths = np.concatenate(all_row_lengths)
    merged_indptr = np.empty(len(merged_row_lengths) + 1, dtype=np.int64)
    merged_indptr[0] = 0
    np.cumsum(merged_row_lengths, out=merged_indptr[1:])

    return {
        "data": np.concatenate(all_data),
        "indices": np.concatenate(all_indices),
        "indptr": merged_indptr,
    }


def perform_dataset_join(dataset: str, tira: Client, tira_dir: str) -> Path:
    join_path = Path(f"{tira_dir}/extracted_datasets/lsr-benchmark/{dataset}/")

    if join_path.exists():
        return join_path

    joint_dataset = JOINT_TO_DATASETS[dataset]
    individual_datasets = joint_dataset["datasets"]
    duplicate_behaviour = joint_dataset["settings"]
    mappings = [f"d{i}" for i in range(len(individual_datasets))]

    dataset_paths = [tira.download_dataset("lsr-benchmark", d) for d in tqdm(individual_datasets, desc="Datasets")]

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        seen_ids = set()
        with open(tmp / "queries.jsonl", "w") as out:
            for i, (mapping, path) in tqdm(
                enumerate(zip(mappings, dataset_paths)), total=len(mappings), desc="Joining Queries"
            ):
                with open(path / "queries.jsonl", "r") as file:
                    prefix_json(
                        file,
                        out,
                        duplicate_behaviour.query,
                        "qid",
                        seen_ids,
                        mapping,
                        desc=f"Prefixing {individual_datasets[i]}",
                    )

        seen_ids = set()
        with gzip.open(tmp / "corpus.jsonl.gz", "wt") as out:
            for i, (mapping, path) in tqdm(
                enumerate(zip(mappings, dataset_paths)), total=len(mappings), desc="Joining Corpora"
            ):
                with gzip.open(path / "corpus.jsonl.gz", "rt") as file:
                    prefix_json(
                        file,
                        out,
                        duplicate_behaviour.doc,
                        "doc_id",
                        seen_ids,
                        mapping,
                        desc=f"Prefixing {individual_datasets[i]}",
                    )

        shutil.copytree(tmp, join_path)

    return join_path


def perform_embedding_join(dataset: str, embedding: str, tira: Client, tira_dir: str) -> Path:
    emb_result_path = Path(f"{tira_dir}/extracted_runs/lsr-benchmark/{dataset}/{embedding}")

    if emb_result_path.exists():
        return emb_result_path

    joint_dataset = JOINT_TO_DATASETS[dataset]
    individual_datasets = joint_dataset["datasets"]
    duplicate_handling = joint_dataset["settings"]
    mappings = [f"d{i}" for i in range(len(individual_datasets))]

    tira = Client()
    embedding_paths = [download_embeddings(embedding, d, tira) for d in individual_datasets]

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "doc").mkdir(parents=True, exist_ok=True)
        (tmp / "query").mkdir(exist_ok=True)
        keep_masks = {"doc": [], "query": []}

        for emb_type in ["doc", "query"]:
            id_file = f"{emb_type}/{emb_type}-ids.txt"
            behaviour = getattr(duplicate_handling, emb_type)
            seen_ids = set()
            with open(tmp / id_file, "w") as out:
                for mapping, path in zip(mappings, embedding_paths):
                    mask = []
                    with open(path / id_file, "r") as file:
                        for line in file:
                            id = line.strip()
                            out_line = f"{id}\n"
                            keep = True
                            if behaviour == DuplicateBehaviour.FAIL:
                                if id in seen_ids:
                                    raise ValueError(f"Duplicate id '{line}' found while joining embeddings.")
                                seen_ids.add(id)
                            elif behaviour == DuplicateBehaviour.PREFIX:
                                out_line = f"{mapping}-{id}\n"
                            elif behaviour == DuplicateBehaviour.SKIP:
                                if id in seen_ids:
                                    keep = False
                                else:
                                    seen_ids.add(id)

                            mask.append(keep)
                            if keep:
                                out.write(out_line)

                    keep_masks[emb_type].append(np.array(mask, dtype=bool))

            meta_file = f"{emb_type}/{emb_type}-ir-metadata.yml"
            meta_out_dir = tmp / emb_type
            for mapping, path in zip(mappings, embedding_paths):
                src_meta = path / meta_file
                dest_meta = meta_out_dir / f"{mapping}-{emb_type}-ir-metadata.yml"
                shutil.copy(src_meta, dest_meta)
                with open(dest_meta, "r") as f:
                    meta = yaml.safe_load(f)
                meta["data"]["test collection"]["subsample of"] = dataset
                with open(dest_meta, "w") as f:
                    yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

        for emb_type, emb_file in [("doc", "doc/doc-embeddings.npz"), ("query", "query/query-embeddings.npz")]:
            merged_embeddings = load_and_merge_embeddings(embedding_paths, emb_file, keep_masks[emb_type])
            np.savez_compressed(
                tmp / emb_file,
                data=merged_embeddings["data"],
                indices=merged_embeddings["indices"],
                indptr=merged_embeddings["indptr"],
            )

        shutil.copytree(tmp, emb_result_path)

    return emb_result_path


def perform_quantization(
    embedding_path: Path,
    precision: Literal["fp16", "int8", "uint8", "ternary", "binary"],
    quant_range: int | None,
    keep_dtype: bool,
    dataset: str,
    embedding: str,
    tira_dir: str,
) -> Path:
    range_suffix = f"-{quant_range}" if quant_range and precision.endswith("int8") else ""
    keep_suffix = "-original_dtype" if keep_dtype else ""
    emb_result_path = Path(
        f"{tira_dir}/extracted_runs/lsr-benchmark/{dataset}-{precision}{range_suffix}{keep_suffix}/{embedding}"
    )

    if emb_result_path.exists():
        return emb_result_path

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "doc").mkdir(parents=True, exist_ok=True)
        (tmp / "query").mkdir(exist_ok=True)
        for directory in ["doc", "query"]:
            dirPath = f"{directory}/{directory}-embeddings.npz"
            with np.load(embedding_path / dirPath) as npz:
                data = npz["data"]
                indices = npz["indices"]
                indptr = npz["indptr"]

            original_dtype = data.dtype

            if precision == "fp16":
                data = data.astype(np.float16)

            if precision.endswith("int8"):
                n = len(indptr) - 1
                d = indptr[1] - indptr[0]
                data = data.reshape(n, d)

                if quant_range:
                    min = np.percentile(data, 100 - quant_range, axis=0)
                    max = np.percentile(data, quant_range, axis=0)
                else:
                    min = np.min(data, axis=0)
                    max = np.max(data, axis=0)

                scale = (max - min) / 255
                scale = np.where(scale == 0, 1, scale)

                data = np.clip(np.floor((data - min) / scale), 0, 255)
                data = data.flatten()

                if precision == "uint8":
                    data = data.astype(np.uint8)
                elif precision == "int8":
                    data = (data - 128).astype(np.int8)

            if precision == "ternary":
                data = np.sign(data).astype(np.int8)

            if precision == "binary":
                data = (data > 0).astype(np.int8)

            if keep_dtype:
                data = data.astype(original_dtype)

            np.savez_compressed(tmp / dirPath, data=data, indices=indices, indptr=indptr)
            shutil.copy(embedding_path / directory / f"{directory}-ids.txt", tmp / directory)

            if dataset in JOINT_TO_DATASETS:
                nmb_of_datasets = len(JOINT_TO_DATASETS[dataset]["datasets"])
                meta_files = [f"d{i}-{directory}-ir-metadata.yml" for i in range(nmb_of_datasets)]
            else:
                meta_files = [f"{directory}-ir-metadata.yml"]
            for file in meta_files:
                with open(embedding_path / directory / file, "r") as f:
                    meta = yaml.safe_load(f)
                meta["data"]["test collection"]["quantization"] = precision
                if keep_dtype:
                    meta["data"]["test collection"]["original datatype"] = True
                with open(tmp / directory / file, "w") as f:
                    yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

        shutil.copytree(tmp, emb_result_path)

    return emb_result_path


@click.argument(
    "datasets", type=click.Choice(list(JOINT_TO_DATASETS.keys()) + list(all_datasets()) + ["all"]), nargs=-1
)
@click.option(
    "--embedding",
    type=click.Choice(all_embeddings() + list(all_dense_embeddings()) + ["all"]),
    required=True,
    multiple=True,
    help="The embeddings to run on",
)
@click.option("-j", "--join-embeddings", is_flag=True)
@click.option("-c", "--join-corpora", is_flag=True)
@click.option(
    "-q",
    "--quantization",
    type=click.Choice(["fp16", "int8", "uint8", "ternary", "binary", "all"]),
    multiple=True,
    help="Number of bits to quantize data to",
)
@click.option(
    "-r", "--quant-range", type=int, help="Percentage of values to include while quantizing to avoid outliers"
)
@click.option(
    "-k", "--keep-dtype", type=bool, is_flag=True, help="Whether to keep the original datatype after quantizing."
)
def modify_data(
    datasets: list[str],
    embedding: list[str],
    join_corpora: bool,
    join_embeddings: bool,
    quantization: list[str],
    quant_range: int | None,
    keep_dtype: bool,
) -> int:
    if not join_corpora and not join_embeddings and not quantization:
        raise click.UsageError("No modification chosen! Aborting.")

    if join_corpora or join_embeddings:
        for d in datasets:
            if d not in JOINT_TO_DATASETS:
                choices_str = ", ".join([f"'{choice}'" for choice in JOINT_TO_DATASETS.keys()])
                raise click.UsageError(f"Can't create joint dataset {d!r}.\nChoose one of {choices_str}")

    tira = Client()
    tira_dir = default_tira_cache_dir()

    if "all" in datasets:
        datasets = list(JOINT_TO_DATASETS.keys()) + list(all_datasets())
    if "all" in embedding:
        embedding = all_embeddings() + list(all_dense_embeddings())
    if "all" in quantization:
        quantization = ["fp16", "int8", "uint8", "ternary", "binary"]

    created_dataset_dirs = []
    created_embedding_dirs = []

    for dataset in tqdm(datasets, desc="Joining"):
        if join_corpora:
            created_dataset_dirs.append(perform_dataset_join(dataset, tira, tira_dir))
        if join_embeddings:
            for emb in tqdm(embedding, desc="Processing Embeddings"):
                created_embedding_dirs.append(perform_embedding_join(dataset, emb, tira, tira_dir))
    if quantization:
        for dataset in tqdm(datasets, desc="Quantizing"):
            for emb in tqdm(embedding, desc="Processing Embeddings"):
                if dataset in JOINT_TO_DATASETS:
                    emb_path = Path(f"{tira_dir}/extracted_runs/lsr-benchmark/{dataset}/{emb}")
                    if not emb_path.exists():
                        raise click.UsageError(
                            f"'{emb}' embeddings don't exist yet for '{dataset}'. Retry with '-j' or '--join'. Aborting!"
                        )
                else:
                    emb_path = download_embeddings(emb, dataset, tira)

                for level in quantization:
                    created_embedding_dirs.append(
                        perform_quantization(emb_path, level, quant_range, keep_dtype, dataset, emb, tira_dir)
                    )

    click.echo("\nFollowing paths have been created:")
    if created_dataset_dirs:
        click.echo("Dataset:")
        for path in sorted(created_dataset_dirs):
            click.echo(f"  - {path}")

    if created_embedding_dirs:
        click.echo("Embeddings:")
        for path in sorted(created_embedding_dirs):
            click.echo(f"  - {path}")

    return 0
