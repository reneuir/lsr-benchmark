"""
Attention, this file was created with the help of an coding agent
"""
import gzip
import json
from pathlib import Path

import numpy as np
import click
from shutil import copy2
from tira.io_utils import patch_ir_metadata
import yaml
from tira.rest_api_client import Client
from lsr_benchmark.datasets import IR_DATASET_TO_TIRA_DATASET

SISAP_RETRIEVAL_DEPTH = 30
SISAP_GROUND_TRUTH_DEPTH = 100


class MissingSisapDependencyError(RuntimeError):
    pass


def _copy_sisap_embedding_metadata(
    source_dir: Path,
    target_dir: Path,
    dataset: str,
    tira_dataset: str | None,
    embedding_tira_id: str | None = None,
):
    metadata_files = {
        source_dir / "doc" / "doc-ir-metadata.yml": target_dir / "doc-ir-metadata.yml",
        source_dir / "query" / "query-ir-metadata.yml": target_dir / "query-ir-metadata.yml",
    }
    for source_path, target_path in metadata_files.items():
        if not source_path.exists():
            raise click.ClickException(f"Expected SISAP metadata file is missing: {source_path}")
        copy2(source_path, target_path)
    if tira_dataset is not None:
        patch_ir_metadata(
            str(target_dir),
            {"data": {"test collection": {"name": "/tira-data/input"}}},
            {"data": {"test collection": {"name": tira_dataset}}},
        )
    embedding_model_metadata = {}
    if embedding_tira_id:
        tira = Client()
        _, embedding_team, embedding_system = embedding_tira_id.split("/", 2)
        system_details = tira.public_system_details(embedding_team, embedding_system)
        embedding_model_metadata = {
            "name": system_details["command"].split("--model")[1].strip(),
            "tira-embedding-software": embedding_tira_id
        }

    for metadata_path in metadata_files.values():
        content = yaml.safe_load(metadata_path.read_text())
        content["data"]["test collection"]["ir-datasets-id"] = dataset
        if embedding_tira_id is not None:
            content["data"]["embedding model"] = embedding_model_metadata
        metadata_path.write_text(yaml.safe_dump(content))
    (target_dir / "doc-ir-metadata.yml").replace(target_dir / "document-embedding-metadata.yml")
    (target_dir / "query-ir-metadata.yml").replace(target_dir / "query-embedding-metadata.yml")


@click.command()
@click.option(
    "--results",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="The SISAP results.h5 file or a TIRA-style results directory containing .h5 files to convert.",
)
@click.option(
    "--embeddings",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="The SISAP embedding bundle directory with ID mapping files.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="The output directory to write run.txt.gz and forwarded metadata files to.",
)
@click.option(
    "--system-name",
    type=str,
    required=False,
    default="sisap",
    help="The system name to write in the run file.",
)
def sisap_to_trec_run(results: Path, embeddings: Path, output: Path, system_name: str):
    try:
        ret = convert_sisap_results_to_trec_run(results, embeddings, output, system_name)
    except MissingSisapDependencyError as exc:
        raise click.ClickException(str(exc)) from exc
    print(ret)


@click.command()
@click.option(
    "--truths",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="The SISAP truths bundle directory with config.json and the ground-truth HDF5 file.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="The TREC qrels file to write. Use .gz for gzip output.",
)
def sisap_to_qrels(truths: Path, output: Path):
    try:
        ret = convert_sisap_truths_to_qrels(truths, output)
    except MissingSisapDependencyError as exc:
        raise click.ClickException(str(exc)) from exc
    print(ret)


def export_embeddings_to_sisap(
    source_dir: Path,
    target_dir: Path,
    dataset: str,
    embedding_tira_id: str | None = None,
    preserve_source_metadata: bool = False,
) -> Path:
    h5py = _import_h5py()

    doc_data, doc_indices, doc_indptr, doc_ids = _load_embedding_matrix(source_dir, "doc")
    query_data, query_indices, query_indptr, query_ids = _load_embedding_matrix(source_dir, "query")
    feature_count = _feature_count(doc_indices, query_indices)

    ground_truth_depth = min(SISAP_GROUND_TRUTH_DEPTH, len(doc_ids))
    run_path = _write_ground_truth_run_file(source_dir, target_dir, dataset)
    knns, dists, algo = _build_ground_truth_from_run_file(run_path, query_ids, doc_ids, ground_truth_depth)

    dataset_name = dataset.replace("/", "-")
    h5_filename = f"benchmark-dev-{dataset_name}.h5"

    with h5py.File(target_dir / h5_filename, "w") as h5_file:
        train_group = h5_file.create_group("train")
        train_group.create_dataset("data", data=doc_data, dtype=np.float32)
        train_group.create_dataset("indices", data=doc_indices, dtype=np.int64)
        train_group.create_dataset("indptr", data=doc_indptr, dtype=np.int64)
        train_group.attrs["shape"] = np.array([len(doc_ids), feature_count], dtype=np.int64)

        otest_group = h5_file.create_group("otest")
        otest_group.attrs["algo"] = algo
        otest_group.attrs["querytime"] = np.float64(0.0)
        queries_group = otest_group.create_group("queries")
        queries_group.create_dataset("data", data=query_data, dtype=np.float32)
        queries_group.create_dataset("indices", data=query_indices, dtype=np.int64)
        queries_group.create_dataset("indptr", data=query_indptr, dtype=np.int64)
        queries_group.attrs["shape"] = np.array([len(query_ids), feature_count], dtype=np.int64)
        otest_group.create_dataset("knns", data=knns, dtype=np.int32)
        otest_group.create_dataset("dists", data=dists, dtype=np.float32)

    config = {
        "task": "task3",
        "data": "train",
        "queries": "otest/queries",
        "gt_I": "otest/knns",
        "k": SISAP_RETRIEVAL_DEPTH,
        "sparse": True,
        "dataset_name": dataset_name,
        "filename": h5_filename,
    }
    (target_dir / "config.json").write_text(json.dumps(config, indent=4) + "\n")

    _write_json_gz(target_dir / "query-id-to-index.json.gz", {query_id: idx for idx, query_id in enumerate(query_ids)})
    _write_json_gz(target_dir / "document-id-to-index.json.gz", {doc_id: idx for idx, doc_id in enumerate(doc_ids)})

    _copy_sisap_embedding_metadata(
        source_dir,
        target_dir,
        dataset,
        None if preserve_source_metadata else IR_DATASET_TO_TIRA_DATASET.get(dataset),
        embedding_tira_id=embedding_tira_id,
    )

    return target_dir


def _write_ground_truth_run_file(source_dir: Path, target_dir: Path, dataset: str) -> Path:
    from tira.io_utils import docker_supported_target_platform

    from ._retrieval import run_retrieval_engine
    from ._verify_installation import EXAMPLE_RETRIEVAL_ENGINE

    platform = docker_supported_target_platform()
    if platform not in EXAMPLE_RETRIEVAL_ENGINE:
        raise ValueError(f"The platform {platform} is not supported for SISAP ground-truth export.")

    retrieval_engine = EXAMPLE_RETRIEVAL_ENGINE[platform]["pyterrier-splade-pisa"]
    run_retrieval_engine(
        retrieval_engine["image"],
        retrieval_engine["command"] + " --k 100",
        dataset,
        source_dir,
        target_dir / "exact-retrieval-run",
        platform=platform,
    )

    ret = target_dir / "exact-retrieval-run" / "run.txt"

    if ret.exists():
        return ret
    else:
        raise ValueError("The retrieval command did not produce run.txt")


def convert_sisap_results_to_trec_run(
    results_path: Path,
    embeddings_dir: Path,
    output_dir: Path,
    system_name: str,
) -> Path:
    if results_path.is_dir():
        result_files = sorted(path for path in results_path.rglob("*.h5") if path.is_file())
        if not result_files:
            raise ValueError(f"Did not find any .h5 result files below {results_path}.")
        for result_file in result_files:
            _convert_single_sisap_results_to_trec_run(
                result_file,
                embeddings_dir,
                _output_dir_for_result_file(results_path, output_dir, result_file),
                system_name,
            )
        return output_dir

    return _convert_single_sisap_results_to_trec_run(results_path, embeddings_dir, output_dir, system_name)


def _convert_single_sisap_results_to_trec_run(
    results_path: Path,
    embeddings_dir: Path,
    output_dir: Path,
    system_name: str,
) -> Path:
    h5py = _import_h5py()
    query_ids = _invert_id_mapping(_read_json_gz(embeddings_dir / "query-id-to-index.json.gz"), "query")
    document_ids = _invert_id_mapping(_read_json_gz(embeddings_dir / "document-id-to-index.json.gz"), "document")

    with h5py.File(results_path, "r") as results_file:
        knns = results_file["knns"][:]
        dists = results_file["dists"][:]

    if knns.ndim != 2:
        raise ValueError(f"Expected knns and dists to be two-dimensional, got {knns.ndim} dimensions.")
    if knns.shape != dists.shape:
        raise ValueError(f"Expected knns and dists to share the same shape, got {knns.shape} and {dists.shape}.")
    if knns.shape[0] != len(query_ids):
        raise ValueError(
            f"Expected {len(query_ids)} query rows from the SISAP embeddings, but got {knns.shape[0]} rows in {results_path}."
        )

    output_path = output_dir / "run.txt.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text_for_write(output_path) as output_file:
        for query_idx, query_id in enumerate(query_ids):
            query_scores = _decreasing_scores_for_ranking(knns.shape[1])
            for rank, (doc_idx, score) in enumerate(zip(knns[query_idx], query_scores), start=1):
                doc_idx = int(doc_idx) - 1
                if doc_idx < 0 or doc_idx >= len(document_ids):
                    raise ValueError(
                        f"Document index {doc_idx + 1} at query row {query_idx}, rank {rank} is out of range for "
                        f"{len(document_ids)} one-based SISAP document IDs."
                    )
                output_file.write(f"{query_id} Q0 {document_ids[doc_idx]} {rank} {float(score)} {system_name}\n")

    _forward_embedding_metadata(embeddings_dir, output_dir)
    return output_dir


def convert_sisap_truths_to_qrels(truths_dir: Path, output_path: Path) -> Path:
    h5py = _import_h5py()
    config = json.loads((truths_dir / "config.json").read_text())
    truths_path = truths_dir / config["filename"]

    with h5py.File(truths_path, "r") as truths_file:
        gt_I = truths_file[config["gt_I"]][:]

    if gt_I.ndim != 2:
        raise ValueError(f"Expected gt_I to be two-dimensional, got {gt_I.ndim} dimensions.")
    if config["k"] > gt_I.shape[1]:
        raise ValueError(f"Expected gt_I to have at least {config['k']} columns, got {gt_I.shape[1]}.")

    normalized_gt_I = _normalize_truth_doc_ids(gt_I)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text_for_write(output_path) as output_file:
        for query_idx, doc_ids in enumerate(normalized_gt_I[:, : config["k"]], start=1):
            for doc_id in doc_ids:
                output_file.write(f"{query_idx} 0 {int(doc_id)} 1\n")

    return output_path


def _build_ground_truth_from_run_file(
    run_path: Path,
    query_ids: list[str],
    doc_ids: list[str],
    depth: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if depth == 0:
        return (
            np.empty((len(query_ids), 0), dtype=np.int32),
            np.empty((len(query_ids), 0), dtype=np.float32),
            "",
        )

    doc_id_to_index = {doc_id: idx + 1 for idx, doc_id in enumerate(doc_ids)}
    rankings: dict[str, list[tuple[int, int, float]]] = {query_id: [] for query_id in query_ids}
    seen_ranks: dict[str, set[int]] = {query_id: set() for query_id in query_ids}
    software_name = None

    with _open_text_for_read(run_path) as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            parts = stripped_line.split()
            if len(parts) != 6:
                raise ValueError(
                    f"Expected {run_path.name} line {line_number} to have 6 whitespace-separated columns, "
                    f"got {len(parts)}."
                )

            query_id, _, doc_id, rank_text, score_text, current_software_name = parts
            if software_name is None:
                software_name = current_software_name
            elif current_software_name != software_name:
                raise ValueError(
                    f"Expected {run_path.name} to contain exactly one software, got "
                    f"{software_name!r} and {current_software_name!r}."
                )

            if query_id not in rankings:
                raise ValueError(f"Unexpected query ID {query_id!r} in {run_path.name}.")
            if doc_id not in doc_id_to_index:
                raise ValueError(f"Unexpected document ID {doc_id!r} in {run_path.name}.")

            rank = int(rank_text)
            if rank <= 0:
                raise ValueError(f"Expected positive ranks in {run_path.name}, got {rank} on line {line_number}.")
            if rank in seen_ranks[query_id]:
                raise ValueError(f"Duplicate rank {rank} for query {query_id!r} in {run_path.name}.")
            seen_ranks[query_id].add(rank)

            rankings[query_id].append((rank, doc_id_to_index[doc_id], float(score_text)))

    if software_name is None:
        raise ValueError(f"Expected at least one run entry in {run_path.name}.")

    knns = np.empty((len(query_ids), depth), dtype=np.int32)
    dists = np.empty((len(query_ids), depth), dtype=np.float32)
    for query_idx, query_id in enumerate(query_ids):
        ranked_docs = sorted(rankings[query_id], key=lambda item: item[0])
        if len(ranked_docs) < depth:
            raise ValueError(
                f"Expected at least {depth} run entries for query {query_id!r} in {run_path.name}, "
                f"got {len(ranked_docs)}."
            )
        knns[query_idx] = np.asarray([doc_idx for _, doc_idx, _ in ranked_docs[:depth]], dtype=np.int32)
        dists[query_idx] = np.asarray([score for _, _, score in ranked_docs[:depth]], dtype=np.float32)

    return knns, dists, software_name


def _load_embedding_matrix(embedding_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    matrix = np.load(embedding_dir / prefix / f"{prefix}-embeddings.npz")
    ids = (embedding_dir / prefix / f"{prefix}-ids.txt").read_text().strip().splitlines()
    return (
        matrix["data"].astype(np.float32),
        matrix["indices"].astype(np.int64),
        matrix["indptr"].astype(np.int64),
        ids,
    )


def _build_postings(
    doc_data: np.ndarray,
    doc_indices: np.ndarray,
    doc_indptr: np.ndarray,
) -> dict[int, list[tuple[int, float]]]:
    postings: dict[int, list[tuple[int, float]]] = {}

    for doc_idx, (start, end) in enumerate(zip(doc_indptr[:-1], doc_indptr[1:])):
        for term_id, doc_weight in zip(doc_indices[start:end], doc_data[start:end]):
            postings.setdefault(int(term_id), []).append((doc_idx, float(doc_weight)))

    return postings


def _forward_embedding_metadata(embeddings_dir: Path, output_dir: Path) -> None:
    metadata_files = (
        "document-embedding-metadata.yml",
        "query-embedding-metadata.yml",
    )
    for metadata_file in metadata_files:
        source_path = embeddings_dir / metadata_file
        if source_path.exists():
            (output_dir / metadata_file).write_bytes(source_path.read_bytes())


def _output_dir_for_result_file(results_root: Path, output_root: Path, result_file: Path) -> Path:
    relative_path = result_file.relative_to(results_root)
    if relative_path.parent.name == "output":
        return output_root / relative_path.parent.parent / result_file.stem

    return output_root / relative_path.parent / result_file.stem


def _feature_count(doc_indices: np.ndarray, query_indices: np.ndarray) -> int:
    max_doc_index = int(doc_indices.max()) if len(doc_indices) > 0 else -1
    max_query_index = int(query_indices.max()) if len(query_indices) > 0 else -1
    return max(max_doc_index, max_query_index) + 1


def _rank_documents_for_query(
    query_indices: np.ndarray,
    query_data: np.ndarray,
    postings: dict[int, list[tuple[int, float]]],
    doc_count: int,
    k: int,
) -> tuple[list[int], list[float]]:
    scores: dict[int, float] = {}

    for term_id, query_weight in zip(query_indices, query_data):
        for doc_idx, doc_weight in postings.get(int(term_id), []):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + (float(query_weight) * doc_weight)

    positives = sorted(
        ((doc_idx, score) for doc_idx, score in scores.items() if score > 0.0),
        key=lambda item: (-item[1], item[0]),
    )
    negatives = sorted(
        ((doc_idx, score) for doc_idx, score in scores.items() if score < 0.0),
        key=lambda item: (-item[1], item[0]),
    )

    ranked_indices = [doc_idx for doc_idx, _ in positives[:k]]
    ranked_scores = [float(score) for _, score in positives[:k]]

    if len(ranked_indices) < k:
        for doc_idx in range(doc_count):
            if doc_idx not in scores or scores[doc_idx] == 0.0:
                ranked_indices.append(doc_idx)
                ranked_scores.append(0.0)
                if len(ranked_indices) == k:
                    return ranked_indices, ranked_scores

    if len(ranked_indices) < k:
        for doc_idx, score in negatives:
            ranked_indices.append(doc_idx)
            ranked_scores.append(float(score))
            if len(ranked_indices) == k:
                break

    return ranked_indices, ranked_scores


def _write_json_gz(path: Path, payload: dict[str, int]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as output_file:
        json.dump(payload, output_file, sort_keys=True)


def _read_json_gz(path: Path) -> dict[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        return json.load(input_file)


def _invert_id_mapping(mapping: dict[str, int], label: str) -> list[str]:
    if not mapping:
        return []

    max_index = max(mapping.values())
    ids = [None] * (max_index + 1)
    for identifier, idx in mapping.items():
        if idx < 0 or idx > max_index:
            raise ValueError(f"Invalid {label} index {idx} for identifier {identifier!r}.")
        if ids[idx] is not None:
            raise ValueError(f"Duplicate {label} index {idx} for identifiers {ids[idx]!r} and {identifier!r}.")
        ids[idx] = identifier

    missing_indices = [idx for idx, identifier in enumerate(ids) if identifier is None]
    if missing_indices:
        raise ValueError(f"Missing {label} identifiers for indices {missing_indices}.")

    return ids


def _open_text_for_write(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def _open_text_for_read(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _normalize_truth_doc_ids(gt_I: np.ndarray) -> np.ndarray:
    if np.any(gt_I < 0):
        raise ValueError("Expected gt_I to contain non-negative document indices.")
    if np.any(gt_I == 0):
        return gt_I + 1
    return gt_I


def _decreasing_scores_for_ranking(length: int) -> list[float]:
    return [float(length - rank) for rank in range(length)]


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise MissingSisapDependencyError(
            "SISAP export requires the optional dependency group. Install it with `pip install lsr-benchmark[sisap]`."
        ) from exc

    return h5py
