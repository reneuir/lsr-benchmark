#!/usr/bin/env python3
import gzip

import click
import faiss
import numpy as np
from tirex_tracker import ExportFormat, register_metadata, tracking

import lsr_benchmark
from lsr_benchmark.click import retrieve_command
from lsr_benchmark.irds import embeddings as load_embeddings


def determine_dimension(*embedding_collections):
    dimension = 0
    for embeddings in embedding_collections:
        for _, tokens, _ in embeddings:
            if len(tokens):
                dimension = max(dimension, max(int(token) for token in tokens) + 1)
    return dimension


def to_dense_matrix(embeddings, dimension):
    embedding_ids = []
    matrix = np.zeros((len(embeddings), dimension), dtype=np.float32)

    for row, (embedding_id, tokens, values) in enumerate(embeddings):
        embedding_ids.append(embedding_id)
        if len(tokens):
            matrix[row, np.asarray(tokens, dtype=np.int64)] = np.asarray(values, dtype=np.float32)

    return embedding_ids, matrix


def build_index(doc_embeddings):
    if doc_embeddings.ndim != 2 or doc_embeddings.shape[1] == 0:
        raise ValueError("Document embeddings must have at least one dimension.")

    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(np.ascontiguousarray(doc_embeddings, dtype=np.float32))
    return index


def retrieve(index, query_ids, query_embeddings, doc_ids, k):
    if k < 1:
        raise ValueError("k must be at least 1.")
    if index.ntotal != len(doc_ids):
        raise ValueError("The number of indexed vectors must match the number of document IDs.")
    if not doc_ids:
        return [[] for _ in query_ids]

    scores, indices = index.search(
        np.ascontiguousarray(query_embeddings, dtype=np.float32),
        min(k, len(doc_ids)),
    )
    results = []
    for query_id, query_scores, query_indices in zip(query_ids, scores, indices):
        ranking = []
        for score, doc_index in zip(query_scores, query_indices):
            if doc_index < 0 or score <= 0:
                continue
            ranking.append((query_id, float(score), doc_ids[doc_index]))
        results.append(ranking)
    return results


@retrieve_command()
@click.option("--batch-size", type=click.IntRange(min=1), default=128, show_default=True)
def main(dataset, embedding, output, k, batch_size):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    register_metadata(
        {
            "actor": {"team": "reneuir-baselines"},
            "tag": f"faiss-{embedding.replace('/', '-')}-{k}",
        }
    )

    print("Load embeddings...")
    doc_embedding_data = load_embeddings(dataset, embedding, "doc")
    query_embedding_data = load_embeddings(dataset, embedding, "query")
    dimension = determine_dimension(doc_embedding_data, query_embedding_data)
    doc_ids, doc_embeddings = to_dense_matrix(doc_embedding_data, dimension)
    query_ids, query_embeddings = to_dense_matrix(query_embedding_data, dimension)
    print("Done loading embeddings.")

    with tracking(
        export_file_path=output / "index-metadata.yml",
        export_format=ExportFormat.IR_METADATA,
    ):
        index = build_index(doc_embeddings)

    with tracking(
        export_file_path=output / "retrieval-metadata.yml",
        export_format=ExportFormat.IR_METADATA,
    ):
        results = []
        for start in range(0, len(query_ids), batch_size):
            end = start + batch_size
            results.extend(
                retrieve(
                    index,
                    query_ids[start:end],
                    query_embeddings[start:end],
                    doc_ids,
                    k,
                )
            )

    with gzip.open(output / "run.txt.gz", "wt") as run_file:
        for ranking in results:
            for rank, (query_id, score, doc_id) in enumerate(ranking, start=1):
                run_file.write(f"{query_id} Q0 {doc_id} {rank} {score} faiss\n")


if __name__ == "__main__":
    main()
