#!/usr/bin/env python3
import gzip
from array import array
from shutil import rmtree

import click
import ir_datasets
import numpy as np
import torch
from tirex_tracker import ExportFormat, register_metadata, tracking

import lsr_benchmark
from lsr_benchmark.click import retrieve_command


class EmbeddingsToSparseTensor:

    def __init__(self) -> None:
        self.data = array("f")
        self.indices = array("i")
        self.indptr = array("L", [0])

    def add(self, components: np.ndarray, values: np.ndarray) -> None:
        self.data.extend(values)
        self.indices.extend(components.astype(np.int32))
        self.indptr.append(self.indptr[-1] + len(components))

    def to_tensor(self, device: torch.device | None = None, size: tuple[int, int] | None = None) -> torch.Tensor:
        return (
            torch.sparse_csr_tensor(
                crow_indices=torch.frombuffer(self.indptr, dtype=torch.int64),
                col_indices=torch.frombuffer(self.indices, dtype=torch.int32),
                values=torch.frombuffer(self.data, dtype=torch.float32),
                size=size if size else (len(self.indptr) - 1, max(self.indices) + 1 if self.indices else 0),
                device=device,
            )
            .to_sparse_coo()
            .to_sparse_csr()  # needed for some reason to work on GPU
        )

@retrieve_command()
@click.option("--use_gpu", is_flag=True, help="Whether to use a GPU if available.")
@click.option("--batch_size", type=int, required=False, default=32, help="Batch size for processing.")
def main(dataset, embedding, output, k, use_gpu, batch_size):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")
    register_metadata(
        {
            "actor": {"team": "reneuir-baselines"},
            "tag": f"pytorch-{embedding.replace('/', '-')}-{k}",
        }
    )

    if use_gpu and not torch.cuda.is_available():
        raise ValueError("No GPU available, but --use_gpu was set.")
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    doc_embeddings = EmbeddingsToSparseTensor()
    doc_ids = []
    for doc_id, tokens, values in ir_dataset.doc_embeddings(model_name=embedding):
        doc_ids.append(doc_id)
        doc_embeddings.add(tokens, values)

    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
        index = doc_embeddings.to_tensor(device=device)

    query_embeddings = ir_dataset.query_embeddings(model_name=embedding)

    rmtree(output / ".tirex-tracker")

    query_embeddings = EmbeddingsToSparseTensor()
    query_ids = []
    for query_id, tokens, values in ir_dataset.query_embeddings(model_name=embedding):
        query_ids.append(query_id)
        query_embeddings.add(tokens, values)

    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        query_tensor = query_embeddings.to_tensor(device=device, size=(len(query_ids), index.shape[1])).to_dense()
        results = []

        offset = 0
        for _query_tensor in query_tensor.split(batch_size):
            scores = index.matmul(_query_tensor.T).T
            topk_scores, topk_indices = torch.topk(scores, k=min(k, scores.shape[1]), dim=-1)
            batch_query_ids = query_ids[offset : offset + _query_tensor.shape[0]]
            offset += _query_tensor.shape[0]
            for query_id, scores, indices in zip(batch_query_ids, topk_scores.cpu().numpy(), topk_indices.cpu().numpy()):
                ranking_for_query = []
                for score, doc_idx in zip(scores, indices):
                    if score == 0:
                        continue
                    ranking_for_query.append((query_id, float(score), doc_ids[doc_idx]))
                results.append(ranking_for_query)

    rmtree(output / ".tirex-tracker")
    with gzip.open(output / "run.txt.gz", "wt") as f:
        for ranking_for_query in results:
            rank = 1
            for qid, score, docno in ranking_for_query:
                f.write(f"{qid} Q0 {docno} {rank} {score} seismic\n")
                rank += 1


if __name__ == "__main__":
    main()
