#!/usr/bin/env python3
import ir_datasets
import lsr_benchmark
import click

from tqdm import tqdm
from tirex_tracker import tracking, ExportFormat, register_metadata
from shutil import rmtree
from lsr_benchmark.click import retrieve_command
import gzip
import numpy as np

from kannolo import SparsePlainHNSW

class KannoloDatasetBuffer():
    def __init__(self):
        self.doc_ids = []
        self.tokens = []
        self.values = []
        self.offsets = [0]

    def add_document(self, doc_id, tokens, values):
        self.doc_ids.append(doc_id)
        self.tokens.append(np.fromiter(map(lambda x: int(x), tokens), dtype=np.int32))
        self.values.append(np.fromiter(values, dtype=np.float32))
        self.offsets.append(self.offsets[-1] + len(tokens))

    def __len__(self):
        return len(self.doc_ids)

    def finalize(self):
        self.doc_ids = np.array(self.doc_ids)
        self.tokens = np.ascontiguousarray(np.concatenate(self.tokens, dtype=np.int32).flatten())
        self.values = np.ascontiguousarray(np.concatenate(self.values, dtype=np.float32).flatten())
        self.offsets = np.ascontiguousarray(np.array(self.offsets, dtype=np.int64).flatten())


@retrieve_command()
@click.option("--ef-search", type=int, required=False, default=200, help="TBD.")
def main(dataset, embedding, output, ef_search, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")

    kannolo_dataset = KannoloDatasetBuffer()

    register_metadata({"actor": {"team": "reneuir-baselines"}, "tag": f"kannolo-{embedding.replace('/', '-')}-{ef_search}-{k}"})
    for (doc_id, tokens, values) in tqdm(ir_dataset.doc_embeddings(model_name=embedding), "create kannolo dataset"):
        kannolo_dataset.add_document(doc_id, tokens, values)

    kannolo_dataset.finalize()
    
    print("Number of document:", len(kannolo_dataset))
    print("Number of unique tokens:", len(set(kannolo_dataset.tokens)))
    print("Max token id:", max(kannolo_dataset.tokens))
        
    print("Documents added to the KannoloDataset. Now indexing..")
    
    efConstruction = 200
    m = 32 
    metric = "ip" 

    with tracking(export_file_path=output / "index-metadata.yml", export_format=ExportFormat.IR_METADATA, ):
        index = SparsePlainHNSW.build_from_arrays(kannolo_dataset.tokens, kannolo_dataset.values, kannolo_dataset.offsets, m=m, ef_construction=efConstruction, metric=metric)

    query_embeddings = ir_dataset.query_embeddings(model_name=embedding)
    
    rmtree(output / ".tirex-tracker")
    results = []

    with tracking(export_file_path=output / "retrieval-metadata.yml", export_format=ExportFormat.IR_METADATA):
        for query_id, query_components, query_values in query_embeddings:
            int_query_components = np.ascontiguousarray(np.fromiter(map(lambda x: int(x), query_components), dtype=np.int32))
            float_query_values = np.ascontiguousarray(np.fromiter(query_values, dtype=np.float32))
            query_offsets = np.array([0, len(int_query_components)], dtype=np.int64)
            dist, ids = index.search(int_query_components, float_query_values, query_offsets,k=k, ef_search=ef_search)
            seen = set()
            unique_dist = []
            unique_docnos = []
            for d, i in zip(dist, ids):
                if int(i) < 0:
                    continue
                docno = kannolo_dataset.doc_ids[int(i)]
                if docno in seen:
                    continue
                seen.add(docno)
                unique_dist.append(d)
                unique_docnos.append(docno)
            results.append(([query_id] * len(unique_dist), unique_dist, unique_docnos))

    rmtree(output / ".tirex-tracker")
    with gzip.open(output/"run.txt.gz", "wt") as f:
        for ranking_for_query in results:
            qids, scores, docnos = ranking_for_query
            rank = 1
            for qid, score, docno in zip(qids, scores, docnos):
                f.write(f"{qid} Q0 {docno} {rank} {score} kannolo\n")
                rank += 1


if __name__ == "__main__":
    main()
