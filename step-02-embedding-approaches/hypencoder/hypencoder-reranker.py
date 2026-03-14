from pathlib import Path

import click
import ir_datasets
from tirex_tracker import ExportFormat, register_metadata, tracking

from hypencoder_cb.modeling.hypencoder import Hypencoder, HypencoderDualEncoder, TextEncoder
from transformers import AutoTokenizer

import lsr_benchmark
from lsr_benchmark.click import option_lsr_dataset


@click.command()
@option_lsr_dataset()
@click.option("--model", type=str, required=False, default="jfkback/hypencoder.6_layer", help="The hypencoder model.")
def main(dataset: str, model: str, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")

    dual_encoder = HypencoderDualEncoder.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)

    query_encoder: Hypencoder = dual_encoder.query_encoder
    passage_encoder: TextEncoder = dual_encoder.passage_encoder

    queries = [{"docno": i.query_id, "text": i.default_text()} for i in ir_dataset.queries_iter()]
    documents = [{"docno": i.doc_id, "text": i.default_text()} for i in ir_dataset.docs_iter()]

    query_inputs = tokenizer([i["text"] for i in queries], return_tensors="pt", padding=True, truncation=True)
    document_inputs = tokenizer([i["text"] for i in documents], return_tensors="pt", padding=True, truncation=True)

    with tracking(export_file_path=output / f"embedding-ir-metadata.yml"):
        q_nets = query_encoder(input_ids=query_inputs["input_ids"], attention_mask=query_inputs["attention_mask"]).representation
        document_embeddings = passage_encoder(input_ids=document_inputs["input_ids"], attention_mask=document_inputs["attention_mask"]).representation


    #document_embeddings_single = document_embeddings.unsqueeze(1)
    print(dir(q_nets))

if __name__ == '__main__':
    main()