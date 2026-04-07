from pathlib import Path

import click
import ir_datasets
from tirex_tracker import ExportFormat, register_metadata, tracking

from tqdm import tqdm
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

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Use device", device)
    query_encoder: Hypencoder = dual_encoder.query_encoder
    query_encoder.to(device)
    passage_encoder: TextEncoder = dual_encoder.passage_encoder
    passage_encoder.to(device)

    queries = [{"qid": i.query_id, "text": i.default_text()} for i in ir_dataset.queries_iter()]
    documents = [{"docno": i.doc_id, "text": i.default_text()} for i in ir_dataset.docs_iter()]

    query_inputs = tokenizer([i["text"] for i in queries], return_tensors="pt", padding=True, truncation=True)
    query_inputs.to(device)

    #document_inputs = tokenizer([i["text"] for i in documents], return_tensors="pt", padding=True, truncation=True)
    #document_inputs.to(device)
    qnets = {}

    with tracking(export_file_path=output / f"embedding-ir-metadata.yml"):
        for idx in tqdm(range(len(queries)), "encode queries"):
            input_ids = torch.tensor([query_inputs["input_ids"][0][idx]])
            input_ids.to(device)
            attention_mask = torch.tensor([query_inputs["attention_mask"][0][idx]])
            attention_mask.to(device)

            q_net = query_encoder(input_ids=input_ids, attention_mask=attention_mask).representation
            qnets[queries[idx]["qid"]] = q_net.state_dict()

        #document_embeddings = passage_encoder(input_ids=document_inputs["input_ids"], attention_mask=document_inputs["attention_mask"]).representation


    #document_embeddings_single = document_embeddings.unsqueeze(1)
    print(dir(q_nets))
    print(type(q_nets))

if __name__ == '__main__':
    main()