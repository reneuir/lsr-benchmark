#!/usr/bin/env python3
import gzip
import math
import struct
import subprocess
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import click
from tirex_tracker import ExportFormat, register_metadata, tracking

import lsr_benchmark
from lsr_benchmark.click import retrieve_command
from lsr_benchmark.irds import embeddings as load_embeddings


MAX_SCORE = (1 << 16) - 1
IOQP_CREATE = "/usr/local/bin/ioqp-create"
IOQP_QUERY = "/usr/local/bin/ioqp-query"


def encode_varint(value):
    if value < 0:
        raise ValueError("Varints must be non-negative.")
    encoded = bytearray()
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def encode_int(field_number, value):
    return encode_varint(field_number << 3) + encode_varint(value)


def encode_bytes(field_number, value):
    return encode_varint((field_number << 3) | 2) + encode_varint(len(value)) + value


def encode_string(field_number, value):
    return encode_bytes(field_number, value.encode())


def encode_message(field_number, value):
    return encode_bytes(field_number, value)


def encode_double(field_number, value):
    return encode_varint((field_number << 3) | 1) + struct.pack("<d", value)


def write_delimited(output, message):
    output.write(encode_varint(len(message)))
    output.write(message)


def determine_quantization_levels(query_embeddings, max_document_impact, max_query_weight):
    max_query_terms = max(
        (
            len({str(token) for token, value in zip(tokens, values) if float(value) > 0})
            for _, tokens, values in query_embeddings
        ),
        default=0,
    )
    if max_query_terms == 0:
        return max_document_impact, max_query_weight

    max_product = MAX_SCORE // max_query_terms
    query_weight = min(max_query_weight, max(1, math.isqrt(max_product)))
    document_impact = min(max_document_impact, max(1, max_product // query_weight))
    return document_impact, query_weight


def quantize(values_by_token, maximum, reference_max=None):
    positive_values = {
        str(token): float(value)
        for token, value in values_by_token.items()
        if float(value) > 0
    }
    if not positive_values:
        return {}

    max_value = reference_max if reference_max is not None else max(positive_values.values())
    if max_value <= 0:
        raise ValueError("The quantization reference must be positive.")
    return {
        token: min(maximum, max(1, math.ceil(value * maximum / max_value)))
        for token, value in positive_values.items()
    }


def merge_embedding(tokens, values):
    if len(tokens) != len(values):
        raise ValueError("Embedding tokens and values must have the same length.")
    merged = defaultdict(float)
    for token, value in zip(tokens, values):
        merged[str(token)] += float(value)
    return merged


def write_ciff(path, document_embeddings, max_document_impact):
    postings = defaultdict(list)
    document_records = []
    total_terms = 0
    merged_documents = [
        (doc_id, merge_embedding(tokens, values))
        for doc_id, tokens, values in document_embeddings
    ]
    max_document_value = max(
        (
            float(value)
            for _, embedding in merged_documents
            for value in embedding.values()
            if float(value) > 0
        ),
        default=0,
    )

    for doc_index, (doc_id, embedding) in enumerate(merged_documents):
        quantized = quantize(embedding, max_document_impact, max_document_value)
        document_length = sum(quantized.values())
        total_terms += document_length
        document_records.append((doc_index, str(doc_id), document_length))
        for token, impact in quantized.items():
            postings[token].append((doc_index, impact))

    if not document_records:
        raise ValueError("IOQP requires at least one document.")
    if not postings:
        raise ValueError("IOQP requires at least one positive document embedding value.")

    header = b"".join(
        [
            encode_int(1, 1),
            encode_int(2, len(postings)),
            encode_int(3, len(document_records)),
            encode_int(4, len(postings)),
            encode_int(5, len(document_records)),
            encode_int(6, total_terms),
            encode_double(7, total_terms / len(document_records)),
            encode_string(8, "lsr-benchmark IOQP index"),
        ]
    )

    with Path(path).open("wb") as output:
        write_delimited(output, header)
        for token in sorted(postings):
            posting_messages = []
            previous_doc_id = 0
            collection_frequency = 0
            for doc_id, impact in postings[token]:
                posting_messages.append(
                    encode_message(
                        4,
                        encode_int(1, doc_id - previous_doc_id) + encode_int(2, impact),
                    )
                )
                previous_doc_id = doc_id
                collection_frequency += impact
            posting_list = b"".join(
                [
                    encode_string(1, token),
                    encode_int(2, len(postings[token])),
                    encode_int(3, collection_frequency),
                    *posting_messages,
                ]
            )
            write_delimited(output, posting_list)

        for doc_id, collection_doc_id, document_length in document_records:
            record = b"".join(
                [
                    encode_int(1, doc_id),
                    encode_string(2, collection_doc_id),
                    encode_int(3, document_length),
                ]
            )
            write_delimited(output, record)


def write_queries(path, query_embeddings, max_query_weight):
    query_ids = {}
    with Path(path).open("w") as output:
        internal_query_id = 0
        for query_id, tokens, values in query_embeddings:
            quantized = quantize(merge_embedding(tokens, values), max_query_weight)
            if not quantized:
                continue
            weighted_tokens = []
            for token, weight in sorted(quantized.items()):
                weighted_tokens.extend([token] * weight)
            query_ids[internal_query_id] = str(query_id)
            output.write(f"{internal_query_id}:{' '.join(weighted_tokens)}\n")
            internal_query_id += 1
    return query_ids


def create_ioqp_index(ciff_path, index_path):
    subprocess.run(  # noqa: S603
        [IOQP_CREATE, "--input", str(ciff_path), "--output", str(index_path)],
        check=True,
    )


def query_ioqp(index_path, query_path, run_path, k, mode):
    subprocess.run(  # noqa: S603
        [
            IOQP_QUERY,
            "--index",
            str(index_path),
            "--queries",
            str(query_path),
            "--output-file",
            str(run_path),
            "--k",
            str(k),
            "--mode",
            mode,
            "--weighted",
        ],
        check=True,
    )


def parse_run(path, query_ids):
    rankings = defaultdict(list)
    with Path(path).open() as run_file:
        for line in run_file:
            internal_query_id, _, doc_id, _, score, _ = line.split()
            numeric_score = int(score)
            if numeric_score > 0:
                rankings[int(internal_query_id)].append((doc_id, numeric_score))
    return [
        (query_ids[internal_query_id], rankings[internal_query_id])
        for internal_query_id in sorted(query_ids)
    ]


@retrieve_command()
@click.option("--rho", type=click.FloatRange(min=0.0, max=1.0), default=1.0, show_default=True)
@click.option("--postings-budget", type=click.IntRange(min=1), default=None)
@click.option("--max-document-impact", type=click.IntRange(min=1, max=65535), default=255, show_default=True)
@click.option("--max-query-weight", type=click.IntRange(min=1, max=32), default=32, show_default=True)
def main(
    dataset,
    embedding,
    output,
    k,
    rho,
    postings_budget,
    max_document_impact,
    max_query_weight,
):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    register_metadata(
        {
            "actor": {"team": "reneuir-baselines"},
            "tag": f"ioqp-{embedding.replace('/', '-')}-{rho}-{postings_budget}-{k}",
        }
    )

    document_embeddings = load_embeddings(dataset, embedding, "doc")
    query_embeddings = load_embeddings(dataset, embedding, "query")
    document_impact, query_weight = determine_quantization_levels(
        query_embeddings,
        max_document_impact,
        max_query_weight,
    )

    with TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)
        ciff_path = temporary_directory / "index.ciff"
        index_path = temporary_directory / "index.ioqp"
        query_path = temporary_directory / "queries.txt"
        raw_run_path = temporary_directory / "run.txt"

        with tracking(
            export_file_path=output / "index-metadata.yml",
            export_format=ExportFormat.IR_METADATA,
        ):
            write_ciff(ciff_path, document_embeddings, document_impact)
            create_ioqp_index(ciff_path, index_path)

        with tracking(
            export_file_path=output / "retrieval-metadata.yml",
            export_format=ExportFormat.IR_METADATA,
        ):
            query_ids = write_queries(query_path, query_embeddings, query_weight)
            if query_ids:
                mode = f"fixed-{postings_budget}" if postings_budget else f"fraction-{rho}"
                query_ioqp(
                    index_path,
                    query_path,
                    raw_run_path,
                    min(k, len(document_embeddings)),
                    mode,
                )
                results = parse_run(raw_run_path, query_ids)
            else:
                results = []

    with gzip.open(output / "run.txt.gz", "wt") as run_file:
        for query_id, ranking in results:
            for rank, (doc_id, score) in enumerate(ranking, start=1):
                run_file.write(f"{query_id} Q0 {doc_id} {rank} {score} ioqp\n")


if __name__ == "__main__":
    main()