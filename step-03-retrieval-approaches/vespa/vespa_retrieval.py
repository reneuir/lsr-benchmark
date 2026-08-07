#!/usr/bin/env python3
import gzip
import json
import math
import os
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import click
from tirex_tracker import ExportFormat, register_metadata, tracking

import lsr_benchmark
from lsr_benchmark.click import retrieve_command
from lsr_benchmark.irds import embeddings as load_embeddings


VESPA_START = "/usr/local/bin/start-container.sh"
VESPA_DEPLOY = "/opt/vespa/bin/vespa-deploy"
CONFIG_URL = "http://127.0.0.1:19071"
DOCUMENT_URL = "http://127.0.0.1:8080/document/v1/lsr/sparse/docid"
QUERY_URL = "http://127.0.0.1:8080/search/"
MAX_TOKEN_ID = (1 << 31) - 1
MAX_VESPA_WEIGHT = (1 << 31) - 1


def vespa_hostname():
    return socket.getfqdn()


@dataclass(frozen=True)
class VespaIndex:
    document_count: int
    indexed_document_count: int
    document_scale: float
    max_weight: int


def merge_embedding(tokens, values):
    if len(tokens) != len(values):
        raise ValueError("Embedding tokens and values must have the same length.")

    merged = defaultdict(float)
    for token, value in zip(tokens, values):
        try:
            token_id = int(token)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Sparse vector index {token!r} is not an integer.") from error
        if not 0 <= token_id <= MAX_TOKEN_ID:
            raise ValueError("Vespa weighted-set keys must fit in a signed 32-bit integer.")

        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("Embedding values must be finite.")
        if numeric_value > 0:
            merged[token_id] += numeric_value
    return dict(merged)


def quantization_scale(embeddings, max_weight):
    if not 1 <= max_weight <= MAX_VESPA_WEIGHT:
        raise ValueError("Maximum Vespa weight must be between 1 and 2,147,483,647.")
    maximum = max(
        (
            value
            for _, tokens, values in embeddings
            for value in merge_embedding(tokens, values).values()
        ),
        default=0,
    )
    if maximum <= 0:
        return 1.0
    return max_weight / maximum


def quantize_embedding(embedding, scale, max_weight):
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("Quantization scale must be finite and positive.")
    return {
        token_id: min(max_weight, max(1, round(value * scale)))
        for token_id, value in embedding.items()
    }


def request_json(url, method="GET", payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"Vespa returned HTTP {error.code}: {body}") from error


class VespaClient:
    def deploy(self, application_path):
        deploy_environment = {
            **os.environ,
            "VESPA_CONFIGSERVERS": vespa_hostname(),
        }
        subprocess.run(  # noqa: S603
            [VESPA_DEPLOY, "prepare", str(application_path)],
            check=True,
            env=deploy_environment,
        )
        subprocess.run(  # noqa: S603
            [VESPA_DEPLOY, "activate"],
            check=True,
            env=deploy_environment,
        )
        wait_for_http(QUERY_URL, timeout=300)

    def feed(self, internal_id, document_id, vector):
        return request_json(
            f"{DOCUMENT_URL}/{internal_id}",
            method="POST",
            payload={
                "fields": {
                    "doc_id": document_id,
                    "embedding": {str(token): weight for token, weight in vector.items()},
                }
            },
        )

    def query(self, vector, k):
        token_weights = ",".join(
            f"{token}:{weight}" for token, weight in sorted(vector.items())
        )
        return request_json(
            QUERY_URL,
            method="POST",
            payload={
                "yql": (
                    "select doc_id from sparse where "
                    f"({{targetHits:{k}}}wand(embedding,@tokens))"
                ),
                "tokens": f"{{{token_weights}}}",
                "ranking": "sparse",
                "hits": k,
                "timeout": "300s",
            },
        )


def write_application_package(path):
    path = Path(path)
    schemas = path / "schemas"
    schemas.mkdir(parents=True)
    hostname = vespa_hostname()
    (path / "hosts.xml").write_text(
        "\n".join(
            [
                "<hosts>",
                f'  <host name="{hostname}">',
                "    <alias>node1</alias>",
                "  </host>",
                "</hosts>",
                "",
            ]
        )
    )
    (path / "services.xml").write_text(
        "\n".join(
            [
                '<services version="1.0">',
                '  <container id="query" version="1.0">',
                "    <search/>",
                "    <document-api/>",
                "    <nodes>",
                '      <jvm options="-Xms32M -Xmx128M"/>',
                '      <node hostalias="node1"/>',
                "    </nodes>",
                "  </container>",
                '  <content id="content" version="1.0">',
                "    <redundancy>1</redundancy>",
                "    <documents>",
                '      <document type="sparse" mode="index"/>',
                "    </documents>",
                "    <tuning>",
                "      <resource-limits>",
                "        <disk>0.99</disk>",
                "      </resource-limits>",
                "    </tuning>",
                "    <nodes>",
                '      <node distribution-key="0" hostalias="node1"/>',
                "    </nodes>",
                "  </content>",
                "</services>",
                "",
            ]
        )
    )
    (schemas / "sparse.sd").write_text(
        "\n".join(
            [
                "schema sparse {",
                "  document sparse {",
                "    field doc_id type string {",
                "      indexing: summary | attribute",
                "    }",
                "    field embedding type weightedset<int> {",
                "      indexing: attribute",
                "      attribute: fast-search",
                "    }",
                "  }",
                "  rank-profile sparse {",
                "    first-phase {",
                "      expression: rawScore(embedding)",
                "    }",
                "  }",
                "}",
                "",
            ]
        )
    )


def build_index(client, document_embeddings, max_weight, feed_workers):
    if feed_workers < 1:
        raise ValueError("Feed workers must be at least 1.")
    documents = list(document_embeddings)
    document_scale = quantization_scale(documents, max_weight)
    seen_document_ids = set()
    feed_items = []

    for internal_id, (document_id, tokens, values) in enumerate(documents):
        document_id = str(document_id)
        if document_id in seen_document_ids:
            raise ValueError(f"Document IDs must be unique; found duplicate {document_id!r}.")
        seen_document_ids.add(document_id)
        embedding = merge_embedding(tokens, values)
        if not embedding:
            continue
        feed_items.append(
            (
                internal_id,
                document_id,
                quantize_embedding(embedding, document_scale, max_weight),
            )
        )

    with ThreadPoolExecutor(max_workers=feed_workers) as executor:
        list(executor.map(lambda item: client.feed(*item), feed_items))

    return VespaIndex(
        document_count=len(documents),
        indexed_document_count=len(feed_items),
        document_scale=document_scale,
        max_weight=max_weight,
    )


def retrieve(client, index, query_embeddings, k):
    if k < 1:
        raise ValueError("k must be at least 1.")
    if index.document_scale <= 0:
        raise ValueError("Vespa index metadata contains an invalid document scale.")

    depth = min(k, index.indexed_document_count)
    results = []
    for query_id, tokens, values in query_embeddings:
        embedding = merge_embedding(tokens, values)
        if not embedding or depth == 0:
            results.append((str(query_id), []))
            continue

        query_scale = index.max_weight / max(embedding.values())
        vector = quantize_embedding(embedding, query_scale, index.max_weight)
        response = client.query(vector, depth)
        ranking = []
        for hit in response.get("root", {}).get("children", []):
            document_id = hit.get("fields", {}).get("doc_id")
            raw_score = float(hit.get("relevance", 0))
            score = raw_score / (index.document_scale * query_scale)
            if document_id is None or score <= 0:
                continue
            ranking.append((str(document_id), score))
        results.append(
            (
                str(query_id),
                sorted(ranking, key=lambda result: result[1], reverse=True)[:depth],
            )
        )
    return results


def wait_for_http(url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2).close()
            return
        except urllib.error.HTTPError:
            return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    raise TimeoutError(f"Vespa endpoint {url!r} was not ready within {timeout} seconds.")


def read_server_log(log_path):
    try:
        return log_path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def is_tracker_permission_error(error):
    if not isinstance(error, PermissionError):
        return False
    if error.filename is not None:
        return Path(error.filename).name == ".tirex-tracker"
    return ".tirex-tracker" in str(error)


@contextmanager
def safe_tracking(**kwargs):
    tracker = tracking(**kwargs)
    entered = False
    try:
        tracker.__enter__()
        entered = True
    except PermissionError as error:
        if not is_tracker_permission_error(error):
            raise
    try:
        yield
    except BaseException as error:
        if entered:
            suppress_exception = tracker.__exit__(
                type(error),
                error,
                error.__traceback__,
            )
            if suppress_exception:
                return
        raise
    else:
        if entered:
            tracker.__exit__(None, None, None)


@contextmanager
def vespa_server(storage_path, startup_timeout=300):
    storage_path = Path(storage_path)
    log_path = storage_path / "vespa.log"
    environment = os.environ.copy()
    environment.update(
        {
            "VESPA_CONFIGSERVERS": vespa_hostname(),
            "VESPA_CONFIGSERVER_JVMARGS": "-Xms32M -Xmx128M",
            "VESPA_CONFIGPROXY_JVMARGS": "-Xms32M -Xmx32M",
            "VESPA_LOG_STDOUT": "false",
        }
    )
    with log_path.open("w") as server_log:
        process = subprocess.Popen(  # noqa: S603
            [VESPA_START],
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )

    try:
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Vespa exited during startup with code {exit_code}.\n"
                    f"{read_server_log(log_path)}"
                )
            try:
                urllib.request.urlopen(CONFIG_URL, timeout=2).close()
                break
            except (OSError, urllib.error.URLError):
                time.sleep(0.25)
        else:
            raise TimeoutError(
                f"Vespa did not start within {startup_timeout} seconds.\n"
                f"{read_server_log(log_path)}"
            )
        yield VespaClient()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@retrieve_command()
@click.option(
    "--max-weight",
    type=click.IntRange(min=1, max=MAX_VESPA_WEIGHT),
    default=10000,
    show_default=True,
    help="Maximum integer weight used for document and query quantization.",
)
@click.option(
    "--feed-workers",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Number of concurrent Vespa document feed requests.",
)
def main(dataset, embedding, output, k, max_weight, feed_workers):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    register_metadata(
        {
            "actor": {"team": "reneuir-baselines"},
            "tag": (
                f"vespa-{embedding.replace('/', '-')}-{max_weight}-"
                f"{feed_workers}-{k}"
            ),
        }
    )

    document_embeddings = load_embeddings(dataset, embedding, "doc")
    query_embeddings = load_embeddings(dataset, embedding, "query")

    with TemporaryDirectory() as temporary_directory:
        application_path = Path(temporary_directory) / "application"
        write_application_package(application_path)
        with vespa_server(temporary_directory) as client:
            client.deploy(application_path)
            with safe_tracking(
                export_file_path=output / "index-metadata.yml",
                export_format=ExportFormat.IR_METADATA,
            ):
                index = build_index(
                    client,
                    document_embeddings,
                    max_weight,
                    feed_workers,
                )

            with safe_tracking(
                export_file_path=output / "retrieval-metadata.yml",
                export_format=ExportFormat.IR_METADATA,
            ):
                results = retrieve(client, index, query_embeddings, k)

    with gzip.open(output / "run.txt.gz", "wt") as run_file:
        for query_id, ranking in results:
            for rank, (document_id, score) in enumerate(ranking, start=1):
                run_file.write(
                    f"{query_id} Q0 {document_id} {rank} {score} vespa\n"
                )


if __name__ == "__main__":
    main()
