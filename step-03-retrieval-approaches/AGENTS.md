# Adding a Retrieval Approach

This guide describes the repository workflow for coding agents that add a
retrieval engine under `step-03-retrieval-approaches/`. Keep each approach
self-contained so that it can be developed locally, built as a container, and
submitted to TIRA independently.

## Expected Directory Layout

Create one directory named after the retrieval approach:

```text
step-03-retrieval-approaches/<approach>/
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile
├── Dockerfile
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── <approach>_retrieval.py
└── test_<approach>_retrieval.py
```

Use the [FAISS implementation](faiss/) as a compact reference. An approach may
need additional source files, system packages, or configuration, but the
development and runtime environments must remain reproducible.

## Agent Workflow

1. Inspect related retrieval implementations and the current Git status. Reuse
   existing loading, tracking, output, testing, and container patterns instead
   of introducing another interface.
2. Create the dedicated `.devcontainer/` first. Put all development tools and
   libraries in that container, then perform implementation and validation
   inside it.
3. Separate runtime dependencies in `requirements.txt` from development-only
   dependencies such as `pytest` and `ruff` in `requirements-dev.txt`.
4. Implement the retrieval command and focused unit tests.
5. Add the minimal runtime `Dockerfile`, ensuring that
   `/index-and-retrieve.py` exists and is executable.
6. Document the algorithm, development commands, usage, options, and TIRA
   submission command in the approach's `README.md`.
7. Add the approach to this directory's [README](README.md) and to
   [the retrieval integration workflow](../.github/workflows/retrieval-integration-tests.yml).
8. Build both containers and run tests, linting, and the runtime entrypoint
   from Docker before finishing.

Do not install approach dependencies on the host. Running Docker itself on the
host is expected; all Python commands and retrieval code should execute in the
development or runtime container.

## Retrieval Command Contract

Use `@retrieve_command()` from `lsr_benchmark.click`. It supplies the common
arguments:

```text
--dataset <dataset ID or directory>
--embedding <embedding ID or directory>
--output <output directory>
--k <retrieval depth>
```

Approach-specific options can be added with `click.option`. A minimal command
has this shape:

```python
import gzip

from tirex_tracker import ExportFormat, register_metadata, tracking

import lsr_benchmark
from lsr_benchmark.click import retrieve_command
from lsr_benchmark.irds import embeddings as load_embeddings


@retrieve_command()
def main(dataset, embedding, output, k):
    output.mkdir(parents=True, exist_ok=True)
    lsr_benchmark.register_to_ir_datasets(dataset)
    register_metadata({
        "actor": {"team": "reneuir-baselines"},
        "tag": f"<approach>-{embedding.replace('/', '-')}-{k}",
    })

    documents = load_embeddings(dataset, embedding, "doc")
    queries = load_embeddings(dataset, embedding, "query")

    with tracking(
        export_file_path=output / "index-metadata.yml",
        export_format=ExportFormat.IR_METADATA,
    ):
        index = build_index(documents)

    with tracking(
        export_file_path=output / "retrieval-metadata.yml",
        export_format=ExportFormat.IR_METADATA,
    ):
        results = search(index, queries, k)

    with gzip.open(output / "run.txt.gz", "wt") as run_file:
        for query_id, ranking in results:
            for rank, (doc_id, score) in enumerate(ranking, start=1):
                run_file.write(
                    f"{query_id} Q0 {doc_id} {rank} {score} <approach>\n"
                )


if __name__ == "__main__":
    main()
```

Adapt the data conversion and result structures to the retrieval library. Do
not change the input or output contract.

## Embedding and Scoring Requirements

`load_embeddings` returns `(item_id, tokens, values)` tuples. Tokens identify
sparse vector dimensions and values contain their weights.

- Preserve document and query IDs exactly.
- Use the same vector dimension for document and query representations.
- Use `float32` when required by the retrieval library.
- Learned sparse retrieval is scored with an inner product unless an approach
  explicitly documents another benchmark-compatible scoring method.
- Return at most `k` documents per query in descending score order.
- Omit invalid result indices and non-positive scores.
- Handle `k` values larger than the corpus without failing.
- Batch queries when the backend supports it and expose the batch size when it
  materially affects memory use.

## Tracking and Output Requirements

Index construction must be measured in `index-metadata.yml`, and query
retrieval must be measured separately in `retrieval-metadata.yml`. The ranking
must be written to `run.txt.gz` in TREC format:

```text
<query-id> Q0 <document-id> <rank> <score> <approach>
```

Ranks start at 1 for every query. The final column must consistently identify
the new approach.

## Tests

Keep algorithmic code in importable functions rather than only in the Click
command. Cover at least:

- conversion from benchmark embeddings to the backend representation;
- the best matching document and descending score order;
- retrieval depth, including `k` larger than the corpus;
- multiple queries;
- filtering invalid or non-positive results;
- invalid option values or inconsistent index metadata;
- generation of a valid compressed TREC run by the command callback.

Mock dataset loading and TiReX tracking in command-level tests. Unit tests
should not require network access or benchmark downloads.

## Container Requirements

The development container installs `requirements-dev.txt`. The runtime image
installs only `requirements.txt`, copies the command to
`/index-and-retrieve.py`, and makes it executable:

```dockerfile
FROM python:3.12-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY <approach>_retrieval.py /index-and-retrieve.py
RUN chmod +x /index-and-retrieve.py

ENTRYPOINT ["/index-and-retrieve.py"]
```

Choose a base image and Python version supported by all required libraries and
by both `linux/amd64` and `linux/arm64`, unless the integration test is
explicitly placed in the AMD64-only workflow.

## Validation Commands

Run the equivalent commands from the approach directory:

```bash
docker build \
    -f .devcontainer/Dockerfile \
    -t lsr-benchmark-<approach>-dev \
    .

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$PWD:/workspace" \
    -w /workspace \
    lsr-benchmark-<approach>-dev \
    sh -c 'pytest -v && ruff check .'

docker build -t lsr-benchmark-<approach> .

docker run --rm \
    --entrypoint /index-and-retrieve.py \
    lsr-benchmark-<approach> \
    --help

tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --dataset tiny-example-20251002_0-training \
    --command '/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir' \
    --mount-directory '$embeddings=lsr-benchmark/lightning-ir/naver-splade-v3-doc' \
    --platform host \
    --dry-run
```

Run the TIRA dry-run on the host. It builds the runtime image, downloads the
tiny example inputs, executes the retrieval command with mounted embeddings,
and validates its outputs. All direct Python development and tests remain
inside the development container.

Finally, run `git diff --check` and remove generated caches. Do not modify
unrelated files or overwrite pre-existing worktree changes.

## Integration Checklist

Before declaring the approach complete, confirm:

- the dedicated development container builds;
- tests and linting pass inside that container;
- the runtime image builds;
- `/index-and-retrieve.py --help` works in the runtime image;
- the host-side `tira-cli code-submission --dry-run` completes on the tiny
  example;
- the runtime image contains no development-only dependencies;
- the approach README contains a working dry-run submission command;
- the retrieval overview lists the approach;
- the integration workflow tests the approach on the tiny example dataset;
- generated caches and temporary files are absent from the change set.
