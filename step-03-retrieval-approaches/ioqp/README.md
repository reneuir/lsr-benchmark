# IOQP Retrieval

Learned sparse retrieval with
[IOQP](https://github.com/jmmackenzie/ioqp), an impact-ordered query processor
written in Rust.

The adapter converts benchmark document embeddings to CIFF, builds an IOQP
index, converts query weights to IOQP's weighted query format, and rewrites the
numeric internal query IDs back to the original benchmark IDs.
IOQP is licensed under the Apache License 2.0; its license is included in the
development and runtime images.

## Quantization

IOQP stores impacts and accumulated scores as unsigned 16-bit integers, while
the benchmark embeddings contain floating-point values. Document impacts and
query weights are therefore quantized before indexing and retrieval.

The adapter automatically lowers the configured quantization levels when
necessary so that the worst-case query score fits into IOQP's 16-bit
accumulator. The upper bounds can be configured with:

```text
--max-document-impact 255
--max-query-weight 32
```

IOQP supports exhaustive processing with `--rho 1`, approximate fractional
processing with a lower `--rho`, or a fixed processing budget with
`--postings-budget`.

## Development

The dedicated development container contains the pinned IOQP `create` and
`query` binaries as `ioqp-create` and `ioqp-query`.

```bash
docker build \
    -f .devcontainer/Dockerfile \
    -t lsr-benchmark-ioqp-dev \
    .

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$PWD:/workspace" \
    -w /workspace \
    lsr-benchmark-ioqp-dev \
    sh -c 'pytest -v && ruff check .'
```

The upstream IOQP query processor assumes an initial block of at least 128
documents. [`ioqp-small-corpora.patch`](ioqp-small-corpora.patch) bounds that
block by the corpus size so unit tests and the benchmark's tiny example can be
processed.

## Usage

```bash
python ioqp_retrieval.py \
    --dataset <dataset> \
    --embedding <embedding> \
    --output <output-dir> \
    --k 1000 \
    --rho 1
```

## Architecture Support

This implementation is AMD64-only because the upstream IOQP crate enables
x86-specific nightly Rust features.

## Submission

```bash
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id reneuir-baselines \
    --dataset tiny-example-20251002_0-training \
    --command '/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir' \
    --mount-directory '$embeddings=lsr-benchmark/lightning-ir/naver-splade-v3-doc' \
    --platform host \
    --dry-run
```
