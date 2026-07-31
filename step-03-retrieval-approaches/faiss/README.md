# FAISS Retrieval

Exact inner-product retrieval with [FAISS](https://github.com/facebookresearch/faiss).
Sparse learned-retrieval embeddings are converted to dense `float32` vectors and
indexed with `faiss.IndexFlatIP`.

## Development

Open this directory in a Dev Container, or build and run it directly:

```bash
docker build \
    -f .devcontainer/Dockerfile \
    -t lsr-benchmark-faiss-dev \
    .
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$PWD:/workspace" \
    -w /workspace \
    lsr-benchmark-faiss-dev \
    sh -c 'pytest -v && ruff check .'
```

## Usage

```bash
python faiss_retrieval.py \
    --dataset <dataset> \
    --embedding <embedding> \
    --output <output-dir> \
    --k 1000
```

Use `--batch-size` to control how many queries FAISS searches at once.

## Submission

```bash
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id reneuir-baselines \
    --dataset tiny-example-20251002_0-training \
    --command '/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir' \
    --mount-directory '$embeddings=lsr-benchmark/lightning-ir/naver-splade-v3-doc' \
    --dry-run
```
