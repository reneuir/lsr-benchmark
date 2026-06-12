# Numpy Exhaustive Retrieval

Exhaustive nearest-neighbor retrieval using cosine similarity implemented with numpy only, without additional dependencies.

## Submission

```
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id reneuir-baselines \
    --dataset tiny-example-20251002_0-training \
    --command '/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir' \
    --mount-directory '$embeddings=lsr-benchmark/lightning-ir/naver-splade-v3-doc' \
    --dry-run
```

## Usage

```bash
python numpy_exhaustive_search.py \
    --dataset <path-to-dataset> \
    --embedding <path-to-embeddings> \
    --output <output-dir> \
    --k 1000
```

## Run Unit Tests

```bash
pip install pytest numpy
python -m pytest test_retrieval.py -v
```
