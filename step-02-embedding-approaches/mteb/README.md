# MTEB Embedding Approach

This approach uses [MTEB](https://github.com/embeddings-benchmark/mteb) to embed
queries and documents the same way as on the RTEB/MTEB leaderboard, i.e. with
the correct per-model (and per-dataset) instruction prompts. Queries are encoded
with the query prompt, documents with the document prompt.

## Usage

```bash
python mteb_embeddings.py \
    --dataset <path-to-dataset> \
    --model <model-name> \
    --batch_size 32 \
    --output <output-dir>
```

For the four RTEB datasets the matching MTEB task (and thus its dataset-specific
instruction) is selected automatically; use `--mteb-task <TaskName>` to override.

## Run Unit Tests

```bash
pip install pytest mteb
python -m pytest test_embeddings.py -v
```

## Submission

Code submission to tira via (remove the --dry-run for upload):

```
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id mteb \
    --dataset tiny-example-20251002_0-training \
    --command '/mteb_embeddings.py --dataset $inputDataset --output $outputDir --model intfloat/e5-mistral-7b-instruct' \
    --mount-hf-model intfloat/e5-mistral-7b-instruct \
    --dry-run
```
