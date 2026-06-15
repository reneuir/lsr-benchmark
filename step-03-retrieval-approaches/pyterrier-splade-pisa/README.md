# PyTerrier-Splade-Pisa Baseline for LSR

This approach does retrieval as [https://github.com/cmacdonald/pyt_splade](pyt_splade) would do with PISA, using pre-tokenized queries and embeddings.


## Submission

```
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id reneuir-baselines \
    --dataset tiny-example-20251002_0-training \
    --command '/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir' \
    --dry-run
```

## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt`.

To make predictions on a dataset, run:

```
./run-pyterrier-splade.py --dataset clueweb09/en/trec-web-2009 --output output-dir
```

