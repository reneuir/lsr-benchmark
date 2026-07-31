# Retrieval Engines in the lsr_benchmark

This directory contains the retrieval engines that we currently have in the lsr_benchmark. We aim to organize the lsr_benchmark as mono-repo that is fully self contained with simple and clean implementations, for that reason, if you want to contribute new retrieval engines (we would be very happy about that), please make a pull request.

Instructions for contributors and coding agents are available in
[Adding a Retrieval Approach](AGENTS.md).

Currently, we have 11 retrieval engines that can run lsr retrieval:

- [duckdb](duckdb)
- [faiss](faiss)
- [ioqp](ioqp)
- [kannolo](kannolo)
- [naive-search](naive-search)
- [pyserini-lsr](pyserini-lsr)
- [pyterrier-splade](pyterrier-splade)
- [pyterrier-splade-pisa](pyterrier-splade-pisa)
- [pytorch-naive](pytorch-naive)
- [seismic](seismic)
- [sqlite](sqlite)

Additionally, we have two lexical retrieval engines as baselines:

- [lexical/pyterrier-naive](lexical/pyterrier-naive)
- [lexical/pyterrier-pisa](lexical/pyterrier-pisa)

## Running all Retrieval Engines

The following code snippet runs all lsr retrieval engines on all embeddings and all datasets and stores the outputs in a directory `../runs`:

```
lsr-benchmark retrieval -o ../runs duckdb faiss ioqp kannolo naive-search pyterrier-splade pyterrier-splade-pisa seismic sqlite pytorch-naive pyserini-lsr
```

The following snippet runs all lexical retrieval engines on all datasets and stores the outputs in a directory `../runs`:

```
lsr-benchmark retrieval -o ../runs pyterrier-naive/ pyterrier-pisa/ --embedding none
```

## Adding a Retrieval Engine to CI

Every new retrieval engine must have a TIRA dry-run in the CI runner matrix.
Add a step to
[`.github/workflows/retrieval-integration-tests.yml`](../.github/workflows/retrieval-integration-tests.yml)
when its container supports both `linux/amd64` and `linux/arm64`:

```yaml
- name: Test <approach>
  run: |
    tira-cli code-submission \
        --path step-03-retrieval-approaches/<approach>/ \
        --task lsr-benchmark \
        --dataset tiny-example-20251002_0-training \
        --command '/index-and-retrieve.py --dataset $inputDataset --embedding $embeddings --output $outputDir' \
        --mount-directory '$embeddings=lsr-benchmark/lightning-ir/naver-splade-v3-doc' \
        --platform host \
        --dry-run
```

This workflow executes the step on both `ubuntu-latest` and
`ubuntu-24.04-arm`. If a dependency is only available for AMD64, add the same
step to
[`.github/workflows/retrieval-integration-tests-amd64-only.yml`](../.github/workflows/retrieval-integration-tests-amd64-only.yml)
instead and document the architecture restriction in the approach README.
Do not add an AMD64-only approach to the multi-architecture workflow.

The dry-run must build the approach's runtime `Dockerfile`, execute
`/index-and-retrieve.py` on `tiny-example-20251002_0-training`, mount the
standard SPLADE embeddings, and produce a valid `run.txt` or `run.txt.gz`.

## Remaining Retrieval Engines

We are in the progress of adding the following remaining retrieval engines:

- [ ] anserini: Carlos
- [ ] naive with dictionaries or with rust: Cosimo
- [ ] opensearch (Maybe a testcontainer as starting point?): Carlos
- [ ] opensearch seismic (would be interesting to compare the plain seismic with a "production ready" variant"): Carlos
