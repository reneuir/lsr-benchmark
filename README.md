<!-- markdownlint-disable MD033 MD041 -->

<img width="100%" src="assets/banner.png" alt="The lsr-benchmark banner image">

<h1 align="center">lsr-benchmark</h1>

[![CI](https://img.shields.io/github/actions/workflow/status/reneuir/lsr-benchmark/ci.yml?branch=master&style=flat-square)](https://github.com/reneuir/lsr-benchmark/actions/workflows/ci.yml)
[![Maintenance](https://img.shields.io/maintenance/yes/2026?style=flat-square)](https://github.com/reneuir/lsr-benchmark/graphs/contributors)
[![Code coverage](https://img.shields.io/codecov/c/github/reneuir/lsr-benchmark?style=flat-square)](https://codecov.io/github/reneuir/lsr-benchmark/)
\
[![Release](https://img.shields.io/github/v/tag/reneuir/lsr-benchmark?style=flat-square&label=library)](https://github.com/reneuir/lsr-benchmark/releases/)
[![PyPi](https://img.shields.io/pypi/v/lsr-benchmark?style=flat-square)](https://pypi.org/project/lsr-benchmark/)
[![Downloads](https://img.shields.io/pypi/dm/lsr-benchmark?style=flat-square)](https://pypi.org/project/lsr-benchmark/)
[![Commit activity](https://img.shields.io/github/commit-activity/m/reneuir/lsr-benchmark?style=flat-square)](https://github.com/reneuir/lsr-benchmark/commits)

[CLI](#command-line-tool)&emsp;•&emsp;[Python API](#cc-api)&emsp;•&emsp;[Citation](#citation)

The lsr-benchmark aims to support holistic evaluations of the learned sparse retrieval paradigm to contrast efficiency and effectiveness across diverse retrieval scenarios. Please see the [corresponding paper](https://webis.de/publications.html?q=reneuir#froebe_2026a) for an overview of the methodology.

# Task

The learned sparse retrieval paradigm conducts retrieval in three steps:

1. Documents are segmented into passages so that the passages can be processed by pre-trained transformers.
2. Documents and queries are embedded into a sparse learned embedding.
3. Retrieval systems create an index of the document embeddings to return a ranking for each embedded query.

You can submit solutions to step 2 (i.e., models that embed documents and queries into sparse embeddings) and/or solutions to step 3 (i.e., retrieval systems). The idea is then to validate all combinations of embeddings with all retrieval systems to identify which solutions work well for which use case, taking different notions of efficiency/effectiveness trade-offs into consideration. The passage segmentation for step 1 is open source (i.e., created via `lsr-benchmark segment-corpus <IR-DATASETS-ID>`) but fixed for this task.


# Installation

You can install the lsr-benchmark via:

```shell
pip3 install lsr-benchmark
```

If you want the latest features, you can install from the main branch:

```shell
pip3 install git+https://github.com/reneuir/lsr-benchmark.git
```

# Supported Corpora and Embeddings

Please run `lsr-benchmark overview` for an up-to-date overview over all datasets and all embeddings. Alternatively, [online overview in TIRA](https://archive.tira.io/task-overview/lsr-benchmark/) provides an overview.

# Retrieval Suites

Predefined suites select the datasets, embeddings, and retrieval engines for a benchmark run:

```shell
lsr-benchmark retrieval --suite reneuir-2026/full --out my-reneuir-2026-results
```

Suites are maintained in [`lsr_benchmark/retrieval_suites.py`](lsr_benchmark/retrieval_suites.py). A suite cannot be combined with positional retrieval engines, `--dataset`, or `--embedding`.

# Running Tests

We have a suite of unit tests that you can run via:

```shell
# first install the local version of the lsr-benchmark
pip3 install -e .[dev,test]
# then run the unit tests
pytest .
```

# Documentation and Tutorials

We have a set of [tutorials available](tutorials).

The `lsr-benchmark --help` command serves as entrypoint to the documentation.

Instructions to add new datasets are available in the [data directory](data).
Instructions to add retrieval engines are available in
[`step-03-retrieval-approaches/AGENTS.md`](step-03-retrieval-approaches/AGENTS.md).

- ToDo: Write how to add new embeddings and evaluations
  - short video

# Data

The formats for data inputs and outputs aim to support slicing and dicing diverse query and document distributions while enabling caching, allowing for GreenIR research.

You can slice and dice the document texts and document embeddings via the API. The document texts for private corpora are only available within the [TIRA sandbox](https://docs.tira.io/participants/python-client.html) whereas the document embeddings are publicly available for all corpora (as one can not re-construct the original documents from sparse embeddings).

```python
dataset = lsr_benchmark.load('<IR-DATASETS-ID>')

# process the document embeddings:
for doc in dataset.docs_iter(embedding='<EMBEDDING-MODEL>', passage_aggregation="first-passage"):
    doc # namedtuple<doc_id, embedding>

# process the document embeddings for all segments:
for doc in dataset.docs_iter(embedding='<EMBEDDING-MODEL>'):
    doc # namedtuple<doc_id, segments.embedding>

# process the document texts:
for doc in dataset.docs_iter(embedding=None):
    doc # namedtuple<doc_id, segments.text>

# process the document texts via segmented versions in ir_datasets
lsr_benchmark.register_to_ir_datasets()
for segmented_doc in ir_datasets.load(f"lsr-benchmark/{dataset}/segmented")
    doc # namedtuple<doc_id, segment>
```

## Format of Document Texts

Inspired by the processing of [MS MARCO v2.1](https://trec-rag.github.io/annoucements/2024-corpus-finalization/), each document consists of a `doc_id` and a list of text `segments` that are short enough to be processed by pre-trained transformers. For instance, a document that consists of 4 passages (e.g., `"text-of-passage-1 text-of-passage-2 text-of-passage-3 text-of-passage-4"`) would be represented as:

- doc_id: 12fd3396-e4d7-4c0f-b468-5a82402b5336
- segments:
  - {"start": 1, "end": 2, "text": "text-of-passage-1 text-of-passage-2"}
  - {"start": 2, "end": 3, "text": "text-of-passage-2 text-of-passage-3"}
  - {"start": 3, "end": 4, "text": "text-of-passage-3 text-of-passage-4"}

## Format of Document Embeddings

Each document consists of a `doc_id` and a list of text `segments` that are short enough to be processed by pre-trained transformers. For instance, a document that consists of 4 passages would be represented as:

- doc_id: 12fd3396-e4d7-4c0f-b468-5a82402b5336
- segments:
  - {"start": 1, "end": 2, "embedding": {"term-1": 0.123, "term-2": 0.912}}
  - {"start": 2, "end": 3, "embedding": {"term-1": 0.421, "term-3": 0.743}}
  - {"start": 3, "end": 4, "embedding": {"term-2": 0.108, "term-4": 0.043}}

# Evaluation

The [online overview in TIRA](https://archive.tira.io/task-overview/lsr-benchmark/) provides an overview of aggregated evaluations. Alternatively, all data and further custom evaluations are available in the [step-04-evaluation](step-04-evaluation) directory of this repository.

Our evaluation methodology encourages the development of diverse and novel measures for lsr models that take efficiency and effectiveness into consideration. We assume that a suitable interpretation of efficiency for a target task highly depends on the application and its context. Therefore, we aim to measure as many efficiency-oriented aspects as possible in a standardized way with the [tirex-tracker](https://github.com/tira-io/tirex-tracker/) to ensure that different efficiency/effectiveness interpretations can be evaluated post-hoc. This methodology and related aspects were developed as part of the [ReNeuIR workshop series](https://reneuir.org/) held at SIGIR [2022](https://dl.acm.org/doi/abs/10.1145/3477495.3531704), [2023](https://dl.acm.org/doi/abs/10.1145/3539618.3591922), [2024](https://dl.acm.org/doi/abs/10.1145/3626772.3657994), and [2025](https://reneuir.org/).
