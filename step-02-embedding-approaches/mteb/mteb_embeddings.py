#!/usr/bin/env python3
from pathlib import Path
import click
import numpy as np
from tirex_tracker import register_metadata, tracking
import lsr_benchmark
from lsr_benchmark.click import option_lsr_dataset
from lsr_benchmark.datasets import TIRA_DATASET_ID_TO_IR_DATASET_ID
import ir_datasets

from mteb import get_model, get_task
from mteb._create_dataloaders import create_dataloader
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import PromptType
from datasets import Dataset

# Optional upper bound on input characters. None disables truncation so that
# each model truncates at its own (token-level) max_seq_length, exactly as
# MTEB/RTEB do. Set e.g. 512 to mimic the sentence-transformers engine.
TRUNCATE_LENGTH = None

# Team the embeddings are namespaced under (mirrors the "actor.team" the other
# engines set; used for the run tag and for tira lookup).
TEAM = "mteb"

# Generic retrieval instruction, used for datasets that have no MTEB/RTEB task
# (e.g. TREC-DL, Robust04, ClueWeb). Mirrors the default retrieval instruction
# of MTEB/E5-style instruct models.
GENERIC_RETRIEVAL_PROMPT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# ir-datasets path -> MTEB task name.
# Uses lsr_benchmark.datasets.TIRA_DATASET_ID_TO_IR_DATASET_ID to resolve
# lsr-benchmark dataset ids to ir-datasets paths first, so we only need to
# map the 4 RTEB datasets here rather than duplicating the full id table.
IR_DATASET_TO_MTEB_TASK = {
    "rteb/aila/statutes":       "AILAStatutes",
    "rteb/aila/casedocs":       "AILACasedocs",
    "rteb/legal-summarization": "LegalSummarization",
    "rteb/financebench":        "FinanceBenchRetrieval",
}


def convert_embeddings_dense(embeddings: np.ndarray):
    n_docs, n_dims = embeddings.shape
    row_idcs = np.repeat(np.arange(n_docs), n_dims)
    col_idcs = np.tile(np.arange(n_dims), n_docs)
    values = embeddings.flatten()
    row_indices = np.bincount(row_idcs + 1, minlength=n_docs + 1).cumsum()
    return values, col_idcs, row_indices


def truncate_texts(texts, truncate_length=TRUNCATE_LENGTH):
    if not truncate_length:
        return texts
    # a guess on the upper bound for transformer tokens is 10 characters per token.
    return [t[:(10 * truncate_length)] for t in texts]


def resolve_task_metadata(dataset: str, mteb_task: str = None) -> TaskMetadata:
    """The MTEB TaskMetadata that drives prompt selection.

    Precedence:
    1. explicit --mteb-task argument
    2. known RTEB datasets via lsr-benchmark's TIRA_DATASET_ID_TO_IR_DATASET_ID
       + IR_DATASET_TO_MTEB_TASK
    3. dataset name that happens to be a valid MTEB task name
    4. generic Retrieval task carrying the default retrieval instruction
    """
    if mteb_task:
        return get_task(mteb_task).metadata

    # resolve lsr-benchmark id -> ir-datasets path -> MTEB task name
    ir_id = TIRA_DATASET_ID_TO_IR_DATASET_ID.get(dataset, dataset)
    name = IR_DATASET_TO_MTEB_TASK.get(ir_id)
    if name:
        return get_task(name).metadata

    try:  # the dataset might itself be a valid MTEB task name
        return get_task(dataset).metadata
    except Exception:
        pass

    return TaskMetadata(
        dataset={"path": f"lsr-benchmark/{dataset}", "revision": "1"},
        name=f"lsr-benchmark/{dataset}",
        description="Generic lsr-benchmark retrieval task.",
        prompt=GENERIC_RETRIEVAL_PROMPT,
        type="Retrieval",
        modalities=["text"],
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
    )


def should_normalize(model, override=None) -> bool:
    """lsr-benchmark scores with a dot product. Normalize when the model
    declares cosine similarity (or none), so the dot product equals the model's
    intended score. Models that declare dot similarity are stored unnormalized.
    """
    if override is not None:
        return override
    sim = getattr(getattr(model, "mteb_model_meta", None), "similarity_fn_name", None)
    sim = getattr(sim, "value", sim)
    return sim is None or str(sim).lower() == "cosine"


def embedd_text_with_model(model, texts, ids, output, task_metadata,
                           prompt_type, batch_size=32, normalize=True,
                           truncate_length=TRUNCATE_LENGTH):
    output.parent.mkdir(parents=True, exist_ok=True)

    # not all models honour their tokenizer.model_max_length
    texts = truncate_texts(texts, truncate_length)

    dataloader = create_dataloader(
        Dataset.from_dict({"id": ids, "text": texts}),
        task_metadata=task_metadata,
        prompt_type=prompt_type,
        batch_size=batch_size,
    )

    with tracking(export_file_path=str(output).replace("-embeddings.npz", "-ir-metadata.yml")):
        embeddings = model.encode(
            dataloader,
            task_metadata=task_metadata,
            hf_split=task_metadata.eval_splits[0] if task_metadata.eval_splits else "test",
            # RTEB/MTEB retrieval tasks expose a single (default) subset; the
            # instruction is selected via task_metadata + prompt_type, not the subset.
            hf_subset="default",
            prompt_type=prompt_type,
            batch_size=batch_size,
            show_progress_bar=True,
        )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    data, indices, indptr = convert_embeddings_dense(embeddings)
    np.savez_compressed(output, data=data, indices=indices, indptr=indptr)
    Path(str(output).replace("-embeddings.npz", "-ids.txt")).write_text("\n".join(ids))


@click.command()
@option_lsr_dataset()
@click.option("--model", type=str, required=True, help="The model name (MTEB registry / HF).")
@click.option("--batch_size", type=int, default=32, help="Batch size.")
@click.option("--device", type=str, default=None, help="Device to load the model on (e.g. 'cuda').")
@click.option("--mteb-task", "mteb_task", type=str, default=None,
              help="Override the MTEB task name used for prompt selection (e.g. 'AILACasedocs').")
@click.option("--normalize/--no-normalize", "normalize", default=None,
              help="Force (no-)L2-normalization. Default: normalize iff the model's similarity is cosine.")
@click.option("--truncate-length", "truncate_length", type=int, default=None,
              help="Optional character truncation (tokens*10). Default: none (model handles it).")
def main(dataset: str, model: str, batch_size: int, device: str, mteb_task: str,
         normalize: bool, truncate_length: int, output: Path):
    lsr_benchmark.register_to_ir_datasets(dataset)
    module = get_model(model, device=device)
    register_metadata({"actor": {"team": TEAM}, "tag": model.replace('/', '-')})

    task_metadata = resolve_task_metadata(dataset, mteb_task)
    normalize = should_normalize(module, normalize)

    ir_dataset = ir_datasets.load(f"lsr-benchmark/{dataset}")

    queries = list(ir_dataset.queries_iter())
    embedd_text_with_model(
        module,
        [q.default_text() for q in queries],
        [q.query_id for q in queries],
        Path(output) / "query" / "query-embeddings.npz",
        task_metadata, PromptType.query, batch_size, normalize, truncate_length,
    )

    docs = list(ir_dataset.docs_iter())
    embedd_text_with_model(
        module,
        [d.default_text() for d in docs],
        [d.doc_id for d in docs],
        Path(output) / "doc" / "doc-embeddings.npz",
        task_metadata, PromptType.document, batch_size, normalize, truncate_length,
    )


if __name__ == "__main__":
    main()
