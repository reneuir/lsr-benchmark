import os
import zipfile
from glob import glob
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import BaseDocs, BaseQueries, GenericQuery, TrecQrels
from ir_datasets.util import MetadataComponent
from tira.check_format import JsonlFormat, QueryProcessorFormat
from tira.third_party_integrations import in_tira_sandbox
from tqdm import tqdm

from lsr_benchmark.datasets import TIRA_DATASET_ID_TO_IR_DATASET_ID


TIRA_LSR_TASK_ID = "lsr-benchmark"


_IR_DATASETS_FROM_TIRA = None
_TIRA_CLIENT = None

def embeddings(
        dataset_id: str, model_name: str, text_type: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:

    if Path(model_name).is_dir() and (Path(model_name) / text_type).is_dir() and (Path(model_name) / text_type / f"{text_type}-embeddings.npz").exists():
        embedding_dir = Path(model_name) / text_type
    else:
        global _TIRA_CLIENT
        if _TIRA_CLIENT is None:
            from tira.rest_api_client import Client
            _TIRA_CLIENT = Client()
        tira = _TIRA_CLIENT

        team_and_model = model_name.split('/')
        team_name = team_and_model[0]
        model_name = '-'.join(team_and_model[1:])
        if in_tira_sandbox():
             embedding_dir = tira.input_run_in_sandbox(f"{TIRA_LSR_TASK_ID}/{team_name}/{model_name}")
             if not embedding_dir:
                 raise ValueError("not mounted")
             embedding_dir = Path(embedding_dir) / text_type
        else:
            embedding_dir = tira.get_run_output(f"{TIRA_LSR_TASK_ID}/{team_name}/{model_name}", dataset_id) / text_type

    embeddings = np.load(embedding_dir / f"{text_type}-embeddings.npz")

    try:
        from tirex_tracker import register_file
        for i in glob(f"{embedding_dir}/*.yml") + glob(f"{embedding_dir}/*.yaml"):
            register_file(embedding_dir, i.split("/")[-1], subdirectory=".embedding")
    except Exception as e:
        print(f"Warning: Could not register file due to: {e}")

    ids = (embedding_dir / f"{text_type}-ids.txt").read_text().strip().split("\n")

    ret = []
    indices = embeddings["indices"].astype("U30")
    data = embeddings["data"]

    ptr_start = 0
    for doc_id, ptr_end in tqdm(list(zip(ids, embeddings["indptr"][1:])), "load doc embeddings"):
        tokens = indices[ptr_start:ptr_end]
        values = data[ptr_start:ptr_end]
        ptr_start = ptr_end
        ret.append((doc_id, tokens, values))
    return ret


def ir_datasets_from_tira(force_reload=False):
    global _IR_DATASETS_FROM_TIRA
    
    if in_tira_sandbox():
        return []
    
    if _IR_DATASETS_FROM_TIRA is None or force_reload:
        global _TIRA_CLIENT
        if _TIRA_CLIENT is None:
            from tira.rest_api_client import Client
            _TIRA_CLIENT = Client()
        tira = _TIRA_CLIENT
        _IR_DATASETS_FROM_TIRA = list(tira.datasets(TIRA_LSR_TASK_ID, force_reload).keys())

    return _IR_DATASETS_FROM_TIRA


def _dowload_from_tira(ir_datasets_id, truth_dataset):
    if os.path.isdir(ir_datasets_id):
        return Path(ir_datasets_id)

    global _TIRA_CLIENT
    if _TIRA_CLIENT is None:
        from tira.rest_api_client import Client
        _TIRA_CLIENT = Client()
    tira = _TIRA_CLIENT

    if '-train' not in ir_datasets_id and not in_tira_sandbox() and not tira.api_key_is_valid():
        raise ValueError(f"The dataset {ir_datasets_id} is private, you can not access the raw data.")
    return tira.download_dataset(task=None, dataset=ir_datasets_id, truth_dataset=truth_dataset)


class Segment(NamedTuple):
    offset_start: int
    offset_end: int
    text: str


class LsrBenchmarkDocument(NamedTuple):
    doc_id: str
    segments: List[Segment]
    text: str

    @staticmethod
    def _from_json(json_doc):
        segments = [Segment(int(i["start"]), int(i["end"]), i["text"]) for i in json_doc["segments"]]
        return LsrBenchmarkDocument(json_doc["doc_id"], segments, json_doc["default_text"])

    def default_text(self):
        return self.text


class LsrBenchmarkDocumentEmbedding(NamedTuple):
    doc_id: str
    embedding: np.array


class LsrBenchmarkSegmentedDocument(NamedTuple):
    doc_id: str
    segment: Segment

    def default_text(self):
        return self.segment.text



class LsrBenchmarkQueries(BaseQueries):
    def __init__(self, ir_datasets_id):
        self.__irds_id = ir_datasets_id
        self.__queries_file = None

    def queries_iter(self):
        if not self.__queries_file:
            self.__queries_file = _dowload_from_tira(self.__irds_id, False) / "queries.jsonl"

        for q in QueryProcessorFormat().all_lines(self.__queries_file):
            yield GenericQuery(q["qid"], q["query"])


class LsrBenchmarkQueryEmbedding(NamedTuple):
    query_id: str
    embedding: np.array


class LsrBenchmarkDocuments(BaseDocs):
    def __init__(self, irds_id):
        self.__corpus_file = None
        self.__irds_id = irds_id

    def docs_iter(self):
        for doc in self.docs():
            yield LsrBenchmarkDocument._from_json(doc)

    def docs(self):
        if not self.__corpus_file:
            self.__corpus_file = _dowload_from_tira(self.__irds_id, False) / "corpus.jsonl.gz"

        reader = JsonlFormat()
        reader.apply_configuration_and_throw_if_invalid(
            {"required_fields": ["doc_id", "segments"], "max_size_mb": 2500}
        )

        yield from reader.yield_next_entry(self.__corpus_file)

    def docs_count(self):
        return None


class LsrBenchmarkSegmentedDocuments(LsrBenchmarkDocuments):
    def docs_iter(self, embedding=None):
        for doc in super().docs_iter(embedding):
            for idx, segment in zip(range(len(doc.segments)), doc.segments):
                yield LsrBenchmarkSegmentedDocument(f"{doc.doc_id}___{idx}___", segment)


class LsrBenchmarkDataset(Dataset):
    def __init__(self, ir_datasets_id, segmented=False, documentation=None):
        self.__irds_id = ir_datasets_id

        queries = LsrBenchmarkQueries(ir_datasets_id)

        if in_tira_sandbox():
            qrels_obj = None
        
        else:
            class QrelsObj:
                def stream(self):
                    qrels_file = _dowload_from_tira(ir_datasets_id, True) / "qrels.txt"
                    return qrels_file.open("rb")

            class TmpTrecQrels(TrecQrels):
                def qrels_iter(self):
                    try:
                        _dowload_from_tira(ir_datasets_id, True)    
                    except Exception:
                        # Dataser could no be found on TIRA, fallback to ir_datasets
                        import ir_datasets
                        if ir_datasets_id in TIRA_DATASET_ID_TO_IR_DATASET_ID:
                            ds = ir_datasets.load(TIRA_DATASET_ID_TO_IR_DATASET_ID[ir_datasets_id])
                        else:
                            ds = ir_datasets.load(ir_datasets_id)
                        yield from ds.qrels_iter()
                    else:
                        yield from super().qrels_iter()

            qrels_obj = TmpTrecQrels(QrelsObj(), {0: "Not Relevant", 1: "Relevant"})

        if segmented:
            docs = LsrBenchmarkSegmentedDocuments(ir_datasets_id)
        else:
            docs = LsrBenchmarkDocuments(ir_datasets_id)

        super().__init__(docs, queries, qrels_obj, documentation)
        self.metadata = MetadataComponent(ir_datasets_id, self)

    def query_embeddings(self, model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        return embeddings(self.__irds_id, model_name, "query")

    def doc_embeddings(self, model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        return embeddings(self.__irds_id, model_name, "doc")

    def dataset_id(self):
        return "lsr-benchmark/" + TIRA_DATASET_ID_TO_IR_DATASET_ID.get(self.__irds_id, self.__irds_id)


def extract_zip(zip_file: Path, target_directory: Path):
    if target_directory.exists():
        return

    if not zipfile.is_zipfile(zip_file):
        raise ValueError(f"I expected that {zip_file} is not a valid ZIP archive.")

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        target_directory.mkdir(parents=True, exist_ok=True)
        zip_ref.extractall(target_directory)


def build_dataset(ir_datasets_id: str, segmented: bool):
    try:
        from tirex_tracker import register_metadata
        register_metadata({"data": {"test collection": {"name": ir_datasets_id}}})
    except Exception as e:
        print(f"Warning: Could not register metadata due to: {e}")

    return LsrBenchmarkDataset(
        ir_datasets_id=ir_datasets_id,
        segmented=segmented,
    )
