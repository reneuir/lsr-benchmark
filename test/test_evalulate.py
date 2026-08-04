from unittest.mock import MagicMock, patch

import pytest
from ir_measures import ScoredDoc, parse_measure
from ir_datasets.formats import TrecQrel

from lsr_benchmark._commands import _evaluate as evaluator

MODULE = "lsr_benchmark._commands._evaluate"


@patch(f"{MODULE}.Path.is_dir", return_value=True)
@patch(f"{MODULE}.lines_if_valid")
@patch(f"{MODULE}.Path.is_file", return_value=True)
@patch(f"{MODULE}.Path.read_text", return_value="dummy_run_content")
@patch(f"{MODULE}.ir_measures.read_trec_run", return_value=["run_data"])
def test_metadata_parsing(mock_read_run, mock_read_text, mock_is_file, mock_lines, mock_is_dir):
    mock_lines.return_value = [
        {"name": "myapproach-doc-123", "content": "doc_meta"},
        {"name": "myapproach-query-123", "content": "query_meta"},
        {"name": "myapproach-other-123", "content": "standard_meta"},
    ]

    metadata, _ = evaluator.__read_metrics("dummy_dir")

    assert metadata["myapproach-doc"] == "doc_meta"
    assert metadata["myapproach-query"] == "query_meta"
    assert metadata["myapproach"] == "standard_meta"


@patch(f"{MODULE}.__read_metrics", return_value=({"group": {}}, [ScoredDoc("q1", "d1", 1.0)]))
@patch(f"{MODULE}.__get_dataset_name", return_value="dataset-1")
@patch(f"{MODULE}.__get_embedding_name", return_value="emb-1")
@patch(f"{MODULE}.lsr_benchmark")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_aggregated_evaluation(mock_lsr, mock_get_emb, mock_get_ds, mock_read):
    mock_lsr.load.return_value.has_qrels.return_value = True
    mock_lsr.load.return_value.qrels = [TrecQrel(query_id="q1", doc_id="d1", relevance=1, iteration=0)]

    measure = parse_measure("P@10")

    result = evaluator.evaluate_approach(
        "dummy", [("P_10", "ir_measure", measure)], per_query=False, exhaustive_truths=False
    )

    assert result["P@10"] == pytest.approx(0.1)
    assert result["tira-dataset-id"] == "dataset-1"
    assert "micro-averages" not in result
    assert "macro-averages" not in result


@patch(f"{MODULE}.__read_metrics", return_value=({"group": {}}, ["run"]))
@patch(f"{MODULE}.__get_dataset_name", return_value="joint_dataset")
@patch(f"{MODULE}.__get_embedding_name", return_value="emb-1")
@patch(f"{MODULE}.lsr_benchmark")
@patch(f"{MODULE}.ir_measures.calc")
@patch.dict(f"{MODULE}.JOINT_TO_DATASETS", {"joint_dataset": {"datasets": ["sub1", "sub2"]}}, clear=True)
def test_per_query_and_joint_evaluation(mock_calc, mock_lsr, mock_get_emb, mock_get_ds, mock_read):
    mock_lsr.load.return_value.has_qrels.return_value = True

    measure_mock = MagicMock()
    measure_mock.__str__.return_value = "P@10"

    class MockMetric:
        def __init__(self, val, qid):
            self.measure = measure_mock
            self.value = val
            self.query_id = qid

    mock_calc.side_effect = [
        MagicMock(aggregated={measure_mock: 0.2}, per_query=[MockMetric(0.2, "q1")]),
        MagicMock(
            aggregated={measure_mock: 0.8},
            per_query=[MockMetric(0.8, "q2"), MockMetric(0.8, "q3"), MockMetric(0.8, "q4")],
        ),
    ]

    result = evaluator.evaluate_approach(
        "dummy", [("P_10", "ir_measure", measure_mock)], per_query=True, exhaustive_truths=False
    )

    assert result["sub1"]["P@10"]["q1"] == 0.2
    assert result["sub2"]["P@10"]["q3"] == 0.8

    # Micro Average: (0.2 + 0.8 + 0.8 + 0.8) / 4 queries = 0.65
    assert result["micro-averages"]["P@10"] == pytest.approx(0.65)

    # Macro Average: (0.2 + 0.8) / 2 datasets = 0.5
    assert result["macro-averages"]["P@10"] == pytest.approx(0.5)


@patch(f"{MODULE}.__read_metrics", return_value=({"group": {}}, [ScoredDoc("q1", "d1", 1.0)]))
@patch(f"{MODULE}.__get_dataset_name", return_value="dataset-1")
@patch(f"{MODULE}.__get_embedding_name", return_value="emb-1")
@patch(f"{MODULE}.lsr_benchmark")
@patch(f"{MODULE}.retrieval")
@patch(f"{MODULE}.temporary_directory")
@patch(f"{MODULE}.gzip.open")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_exhaustive_truths_builds_qrels_from_run(
    mock_gzip_open, mock_tmp_dir, mock_retrieval, mock_lsr, mock_get_emb, mock_get_ds, mock_read
):
    mock_lsr.load.return_value.has_qrels.return_value = True
    mock_gzip_open.return_value.__enter__.return_value = ["q1 0 d1 1 1.0 run\n"]

    measure = parse_measure("P@10")

    result = evaluator.evaluate_approach(
        "dummy", [("P_10", "ir_measure", measure)], per_query=False, exhaustive_truths=True
    )

    mock_retrieval.assert_called_once()

    qrels = mock_lsr.load.return_value.qrels
    assert len(qrels) == 1
    assert qrels[0].query_id == "q1"
    assert qrels[0].doc_id == "d1"
    assert qrels[0].relevance == 1

    assert result["P@10"] == pytest.approx(0.1)
