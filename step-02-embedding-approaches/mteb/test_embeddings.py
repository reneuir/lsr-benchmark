import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import mteb
from mteb.types import PromptType

from mteb_embeddings import embedd_text_with_model, resolve_task_metadata, should_normalize


def load_dense(d, text_type):
    """Read the CSR npz + ids written by the engine and reconstruct the dense
    matrix. Columns are stored as 0..n_dims-1 per row, so a reshape suffices.

    We read the npz directly (instead of lsr_benchmark.irds.embeddings) so the
    tests stay independent of tirex_tracker's global file registry.
    """
    base = Path(d) / text_type
    npz = np.load(base / f"{text_type}-embeddings.npz")
    indptr = npz["indptr"]
    n_docs = len(indptr) - 1
    n_dims = int(indptr[1] - indptr[0])
    dense = npz["data"].reshape(n_docs, n_dims)
    ids = (base / f"{text_type}-ids.txt").read_text().strip().split("\n")
    return ids, dense


class TestEmbeddings(unittest.TestCase):
    """Structural tests on a small registered model (all-MiniLM-L6-v2,
    384 dims, cosine) so the test is cheap and runs offline once cached.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
        cls.task_metadata = resolve_task_metadata("some-unknown-dataset", None)
        cls.texts = ["brown fox", "green cat", "yellow rabbit"]
        cls.ids = ["id-1", "id-2", "id-3"]
        cls.n_dims = 384

    def _save(self, d, texts, ids, prompt_type=PromptType.document, text_type="doc"):
        output = Path(d) / text_type / f"{text_type}-embeddings.npz"
        embedd_text_with_model(
            self.model, texts, ids, output, self.task_metadata,
            prompt_type, batch_size=32, normalize=True,
        )
        return output

    def test_01_count_and_ids(self):
        with TemporaryDirectory() as d:
            self._save(d, self.texts, self.ids)
            ids, dense = load_dense(d, "doc")
            self.assertEqual(len(dense), len(self.texts))
            self.assertEqual(ids, self.ids)

    def test_02_unit_norm_when_cosine(self):
        with TemporaryDirectory() as d:
            self._save(d, self.texts, self.ids)
            _ids, dense = load_dense(d, "doc")
            for vec in dense:
                self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)

    def test_03_single_text(self):
        with TemporaryDirectory() as d:
            self._save(d, ["single text"], ["id-1"])
            _ids, dense = load_dense(d, "doc")
            self.assertEqual(len(dense), 1)

    def test_04_different_texts_differ(self):
        with TemporaryDirectory() as d:
            self._save(d, self.texts, self.ids)
            _ids, dense = load_dense(d, "doc")
            self.assertLess(float(np.dot(dense[0], dense[1])), 1.0)
            self.assertLess(float(np.dot(dense[0], dense[2])), 1.0)

    def test_05_query_path_runs(self):
        with TemporaryDirectory() as d:
            self._save(d, self.texts, self.ids, prompt_type=PromptType.query,
                       text_type="query")
            _ids, dense = load_dense(d, "query")
            self.assertEqual(len(dense), len(self.texts))

    def test_06_rteb_task_resolves(self):
        tm = resolve_task_metadata("aila-casedocs-20260426-training", None)
        self.assertEqual(tm.name, "AILACasedocs")

    def test_07_normalize_heuristic(self):
        self.assertTrue(should_normalize(self.model, None))
        self.assertFalse(should_normalize(self.model, False))


if __name__ == "__main__":
    unittest.main()
