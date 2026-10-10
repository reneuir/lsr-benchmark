import os
import subprocess
from pathlib import Path

import pytest
import yaml

# Requires Docker, network access, and TIRA; enabled in the embedd-integration-test workflow.
pytestmark = pytest.mark.skipif(
    not os.environ.get("LSR_EMBEDD_INTEGRATION_TEST"),
    reason="Set LSR_EMBEDD_INTEGRATION_TEST=1 to run the embedd integration test.",
)

DATASET = "tiny-example-20251002_0-training"
SYSTEM = "lsr-benchmark/lightning-ir/webis-splade"


def test_embedd_produces_valid_embeddings(tmp_path):
    out = tmp_path / "tmp"
    result = subprocess.run(
        ["lsr-benchmark", "embedd", SYSTEM, "-o", str(out), "--dataset", DATASET],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    embeddings_dir = out / DATASET / "webis-splade"
    for kind in ("doc", "query"):
        directory = embeddings_dir / kind
        assert (directory / f"{kind}-embeddings.npz").stat().st_size > 0
        ids = (directory / f"{kind}-ids.txt").read_text().split()
        assert len(ids) > 0
        metadata = yaml.safe_load((directory / f"{kind}-ir-metadata.yml").read_text())
        assert metadata["data"]["test collection"]["name"]
