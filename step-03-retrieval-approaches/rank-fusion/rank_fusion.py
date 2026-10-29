#!/usr/bin/env python3
import gzip
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import yaml
from ranx import Run, fuse
from tirex_tracker import ExportFormat, tracking

RESOURCE_KEYS = (
    ("cpu", "energy used system"),
    ("gpu", "energy used system"),
    ("ram", "energy used system"),
    ("runtime", "wallclock"),
)


def write_joint_metadata(metadata: list[dict], tag: str, out_path: Path) -> None:
    if not metadata:
        return

    combined_metadata = {
        "actor": metadata[0]["actor"],
        "data": metadata[0]["data"],
        "resources": {resource: {} for resource, _ in RESOURCE_KEYS},
        "tag": tag,
    }

    for resource, key in RESOURCE_KEYS:
        total = 0.0
        unit = None
        all_null = True

        for item in metadata:
            value, current_unit = str(item["resources"][resource][key]).split(maxsplit=1)
            if unit is None:
                unit = current_unit
            elif current_unit != unit:
                raise ValueError(f"Mismatching units for resources.{resource}.{key}: {unit!r} and {current_unit!r}.")

            if value != "null":
                total += float(value)
                all_null = False

        combined_value = "null" if all_null else f"{total:g}"
        combined_metadata["resources"][resource][key] = f"{combined_value} {unit}"

    with out_path.open("w") as f:
        yaml.safe_dump(combined_metadata, f, sort_keys=False)


def write_embedding_metadata(run_paths: list[Path], target_dir: Path) -> None:
    target_dir.mkdir()
    next_index = {"doc": 0, "query": 0}
    for run_path in run_paths:
        for path in sorted((run_path / ".embedding").iterdir(), key=lambda p: (len(p.name), p.name)):
            meta_type = "doc" if path.name.endswith("doc-ir-metadata.yml") else "query"
            shutil.copy(path, target_dir / f"d{next_index[meta_type]}-{meta_type}-ir-metadata.yml")
            next_index[meta_type] += 1


def warm_up_ranx(runs: list[Run], norm: str, method: str) -> None:
    """Fuse tiny runs with the same numba types as `runs`, so ranx's numba
    compilation happens before the measured fusion.

    numba compiles one version per argument type, and ranx stores each run's
    doc ids as fixed-length strings sized to that run's longest doc id. The
    tiny runs therefore use doc ids of exactly that length.
    """
    tiny_runs = []
    for run in runs:
        doc_id_length = run.run._numba_type_.value_type.key_type.count
        tiny_runs.append(Run({"q": {"a" * doc_id_length: 1.0, "b" * doc_id_length: 0.5}}))
    fuse(runs=tiny_runs, norm=norm, method=method)


@click.command()
@click.argument("run-paths", type=click.Path(exists=True, resolve_path=True, path_type=Path), nargs=-1, required=True)
@click.option(
    "-n",
    "--norm",
    type=click.Choice(["min-max", "min-max-inverted", "max", "sum", "zmuv", "rank", "borda"]),
    default="min-max",
)
@click.option(
    "-m",
    "--method",
    type=click.Choice(["min", "med", "anz", "log_isr", "bordafuse", "condorcet", "max", "sum", "mnz", "lsr"]),
    default="min",
)
@click.option("--output", type=Path, required=True)
def main(run_paths, norm, method, output):
    metadata = ([], [])
    for run_path in run_paths:
        for i, meta_type in enumerate(["index", "retrieval"]):
            meta_path = run_path / f"{meta_type}-metadata.yml"
            if meta_path.exists():
                metadata[i].append(yaml.safe_load(meta_path.read_text()))
    output.mkdir(parents=True, exist_ok=True)
    tag = f"rank-fusion-{norm}_norm-{method}_method"

    runs = [Run.from_file(path / "run.txt.gz") for path in run_paths]
    warm_up_ranx(runs, norm, method)
    retrieval_metadata_path = output / "retrieval-metadata.yml"
    with tracking(
        export_file_path=retrieval_metadata_path,
        export_format=ExportFormat.IR_METADATA,
    ):
        fused = fuse(runs=runs, norm="min-max", method="min")
    with open(retrieval_metadata_path) as f:
        metadata[1].append(yaml.safe_load(f))

    with TemporaryDirectory() as tmp:
        run_txt = Path(tmp) / "run.txt"
        fused.save(run_txt)
        with open(run_txt, "rb") as f_in, gzip.open(output / "run.txt.gz", "wb") as f_out:
            f_out.writelines(f_in)

        embedding_dir = Path(tmp) / ".embedding"
        write_embedding_metadata(run_paths, embedding_dir)
        shutil.copytree(embedding_dir, output / ".embedding", dirs_exist_ok=True)

    write_joint_metadata(metadata[0], tag, output / "index-metadata.yml")
    write_joint_metadata(metadata[1], tag, retrieval_metadata_path)


if __name__ == "__main__":
    main()
