#!/usr/bin/env python3
"""Run independent N/R/P diagnostics, with one baseline per fresh process.

This runner does not configure/build or establish performance thresholds. Source
verification/environments.sh first when the executable needs toolchain libraries.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys

MODES = ("native", "composite", "materialized", "flattened")


def integers(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("use comma-separated positive integers")
    return list(dict.fromkeys(values))


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new results directory")
    parser.add_argument("--cells", type=integers, default=[8, 16, 32], help="axis-cell N sweep")
    parser.add_argument("--regions", type=integers, default=[1, 2, 4, 8, 16])
    parser.add_argument("--ranks", type=integers, default=[1, 2, 4])
    parser.add_argument("--fixed-cells", type=int, default=16)
    parser.add_argument("--fixed-regions", type=int, default=2)
    parser.add_argument("--fixed-ranks", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3, help="fresh processes per case")
    parser.add_argument("--repeats", type=int, default=5, help="in-process traversal/application repeats")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--mpi-args", default="", help="shell-quoted launcher flags, e.g. '--bind-to core'")
    parser.add_argument("--timeout", type=int, default=300, help="seconds per fresh process")
    parser.add_argument("--allow-non-release", action="store_true", help="smoke checks only")
    return parser.parse_args()


def summarize(output: Path, rows: list[dict], unsupported: list[dict]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with (output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in ("axis_cells", "regions_requested", "ranks", "mode", "stage"))
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, samples in sorted(grouped.items()):
        entry = dict(zip(("axis_cells", "regions", "ranks", "mode", "stage"), key))
        entry["samples"] = len(samples)
        for metric in ("seconds_max_rank", "peak_rss_bytes_max_rank", "measured_mesh_bytes_max_rank",
                       "compatibility_bytes_max_rank", "ghost_owned_ratio_max_rank"):
            values = [sample[metric] for sample in samples]
            for name, function in (("median", statistics.median), ("min", min), ("max", max)):
                entry[f"{metric}_{name}"] = function(values)
        summaries.append(entry)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    text = ["# Region scaling diagnostics", "",
            "Each timing is the median of maximum-rank elapsed times. N is the physical cell count; "
            "R is the requested x-slab count. Native and flattened represent one region. "
            "Peak RSS is the largest process high-water mark, including runtime and earlier phases; "
            "it is not a simultaneous sum or a mesh-only allocation.", "",
            "| N | R | P | Mode | Samples | Construction ms | Traversal ms | Assembly ms | Apply ms | Solve ms | Output ms | Peak MiB | Mesh KiB | Ghost/owned max |",
            "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    cases = sorted({key[:4] for key in grouped})
    lookup = {(entry["axis_cells"], entry["regions"], entry["ranks"], entry["mode"], entry["stage"]): entry
              for entry in summaries}
    phases = ("construction", "traversal_average", "assembly", "application_average",
              "solve_including_preconditioner", "output")
    for case in cases:
        if any((*case, phase) not in lookup for phase in phases):
            continue
        values = [lookup[(*case, phase)] for phase in phases]
        n, regions, ranks, mode = case
        timings = " | ".join(f"{value['seconds_max_rank_median'] * 1000:.4f}" for value in values)
        final = values[-1]
        text.append(f"| {n**3} | {regions} | {ranks} | {mode} | {final['samples']} | {timings} | "
                    f"{final['peak_rss_bytes_max_rank_median'] / 2**20:.3f} | "
                    f"{final['measured_mesh_bytes_max_rank_median'] / 1024:.3f} | "
                    f"{final['ghost_owned_ratio_max_rank_median']:.4f} |")
    text += ["", f"Unsupported cases: {len(unsupported)} (flattened MPI requires an explicitly partitioned reference).",
             "", "No speedup threshold is applied. See measurements.csv for view acquisition, structural storage "
             "components, sums, checksums and solver residuals; summary.csv includes sample ranges. "
             "output timings cover native geometry VTU generation/writing, without fsync.", ""]
    (output / "report.md").write_text("\n".join(text))


def main() -> int:
    args = arguments()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise ValueError(f"missing executable: {binary}")
    if min(args.fixed_cells, args.fixed_regions, args.fixed_ranks, args.samples, args.repeats, args.timeout) < 1:
        raise ValueError("fixed dimensions, samples, repeats and timeout must be positive")
    cases: dict[tuple[int, int, int], set[str]] = {}
    for dimension, values in (("N", args.cells), ("R", args.regions), ("P", args.ranks)):
        for value in values:
            case = (value if dimension == "N" else args.fixed_cells,
                    value if dimension == "R" else args.fixed_regions,
                    value if dimension == "P" else args.fixed_ranks)
            n, regions, ranks = case
            if n < 2 or n > 512 or regions > n or n % regions:
                raise ValueError(f"invalid N/R/P case {case}: require 2 <= n <= 512 and R divides n")
            cases.setdefault(case, set()).add(dimension)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "host": platform.node(),
        "platform": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "binary": str(binary), "binary_sha256": digest(binary), "source_commit": git(root, "rev-parse", "HEAD"),
        "source_status": git(root, "status", "--porcelain"), "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {key: value for key, value in os.environ.items() if key in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "KOKKOS_NUM_THREADS") or key.startswith("OMPI_MCA_")},
        "cases": [], "unsupported": [],
    }
    manifest["project_shared_libraries"] = {}
    library_dirs = (binary.parent.parent / "lib", binary.parent.parent.parent / "lib" / binary.parent.name)
    for directory in library_dirs:
        if directory.is_dir():
            for library in sorted(directory.glob("*SimpleFluid*")):
                if library.is_file() and any(suffix in library.name for suffix in (".so", ".dylib", ".dll")):
                    manifest["project_shared_libraries"][str(library.resolve())] = digest(library)
    # Digests distinguish the executable and project libraries when sources are dirty.
    (output / "source.diff").write_text(git(root, "diff", "HEAD", "--binary") + "\n")
    manifest["new_source_files"] = {}
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
    for name in untracked:
        relative = Path(name)
        if (relative.parts and relative.parts[0] in ("src", "cmake", "verification")
                and relative.suffix in (".cc", ".hh", ".hpp", ".tcc", ".ipp", ".cmake", ".py", ".sh")):
            target = output / "source-new" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / relative).read_bytes())
            manifest["new_source_files"][name] = digest(target)
    try:
        manifest["mpi_launcher_version"] = subprocess.check_output(
            [args.mpiexec, "--version"], text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        manifest["mpi_launcher_version"] = f"unavailable: {error}"
    rows: list[dict] = []
    try:
        for (n, regions, ranks), dimensions in cases.items():
            for mode in args.modes:
                if mode == "flattened" and ranks != 1:
                    manifest["unsupported"].append({"axis_cells": n, "regions": regions, "ranks": ranks,
                                                    "mode": mode, "reason": "explicit unstructured MPI requires a separately partitioned reference"})
                    continue
                for sample in range(args.samples):
                    name = f"n{n}_r{regions}_p{ranks}_{mode}_s{sample}"
                    folder = output / name
                    folder.mkdir()
                    command = [str(binary), mode, str(n), str(regions), str(args.repeats), str(folder / "mesh.vtu")]
                    if ranks > 1:
                        command = [args.mpiexec, *shlex.split(args.mpi_args), "-n", str(ranks), *command]
                    record = {"name": name, "command": command, "dimensions": sorted(dimensions)}
                    manifest["cases"].append(record)
                    print(f"[{len(manifest['cases'])}] {name}", flush=True)
                    with (folder / "stdout.jsonl").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
                        process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=args.timeout, check=False)
                    record["returncode"] = process.returncode
                    if process.returncode:
                        raise RuntimeError(f"{name} failed ({process.returncode}); see {folder / 'stderr.log'}")
                    events = [json.loads(line) for line in (folder / "stdout.jsonl").read_text().splitlines() if line.startswith("{")]
                    metadata = next(event for event in events if event["type"] == "metadata")
                    if metadata["build"] != "Release" and not args.allow_non_release:
                        raise RuntimeError("performance qualification requires a Release executable; --allow-non-release is for smoke checks")
                    result = next(event for event in events if event["type"] == "result")
                    if not result["converged"]:
                        raise RuntimeError(f"{name}: unconverged result")
                    for event in events:
                        if event["type"] == "checkpoint":
                            row = {**metadata, **result, **event, "sample": sample, "dimensions": ",".join(sorted(dimensions))}
                            del row["type"]
                            rows.append(row)
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        summarize(output, rows, manifest["unsupported"])
    print(f"Report: {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        sys.exit(str(error))
