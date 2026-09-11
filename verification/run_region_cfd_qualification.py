#!/usr/bin/env python3
"""Repeat the reported two-rank SST/bubbly-convection windows with unchanged gates.

The reference directory is an existing region-operator report with retained
inputs, full CFD measurements, and merged native/IsoRegion/OpenFOAM outputs.
Only the candidate is executed. Reference timings are explicitly retained
observations, not newly measured paired timings. Output directories must be new.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification/openfoam"))
from compare_verification import ComparisonError, compare, load_manifest
from plot_water_fields import load_fields
from run_performance import check_artifacts, merge_outputs, shared_library_hashes


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_changes(reference: Path, actual: Path) -> dict:
    with reference.open(newline="") as stream:
        reader = csv.DictReader(stream)
        names, left = reader.fieldnames, list(reader)
    with actual.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if names != reader.fieldnames:
            raise ComparisonError(f"CSV columns changed: {actual}")
        right = list(reader)
    if len(left) != len(right):
        raise ComparisonError(f"CSV row count changed: {actual}")
    maxima = {name: 0.0 for name in names}
    for a, b in zip(left, right):
        for name in names:
            try:
                delta = abs(float(a[name]) - float(b[name]))
            except ValueError:
                if a[name] != b[name]:
                    raise ComparisonError(f"CSV labels changed: {actual}, {name}")
                continue
            if not math.isfinite(delta):
                raise ComparisonError(f"Nonfinite comparison: {actual}, {name}")
            maxima[name] = max(maxima[name], delta)
    return {"exact_numeric_match": all(value == 0 for value in maxima.values()),
            "max_absolute_changes": maxima}


def summarize(output: Path, records: list[dict], references: list[dict]) -> None:
    summary = []
    records = [r for r in records if not r.get("profiled")]
    for cells, variant in sorted({(r["cells"], r["variant"]) for r in records if r.get("passed")}):
        current = [r for r in records if r["cells"] == cells and r["variant"] == variant and r.get("passed")]
        old = [r for r in references if r["cells"] == cells and r["variant"] == variant]
        elapsed = [r["loop_wall_s"] for r in current]
        baseline = statistics.median(r["loop_wall_s"] for r in old)
        median = statistics.median(elapsed)
        summary.append({"cells": cells, "variant": variant, "ranks": 2,
                        "retained_baseline_samples": len(old), "candidate_samples": len(current),
                        "baseline_median_s": baseline, "candidate_median_s": median,
                        "candidate_min_s": min(elapsed), "candidate_max_s": max(elapsed),
                        "baseline_over_candidate": baseline / median})
    write_json(output / "summary.json", summary)
    lines = ["# Region CFD execution qualification", "",
             "Candidate runs use the unchanged two-rank, 10-step / 0.2-second windows. "
             "Times include loop diagnostics/output and exclude initialization/merging. "
             "Baseline medians are retained observations from the supplied report; "
             "candidate medians are fresh sequential executions. Profiled runs are separate.", "",
             "| Cells | Variant | Baseline samples | Candidate samples | Baseline s | Candidate s [min, max] | Ratio |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary:
        lines.append(f"| {r['cells']} | {r['variant']} | {r['retained_baseline_samples']} | "
                     f"{r['candidate_samples']} | {r['baseline_median_s']:.4f} | {r['candidate_median_s']:.4f} "
                     f"[{r['candidate_min_s']:.4f}, {r['candidate_max_s']:.4f}] | {r['baseline_over_candidate']:.3f} |")
    lines += ["", "The physical comparison uses the retained OpenFOAM reference with its original "
              "manifest. Field/turbulence values and flow/gas iteration counts must match the retained "
              "SimpleFluid baseline exactly. All CSV changes and artifact identities are recorded. "
              "These initial windows do not qualify the full 20-second schedule or developed flow."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cells", nargs="+", type=int, default=[10752, 98304])
    parser.add_argument("--variants", nargs="+", choices=["piso", "iso1piso", "isopiso"],
                        default=["piso", "iso1piso", "isopiso"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--profile", action="store_true", help="Separate profiled observations; use one repeat")
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout < 1:
        parser.error("positive repeats and timeout required")
    if args.profile and args.repeats != 1:
        parser.error("profiled runs must use --repeats 1")
    reference = args.reference_root.resolve(strict=True)
    build = args.build_dir.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = build / "bin/Release/bottom_heated_bubbly_convection"
    binary_hash = digest(executable)
    libraries = shared_library_hashes(build / "lib/Release")
    provenance = json.loads((reference / "provenance.json").read_text())
    # Retained artifacts identify the reported baseline; do not reconfigure it.
    for path, expected in provenance["hashes"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Retained baseline artifact changed: {path}")
    records = [json.loads(line) for line in (reference / "cfd-measurements.jsonl").read_text().splitlines()]
    references = [r for r in records if r["phase"] in ("full", "one-region") and r["returncode"] == 0]
    inputs = [reference / "cfd" / f"input-n{cells}" for cells in args.cells]
    captured = {}
    for folder in inputs:
        for path in folder.iterdir():
            if path.is_file(): captured[str(path)] = digest(path)
    for r in references:
        if r["cells"] in args.cells:
            for path in (Path(r["directory"]) / "merged").glob("*.csv"):
                captured[str(path)] = digest(path)
    env = dict(os.environ)
    env.update({key: "1" for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]})
    env["LD_LIBRARY_PATH"] = str(build / "lib/Release") + ":" + env.get("LD_LIBRARY_PATH", "")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
                "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "executable": str(executable), "executable_sha256": binary_hash,
                "shared_libraries_sha256": libraries, "retained_provenance": provenance,
                "input_and_reference_sha256": captured,
                "cmake_cache_sha256": digest(build / "CMakeCache.txt"), "runs": []}
    (output / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT))
    names = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT, text=True).split("\0")
    manifest["new_source_sha256"] = {}
    for name in filter(None, names):
        path = ROOT / name
        if not name.startswith(("src/", "verification/")) or path.suffix not in (".hh", ".cc", ".tcc", ".py", ".sh"):
            continue
        target = output / "source-new" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        manifest["new_source_sha256"][name] = digest(path)
    try:
        for cells in args.cells:
            settings = load_manifest(reference / "cfd" / f"input-n{cells}" / "manifest.json")
            for repeat in range(1, args.repeats + 1):
                order = args.variants if repeat % 2 else list(reversed(args.variants))
                for variant in order:
                    old = next(r for r in references if r["cells"] == cells and r["variant"] == variant)
                    of = next(r for r in references if r["cells"] == cells and r["variant"] == "openfoam")
                    directory = output / f"n{cells}" / f"{variant}-{repeat}"
                    directory.mkdir(parents=True)
                    inp = reference / "cfd" / f"input-n{cells}"
                    command = [str(executable), "--output", str(directory), "--mesh-file", str(inp / "mesh.dat"),
                               "--properties", str(inp / "reference.properties"), "--water-properties",
                               str(inp / "reference_water.properties"), "--steps", "10", "--transport-solver",
                               "bicgstab", "--transport-preconditioner", "sgs", "--pressure-solver", "pcg",
                               "--pressure-preconditioner", "dic"]
                    if variant != "piso": command += ["--mesh-backend", "isoregion", "--regions", "1" if variant == "iso1piso" else "2"]
                    if args.profile: command = [sys.executable, str(reference / "profile_rank.py"), str(directory), *command]
                    command = ["mpiexec", "--timeout", str(args.timeout), "--bind-to", "core", "--map-by", "core",
                               "--report-bindings", "-n", "2", *command]
                    record = {"cells": cells, "variant": variant, "repeat": repeat, "ranks": 2, "steps": 10,
                              "dt": 0.02, "directory": str(directory), "profiled": args.profile, "command": command}
                    manifest["runs"].append(record)
                    check_artifacts(executable, binary_hash, libraries)
                    record["start_epoch"] = time.time()
                    start = time.perf_counter()
                    with (directory / "launcher.log").open("w") as log:
                        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout + 30)
                    record.update(returncode=process.returncode, launcher_wall_s=time.perf_counter() - start)
                    if process.returncode: raise RuntimeError(f"CFD run failed: {directory}")
                    check_artifacts(executable, binary_hash, libraries)
                    record.update(merge_outputs(directory, 2, record["start_epoch"]))
                    actual = directory / "merged"
                    of_output, old_output = Path(of["directory"]) / "merged", Path(old["directory"]) / "merged"
                    physical = compare(settings, of_output / "history.csv", actual / "history.csv",
                                       min(of["start_epoch"], record["start_epoch"]))
                    load_fields(settings, of_output / "fields.csv", actual / "fields.csv",
                                min(of["start_epoch"], record["start_epoch"]))
                    changes = {p.name: csv_changes(p, actual / p.name) for p in old_output.glob("*.csv")}
                    iteration_changes = changes["linear_solver_statistics.csv"]["max_absolute_changes"]
                    counts_match = all(iteration_changes[name] == 0 for name in
                                       ["flow_solves", "flow_iterations", "gas_solves", "gas_iterations"])
                    fields_match = all(changes[name]["exact_numeric_match"] for name in ["fields.csv", "turbulence.csv"])
                    record.update(passed=physical["passed"] and counts_match and fields_match,
                                  physical=physical, changes=changes, iteration_counts_match=counts_match,
                                  fields_match=fields_match)
                    write_json(directory / "measurement.json", record)
                    print(f"{cells} {variant} repeat {repeat}: {record['loop_wall_s']:.4f}s; passed={record['passed']}", flush=True)
                    if not record["passed"]: raise ComparisonError(f"Unchanged comparison gates failed: {directory}")
    finally:
        write_json(output / "manifest.json", manifest)
        summarize(output, manifest["runs"], references)
    for name, expected in captured.items():
        if digest(Path(name)) != expected: raise RuntimeError(f"Input/reference changed during qualification: {name}")
    print(f"Report: {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
