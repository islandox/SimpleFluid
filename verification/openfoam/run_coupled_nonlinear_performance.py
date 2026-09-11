#!/usr/bin/env python3
"""Matched two-rank nonlinear convection windows using prepared 10k/100k fixtures.

The fixture directory contains input-n<CELLS>, templates/n<CELLS> (meshed and
decomposed OpenFOAM cases), and bin/bottomBubblyConvectionFoam. All timings are
fresh executions; retained outputs are never used as timing references.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from compare_verification import compare, load_manifest, read_samples
from plot_water_fields import load_fields
from run_performance import (check_artifacts, digest, merge_outputs,
                             positive_integer, shared_library_hashes, write_json)

VARIANTS = ["openfoam", "piso", "coupled-assembled", "coupled-composite",
            "nox-assembled", "nox-composite", "nox-composite-streamed",
            "iso2-nox-composite-streamed", "nox-picard-composite",
            "nox-bicgstab-composite"]


def solver_arguments(variant, inputs, directory, steps, restart):
    command = ["--output", str(directory), "--mesh-file", str(inputs / "mesh.dat"),
               "--properties", str(inputs / "reference.properties"),
               "--water-properties", str(inputs / "reference_water.properties"),
               "--steps", str(steps), "--transport-solver", "bicgstab",
               "--transport-preconditioner", "sgs"]
    if variant == "piso":
        return command + ["--coupling", "piso", "--pressure-solver", "pcg",
                          "--pressure-preconditioner", "dic"]
    nonlinear = "nox" in variant
    command += ["--coupling", "nox" if nonlinear else "coupled",
                "--coupled-operator", "assembled" if "assembled" in variant else "block_composite",
                "--coupled-workspace", "streamed_products" if "streamed" in variant else "cached_products"]
    if nonlinear:
        command += ["--nonlinear-linear-solver", "bicgstab" if "bicgstab" in variant else "gmres",
                    "--nonlinear-method", "picard" if "picard" in variant else "newton",
                    "--nonlinear-restart", str(restart)]
    if variant.startswith("iso2"):
        command += ["--mesh-backend", "isoregion", "--regions", "2"]
    return command


def run_one(args, cells, variant, repeat, artifacts):
    directory = args.output / f"n{cells}" / f"{variant}-{repeat}"
    directory.mkdir(parents=True, exist_ok=False)
    inputs = args.fixtures / f"input-n{cells}"
    if variant == "openfoam":
        for name in ["0", "constant", "system", "processor0", "processor1"]:
            shutil.copytree(args.fixtures / "templates" / f"n{cells}" / name, directory / name)
        executable = args.foam_executable
        command = [str(executable), "-case", str(directory), "-parallel", "-steps", str(args.steps)]
    else:
        executable = args.executable
        command = [str(executable), *solver_arguments(variant, inputs, directory, args.steps, args.nox_restart)]
    check_artifacts(executable, artifacts[str(executable)], args.libraries)
    command = ["mpiexec", "--timeout", str(args.timeout), "--bind-to", "core", "--map-by", "core",
               "--report-bindings", "-n", "2", *command]
    environment = dict(os.environ)
    environment.update({key: "1" for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]})
    environment["LD_LIBRARY_PATH"] = str(args.library_dir) + ":" + environment.get("LD_LIBRARY_PATH", "")
    start = time.time()
    begin = time.perf_counter()
    with (directory / "launcher.log").open("w") as stream:
        try:
            result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=args.timeout + 30)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
    record = dict(cells=cells, variant=variant, repeat=repeat, ranks=2, steps=args.steps,
                  start_epoch=start, command=command, directory=str(directory), returncode=returncode,
                  wall_s=time.perf_counter() - begin, qualified=False)
    check_artifacts(executable, artifacts[str(executable)], args.libraries)
    if returncode == 0:
        try:
            record.update(merge_outputs(directory, 2, start, "processor" if variant == "openfoam" else "rank"))
            nonlinear_path = directory / "merged/nonlinear_solver_statistics.csv"
            if nonlinear_path.exists():
                with nonlinear_path.open() as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != args.steps or any(row["converged"] != "1" for row in rows):
                    raise ValueError("missing or unaccepted nonlinear physical steps")
                counts = ["nonlinear_iterations", "linear_solves", "krylov_iterations", "residual_evaluations",
                          "line_search_rejections"]
                seconds = ["total_seconds", "residual_seconds", "linearization_seconds", "linear_solve_seconds"]
                record["nonlinear"] = {key: sum(float(row[key]) for row in rows) for key in counts + seconds}
                record["nonlinear"]["max_continuity_m3_s"] = max(float(row["continuity_max_m3_s"]) for row in rows)
                record["nonlinear"]["max_scaled_residual"] = max(float(row["scaled_residual"]) for row in rows)
        except (ValueError, RuntimeError, OSError) as error:
            record["validation_error"] = str(error)
    write_json(directory / "measurement.json", record)
    print(json.dumps({key: record.get(key) for key in
                      ["cells", "variant", "repeat", "returncode", "loop_wall_s", "validation_error"]}), flush=True)
    return record


def qualify(args, records):
    for cells in args.cells:
        references = [record for record in records if record["cells"] == cells and
                      record["variant"] == "openfoam" and record["returncode"] == 0 and
                      "validation_error" not in record]
        if not references:
            continue
        manifest = load_manifest(args.fixtures / f"input-n{cells}/manifest.json")
        for record in records:
            if record["cells"] != cells or record["returncode"] or "validation_error" in record:
                continue
            reference = next((r for r in references if r["repeat"] == record["repeat"]), references[0])
            left, right = Path(reference["directory"]) / "merged", Path(record["directory"]) / "merged"
            start = min(reference["start_epoch"], record["start_epoch"])
            try:
                if record is reference:
                    rows = read_samples(right / "history.csv", manifest, start)
                    for name, limits in manifest["conservation"].items():
                        if any(abs(row[name]) > limits["absolute_tolerance"] for row in rows.values()):
                            raise ValueError(f"OpenFOAM failed independent {name} gate")
                    record["qualified"] = True
                    write_json(right.parent / "measurement.json", record)
                    continue
                report = compare(manifest, left / "history.csv", right / "history.csv", start)
                load_fields(manifest, left / "fields.csv", right / "fields.csv", start)
                write_json(right.parent / "comparison.json", report)
                record["qualified"] = bool(report["passed"])
                if not record["qualified"]:
                    record["validation_error"] = "physical comparison failed; see comparison.json"
            except (ValueError, RuntimeError, OSError) as error:
                record["validation_error"] = str(error)
            write_json(right.parent / "measurement.json", record)


def summarize(args, records):
    summary = []
    for cells in args.cells:
        for variant in args.variants:
            subset = [r for r in records if r["cells"] == cells and r["variant"] == variant]
            valid = [r for r in subset if r["qualified"]]
            row = dict(cells=cells, variant=variant, attempts=len(subset), qualified=len(valid))
            if valid:
                durations = [r["loop_wall_s"] for r in valid]
                row.update(median_s=statistics.median(durations), minimum_s=min(durations), maximum_s=max(durations))
                if "nonlinear" in valid[0]:
                    row["nonlinear_medians"] = {key: statistics.median(r["nonlinear"][key] for r in valid)
                                                for key in valid[0]["nonlinear"]}
            summary.append(row)
    write_json(args.output / "measurements.json", records)
    write_json(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--config", default="Release")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=positive_integer, nargs="+", default=[10752, 98304])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+",
                        default=["openfoam", "piso", "coupled-composite", "nox-assembled", "nox-composite"])
    parser.add_argument("--repeats", type=positive_integer, default=3)
    parser.add_argument("--steps", type=positive_integer, default=10)
    parser.add_argument("--timeout", type=positive_integer, default=900)
    parser.add_argument("--nox-restart", type=positive_integer, default=80)
    args = parser.parse_args()
    args.fixtures, args.build_dir, args.output = (path.resolve() for path in
                                                 [args.fixtures, args.build_dir, args.output])
    args.output.mkdir(parents=True, exist_ok=False)
    args.executable = args.build_dir / "bin" / args.config / "bottom_heated_bubbly_convection"
    args.foam_executable = args.fixtures / "bin/bottomBubblyConvectionFoam"
    args.library_dir = args.build_dir / "lib" / args.config
    args.libraries = shared_library_hashes(args.library_dir)
    artifacts = {str(path): digest(path) for path in [args.executable, args.foam_executable]}
    provenance = dict(arguments={key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()}, artifacts=artifacts,
                      source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    provenance["inputs"] = {str(path): digest(path) for cells in args.cells
                            for path in sorted((args.fixtures / f"input-n{cells}").glob("*")) if path.is_file()}
    provenance["build_cache"] = digest(args.build_dir / "CMakeCache.txt")
    provenance["environment"] = {key: os.environ.get(key, "") for key in
                                 ["PATH", "LD_LIBRARY_PATH", "WM_PROJECT_VERSION", "WM_OPTIONS"]}
    provenance["machine"] = subprocess.check_output(["lscpu", "-J"], text=True)
    write_json(args.output / "provenance.json", provenance)
    (args.output / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
    write_json(args.output / "untracked-sources.json", {path: Path(path).read_text() for path in untracked
                                                       if Path(path).suffix in {".hh", ".cc", ".py", ".md"}})
    records = []
    rejected = set()
    for repeat in range(1, args.repeats + 1):
        for cells in args.cells:
            order = args.variants if repeat % 2 else list(reversed(args.variants))
            for variant in order:
                if (cells, variant) in rejected:
                    continue
                record = run_one(args, cells, variant, repeat, artifacts)
                records.append(record)
                if record["returncode"] or "validation_error" in record:
                    rejected.add((cells, variant))
                write_json(args.output / "measurements.json", records)
            qualify(args, records)
            for record in records:
                if "validation_error" in record:
                    rejected.add((record["cells"], record["variant"]))
            summarize(args, records)
    print(json.dumps(summarize(args, records), indent=2), flush=True)
    return int(any(not record["qualified"] for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
