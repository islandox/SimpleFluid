#!/usr/bin/env python3
"""Compare existing SimpleFluid builds on explicitly selected full fixtures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import time

from compare_verification import ComparisonError, compare, load_manifest
from plot_water_fields import load_fields


HERE = Path(__file__).resolve().parent
# Family, physical mode, executable, compared CSV, recommended MPI ranks.
CASES = {
    "convection": ("bottomHeatedBubblyConvection", "transient",
                   "bottom_heated_bubbly_convection", "history.csv", 2),
    "bubble-steady": ("dispersedBubbleFlow", "steady",
                      "dispersed_bubble_verification", "profiles.csv", 1),
    "bubble-transient": ("dispersedBubbleFlow", "transient",
                         "dispersed_bubble_verification", "profiles.csv", 1),
    "ale-steady": ("planarALE", "steady", "planar_ale_comparison", "history.csv", 1),
    "ale-transient": ("planarALE", "transient", "planar_ale_comparison", "history.csv", 1),
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def shared_library_hashes(directory: Path | None) -> dict:
    if directory is None:
        return {}
    return {str(path): digest(path) for path in sorted(directory.resolve().glob("*.so*"))
            if path.is_file()}


def check_artifacts(executable: Path, expected_hash: str, libraries: dict) -> None:
    for path, checksum in {str(executable): expected_hash, **libraries}.items():
        if digest(Path(path)) != checksum:
            raise RuntimeError(f"executable or shared library changed during benchmark: {path}")


def solver_arguments(case: str, inputs: Path, policy: str) -> list[str]:
    family, mode, _, _, _ = CASES[case]
    command = ["--mesh-file", str(inputs / "mesh.dat"),
               "--water-properties", str(inputs / "reference_water.properties")]
    if family == "bottomHeatedBubblyConvection":
        command += ["--properties", str(inputs / "reference.properties")]
    else:
        command += ["--mode", mode]
        if family == "dispersedBubbleFlow":
            command += ["--parameters", str(inputs / "reference.properties")]
    if policy in ("recommended", "dic-sgs"):
        if family != "dispersedBubbleFlow":
            command += ["--pressure-solver", "pcg", "--pressure-preconditioner", "dic"]
        if family != "planarALE":
            command += ["--transport-solver", "bicgstab", "--transport-preconditioner", "sgs"]
    elif policy == "pressure-dic" and family != "dispersedBubbleFlow":
        command += ["--pressure-solver", "pcg", "--pressure-preconditioner", "dic"]
    return command


def merge_outputs(directory: Path, ranks: int, started: float, prefix: str = "rank") -> dict:
    """Merge owned samples, checking repeated global diagnostics before dropping them."""
    paths = [directory] if ranks == 1 else [directory / f"{prefix}{i}" for i in range(ranks)]
    output = directory / "merged"
    output.mkdir()
    names = {path.name for path in paths[0].glob("*.csv")}
    if not {"fields.csv", "history.csv"}.issubset(names):
        raise ComparisonError(f"missing required CSVs in {paths[0]}")
    timings = []
    for rank, path in enumerate(paths):
        if {file.name for file in path.glob("*.csv")} != names:
            raise ComparisonError(f"rank CSV sets differ: {path}")
        timing_path = path / "timing.json"
        if timing_path.stat().st_mtime < started:
            raise ComparisonError(f"stale rank timing: {timing_path}")
        timing = json.loads(timing_path.read_text())
        if timing["rank"] != rank or timing["ranks"] != ranks:
            raise ComparisonError(f"wrong rank timing: {timing_path}")
        if any(not math.isfinite(timing[key]) or timing[key] < 0
               for key in ("loop_wall_s", "loop_cpu_s")):
            raise ComparisonError(f"invalid rank timing: {timing_path}")
        timings.append(timing)
    for name in sorted(names):
        header, parts = None, []
        for path in paths:
            source = path / name
            if source.stat().st_mtime < started:
                raise ComparisonError(f"stale CSV: {source}")
            with source.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or (header is not None and header != reader.fieldnames):
                    raise ComparisonError(f"inconsistent CSV columns: {source}")
                header = reader.fieldnames
                parts.append(list(reader))
        if name in ("fields.csv", "profiles.csv", "turbulence.csv"):
            rows = [row for part in parts for row in part]
            rows.sort(key=lambda row: (float(row["time_s"]), int(row["sample"])))
            if len({(float(row["time_s"]), row["sample"]) for row in rows}) != len(rows):
                raise ComparisonError(f"duplicate partition samples: {name}")
        else:
            rows = parts[0]
            for part in parts[1:]:
                if len(part) != len(rows):
                    raise ComparisonError(f"rank row counts differ: {name}")
                for reference, actual in zip(rows, part):
                    for key in header:
                        try:
                            agrees = math.isclose(float(reference[key]), float(actual[key]),
                                                 rel_tol=1e-12, abs_tol=1e-20)
                        except (TypeError, ValueError):
                            agrees = reference[key] == actual[key]
                        if not agrees:
                            raise ComparisonError(f"rank global diagnostics differ: {name}, {key}")
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
    return {"loop_wall_s": max(item["loop_wall_s"] for item in timings),
            "loop_cpu_sum_s": sum(item["loop_cpu_s"] for item in timings),
            "rank_timings": timings}


def run(args, case: str, label: str, executable: Path, expected_hash: str,
        inputs: Path, ranks: int, repeat: int, environment: dict,
        libraries: dict | None = None) -> dict:
    directory = args.output / case / f"p{ranks}" / f"{label}-{repeat}"
    directory.mkdir(parents=True)
    command = [args.mpiexec, *shlex.split(args.mpi_args), "--timeout", str(args.timeout),
               "-n", str(ranks), str(executable), "--output", str(directory),
               *solver_arguments(case, inputs, args.policy)]
    record = {"case": case, "label": label, "ranks": ranks, "repeat": repeat,
              "policy": args.policy, "command": command, "directory": str(directory),
              "executable_sha256": expected_hash, "shared_libraries_sha256": libraries or {}}
    try:
        check_artifacts(executable, expected_hash, libraries or {})
        record["start_epoch"] = time.time()
        start = time.perf_counter()
        with (directory / "launcher.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, timeout=args.timeout + 30)
        record["launcher_wall_s"] = time.perf_counter() - start
        record["returncode"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"solver failed; see {directory / 'launcher.log'}")
        check_artifacts(executable, expected_hash, libraries or {})
        record.update(merge_outputs(directory, ranks, record["start_epoch"]))
    except Exception as error:
        record["error"] = str(error)
        raise
    finally:
        write_json(directory / "measurement.json", record)
        with (args.output / "measurements.jsonl").open("a") as ledger:
            ledger.write(json.dumps(record, allow_nan=False) + "\n")
    print(f"{case} {label} p{ranks} repeat {repeat}: {record['loop_wall_s']:.6g} loop s", flush=True)
    return record


def compare_pair(manifest: dict, filename: str, baseline: dict, current: dict) -> dict:
    reference = Path(baseline["directory"]) / "merged"
    actual = Path(current["directory"]) / "merged"
    started = min(baseline["start_epoch"], current["start_epoch"])
    report = compare(manifest, reference / filename, actual / filename, started)
    # Reuse the established physical gates; make the comparator's slot labels explicit.
    report["input_labels"] = {"openfoam": "SimpleFluid baseline", "simplefluid": "SimpleFluid current"}
    fields = load_fields(manifest, reference / "fields.csv", actual / "fields.csv", started)
    report["field_max_absolute_changes"] = {
        name: max(abs(row[name] - fields[1][key][name]) for key, row in fields[0].items())
        for name in fields[0][next(iter(fields[0]))] if name not in ("time_s", "sample")}
    write_json(actual.parent / "comparison.json", report)
    if not report["passed"]:
        raise ComparisonError(f"physical comparison failed: {actual.parent / 'comparison.json'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bin-dir", type=Path, required=True)
    parser.add_argument("--current-bin-dir", type=Path, required=True)
    parser.add_argument("--baseline-lib-dir", type=Path, help="prepend baseline shared-library directory to LD_LIBRARY_PATH")
    parser.add_argument("--current-lib-dir", type=Path, help="prepend current shared-library directory to LD_LIBRARY_PATH")
    parser.add_argument("--output", type=Path, required=True, help="new output directory, never overwritten")
    parser.add_argument("--cases", choices=CASES, nargs="+", required=True)
    parser.add_argument("--ranks", type=positive_integer, nargs="+", help="override every selected fixture's rank count")
    parser.add_argument("--policy", choices=("recommended", "default", "pressure-dic", "dic-sgs"), default="recommended")
    parser.add_argument("--repeats", type=positive_integer, default=3)
    parser.add_argument("--timeout", type=positive_integer, default=1800, help="OpenMPI timeout per solver in seconds")
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--mpi-args", default="--bind-to core --map-by core --report-bindings")
    args = parser.parse_args()
    args.output = args.output.resolve()
    binaries = {label: {case: (directory / CASES[case][2]).resolve() for case in args.cases}
                for label, directory in (("baseline", args.baseline_bin_dir), ("current", args.current_bin_dir))}
    hashes = {label: {case: digest(binary) for case, binary in items.items()}
              for label, items in binaries.items()}
    environment = dict(os.environ)
    environment.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    environments, library_hashes = {}, {}
    for label in ("baseline", "current"):
        environments[label] = dict(environment)
        library = getattr(args, f"{label}_lib_dir")
        if library is not None:
            if not library.is_dir():
                parser.error(f"shared-library directory does not exist: {library}")
            original = environment.get("LD_LIBRARY_PATH", "")
            environments[label]["LD_LIBRARY_PATH"] = str(library.resolve()) + (":" + original if original else "")
        library_hashes[label] = shared_library_hashes(library)
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {"platform": platform.platform(), "python": platform.python_version(),
                  "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  "binaries": {label: {case: {"path": str(path), "sha256": hashes[label][case]}
                                        for case, path in items.items()} for label, items in binaries.items()},
                  "shared_libraries_sha256": library_hashes,
                  "environment": {label: {name: selected.get(name) for name in
                                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                           "LD_LIBRARY_PATH", "SIMPLEFLUID_BUILD_CONFIG", "SIMPLEFLUID_COMPILER")}
                                  for label, selected in environments.items()}}
    if hasattr(os, "sched_getaffinity"):
        provenance["allowed_cpus"] = sorted(os.sched_getaffinity(0))
    write_json(args.output / "provenance.json", provenance)
    summary = []
    for case in dict.fromkeys(args.cases):
        family, mode, _, filename, recommended_ranks = CASES[case]
        inputs = args.output / case / "inputs"
        inputs.mkdir(parents=True)
        for source in (HERE / family / "mesh.dat", HERE / "reference_water.properties",
                       HERE / family / f"{mode}.json"):
            shutil.copy2(source, inputs / source.name)
        if (HERE / family / "reference.properties").exists():
            shutil.copy2(HERE / family / "reference.properties", inputs / "reference.properties")
        manifest = load_manifest(inputs / f"{mode}.json")
        for ranks in dict.fromkeys(args.ranks or [recommended_ranks]):
            records = []
            for repeat in range(1, args.repeats + 1):
                order = ("baseline", "current") if repeat % 2 else ("current", "baseline")
                pair = {label: run(args, case, label, binaries[label][case], hashes[label][case],
                                   inputs, ranks, repeat, environments[label], library_hashes[label])
                        for label in order}
                compare_pair(manifest, filename, pair["baseline"], pair["current"])
                records.extend(pair.values())
            result = {"case": case, "ranks": ranks, "policy": args.policy, "repeats": args.repeats}
            for label in ("baseline", "current"):
                selected = [row for row in records if row["label"] == label]
                result[label] = {key: {"median": statistics.median(row[key] for row in selected),
                                      "min": min(row[key] for row in selected),
                                      "max": max(row[key] for row in selected)}
                                 for key in ("loop_wall_s", "loop_cpu_sum_s", "launcher_wall_s")}
            result["loop_speedup"] = result["baseline"]["loop_wall_s"]["median"] / result["current"]["loop_wall_s"]["median"]
            summary.append(result)
            write_json(args.output / "summary.json", summary)
    print(f"Accepted timings and comparisons: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
