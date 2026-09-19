#!/usr/bin/env python3
"""Run per-rank partition diagnostics and validate topology/numerical invariants.

Every measured process must own at least 10,000 cells. Timing is observational:
no performance threshold is a correctness assertion.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import statistics
import subprocess
import time

CASES = [("ordered", 2), ("ordered", 8), ("shuffled", 8), ("refined", 2), ("refined", 3)]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-11)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, mode, n, parameter, ranks, repeats):
    metadata = [r for r in records if r["type"] == "metadata"]
    validation = [r for r in records if r["type"] == "validation"]
    parts = sorted((r for r in records if r["type"] == "partition"), key=lambda r: r["rank"])
    samples = [r for r in records if r["type"] == "sample"]
    require(len(metadata) == len(validation) == 1, "Missing unique metadata/validation")
    meta, valid = metadata[0], validation[0]
    require(valid["passed"], "Numerical validation failed")
    require((meta["mode"], meta["axis_cells"], meta["parameter"], meta["ranks"], meta["repeats"])
            == (mode, n, parameter, ranks, repeats), "Wrong run metadata")
    require([p["rank"] for p in parts] == list(range(ranks)), "Missing rank diagnostics")
    require(len(samples) == ranks * repeats, "Missing timing samples")
    require({(s["rank"], s["repeat"]) for s in samples}
            == {(p, i) for p in range(ranks) for i in range(repeats)}, "Duplicate/missing sample")
    require(min(p["owned_cells"] for p in parts) >= 10000, "Under 10k owned cells per rank")
    require(sum(p["owned_cells"] for p in parts) == meta["global_cells"], "Cell count mismatch")
    require(sum(p["matrix_nnz"] for p in parts) == valid["global_nnz"], "Matrix count mismatch")
    require(max(p["owned_cells"] for p in parts) - min(p["owned_cells"] for p in parts) <= 1,
            "Contiguous distribution lost equal cell counts")
    for name in ("cut_faces", "cut_region_faces"):
        require(sum(p[name] for p in parts) % 2 == 0, "Cut incidence must be counted twice")
    require(all(p["cut_region_faces"] <= p["cut_faces"] for p in parts), "Invalid interface cuts")
    if ranks == 1:
        require(parts[0]["ghost_cells"] == parts[0]["cut_faces"] == 0, "Serial ghosts/cuts")
    # Closed-form slab fixtures independently check the cut-count implementation.
    if parameter == 8 and mode in ("ordered", "shuffled") and ranks in (2, 4):
        planes = ranks - 1 if mode == "ordered" else 7
        require(sum(p["cut_faces"] for p in parts) == 2 * planes * n * n,
                "Cut count disagrees with the slab geometry")
    if mode == "refined":
        require(sum(p["owned_region_faces"] for p in parts) == parameter * n * n,
                "Coarse/fine subface count mismatch")

    identity = dict(mode=mode, axis_cells=n, parameter=parameter, ranks=ranks)
    per_rank = []
    for p in parts:
        row = dict(identity, **{k: v for k, v in p.items() if k != "type"})
        for metric in ("assembly_seconds", "apply_seconds"):
            values = [s[metric] for s in samples if s["rank"] == p["rank"]]
            require(all(math.isfinite(v) and v > 0 for v in values), "Invalid elapsed time")
            row[metric + "_median"] = statistics.median(values)
        per_rank.append(row)

    def imbalance(key):
        values = [p[key] for p in parts]
        return max(values) / statistics.mean(values)

    result = dict(identity, global_cells=meta["global_cells"],
                  owned_cells_min=min(p["owned_cells"] for p in parts),
                  owned_cells_max=max(p["owned_cells"] for p in parts),
                  ghost_cells_sum=sum(p["ghost_cells"] for p in parts),
                  ghost_owned_ratio_max=max(p["ghost_cells"] / p["owned_cells"] for p in parts),
                  unique_cut_faces=sum(p["cut_faces"] for p in parts) // 2,
                  unique_cut_region_faces=sum(p["cut_region_faces"] for p in parts) // 2,
                  unique_cut_area=sum(p["cut_area"] for p in parts) / 2,
                  neighbor_ranks_max=max(p["neighbor_ranks"] for p in parts),
                  nnz_min=min(p["matrix_nnz"] for p in parts),
                  nnz_max=max(p["matrix_nnz"] for p in parts),
                  nnz_max_over_mean=imbalance("matrix_nnz"),
                  incidence_max_over_mean=imbalance("face_incidences"),
                  owned_faces_max_over_mean=imbalance("owned_faces"),
                  max_cell_faces=max(p["max_cell_faces"] for p in parts),
                  max_row_nnz=max(p["max_row_nnz"] for p in parts))
    for metric in ("assembly_seconds", "apply_seconds"):
        # Median of the slowest rank in each synchronized repetition.
        critical = [max(s[metric] for s in samples if s["repeat"] == i) for i in range(repeats)]
        result[metric + "_critical_median"] = statistics.median(critical)
        result[metric + "_critical_min"] = min(critical)
        result[metric + "_critical_max"] = max(critical)
        rank_medians = [p[metric + "_median"] for p in per_rank]
        result[metric + "_rank_median_max_over_mean"] = max(rank_medians) / statistics.mean(rank_medians)
    result.update({k: v for k, v in valid.items() if k not in ("type", "passed")})
    return result, per_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build/gcc/bin/RelWithDebInfo/region_partition_benchmark"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[40, 64])
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=18092026)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--mpi-args", default="--bind-to core --map-by core")
    args = parser.parse_args()
    executable = args.executable.resolve()
    require(executable.is_file(), "Build region_partition_benchmark first")
    require(1 in args.ranks and all(p > 0 for p in args.ranks), "Include serial reference and positive rank counts")
    require(len(set(args.sizes)) == len(args.sizes) and len(set(args.ranks)) == len(args.ranks), "Duplicate sizes/ranks")
    require(1 <= args.repeats <= 100, "Repeats must be in 1..100")
    require(all(8 <= n <= 128 and n % 8 == 0 for n in args.sizes), "Sizes must be multiples of 8 in [8,128]")
    jobs = [(mode, n, parameter, ranks) for n in args.sizes for mode, parameter in CASES for ranks in args.ranks]
    for mode, n, parameter, ranks in jobs:
        cells = n**3 if mode != "refined" else n**3 // 2 * (parameter + 1)
        require(cells // ranks >= 10000, f"{mode}/{n}/{ranks}: each rank must own at least 10000 cells")
    random.Random(args.seed).shuffle(jobs)
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,sm")
    host = dict(platform=platform.platform(), processor=platform.processor(), seed=args.seed,
                executable=str(executable), jobs=jobs, repeats=args.repeats,
                command=shlex.join(os.sys.argv),
                environment={k: environment[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMPI_MCA_pml", "OMPI_MCA_btl")})
    for name, command in (("lscpu", ["lscpu"]), ("mpi_version", [args.mpiexec, "--version"]),
                          ("git_head", ["git", "rev-parse", "HEAD"]), ("git_status", ["git", "status", "--short"])):
        host[name] = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    source_paths = [
        "src/benchmarks/region_partition_benchmark.cc",
        "src/FVM/details/FaceStencilMatrix.hh",
        "src/FVM/details/DiffusionSystemImpl.hh",
        "src/FVM/details/MatrixOperatorsImpl.hh",
        "verification/run_region_partition_diagnostics.py",
    ]
    host["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                             for name in source_paths}
    (args.output / "host.json").write_text(json.dumps(host, indent=2) + "\n")
    summary, rank_rows = [], []
    for index, (mode, n, parameter, ranks) in enumerate(jobs):
        stem = f"{mode}-n{n}-k{parameter}-p{ranks}"
        command = [args.mpiexec, *shlex.split(args.mpi_args), "-n", str(ranks),
                   str(executable), mode, str(n), str(parameter), str(args.repeats)]
        print(f"[{index+1}/{len(jobs)}] {stem}", flush=True)
        start = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=900)
        (args.output / (stem + ".jsonl")).write_text(result.stdout)
        (args.output / (stem + ".stderr")).write_text(result.stderr)
        (args.output / (stem + ".command.json")).write_text(
            json.dumps(dict(command=command, seconds=time.monotonic()-start, exit_code=result.returncode), indent=2) + "\n")
        require(result.returncode == 0, f"{stem} failed; see its stderr")
        records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        row, parts = summarize(records, mode, n, parameter, ranks, args.repeats)
        summary.append(row)
        rank_rows.extend(parts)
        write_csv(args.output / "summary.csv", summary)
        write_csv(args.output / "per_rank.csv", rank_rows)

    indexed = {(r["mode"], r["axis_cells"], r["parameter"], r["ranks"]): r for r in summary}
    for row in summary:
        key = row["mode"], row["axis_cells"], row["parameter"], 1
        serial = indexed[key]
        require(row["global_nnz"] == serial["global_nnz"], "Partition changed global nonzero count")
        for metric in ("action_norm2", "rhs_norm2"):
            require(close(row[metric], serial[metric]), f"Partition changed {metric}: {key}")
        if row["mode"] == "shuffled":
            ordered = indexed["ordered", row["axis_cells"], row["parameter"], row["ranks"]]
            require(row["global_nnz"] == ordered["global_nnz"], "Descriptor order changed global nonzeros")
            for metric in ("action_norm2", "rhs_norm2"):
                require(close(row[metric], ordered[metric]), f"Descriptor order changed {metric}")
    (args.output / "validation.json").write_text(json.dumps(dict(
        passed=True, configurations=len(summary), minimum_owned_cells=min(r["owned_cells_min"] for r in summary),
        numerical_checks="generic assembly parity, constant solution, partition/order-invariant action and RHS norms",
        topology_checks="cell/nnz totals, analytical slab cuts, coarse/fine subface counts",
        timing_thresholds=False), indent=2) + "\n")
    print("All topology and numerical checks passed.", flush=True)


if __name__ == "__main__":
    main()
