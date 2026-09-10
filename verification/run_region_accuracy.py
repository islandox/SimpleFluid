#!/usr/bin/env python3
"""Run the focused region accuracy executable and compare serial/MPI diagnostics.

Build testRegionQualification and source verification/environments.sh first.
This runner executes only its three qualification tests and retains each log.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


RECORD = re.compile(r"REGION_(REFINEMENT|ADVECTION|DECOMPOSITION) ([^\r\n]*)")
KEY_FIELDS = {
    "REFINEMENT": ("n", "ratio", "interface", "normal_refinement"),
    "ADVECTION": ("cells",),
    "DECOMPOSITION": ("regions",),
}
COMPARE_FIELDS = {
    "REFINEMENT": (
        "global_l2", "interface_l2", "conforming_global_l2",
        "conforming_interface_l2", "max_angle",
    ),
    "ADVECTION": ("amplitude", "phase_error", "l2"),
}


def records_from_log(text: str) -> dict[str, dict[str, float]]:
    records = {}
    for kind, payload in RECORD.findall(text):
        record = {}
        for name, value in re.findall(r"(\w+)=([-+0-9.eE]+)", payload):
            record[name] = float(value)
        if not all(math.isfinite(value) for value in record.values()):
            raise ValueError(f"Nonfinite diagnostic: {kind} {payload}")
        key = kind + ":" + ",".join(f"{field}={record[field]:g}" for field in KEY_FIELDS[kind])
        if key in records:
            raise ValueError(f"Duplicate diagnostic record: {key}")
        records[key] = record
    expected = {"REFINEMENT": 12, "ADVECTION": 3, "DECOMPOSITION": 2}
    for kind, count in expected.items():
        found = sum(key.startswith(kind + ":") for key in records)
        if found != count:
            raise ValueError(f"Expected {count} {kind} records, found {found}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    ranks = sorted(set(args.ranks))
    if not ranks or ranks[0] != 1 or any(rank < 1 for rank in ranks):
        parser.error("--ranks must include serial rank 1 and only positive rank counts")
    if not math.isfinite(args.absolute_tolerance) or args.absolute_tolerance <= 0:
        parser.error("--absolute-tolerance must be finite and positive")
    executable = args.executable.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for rank in ranks:
        command = [str(executable), "--gtest_color=no"]
        if rank > 1:
            command = [args.mpiexec, "-n", str(rank), *command]
        print(f"Running region accuracy at {rank} rank(s)", flush=True)
        log = args.output / f"accuracy_{rank}ranks.log"
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired as error:
            output = error.stdout or ""
            log.write_text(output.decode(errors="replace") if isinstance(output, bytes) else output)
            raise RuntimeError(f"Region accuracy timed out at {rank} ranks; see {log}") from error
        log.write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(f"Region accuracy failed at {rank} ranks (exit {completed.returncode}); see {log}")
        results[rank] = records_from_log(completed.stdout)
    comparisons = []
    for rank in ranks[1:]:
        for key, reference in results[1].items():
            kind = key.split(":", 1)[0]
            for field in COMPARE_FIELDS.get(kind, ()):
                difference = abs(results[rank][key][field] - reference[field])
                comparisons.append({"ranks": rank, "case": key, "field": field, "difference": difference})
                if difference > args.absolute_tolerance:
                    raise RuntimeError(
                        f"Serial/{rank}-rank mismatch: {key} {field}, difference={difference:g}, "
                        f"tolerance={args.absolute_tolerance:g}; logs are in {args.output}")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executable": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "ranks": ranks,
        "absolute_tolerance": args.absolute_tolerance,
        "diagnostics": results,
        "comparisons": comparisons,
    }
    report_path = args.output / "accuracy.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    maximum = max((entry["difference"] for entry in comparisons), default=0.0)
    print(f"Region accuracy passed; maximum serial/MPI diagnostic difference={maximum:.3g}; {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
