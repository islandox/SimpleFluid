#!/usr/bin/env python3
"""Compare matched OpenFOAM/SimpleFluid verification CSVs without dependencies.

Each CSV has time_s,sample and the quantity/residual columns named by the
manifest. The expected times and samples define a complete Cartesian product;
matching two truncated runs therefore cannot pass. Quantities use
abs(SimpleFluid - OpenFOAM) <= absolute_tolerance +
relative_tolerance * abs(OpenFOAM). Conservation columns are independently
bounded against zero for both solvers, even when both solvers agree.

Run wrappers should pass --not-before with the epoch timestamp captured before
launching either solver. CSV modification times must be at least that recent.
Numerical agreement does not by itself qualify a physical model or reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


class ComparisonError(ValueError):
    """An input does not meet the verification contract."""


def finite_number(value: Any, context: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ComparisonError(f"{context} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ComparisonError(f"{context} must be a finite number") from error
    if not math.isfinite(number) or (nonnegative and number < 0):
        qualifier = "finite nonnegative" if nonnegative else "finite"
        raise ComparisonError(f"{context} must be a {qualifier} number")
    return number


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ComparisonError(f"duplicate manifest key {name!r}")
        result[name] = value
    return result


def validate_manifest(document: Any) -> dict[str, Any]:
    if (not isinstance(document, dict)
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1):
        raise ComparisonError("manifest must be a schema_version 1 object")
    if not isinstance(document.get("case"), str) or not document["case"].strip():
        raise ComparisonError("manifest case must be a nonempty string")
    if document.get("mode") not in ("steady", "transient"):
        raise ComparisonError("manifest mode must be steady or transient")
    tolerance = finite_number(document.get("time_tolerance_s", 1.0e-10),
                              "time_tolerance_s", nonnegative=True)
    times = document.get("expected_times_s")
    if not isinstance(times, list) or not times:
        raise ComparisonError("expected_times_s must be a nonempty list")
    times = [finite_number(value, "expected_times_s", nonnegative=True)
             for value in times]
    if any(right <= left or right - left <= 2 * tolerance
           for left, right in zip(times, times[1:])):
        raise ComparisonError("expected_times_s must increase strictly and "
                              "have nonoverlapping time tolerance windows")
    samples = document.get("expected_samples")
    if (not isinstance(samples, list) or not samples
            or any(not isinstance(value, str) or not value.strip()
                   or value != value.strip() for value in samples)):
        raise ComparisonError("expected_samples must be nonempty strings "
                              "without surrounding whitespace")
    if len(set(samples)) != len(samples):
        raise ComparisonError("expected_samples contains duplicate samples")
    quantities = document.get("quantities")
    if not isinstance(quantities, dict) or not quantities:
        raise ComparisonError("quantities must be a nonempty object")
    conservation = document.get("conservation", {})
    if not isinstance(conservation, dict):
        raise ComparisonError("conservation must be an object")
    for category, entries in (("quantities", quantities),
                              ("conservation", conservation)):
        for name, limits in entries.items():
            if (not isinstance(name, str) or not name.strip()
                    or name != name.strip() or name in ("time_s", "sample")):
                raise ComparisonError(f"invalid {category} column {name!r}")
            if not isinstance(limits, dict):
                raise ComparisonError(f"{category}.{name} must be an object")
            finite_number(limits.get("absolute_tolerance"),
                          f"{category}.{name}.absolute_tolerance", nonnegative=True)
            if category == "quantities":
                finite_number(limits.get("relative_tolerance"),
                              f"{category}.{name}.relative_tolerance", nonnegative=True)
                if not isinstance(limits.get("units"), str) or not limits["units"]:
                    raise ComparisonError(f"quantities.{name}.units is required")
            elif "relative_tolerance" in limits:
                raise ComparisonError(f"conservation.{name} uses an absolute "
                                      "zero-residual bound only")
    return {**document, "time_tolerance_s": tolerance,
            "expected_times_s": times, "conservation": conservation}


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"),
                              object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"cannot read manifest {path}: {error}") from error
    return validate_manifest(document)


def read_samples(path: Path, manifest: dict[str, Any],
                 not_before: float | None = None) -> dict[tuple[int, str], dict[str, Any]]:
    if not_before is not None:
        timestamp = finite_number(not_before, "not_before", nonnegative=True)
        if path.stat().st_mtime < timestamp:
            raise ComparisonError(f"stale CSV {path}: predates this run")
    columns = set(manifest["quantities"]) | set(manifest["conservation"])
    expected_columns = columns | {"time_s", "sample"}
    expected_times = manifest["expected_times_s"]
    expected_samples = set(manifest["expected_samples"])
    time_tolerance = manifest["time_tolerance_s"]
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers or len(set(headers)) != len(headers):
            raise ComparisonError(f"{path}: missing or duplicate CSV headers")
        if set(headers) != expected_columns:
            missing = sorted(expected_columns - set(headers))
            extra = sorted(set(headers) - expected_columns)
            raise ComparisonError(f"{path}: CSV columns differ: "
                                  f"missing={missing}, unexpected={extra}")
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ComparisonError(f"{path}:{line}: malformed CSV row")
            sample = row["sample"].strip()
            if sample not in expected_samples:
                raise ComparisonError(f"{path}:{line}: unexpected sample {sample!r}")
            actual_time = finite_number(row["time_s"], f"{path}:{line}:time_s",
                                        nonnegative=True)
            matches = [index for index, time in enumerate(expected_times)
                       if abs(actual_time - time) <= time_tolerance]
            if len(matches) != 1:
                raise ComparisonError(f"{path}:{line}: unexpected time {actual_time}")
            key = (matches[0], sample)
            if key in rows:
                raise ComparisonError(f"{path}:{line}: duplicate time/sample {key}")
            rows[key] = {"time_s": actual_time,
                         **{column: finite_number(row[column],
                                                  f"{path}:{line}:{column}")
                            for column in columns}}
    expected = {(index, sample) for index in range(len(expected_times))
                for sample in expected_samples}
    missing = sorted(expected - rows.keys())
    if missing:
        preview = [(expected_times[index], sample) for index, sample in missing[:5]]
        raise ComparisonError(f"{path}: missing {len(missing)} expected "
                              f"time/sample rows; first missing={preview}")
    return rows


def compare(manifest: dict[str, Any], openfoam: Path, simplefluid: Path,
            not_before: float | None = None) -> dict[str, Any]:
    """Return a JSON-safe report; invalid input raises ComparisonError/OSError."""
    manifest = validate_manifest(manifest)
    if os.path.samefile(openfoam, simplefluid):
        raise ComparisonError("OpenFOAM and SimpleFluid CSVs must be distinct files")
    reference = read_samples(openfoam, manifest, not_before)
    actual = read_samples(simplefluid, manifest, not_before)
    failures: list[str] = []
    keys = sorted(reference)
    for key in keys:
        if abs(reference[key]["time_s"] - actual[key]["time_s"]) > manifest["time_tolerance_s"]:
            raise ComparisonError(f"solver time mismatch at sample {key[1]!r}, "
                                  f"expected time {manifest['expected_times_s'][key[0]]}")
    metrics: dict[str, Any] = {}
    for name, limits in manifest["quantities"].items():
        absolute = float(limits["absolute_tolerance"])
        relative = float(limits["relative_tolerance"])
        errors: list[float] = []
        failed = 0
        worst: dict[str, Any] | None = None
        max_excess = -math.inf
        for key in keys:
            ref_value = reference[key][name]
            value = actual[key][name]
            error = finite_number(abs(value - ref_value), f"{name}: absolute error")
            bound = finite_number(absolute + relative * abs(ref_value),
                                  f"{name}: tolerance bound")
            errors.append(error)
            failed += error > bound
            excess = error - bound
            if excess > max_excess:
                max_excess = excess
                worst = {"time_s": manifest["expected_times_s"][key[0]],
                         "sample": key[1], "openfoam": ref_value,
                         "simplefluid": value, "absolute_error": error,
                         "allowed_error": bound}
        metrics[name] = {"units": limits["units"], "samples": len(keys),
                         "absolute_tolerance": absolute, "relative_tolerance": relative,
                         "max_absolute_error": max(errors),
                         "rms_absolute_error": math.hypot(
                             *(error / math.sqrt(len(errors)) for error in errors)),
                         "failed_samples": failed, "worst_sample": worst}
        if failed:
            failures.append(f"{name}: {failed}/{len(keys)} samples exceed abs+rel tolerance")
    conservation: dict[str, Any] = {}
    for name, limits in manifest["conservation"].items():
        bound = float(limits["absolute_tolerance"])
        conservation[name] = {"absolute_tolerance": bound}
        for solver, rows in (("openfoam", reference), ("simplefluid", actual)):
            worst_key = max(keys, key=lambda key: abs(rows[key][name]))
            failed = sum(abs(rows[key][name]) > bound for key in keys)
            conservation[name][solver] = {
                "max_absolute_residual": abs(rows[worst_key][name]),
                "failed_samples": failed,
                "worst_time_s": manifest["expected_times_s"][worst_key[0]],
                "worst_sample": worst_key[1]}
            if failed:
                failures.append(f"{solver} {name}: {failed}/{len(keys)} samples "
                                "exceed independent residual bound")
    return {"schema_version": 1, "case": manifest["case"], "mode": manifest["mode"],
            "passed": not failures, "status": "failed" if failures else "passed",
            "openfoam_csv": str(openfoam.resolve()),
            "simplefluid_csv": str(simplefluid.resolve()),
            "not_before_epoch_s": not_before,
            "expected_rows_per_solver": len(keys),
            "expected_times_s": manifest["expected_times_s"],
            "expected_samples": manifest["expected_samples"],
            "quantities": metrics, "conservation": conservation, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--openfoam", required=True, type=Path)
    parser.add_argument("--simplefluid", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--not-before", type=float,
                        help="Reject solver CSVs older than this epoch timestamp")
    args = parser.parse_args(argv)
    inputs = (args.manifest, args.openfoam, args.simplefluid)
    overwrites_input = any(
        args.report.resolve() == path.resolve()
        or (args.report.exists() and path.exists() and os.path.samefile(args.report, path))
        for path in inputs)
    try:
        if overwrites_input:
            raise ComparisonError("report path must not overwrite an input")
        report = compare(load_manifest(args.manifest), args.openfoam,
                         args.simplefluid, args.not_before)
        code = 0 if report["passed"] else 1
    except (ComparisonError, OSError, csv.Error, UnicodeError) as error:
        report = {"schema_version": 1, "passed": False, "status": "error",
                  "failures": [str(error)]}
        code = 2
    # Do not clobber an input even when an invalid report path caused the error.
    if not overwrites_input:
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                   encoding="utf-8")
        except OSError as error:
            print(f"cannot write comparison report: {error}", file=sys.stderr)
            return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
