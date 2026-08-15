#!/usr/bin/env python3
"""Compare SimpleFluid and OpenFOAM cavity centerline velocity profiles."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


COMPONENTS = {"ux": 1, "uy": 2, "uz": 3}


def nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError(
            "tolerances must be finite and non-negative")
    return result


def read_openfoam(path: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.replace("(", " ").replace(")", " ").split()
            if len(values) < 4:
                continue
            rows.append(tuple(float(value) for value in values[:4]))
    if len(rows) < 2:
        raise ValueError(f"{path} does not contain an OpenFOAM profile")
    return sorted(rows)


def read_simplefluid(path: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"coordinate", "ux", "uy", "uz"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {sorted(required)}")
        for row in reader:
            rows.append(tuple(float(row[name]) for name in
                              ("coordinate", "ux", "uy", "uz")))
    if not rows:
        raise ValueError(f"{path} does not contain a SimpleFluid profile")
    return rows


def interpolate(
    rows: list[tuple[float, float, float, float]],
    coordinate: float,
    component: int,
) -> float:
    if coordinate < rows[0][0] or coordinate > rows[-1][0]:
        raise ValueError(f"coordinate {coordinate} is outside reference range")
    for left, right in zip(rows, rows[1:]):
        if left[0] <= coordinate <= right[0]:
            width = right[0] - left[0]
            if width == 0.0:
                return left[component]
            weight = (coordinate - left[0]) / width
            return left[component] + weight * (
                right[component] - left[component])
    return rows[-1][component]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openfoam", type=Path, required=True)
    parser.add_argument("--simplefluid", type=Path, required=True)
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--max-l2", type=nonnegative_float)
    parser.add_argument("--max-linf", type=nonnegative_float)
    arguments = parser.parse_args()

    openfoam = read_openfoam(arguments.openfoam)
    simplefluid = read_simplefluid(arguments.simplefluid)
    component = COMPONENTS[arguments.component]
    errors = [
        row[component] - interpolate(openfoam, row[0], component)
        for row in simplefluid
    ]

    l2 = math.sqrt(sum(error * error for error in errors) / len(errors))
    maximum = max(abs(error) for error in errors)
    print(f"samples={len(errors)} l2={l2:.12g} linf={maximum:.12g}")

    failures: list[str] = []
    if arguments.max_l2 is not None and l2 > arguments.max_l2:
        failures.append(f"l2 {l2:.12g} exceeds {arguments.max_l2:.12g}")
    if arguments.max_linf is not None and maximum > arguments.max_linf:
        failures.append(
            f"linf {maximum:.12g} exceeds {arguments.max_linf:.12g}")
    if failures:
        print("comparison failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
