#!/usr/bin/env python3
"""Compare pitzDaily velocity profiles from OpenFOAM and SimpleFluid."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path


STATIONS = {"x050": 0.050, "x100": 0.100, "x200": 0.200}


def nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError(
            "tolerances must be finite and non-negative")
    return result


def latest_profile(case: Path, station: str) -> Path:
    candidates = list(
        case.glob(f"postProcessing/profiles/*/{station}_U.xy"))
    if not candidates:
        raise FileNotFoundError(
            f"no {station}_U.xy below {case / 'postProcessing/profiles'}")

    def time_value(path: Path) -> float:
        try:
            return float(path.parent.name)
        except ValueError:
            return -math.inf

    return max(candidates, key=time_value)


def read_openfoam(path: Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.replace("(", " ").replace(")", " ").split()
            if len(values) >= 3:
                rows.append(tuple(float(value) for value in values[:3]))
    if len(rows) < 2:
        raise ValueError(f"{path} does not contain a raw U profile")
    return sorted(rows)


def read_simplefluid(patterns: list[str]) -> list[dict[str, float]]:
    paths = sorted({
        Path(path) for pattern in patterns for path in glob.glob(pattern)
    })
    if not paths:
        raise FileNotFoundError("no SimpleFluid rank CSV files matched")

    rows: list[dict[str, float]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"x", "y", "ux", "uy"}
            if reader.fieldnames is None or not required.issubset(
                    reader.fieldnames):
                raise ValueError(f"{path} must contain {sorted(required)}")
            for row in reader:
                rows.append({name: float(value) for name, value in row.items()})
    if not rows:
        raise ValueError("SimpleFluid CSV files contain no cells")
    return rows


def simplefluid_profile(
    cells: list[dict[str, float]], station: float
) -> tuple[float, list[tuple[float, float, float]]]:
    x = min({round(cell["x"], 12) for cell in cells},
            key=lambda value: abs(value - station))
    selected = [cell for cell in cells if abs(cell["x"] - x) < 5e-11]
    by_y: dict[float, list[dict[str, float]]] = {}
    for cell in selected:
        by_y.setdefault(round(cell["y"], 12), []).append(cell)

    rows = []
    for y, values in sorted(by_y.items()):
        count = len(values)
        rows.append((
            y,
            sum(value["ux"] for value in values) / count,
            sum(value["uy"] for value in values) / count,
        ))
    return x, rows


def interpolate(
    rows: list[tuple[float, float, float]], coordinate: float, component: int
) -> float:
    if coordinate < rows[0][0] or coordinate > rows[-1][0]:
        raise ValueError(f"y={coordinate} is outside the OpenFOAM profile")
    for left, right in zip(rows, rows[1:]):
        if left[0] <= coordinate <= right[0]:
            width = right[0] - left[0]
            if width == 0.0:
                return left[component]
            fraction = (coordinate - left[0]) / width
            return left[component] + fraction * (
                right[component] - left[component])
    return rows[-1][component]


def error_norms(
    reference: list[tuple[float, float, float]],
    result: list[tuple[float, float, float]],
    component: int,
) -> tuple[float, float]:
    errors = [
        row[component] - interpolate(reference, row[0], component)
        for row in result
        if reference[0][0] <= row[0] <= reference[-1][0]
    ]
    if not errors:
        raise ValueError("profiles have no overlapping samples")
    return (
        math.sqrt(sum(error * error for error in errors) / len(errors)),
        max(abs(error) for error in errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openfoam-case", type=Path, required=True)
    parser.add_argument("--simplefluid-glob", nargs="+", required=True)
    parser.add_argument("--max-l2", type=nonnegative_float)
    parser.add_argument("--max-linf", type=nonnegative_float)
    arguments = parser.parse_args()

    cells = read_simplefluid(arguments.simplefluid_glob)
    failures: list[str] = []
    for name, station in STATIONS.items():
        openfoam_path = latest_profile(arguments.openfoam_case, name)
        reference = read_openfoam(openfoam_path)
        actual_x, result = simplefluid_profile(cells, station)
        print(f"{name}: OpenFOAM={openfoam_path}, SimpleFluid x={actual_x:.12g} m")
        for component_name, component in (("ux", 1), ("uy", 2)):
            l2, linf = error_norms(reference, result, component)
            print(
                f"  {component_name}: samples={len(result)} "
                f"l2={l2:.12g} m/s linf={linf:.12g} m/s")
            if arguments.max_l2 is not None and l2 > arguments.max_l2:
                failures.append(
                    f"{name} {component_name} l2 exceeds {arguments.max_l2}")
            if arguments.max_linf is not None and linf > arguments.max_linf:
                failures.append(
                    f"{name} {component_name} linf exceeds {arguments.max_linf}")

    if failures:
        print("comparison failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
