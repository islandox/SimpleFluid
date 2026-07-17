#!/usr/bin/env python3
"""Compare near-pipe axial profiles from OpenFOAM and SimpleFluid."""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path


def latest_openfoam_profile(case: Path) -> Path:
    candidates = list(case.glob("postProcessing/profiles/*/nearPipe_T_U.xy"))
    if not candidates:
        raise FileNotFoundError(
            f"no nearPipe_T_U.xy below {case / 'postProcessing/profiles'}")

    def time_value(path: Path) -> float:
        try:
            return float(path.parent.name)
        except ValueError:
            return -math.inf

    return max(candidates, key=time_value)


def read_openfoam(path: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    theta = math.pi / 8.0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.replace("(", " ").replace(")", " ").split()
            if len(values) < 5:
                continue
            z, temperature, ux, uy, uz = map(float, values[:5])
            ur = ux * math.cos(theta) + uy * math.sin(theta)
            rows.append((z, temperature, ur, uz))
    if len(rows) < 2:
        raise ValueError(f"{path} does not contain a raw T/U set profile")
    return sorted(rows)


def read_simplefluid(
    patterns: list[str], target_radius: float
) -> tuple[float, list[tuple[float, float, float, float]]]:
    paths = sorted({Path(path) for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError("no SimpleFluid rank CSV files matched")

    cells: list[dict[str, float]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                cells.append({name: float(value) for name, value in row.items()})
    if not cells:
        raise ValueError("SimpleFluid CSV files contain no cells")

    radius = min({round(cell["r"], 12) for cell in cells},
                 key=lambda value: abs(value - target_radius))
    selected = [cell for cell in cells if abs(cell["r"] - radius) < 5e-11]
    by_z: dict[float, list[dict[str, float]]] = {}
    for cell in selected:
        by_z.setdefault(round(cell["z"], 12), []).append(cell)

    rows = []
    for z, values in sorted(by_z.items()):
        count = len(values)
        rows.append((
            z,
            sum(value["temperature"] for value in values) / count,
            sum(value["ur"] for value in values) / count,
            sum(value["uz"] for value in values) / count,
        ))
    return radius, rows


def interpolate(
    rows: list[tuple[float, float, float, float]], z: float, component: int
) -> float:
    if z < rows[0][0] or z > rows[-1][0]:
        raise ValueError(f"z={z} lies outside the OpenFOAM sample range")
    for left, right in zip(rows, rows[1:]):
        if left[0] <= z <= right[0]:
            fraction = (z - left[0]) / (right[0] - left[0])
            return left[component] + fraction * (right[component] - left[component])
    return rows[-1][component]


def error_norms(
    reference: list[tuple[float, float, float, float]],
    result: list[tuple[float, float, float, float]],
    component: int,
) -> tuple[float, float]:
    errors = [row[component] - interpolate(reference, row[0], component)
              for row in result if reference[0][0] <= row[0] <= reference[-1][0]]
    if not errors:
        raise ValueError("profiles have no overlapping axial samples")
    return (math.sqrt(sum(error * error for error in errors) / len(errors)),
            max(abs(error) for error in errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openfoam-case", type=Path, required=True)
    parser.add_argument("--simplefluid-glob", nargs="+", required=True)
    parser.add_argument("--radius", type=float, default=0.080)
    arguments = parser.parse_args()

    openfoam_path = latest_openfoam_profile(arguments.openfoam_case)
    openfoam = read_openfoam(openfoam_path)
    radius, simplefluid = read_simplefluid(
        arguments.simplefluid_glob, arguments.radius)

    print(f"OpenFOAM profile: {openfoam_path}")
    print(f"SimpleFluid radial layer: r={radius:.12g} m, samples={len(simplefluid)}")
    for name, component, units in (
        ("temperature", 1, "K"),
        ("radial velocity", 2, "m/s"),
        ("axial velocity", 3, "m/s"),
    ):
        l2, linf = error_norms(openfoam, simplefluid, component)
        print(f"{name}: l2={l2:.12g} {units}, linf={linf:.12g} {units}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
