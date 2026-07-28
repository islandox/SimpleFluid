#!/usr/bin/env python3
"""Map an axisymmetric OpenFOAM Shiri state onto SimpleFluid CSV checkpoints."""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path

from compare_profiles import read_openfoam_cells, rz_averages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openfoam-case", type=Path, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--simplefluid-template-glob", nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--openfoam-theta-cells", type=int, default=1)
    parser.add_argument("--reference-density", type=float, default=1.198)
    parser.add_argument("--thermal-expansion", type=float, default=1.0 / 290.0)
    parser.add_argument("--reference-temperature", type=float, default=290.0)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    arguments = parser.parse_args()

    if arguments.openfoam_theta_cells <= 0:
        raise ValueError("--openfoam-theta-cells must be positive")
    for name in (
        "reference_density",
        "thermal_expansion",
        "reference_temperature",
        "gravity_z",
    ):
        if not math.isfinite(getattr(arguments, name)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if arguments.reference_density <= 0.0:
        raise ValueError("--reference-density must be positive")

    openfoam_rz = rz_averages(
        read_openfoam_cells(arguments.openfoam_case, arguments.time),
        arguments.openfoam_theta_cells,
    )
    template_paths = sorted(
        {
            Path(path)
            for pattern in arguments.simplefluid_template_glob
            for path in glob.glob(pattern)
        }
    )
    if not template_paths:
        raise FileNotFoundError("no SimpleFluid checkpoint templates matched")

    arguments.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for rank, template_path in enumerate(template_paths):
        output_path = Path(f"{arguments.output_prefix}_rank{rank}.csv")
        with template_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError(f"{template_path} has no CSV header")
            required = {
                "r",
                "theta",
                "z",
                "temperature",
                "ur",
                "utheta",
                "uz",
                "pressure",
                "k",
                "epsilon",
                "nut",
            }
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"{template_path} is missing columns {sorted(missing)}"
                )
            with output_path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    key = (round(float(row["r"]), 6), round(float(row["z"]), 6))
                    if key not in openfoam_rz:
                        raise ValueError(
                            "OpenFOAM and SimpleFluid R-Z coordinates differ at "
                            f"r={row['r']}, z={row['z']}"
                        )
                    reference = openfoam_rz[key]
                    temperature = reference["temperature"]
                    relative_density = 1.0 - arguments.thermal_expansion * (
                        temperature - arguments.reference_temperature
                    )
                    # SimpleFluid advances the Boussinesq dynamic pressure
                    # after removing the reference hydrostatic head.  Thus
                    # OpenFOAM's p = p_rgh + rhok*gh becomes
                    # p' = p_rgh + (rhok - 1)*gh.  OpenFOAM's incompressible
                    # RANS pressure also absorbs the isotropic 2k/3 Reynolds
                    # stress, while SimpleFluid keeps -grad(2k/3) explicit.
                    kinematic_pressure = (
                        reference["p_rgh"]
                        + (relative_density - 1.0)
                        * arguments.gravity_z
                        * reference["z"]
                        - (2.0 / 3.0) * reference["k"]
                    )
                    replacements = {
                        "temperature": temperature,
                        "ur": reference["ur"],
                        "utheta": reference["utheta"],
                        "uz": reference["uz"],
                        "pressure": (
                            arguments.reference_density * kinematic_pressure
                        ),
                        "k": reference["k"],
                        "epsilon": reference["epsilon"],
                        "nut": reference["nut"],
                    }
                    if "alphat" in row:
                        replacements["alphat"] = reference["alphat"]
                    for field, value in replacements.items():
                        row[field] = f"{value:.17g}"
                    writer.writerow(row)
                    total_rows += 1

    print(
        f"wrote {total_rows} cells from OpenFOAM time {arguments.time:g} "
        f"to {len(template_paths)} SimpleFluid checkpoint files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
