#!/usr/bin/env python3
"""Compare matched-time turbulent near-pipe profiles."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from pathlib import Path


ProfileRow = dict[str, float]
REQUIRED_OPENFOAM_FIELDS = {"T", "U", "k", "epsilon", "nut", "alphat"}
OPENFOAM_SAMPLE_X = 0.0739104
OPENFOAM_SAMPLE_Y = 0.0306147
OPENFOAM_SAMPLE_RADIUS = math.hypot(OPENFOAM_SAMPLE_X, OPENFOAM_SAMPLE_Y)
OPENFOAM_SAMPLE_THETA = math.atan2(OPENFOAM_SAMPLE_Y, OPENFOAM_SAMPLE_X)
TURBULENT_PRANDTL = 0.85


def profile_fields(path: Path) -> list[str]:
    prefix = "nearPipe_"
    if not path.stem.startswith(prefix):
        return []
    return path.stem.removeprefix(prefix).split("_")


def latest_openfoam_profile(case: Path, expected_time: float) -> Path:
    candidates = [
        path
        for path in case.glob("postProcessing/profiles/*/nearPipe_*.xy")
        if (
            len(profile_fields(path)) == len(REQUIRED_OPENFOAM_FIELDS)
            and set(profile_fields(path)) == REQUIRED_OPENFOAM_FIELDS
        )
    ]
    if not candidates:
        raise FileNotFoundError(
            "no turbulent nearPipe profile containing "
            "T/U/k/epsilon/nut/alphat below "
            f"{case / 'postProcessing/profiles'}"
        )

    def time_value(path: Path) -> float:
        try:
            return float(path.parent.name)
        except ValueError:
            return -math.inf

    latest_time = max(time_value(path) for path in candidates)
    latest = [
        path for path in candidates if math.isclose(time_value(path), latest_time)
    ]
    if len(latest) != 1:
        raise ValueError(
            f"ambiguous OpenFOAM profiles at time {latest_time}: "
            + ", ".join(str(path) for path in latest)
        )
    if not math.isclose(latest_time, expected_time, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"latest OpenFOAM profile time is {latest_time}, "
            f"expected {expected_time}"
        )
    return latest[0]


def read_openfoam(path: Path) -> list[ProfileRow]:
    rows: list[ProfileRow] = []
    fields = profile_fields(path)
    scalar_names = {
        "T": "temperature",
        "k": "k",
        "epsilon": "epsilon",
        "nut": "nut",
        "alphat": "alphat",
    }
    for field in fields:
        if field != "U" and field not in scalar_names:
            raise ValueError(f"unsupported OpenFOAM profile field {field!r}")

    expected_columns = 1 + sum(3 if field == "U" else 1 for field in fields)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.replace("(", " ").replace(")", " ").split()
            if len(tokens) != expected_columns:
                raise ValueError(
                    f"{path}:{line_number} has {len(tokens)} columns; "
                    f"expected {expected_columns}"
                )
            try:
                values = [float(value) for value in tokens]
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number} contains a non-numeric value"
                ) from error
            if any(not math.isfinite(value) for value in values):
                raise ValueError(
                    f"{path}:{line_number} contains a non-finite profile value"
                )
            row: ProfileRow = {"z": values[0]}
            column = 1
            for field in fields:
                if field == "U":
                    ux, uy, uz = values[column : column + 3]
                    row["ur"] = (
                        ux * math.cos(OPENFOAM_SAMPLE_THETA)
                        + uy * math.sin(OPENFOAM_SAMPLE_THETA)
                    )
                    row["utheta"] = (
                        -ux * math.sin(OPENFOAM_SAMPLE_THETA)
                        + uy * math.cos(OPENFOAM_SAMPLE_THETA)
                    )
                    row["uz"] = uz
                    column += 3
                else:
                    row[scalar_names[field]] = values[column]
                    column += 1
            if column != len(values):
                raise ValueError(
                    f"{path} row has {len(values)} values; parsed {column}"
                )
            rows.append(row)

            row["turbulent_diffusivity"] = row["nut"] / TURBULENT_PRANDTL

    required = {
        "z", "temperature", "ur", "utheta", "uz", "k", "epsilon", "nut",
        "alphat", "turbulent_diffusivity",
    }
    if len(rows) < 2 or any(not required.issubset(row) for row in rows):
        raise ValueError(f"{path} does not contain the required turbulent profile")
    rows.sort(key=lambda row: row["z"])
    if any(
        right["z"] <= left["z"] for left, right in zip(rows, rows[1:])
    ):
        raise ValueError(f"{path} axial coordinates are not strictly increasing")
    for row in rows:
        if not 200.0 <= row["temperature"] <= 500.0:
            raise ValueError(f"{path} contains an implausible temperature")
        if any(abs(row[field]) > 1.0e3 for field in ("ur", "utheta", "uz")):
            raise ValueError(f"{path} contains an implausible velocity")
        if any(
            row[field] < 0.0 or row[field] > 1.0e6
            for field in ("k", "epsilon", "nut", "alphat")
        ):
            raise ValueError(f"{path} contains implausible turbulence data")
    return rows


def read_simplefluid(
    patterns: list[str],
    target_radius: float,
    expected_ranks: int | None,
    expected_cells: int | None,
    expected_theta_cells: int | None,
    expected_radial_cells: int | None,
    expected_axial_cells: int | None,
) -> tuple[
    tuple[float, float],
    list[ProfileRow],
    float,
    ProfileRow,
    tuple[float, float],
]:
    paths = sorted({Path(path) for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError("no SimpleFluid rank CSV files matched")
    rank_pattern = re.compile(r"_rank([0-9]+)[.]csv$")
    ranks = []
    for path in paths:
        match = rank_pattern.search(path.name)
        if match is None:
            raise ValueError(f"cannot determine rank from {path}")
        ranks.append(int(match.group(1)))
    if len(set(ranks)) != len(ranks) or sorted(ranks) != list(range(len(ranks))):
        raise ValueError(f"SimpleFluid rank CSVs are not consecutive: {sorted(ranks)}")
    if expected_ranks is not None and len(paths) != expected_ranks:
        raise ValueError(
            f"expected {expected_ranks} SimpleFluid rank CSVs, found {len(paths)}"
        )

    required = {
        "r", "theta", "z", "temperature", "ur", "utheta", "uz",
        "k", "epsilon", "nut", "alphat",
    }
    cells: list[ProfileRow] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"{path} has no CSV header")
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"{path} has duplicate CSV column names")
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"{path} is missing CSV columns: {sorted(missing)}"
                )
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{path}:{line_number} is a malformed CSV row")
                try:
                    cell = {
                        name: float(value)
                        for name, value in row.items()
                        if name is not None and value is not None
                    }
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line_number} contains a non-numeric value"
                    ) from error
                cell["turbulent_diffusivity"] = (
                    cell["nut"] / TURBULENT_PRANDTL
                )
                cells.append(cell)
    if not cells:
        raise ValueError("SimpleFluid CSV files contain no cells")
    if expected_cells is not None and len(cells) != expected_cells:
        raise ValueError(
            f"expected {expected_cells} SimpleFluid cells, found {len(cells)}"
        )
    for path_row, cell in enumerate(cells, start=1):
        if any(not math.isfinite(value) for value in cell.values()):
            raise ValueError(
                f"SimpleFluid cell row {path_row} contains a non-finite value"
            )
        if not 200.0 <= cell["temperature"] <= 500.0:
            raise ValueError(
                f"SimpleFluid cell row {path_row} has an implausible temperature"
            )
        if any(abs(cell[field]) > 1.0e3 for field in ("ur", "utheta", "uz")):
            raise ValueError(
                f"SimpleFluid cell row {path_row} has an implausible velocity"
            )
        if any(
            cell[field] < 0.0 or cell[field] > 1.0e6
            for field in ("k", "epsilon", "nut", "alphat")
        ):
            raise ValueError(
                f"SimpleFluid cell row {path_row} has implausible turbulence data"
            )

    if any(not required.issubset(cell) for cell in cells):
        raise ValueError(
            "SimpleFluid profiles do not contain standard k-epsilon fields"
        )

    radii = sorted({round(cell["r"], 12) for cell in cells})
    axial_coordinates = sorted({round(cell["z"], 12) for cell in cells})
    theta_coordinates = sorted({round(cell["theta"], 12) for cell in cells})
    if expected_radial_cells is not None and len(radii) != expected_radial_cells:
        raise ValueError(
            f"expected {expected_radial_cells} radial cells, found {len(radii)}"
        )
    if expected_axial_cells is not None and len(axial_coordinates) != expected_axial_cells:
        raise ValueError(
            f"expected {expected_axial_cells} axial cells, "
            f"found {len(axial_coordinates)}"
        )
    if (
        expected_theta_cells is not None
        and len(theta_coordinates) != expected_theta_cells
    ):
        raise ValueError(
            f"expected {expected_theta_cells} theta cells, "
            f"found {len(theta_coordinates)}"
        )
    coordinate_keys = {
        (
            round(cell["r"], 12),
            round(cell["theta"], 12),
            round(cell["z"], 12),
        )
        for cell in cells
    }
    if len(coordinate_keys) != len(cells):
        raise ValueError("SimpleFluid CSV files contain duplicate cell coordinates")
    if target_radius < radii[0] or target_radius > radii[-1]:
        raise ValueError(
            f"target radius {target_radius} lies outside SimpleFluid cell centres"
        )
    lower_radius = max(radius for radius in radii if radius <= target_radius)
    upper_radius = min(radius for radius in radii if radius >= target_radius)
    averaged_fields = (
        "temperature", "ur", "utheta", "uz", "k", "epsilon", "nut",
        "turbulent_diffusivity",
    )

    theta_spreads: ProfileRow = {field: 0.0 for field in averaged_fields}

    def theta_averages(radius: float) -> dict[float, ProfileRow]:
        selected = [
            cell for cell in cells if abs(cell["r"] - radius) < 5e-11
        ]
        by_z: dict[float, list[ProfileRow]] = {}
        for cell in selected:
            by_z.setdefault(round(cell["z"], 12), []).append(cell)
        result: dict[float, ProfileRow] = {}
        for z, values in by_z.items():
            if (
                expected_theta_cells is not None
                and len(values) != expected_theta_cells
            ):
                raise ValueError(
                    f"expected {expected_theta_cells} theta samples at "
                    f"r={radius}, z={z}; found {len(values)}"
                )
            row: ProfileRow = {"z": z}
            for field in averaged_fields:
                row[field] = sum(value[field] for value in values) / len(values)
                theta_spreads[field] = max(
                    theta_spreads[field],
                    max(value[field] for value in values)
                    - min(value[field] for value in values),
                )
            result[z] = row
        return result

    lower_rows = theta_averages(lower_radius)
    upper_rows = theta_averages(upper_radius)
    if lower_rows.keys() != upper_rows.keys():
        raise ValueError("SimpleFluid radial layers have different axial samples")
    fraction = (
        0.0
        if lower_radius == upper_radius
        else (target_radius - lower_radius) / (upper_radius - lower_radius)
    )
    rows: list[ProfileRow] = []
    for z in sorted(lower_rows):
        row: ProfileRow = {"z": z}
        for field in averaged_fields:
            row[field] = (
                lower_rows[z][field]
                + fraction * (upper_rows[z][field] - lower_rows[z][field])
            )
        rows.append(row)

    selected_radii = {lower_radius, upper_radius}
    maximum_azimuthal_velocity = max(
        abs(cell["utheta"])
        for cell in cells
        if round(cell["r"], 12) in selected_radii
    )
    temperature_range = (
        min(cell["temperature"] for cell in cells),
        max(cell["temperature"] for cell in cells),
    )
    return (
        (lower_radius, upper_radius),
        rows,
        maximum_azimuthal_velocity,
        theta_spreads,
        temperature_range,
    )


def interpolate(rows: list[ProfileRow], z: float, field: str) -> float:
    if z < rows[0]["z"] or z > rows[-1]["z"]:
        raise ValueError(f"z={z} lies outside the OpenFOAM sample range")
    for left, right in zip(rows, rows[1:]):
        if left["z"] <= z <= right["z"]:
            fraction = (z - left["z"]) / (right["z"] - left["z"])
            return left[field] + fraction * (right[field] - left[field])
    return rows[-1][field]


def error_norms(
    reference: list[ProfileRow],
    result: list[ProfileRow],
    field: str,
    reference_offset: float = 0.0,
) -> tuple[float, float, float]:
    pairs = [
        (row["z"], row[field], interpolate(reference, row["z"], field))
        for row in result
        if reference[0]["z"] <= row["z"] <= reference[-1]["z"]
    ]
    if len(pairs) < 2:
        raise ValueError("profiles have fewer than two overlapping axial samples")
    span = pairs[-1][0] - pairs[0][0]
    if span <= 0.0:
        raise ValueError("profile overlap has no positive axial span")

    error_integral = 0.0
    reference_integral = 0.0
    errors = []
    for z, value, expected in pairs:
        errors.append(value - expected)
    for left, right in zip(pairs, pairs[1:]):
        dz = right[0] - left[0]
        left_error = left[1] - left[2]
        right_error = right[1] - right[2]
        error_integral += 0.5 * (
            left_error * left_error + right_error * right_error
        ) * dz
        left_reference = left[2] - reference_offset
        right_reference = right[2] - reference_offset
        reference_integral += 0.5 * (
            left_reference * left_reference
            + right_reference * right_reference
        ) * dz
    axial_rms = math.sqrt(error_integral / span)
    reference_rms = math.sqrt(reference_integral / span)
    relative_rms = (
        axial_rms / reference_rms if reference_rms > 0.0 else math.inf
    )
    return axial_rms, max(abs(error) for error in errors), relative_rms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openfoam-case", type=Path, required=True)
    parser.add_argument("--simplefluid-glob", nargs="+", required=True)
    parser.add_argument("--radius", type=float, default=OPENFOAM_SAMPLE_RADIUS)
    parser.add_argument("--expected-time", type=float, default=0.4)
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--expected-cells", type=int, default=80000)
    parser.add_argument("--expected-theta-cells", type=int, default=20)
    parser.add_argument("--expected-radial-cells", type=int, default=40)
    parser.add_argument("--expected-axial-cells", type=int, default=100)
    arguments = parser.parse_args()

    if not math.isclose(
        arguments.radius,
        OPENFOAM_SAMPLE_RADIUS,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError(
            f"requested radius {arguments.radius} does not match the fixed "
            f"OpenFOAM sample radius {OPENFOAM_SAMPLE_RADIUS}"
        )
    openfoam_path = latest_openfoam_profile(
        arguments.openfoam_case, arguments.expected_time
    )
    openfoam = read_openfoam(openfoam_path)
    (
        radial_bracket,
        simplefluid,
        maximum_azimuthal_velocity,
        theta_spreads,
        temperature_range,
    ) = read_simplefluid(
        arguments.simplefluid_glob,
        arguments.radius,
        arguments.expected_ranks,
        arguments.expected_cells,
        arguments.expected_theta_cells,
        arguments.expected_radial_cells,
        arguments.expected_axial_cells,
    )

    print(f"OpenFOAM profile: {openfoam_path}")
    print(
        "SimpleFluid radial interpolation: "
        f"target={arguments.radius:.12g} m, "
        f"bracket=[{radial_bracket[0]:.12g}, "
        f"{radial_bracket[1]:.12g}] m, samples={len(simplefluid)}"
    )
    print(
        "SimpleFluid selected-layer max |u_theta|="
        f"{maximum_azimuthal_velocity:.12g} m/s"
    )
    print(
        "SimpleFluid global temperature range: "
        f"[{temperature_range[0]:.12g}, {temperature_range[1]:.12g}] K"
    )
    if (
        temperature_range[0] < 290.0 - 1.0e-3
        or temperature_range[1] > 360.0 + 1.0e-3
    ):
        print(
            "WARNING: SimpleFluid temperature violates the [290, 360] K "
            "initial/boundary bounds."
        )
    for name, field, units, reference_offset in (
        ("temperature", "temperature", "K", 290.0),
        ("radial velocity", "ur", "m/s", 0.0),
        ("azimuthal velocity", "utheta", "m/s", 0.0),
        ("axial velocity", "uz", "m/s", 0.0),
        ("turbulent kinetic energy", "k", "m2/s2", 0.0),
        ("dissipation rate", "epsilon", "m2/s3", 0.0),
        ("turbulent viscosity", "nut", "m2/s", 0.0),
        (
            "interior turbulent thermal diffusivity (nut/Prt)",
            "turbulent_diffusivity",
            "m2/s",
            0.0,
        ),
    ):
        axial_rms, linf, relative_rms = error_norms(
            openfoam, simplefluid, field, reference_offset
        )
        print(
            f"{name}: axial_rms={axial_rms:.12g} {units}, "
            f"linf={linf:.12g} {units}, "
            f"relative_axial_rms={relative_rms:.6%}, "
            f"max_theta_spread={theta_spreads[field]:.12g} {units}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
