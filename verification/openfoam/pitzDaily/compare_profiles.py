#!/usr/bin/env python3
"""Compare pitzDaily velocity profiles from OpenFOAM and SimpleFluid."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


STATIONS = {"x050": 0.050, "x100": 0.100, "x200": 0.200}
COMPONENTS = {"ux": 1, "uy": 2}


class ToleranceManifestError(ValueError):
    """Report an invalid or not-yet-qualified acceptance manifest."""


def nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError(
            "tolerances must be finite and non-negative")
    return result


def _manifest_nonnegative(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ToleranceManifestError(f"{context} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ToleranceManifestError(
            f"{context} must be a number") from error
    if not math.isfinite(result) or result < 0.0:
        raise ToleranceManifestError(
            f"{context} must be finite and non-negative")
    return result


def _manifest_positive(value: Any, context: str) -> float:
    result = _manifest_nonnegative(value, context)
    if result <= 0.0:
        raise ToleranceManifestError(f"{context} must be positive")
    return result


def _manifest_positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ToleranceManifestError(f"{context} must be a positive integer")
    return value


def _validate_source_checksums(
    manifest_path: Path, qualification: dict[str, Any]
) -> None:
    """Require qualified manifests to authenticate their retained sources."""
    checksums = qualification.get("source_files_sha256")
    if not isinstance(checksums, dict) or not checksums:
        raise ToleranceManifestError(
            f"{manifest_path}: qualified manifests require source_files_sha256")

    root = manifest_path.parent.resolve()
    for relative_name, expected_digest in checksums.items():
        if not isinstance(relative_name, str) or not relative_name:
            raise ToleranceManifestError(
                f"{manifest_path}: checksum paths must be non-empty strings")
        if (not isinstance(expected_digest, str)
                or len(expected_digest) != 64
                or any(character not in "0123456789abcdefABCDEF"
                       for character in expected_digest)):
            raise ToleranceManifestError(
                f"{manifest_path}: invalid SHA-256 for {relative_name!r}")

        source_path = (root / relative_name).resolve()
        try:
            source_path.relative_to(root)
        except ValueError as error:
            raise ToleranceManifestError(
                f"{manifest_path}: checksum path escapes its reference "
                f"directory: {relative_name!r}") from error
        try:
            actual_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        except OSError as error:
            raise ToleranceManifestError(
                f"{manifest_path}: cannot read qualified source "
                f"{relative_name!r}: {error}") from error
        if actual_digest != expected_digest.lower():
            raise ToleranceManifestError(
                f"{manifest_path}: SHA-256 mismatch for {relative_name!r}")


def load_tolerance_manifest(
    path: Path, required_scope: str | None = None
) -> dict[str, Any]:
    """Load and validate a station/component acceptance manifest."""
    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ToleranceManifestError(
            f"cannot load tolerance manifest {path}: {error}") from error

    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ToleranceManifestError(
            f"{path} must be a schema_version 1 JSON object")

    qualification = document.get("qualification")
    if not isinstance(qualification, dict):
        raise ToleranceManifestError(
            f"{path} must contain a qualification object")
    status = qualification.get("status")
    scope = qualification.get("scope")
    if status != "qualified":
        reason = qualification.get("reason", "no reason recorded")
        raise ToleranceManifestError(
            f"{path} is not qualified ({status!r}): {reason}")
    if required_scope is not None and scope != required_scope:
        raise ToleranceManifestError(
            f"{path} has scope {scope!r}; expected {required_scope!r}")
    _validate_source_checksums(path, qualification)

    simplefluid_run = None
    reference_definition = document.get("reference_definition")
    if scope == "physical_reference":
        if not isinstance(reference_definition, dict):
            raise ToleranceManifestError(
                f"{path}: qualified physical references require a "
                "reference_definition object")
        for field_name in ("openfoam_case", "simplefluid_source_revision"):
            value = reference_definition.get(field_name)
            if (not isinstance(value, str) or not value.strip()
                    or value.strip().lower() == "unqualified"):
                raise ToleranceManifestError(
                    f"{path}: reference_definition.{field_name} must record "
                    "qualified provenance")
        run_document = reference_definition.get("simplefluid_run")
        if not isinstance(run_document, dict):
            raise ToleranceManifestError(
                f"{path}: reference_definition.simplefluid_run is required")
        simplefluid_run = {
            "mesh_divisor": _manifest_positive_integer(
                run_document.get("mesh_divisor"),
                "reference_definition.simplefluid_run.mesh_divisor"),
            "steps": _manifest_positive_integer(
                run_document.get("steps"),
                "reference_definition.simplefluid_run.steps"),
            "dt_s": _manifest_positive(
                run_document.get("dt_s"),
                "reference_definition.simplefluid_run.dt_s"),
            "mpi_ranks": _manifest_positive_integer(
                run_document.get("mpi_ranks"),
                "reference_definition.simplefluid_run.mpi_ranks"),
        }

    stations = document.get("stations")
    if not isinstance(stations, dict):
        raise ToleranceManifestError(f"{path} must contain a stations object")
    missing = set(STATIONS) - set(stations)
    extra = set(stations) - set(STATIONS)
    if missing or extra:
        raise ToleranceManifestError(
            f"{path} station keys differ from the required set: "
            f"missing={sorted(missing)}, extra={sorted(extra)}")

    normalized: dict[str, Any] = {}
    for station_name, expected_x in STATIONS.items():
        station = stations[station_name]
        if not isinstance(station, dict):
            raise ToleranceManifestError(
                f"{path}: stations.{station_name} must be an object")
        x_m = _manifest_nonnegative(
            station.get("x_m"), f"stations.{station_name}.x_m")
        if not math.isclose(x_m, expected_x, rel_tol=0.0, abs_tol=1.0e-12):
            raise ToleranceManifestError(
                f"{path}: stations.{station_name}.x_m is {x_m}, "
                f"expected {expected_x}")
        max_offset = _manifest_nonnegative(
            station.get("max_station_offset_m"),
            f"stations.{station_name}.max_station_offset_m")
        minimum_samples = _manifest_positive_integer(
            station.get("minimum_samples"),
            f"stations.{station_name}.minimum_samples")
        if minimum_samples < 2:
            raise ToleranceManifestError(
                f"{path}: stations.{station_name}.minimum_samples must be "
                "at least two")
        minimum_span_fraction = _manifest_positive(
            station.get("minimum_reference_span_fraction"),
            f"stations.{station_name}.minimum_reference_span_fraction")
        if minimum_span_fraction > 1.0:
            raise ToleranceManifestError(
                f"{path}: stations.{station_name}."
                "minimum_reference_span_fraction cannot exceed one")
        components = station.get("components")
        if not isinstance(components, dict) or set(components) != set(COMPONENTS):
            raise ToleranceManifestError(
                f"{path}: stations.{station_name}.components must contain "
                f"exactly {sorted(COMPONENTS)}")

        normalized_components: dict[str, dict[str, float]] = {}
        for component_name in COMPONENTS:
            component = components[component_name]
            if not isinstance(component, dict):
                raise ToleranceManifestError(
                    f"{path}: {station_name}.{component_name} must be an object")
            max_l2 = _manifest_nonnegative(
                component.get("max_l2_m_per_s"),
                f"stations.{station_name}.{component_name}.max_l2_m_per_s")
            max_linf = _manifest_nonnegative(
                component.get("max_linf_m_per_s"),
                f"stations.{station_name}.{component_name}.max_linf_m_per_s")
            if max_l2 > max_linf:
                raise ToleranceManifestError(
                    f"{path}: {station_name}.{component_name} max_l2 exceeds "
                    "max_linf")
            normalized_components[component_name] = {
                "max_l2": max_l2,
                "max_linf": max_linf,
            }
        normalized[station_name] = {
            "x_m": x_m,
            "max_station_offset_m": max_offset,
            "minimum_samples": minimum_samples,
            "minimum_reference_span_fraction": minimum_span_fraction,
            "components": normalized_components,
        }

    return {
        "path": path,
        "qualification": qualification,
        "reference_definition": reference_definition,
        "simplefluid_run": simplefluid_run,
        "stations": normalized,
    }


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
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.replace("(", " ").replace(")", " ").split()
            if len(values) < 3:
                raise ValueError(
                    f"{path}:{line_number} must contain y, ux, and uy")
            try:
                row = tuple(float(value) for value in values[:3])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number} contains a non-numeric value") \
                    from error
            if not all(math.isfinite(value) for value in row):
                raise ValueError(
                    f"{path}:{line_number} contains a non-finite value")
            rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"{path} does not contain a raw U profile")
    rows.sort()
    if any(left[0] >= right[0] for left, right in zip(rows, rows[1:])):
        raise ValueError(f"{path} must contain unique increasing y samples")
    return rows


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
            for line_number, row in enumerate(reader, start=2):
                try:
                    parsed = {
                        name: float(row[name]) for name in required
                    }
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{path}:{line_number} contains a non-numeric value") \
                        from error
                if not all(math.isfinite(value) for value in parsed.values()):
                    raise ValueError(
                        f"{path}:{line_number} contains a non-finite value")
                rows.append(parsed)
    if not rows:
        raise ValueError("SimpleFluid CSV files contain no cells")
    return rows


def simplefluid_profile(
    cells: list[dict[str, float]], station: float
) -> tuple[float, list[tuple[float, float, float]]]:
    x = min({round(cell["x"], 12) for cell in cells},
            key=lambda value: abs(value - station))
    selected = [cell for cell in cells if abs(cell["x"] - x) < 5e-11]
    expected_samples = None
    if len(selected) < 2:
        expected_samples = pitz_daily_profile_sample_count(len(cells))
        if expected_samples is None:
            raise ValueError(
                "cannot identify a pitzDaily cell layer at the requested station")
        selected = sorted(
            cells, key=lambda cell: abs(cell["x"] - station)
        )[:expected_samples]
        x = sum(cell["x"] for cell in selected) / len(selected)
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
    if expected_samples is not None and len(rows) != expected_samples:
        raise ValueError(
            "graded pitzDaily layer does not contain the expected transverse samples")
    return x, rows


def pitz_daily_profile_sample_count(cell_count: int) -> int | None:
    """Recover the transverse cell count from a supported pitzDaily mesh."""
    for divisor in range(1, 1001):
        upstream_x = max(1, (18 + divisor - 1) // divisor)
        main_x = max(1, (180 + divisor - 1) // divisor)
        outlet_x = max(1, (25 + divisor - 1) // divisor)
        upper_y = max(1, (30 + divisor - 1) // divisor)
        lower_y = max(1, (27 + divisor - 1) // divisor)
        expected_cells = (
            upstream_x * upper_y
            + main_x * (lower_y + upper_y)
            + outlet_x * (lower_y + upper_y)
        )
        if expected_cells == cell_count:
            return lower_y + upper_y
    return None


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
    if len(result) < 2:
        raise ValueError("profiles require at least two result samples")
    if any(row[0] < reference[0][0] or row[0] > reference[-1][0]
           for row in result):
        raise ValueError(
            "result profile contains samples outside the OpenFOAM y range")
    errors = [
        row[component] - interpolate(reference, row[0], component)
        for row in result
    ]
    if not all(math.isfinite(error) for error in errors):
        raise ValueError("profile comparison produced a non-finite error")
    return (
        math.sqrt(sum(error * error for error in errors) / len(errors)),
        max(abs(error) for error in errors),
    )


def collect_metrics(
    openfoam_case: Path, simplefluid_patterns: list[str]
) -> dict[str, dict[str, Any]]:
    """Read all configured stations and compute component error norms."""
    cells = read_simplefluid(simplefluid_patterns)
    metrics: dict[str, dict[str, Any]] = {}
    for station_name, station_x in STATIONS.items():
        openfoam_path = latest_profile(openfoam_case, station_name)
        reference = read_openfoam(openfoam_path)
        actual_x, result = simplefluid_profile(cells, station_x)
        result_span = result[-1][0] - result[0][0]
        reference_span = reference[-1][0] - reference[0][0]
        if reference_span <= 0.0:
            raise ValueError(
                f"{openfoam_path} must span more than one y coordinate")
        station_metrics: dict[str, Any] = {
            "openfoam_path": openfoam_path,
            "actual_x": actual_x,
            "samples": len(result),
            "reference_span_fraction": result_span / reference_span,
        }
        for component_name, component in COMPONENTS.items():
            l2, linf = error_norms(reference, result, component)
            station_metrics[component_name] = {"l2": l2, "linf": linf}
        metrics[station_name] = station_metrics
    return metrics


def evaluate_metrics(
    metrics: dict[str, dict[str, Any]],
    tolerance_manifest: dict[str, Any] | None = None,
    max_l2: float | None = None,
    max_linf: float | None = None,
) -> list[str]:
    """Return all station/component acceptance failures."""
    failures: list[str] = []
    for station_name, station_x in STATIONS.items():
        station_metrics = metrics[station_name]
        station_limits = None
        if tolerance_manifest is not None:
            station_limits = tolerance_manifest["stations"][station_name]
            offset = abs(station_metrics["actual_x"] - station_x)
            if (not math.isfinite(offset)
                    or offset > station_limits["max_station_offset_m"]):
                failures.append(
                    f"{station_name} sample offset {offset:.12g} m exceeds "
                    f"{station_limits['max_station_offset_m']:.12g} m")
            if station_metrics["samples"] < station_limits["minimum_samples"]:
                failures.append(
                    f"{station_name} has {station_metrics['samples']} samples; "
                    f"requires at least {station_limits['minimum_samples']}")
            span_fraction = station_metrics["reference_span_fraction"]
            if (not math.isfinite(span_fraction)
                    or span_fraction
                    < station_limits["minimum_reference_span_fraction"]):
                failures.append(
                    f"{station_name} spans {span_fraction:.12g} of the "
                    "OpenFOAM y range; requires at least "
                    f"{station_limits['minimum_reference_span_fraction']:.12g}")

        for component_name in COMPONENTS:
            component_metrics = station_metrics[component_name]
            l2_limit = max_l2
            linf_limit = max_linf
            if station_limits is not None:
                component_limits = station_limits["components"][component_name]
                l2_limit = component_limits["max_l2"]
                linf_limit = component_limits["max_linf"]
            if not math.isfinite(component_metrics["l2"]):
                failures.append(
                    f"{station_name} {component_name} l2 is non-finite")
            elif l2_limit is not None and component_metrics["l2"] > l2_limit:
                failures.append(
                    f"{station_name} {component_name} l2 "
                    f"{component_metrics['l2']:.12g} m/s exceeds "
                    f"{l2_limit:.12g} m/s")
            if not math.isfinite(component_metrics["linf"]):
                failures.append(
                    f"{station_name} {component_name} linf is non-finite")
            elif (linf_limit is not None
                  and component_metrics["linf"] > linf_limit):
                failures.append(
                    f"{station_name} {component_name} linf "
                    f"{component_metrics['linf']:.12g} m/s exceeds "
                    f"{linf_limit:.12g} m/s")
    return failures


def print_metrics(
    metrics: dict[str, dict[str, Any]],
    tolerance_manifest: dict[str, Any] | None = None,
) -> None:
    """Print station/component metrics and active limits."""
    for station_name in STATIONS:
        station_metrics = metrics[station_name]
        print(
            f"{station_name}: OpenFOAM={station_metrics['openfoam_path']}, "
            f"SimpleFluid x={station_metrics['actual_x']:.12g} m")
        for component_name in COMPONENTS:
            component_metrics = station_metrics[component_name]
            line = (
                f"  {component_name}: samples={station_metrics['samples']} "
                f"l2={component_metrics['l2']:.12g} m/s "
                f"linf={component_metrics['linf']:.12g} m/s")
            if tolerance_manifest is not None:
                limits = tolerance_manifest["stations"][station_name][
                    "components"][component_name]
                line += (
                    f" limits=({limits['max_l2']:.12g}, "
                    f"{limits['max_linf']:.12g}) m/s")
            print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openfoam-case", type=Path)
    parser.add_argument("--simplefluid-glob", nargs="+")
    parser.add_argument("--tolerances", type=Path)
    parser.add_argument("--required-scope")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--print-simplefluid-settings", action="store_true")
    parser.add_argument("--max-l2", type=nonnegative_float)
    parser.add_argument("--max-linf", type=nonnegative_float)
    arguments = parser.parse_args(argv)

    if arguments.check_only and arguments.tolerances is None:
        parser.error("--check-only requires --tolerances")
    if arguments.print_simplefluid_settings and arguments.tolerances is None:
        parser.error("--print-simplefluid-settings requires --tolerances")
    if arguments.check_only and arguments.print_simplefluid_settings:
        parser.error(
            "--check-only and --print-simplefluid-settings are mutually "
            "exclusive")
    if arguments.required_scope is not None and arguments.tolerances is None:
        parser.error("--required-scope requires --tolerances")
    if arguments.tolerances is not None and (
            arguments.max_l2 is not None or arguments.max_linf is not None):
        parser.error(
            "--tolerances cannot be combined with global --max-l2/--max-linf")
    if (not arguments.check_only
            and not arguments.print_simplefluid_settings and (
            arguments.openfoam_case is None
            or arguments.simplefluid_glob is None)):
        parser.error(
            "comparison requires --openfoam-case and --simplefluid-glob")

    try:
        tolerance_manifest = None
        if arguments.tolerances is not None:
            tolerance_manifest = load_tolerance_manifest(
                arguments.tolerances, arguments.required_scope)
            qualification = tolerance_manifest["qualification"]
            if not arguments.print_simplefluid_settings:
                print(
                    f"qualified tolerances: {arguments.tolerances} "
                    f"(scope={qualification.get('scope')})")
        if arguments.print_simplefluid_settings:
            if tolerance_manifest is None \
                    or tolerance_manifest["simplefluid_run"] is None:
                raise ToleranceManifestError(
                    "qualified manifest has no SimpleFluid run settings")
            settings = tolerance_manifest["simplefluid_run"]
            print(
                settings["mesh_divisor"],
                settings["steps"],
                repr(settings["dt_s"]),
                settings["mpi_ranks"],
            )
            return 0
        if arguments.check_only:
            return 0

        metrics = collect_metrics(
            arguments.openfoam_case, arguments.simplefluid_glob)
        print_metrics(metrics, tolerance_manifest)
        failures = evaluate_metrics(
            metrics,
            tolerance_manifest=tolerance_manifest,
            max_l2=arguments.max_l2,
            max_linf=arguments.max_linf,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"comparison setup failed: {error}", file=sys.stderr)
        return 2

    if failures:
        print("comparison failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
