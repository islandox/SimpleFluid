#!/usr/bin/env python3
"""Render pitzDaily velocity-profile mesh-refinement diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from compare_profiles import (
    COMPONENTS,
    STATIONS,
    interpolate,
    latest_common_profiles,
    read_openfoam,
    read_simplefluid,
    simplefluid_profile,
)
from plot_profile_comparison import (
    HEIGHT,
    WIDTH,
    coordinates,
    document,
    expanded_bounds,
    panel_axes,
    panel_rectangle,
    text_element,
)


COLORS = {
    "openfoam": "#1f77b4",
    "divisor4": "#f4a261",
    "divisor2": "#d95f02",
    "divisor1": "#6a3d9a",
    "d4-d2": "#e76f51",
    "d2-d1": "#6a3d9a",
}


def collect_profiles(
    patterns: dict[str, list[str]],
) -> dict[str, dict[str, tuple[float, list[tuple[float, float, float]]]]]:
    profiles = {}
    for mesh_name, mesh_patterns in patterns.items():
        cells = read_simplefluid(mesh_patterns)
        profiles[mesh_name] = {
            station_name: simplefluid_profile(cells, station_x)
            for station_name, station_x in STATIONS.items()
        }
    return profiles


def render_profiles(
    path: Path,
    openfoam_case: Path,
    profiles: dict[str, dict[str, tuple[float, list[tuple[float, float, float]]]]],
) -> None:
    elements: list[str] = []
    openfoam_paths = latest_common_profiles(openfoam_case)
    for column, (station_name, station_x) in enumerate(STATIONS.items()):
        reference = read_openfoam(openfoam_paths[station_name])
        for row, (component_name, component) in enumerate(COMPONENTS.items()):
            rectangle = panel_rectangle(column, row)
            all_rows = [reference] + [profiles[name][station_name][1] for name in profiles]
            values = [sample[component] for rows in all_rows for sample in rows]
            xmin, xmax = expanded_bounds(values, include_zero=True)
            ymin, ymax = reference[0][0], reference[-1][0]
            bounds = xmin, xmax, ymin, ymax
            panel_axes(
                elements,
                rectangle,
                bounds,
                f"x={station_x:.2f} m — {component_name}",
                f"{component_name} (m/s)",
                column == 0,
            )
            reference_points = coordinates(
                [(sample[component], sample[0]) for sample in reference],
                bounds,
                rectangle,
            )
            elements.append(
                f'<polyline points="{reference_points}" fill="none" '
                f'stroke="{COLORS["openfoam"]}" stroke-width="2.5"/>'
            )
            for mesh_name in ("divisor4", "divisor2", "divisor1"):
                mesh_rows = profiles[mesh_name][station_name][1]
                points = coordinates(
                    [(sample[component], sample[0]) for sample in mesh_rows],
                    bounds,
                    rectangle,
                )
                elements.append(
                    f'<polyline points="{points}" fill="none" '
                    f'stroke="{COLORS[mesh_name]}" stroke-width="1.8"/>'
                )
    legend = [
        ("openfoam", "OpenFOAM v2606"),
        ("divisor4", "divisor 4 (820 cells)"),
        ("divisor2", "divisor 2 (3,122 cells)"),
        ("divisor1", "divisor 1 (12,225 cells)"),
    ]
    start_x = 405
    for index, (name, label) in enumerate(legend):
        x = start_x + index * 255
        elements.append(
            f'<line x1="{x}" y1="852" x2="{x + 38}" y2="852" '
            f'stroke="{COLORS[name]}" stroke-width="3"/>'
        )
        elements.append(text_element(x + 47, 858, label, size=14))
    path.write_text(
        document(
            "pitzDaily velocity mesh refinement",
            "Graded meshes at target Co=0.4; all cases exhausted their prior step ceilings",
            elements,
        ),
        encoding="utf-8",
    )


def profile_delta(
    coarse: list[tuple[float, float, float]],
    fine: list[tuple[float, float, float]],
    component: int,
) -> tuple[list[tuple[float, float]], float, float, float, float]:
    lower = max(coarse[0][0], fine[0][0])
    upper = min(coarse[-1][0], fine[-1][0])
    samples = [sample for sample in coarse if lower <= sample[0] <= upper]
    fine_values = [interpolate(fine, sample[0], component) for sample in samples]
    errors = [sample[component] - value for sample, value in zip(samples, fine_values)]
    rms_error = math.sqrt(sum(error * error for error in errors) / len(errors))
    maximum_error = max(abs(error) for error in errors)
    rms_reference = math.sqrt(sum(value * value for value in fine_values) / len(fine_values))
    peak_reference = max(abs(value) for value in fine_values)
    return (
        [(error, sample[0]) for error, sample in zip(errors, samples)],
        rms_error,
        maximum_error,
        100.0 * rms_error / rms_reference,
        100.0 * maximum_error / peak_reference,
    )


def render_deltas(
    path: Path,
    profiles: dict[str, dict[str, tuple[float, list[tuple[float, float, float]]]]],
    csv_path: Path,
) -> None:
    elements: list[str] = []
    records: list[tuple[object, ...]] = []
    pairs = (("divisor4", "divisor2", "d4-d2"), ("divisor2", "divisor1", "d2-d1"))
    for column, (station_name, station_x) in enumerate(STATIONS.items()):
        for row, (component_name, component) in enumerate(COMPONENTS.items()):
            deltas = {}
            for coarse_name, fine_name, label in pairs:
                result = profile_delta(
                    profiles[coarse_name][station_name][1],
                    profiles[fine_name][station_name][1],
                    component,
                )
                deltas[label] = result
                records.append(
                    (
                        coarse_name,
                        fine_name,
                        station_name,
                        component_name,
                        len(result[0]),
                        result[1],
                        result[2],
                        result[3],
                        result[4],
                    )
                )
            rectangle = panel_rectangle(column, row)
            values = [point[0] for result in deltas.values() for point in result[0]]
            limit = max(abs(value) for value in values) * 1.12
            first_rows = profiles["divisor4"][station_name][1]
            last_rows = profiles["divisor1"][station_name][1]
            ymin = max(first_rows[0][0], last_rows[0][0])
            ymax = min(first_rows[-1][0], last_rows[-1][0])
            bounds = -limit, limit, ymin, ymax
            panel_axes(
                elements,
                rectangle,
                bounds,
                f"x={station_x:.2f} m — {component_name}",
                f"coarse − fine {component_name} (m/s)",
                column == 0,
            )
            for label, result in deltas.items():
                points = coordinates(result[0], bounds, rectangle)
                elements.append(
                    f'<polyline points="{points}" fill="none" '
                    f'stroke="{COLORS[label]}" stroke-width="2.2"/>'
                )
            latest = deltas["d2-d1"]
            elements.append(
                text_element(
                    (rectangle[0] + rectangle[2]) / 2,
                    rectangle[1] + 22,
                    f"D2→D1 RMS={latest[1]:.3g} m/s ({latest[3]:.1f}%)",
                    size=13,
                    anchor="middle",
                )
            )
    for index, (name, label) in enumerate(
        (("d4-d2", "divisor 4 − divisor 2"), ("d2-d1", "divisor 2 − divisor 1"))
    ):
        x = 535 + index * 350
        elements.append(
            f'<line x1="{x}" y1="852" x2="{x + 42}" y2="852" '
            f'stroke="{COLORS[name]}" stroke-width="3"/>'
        )
        elements.append(text_element(x + 52, 858, label, size=14))
    path.write_text(
        document(
            "pitzDaily successive-mesh profile changes",
            "Relative values use the finer-mesh component RMS; signed curves are coarse minus fine",
            elements,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "coarse_mesh",
                "fine_mesh",
                "station",
                "component",
                "samples",
                "rms_change_m_per_s",
                "maximum_change_m_per_s",
                "relative_rms_change_percent",
                "relative_maximum_change_percent",
            )
        )
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openfoam-case", type=Path, required=True)
    parser.add_argument("--divisor4-glob", nargs="+", required=True)
    parser.add_argument("--divisor2-glob", nargs="+", required=True)
    parser.add_argument("--divisor1-glob", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    profiles = collect_profiles(
        {
            "divisor4": arguments.divisor4_glob,
            "divisor2": arguments.divisor2_glob,
            "divisor1": arguments.divisor1_glob,
        }
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = arguments.output_dir / "mesh_convergence_profiles.svg"
    deltas_path = arguments.output_dir / "mesh_convergence_deltas.svg"
    csv_path = arguments.output_dir / "mesh_convergence_summary.csv"
    render_profiles(profiles_path, arguments.openfoam_case, profiles)
    render_deltas(deltas_path, profiles, csv_path)
    print(f"profiles: {profiles_path}")
    print(f"successive differences: {deltas_path}")
    print(f"summary: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
