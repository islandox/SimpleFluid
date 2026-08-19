#!/usr/bin/env python3
"""Render pitzDaily velocity distributions and scale-normalized errors."""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

from compare_profiles import (
    COMPONENTS,
    STATIONS,
    interpolate,
    latest_profile,
    read_openfoam,
    read_simplefluid,
    simplefluid_profile,
)


WIDTH = 1500
HEIGHT = 900
COLORS = {"openfoam": "#1f77b4", "simplefluid": "#d95f02"}


def text_element(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 18,
    anchor: str = "start",
    weight: int = 400,
    transform: str | None = None,
) -> str:
    transform_attribute = "" if transform is None else f' transform="{transform}"'
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#17212b"'
        f'{transform_attribute}>{html.escape(value)}</text>'
    )


def expanded_bounds(values: list[float], include_zero: bool = False) -> tuple[float, float]:
    if include_zero:
        values = [*values, 0.0]
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper):
        padding = max(abs(lower) * 0.1, 1.0)
    else:
        padding = 0.08 * (upper - lower)
    return lower - padding, upper + padding


def coordinates(
    rows: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    rectangle: tuple[float, float, float, float],
) -> str:
    xmin, xmax, ymin, ymax = bounds
    left, top, right, bottom = rectangle
    return " ".join(
        f"{left + (x - xmin) / (xmax - xmin) * (right - left):.2f},"
        f"{bottom - (y - ymin) / (ymax - ymin) * (bottom - top):.2f}"
        for x, y in rows
    )


def panel_axes(
    elements: list[str],
    rectangle: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
    title: str,
    xlabel: str,
    show_ylabel: bool,
) -> None:
    left, top, right, bottom = rectangle
    xmin, xmax, ymin, ymax = bounds
    elements.append(
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{right-left:.2f}" '
        f'height="{bottom-top:.2f}" fill="#ffffff" stroke="#8a949e"/>'
    )
    for index in range(5):
        fraction = index / 4
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        x_value = xmin + fraction * (xmax - xmin)
        y_value = ymin + fraction * (ymax - ymin)
        elements.append(
            f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" '
            f'y2="{bottom:.2f}" stroke="#e3e7eb"/>'
        )
        elements.append(
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" '
            f'y2="{y:.2f}" stroke="#e3e7eb"/>'
        )
        elements.append(text_element(x, bottom + 23, f"{x_value:.3g}", size=13, anchor="middle"))
        if show_ylabel:
            elements.append(text_element(left - 10, y + 5, f"{y_value:.3g}", size=13, anchor="end"))
    if xmin < 0.0 < xmax:
        zero_x = left + (0.0 - xmin) / (xmax - xmin) * (right - left)
        elements.append(
            f'<line x1="{zero_x:.2f}" y1="{top:.2f}" x2="{zero_x:.2f}" '
            f'y2="{bottom:.2f}" stroke="#616b75" stroke-width="1.5"/>'
        )
    elements.append(text_element((left + right) / 2, top - 17, title, size=18, anchor="middle", weight=600))
    elements.append(text_element((left + right) / 2, bottom + 49, xlabel, size=15, anchor="middle"))
    if show_ylabel:
        centre_y = (top + bottom) / 2
        elements.append(
            text_element(
                left - 62,
                centre_y,
                "y (m)",
                size=15,
                anchor="middle",
                transform=f"rotate(-90 {left - 62:.2f} {centre_y:.2f})",
            )
        )


def document(title: str, subtitle: str, elements: list[str]) -> str:
    heading = [
        text_element(WIDTH / 2, 38, title, size=27, anchor="middle", weight=700),
        text_element(WIDTH / 2, 66, subtitle, size=16, anchor="middle"),
    ]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">\n'
        '<rect width="100%" height="100%" fill="#f7f9fb"/>\n'
        + "\n".join(heading + elements)
        + "\n</svg>\n"
    )


def collect_profiles(
    openfoam_case: Path, patterns: list[str]
) -> dict[str, dict[str, object]]:
    cells = read_simplefluid(patterns)
    profiles: dict[str, dict[str, object]] = {}
    for station_name, station in STATIONS.items():
        reference = read_openfoam(latest_profile(openfoam_case, station_name))
        actual_x, result = simplefluid_profile(cells, station)
        profiles[station_name] = {
            "reference": reference,
            "result": result,
            "actual_x": actual_x,
        }
    return profiles


def panel_rectangle(column: int, row: int) -> tuple[float, float, float, float]:
    panel_width = 450
    panel_height = 340
    panel_left = 25 + column * 490
    panel_top = 105 + row * 365
    return panel_left + 82, panel_top + 50, panel_left + panel_width - 25, panel_top + panel_height - 58


def render_distribution(
    output: Path, profiles: dict[str, dict[str, object]]
) -> None:
    elements: list[str] = []
    for column, (station_name, station) in enumerate(STATIONS.items()):
        profile = profiles[station_name]
        reference = profile["reference"]
        result = profile["result"]
        assert isinstance(reference, list) and isinstance(result, list)
        for row, (component_name, component) in enumerate(COMPONENTS.items()):
            rectangle = panel_rectangle(column, row)
            velocity_values = [sample[component] for sample in reference + result]
            xmin, xmax = expanded_bounds(velocity_values, include_zero=True)
            ymin = min(sample[0] for sample in reference)
            ymax = max(sample[0] for sample in reference)
            bounds = xmin, xmax, ymin, ymax
            panel_axes(
                elements,
                rectangle,
                bounds,
                f"x={station:.2f} m — {component_name}",
                f"{component_name} (m/s)",
                column == 0,
            )
            reference_points = coordinates(
                [(sample[component], sample[0]) for sample in reference], bounds, rectangle
            )
            result_points = coordinates(
                [(sample[component], sample[0]) for sample in result], bounds, rectangle
            )
            elements.append(
                f'<polyline points="{reference_points}" fill="none" '
                f'stroke="{COLORS["openfoam"]}" stroke-width="2.5"/>'
            )
            for point in result_points.split():
                x, y = point.split(",")
                elements.append(
                    f'<circle cx="{x}" cy="{y}" r="4" fill="{COLORS["simplefluid"]}" '
                    'stroke="#ffffff" stroke-width="1"/>'
                )
    elements.extend(
        [
            '<line x1="585" y1="852" x2="630" y2="852" '
            f'stroke="{COLORS["openfoam"]}" stroke-width="3"/>',
            text_element(642, 858, "OpenFOAM v2606 (steady)", size=15),
            f'<circle cx="850" cy="852" r="5" fill="{COLORS["simplefluid"]}"/>',
            text_element(864, 858, "SimpleFluid (t=0.002 s)", size=15),
        ]
    )
    output.write_text(
        document(
            "pitzDaily velocity-profile distributions",
            "Converged OpenFOAM versus SimpleFluid diagnostic smoke profile; planar x-y case",
            elements,
        ),
        encoding="utf-8",
    )


def render_relative_error(
    output: Path, profiles: dict[str, dict[str, object]]
) -> list[tuple[str, str, float, float]]:
    elements: list[str] = []
    summary: list[tuple[str, str, float, float]] = []
    for column, (station_name, station) in enumerate(STATIONS.items()):
        profile = profiles[station_name]
        reference = profile["reference"]
        result = profile["result"]
        assert isinstance(reference, list) and isinstance(result, list)
        for row, (component_name, component) in enumerate(COMPONENTS.items()):
            reference_at_result = [
                interpolate(reference, sample[0], component) for sample in result
            ]
            errors = [
                sample[component] - reference_value
                for sample, reference_value in zip(result, reference_at_result)
            ]
            peak_reference = max(abs(value) for value in reference_at_result)
            rms_reference = math.sqrt(
                sum(value * value for value in reference_at_result)
                / len(reference_at_result)
            )
            if peak_reference == 0.0 or rms_reference == 0.0:
                raise ValueError(f"{station_name} {component_name} reference scale is zero")
            relative_points = [
                (100.0 * error / peak_reference, sample[0])
                for error, sample in zip(errors, result)
            ]
            l2 = math.sqrt(sum(error * error for error in errors) / len(errors))
            linf = max(abs(error) for error in errors)
            summary.append(
                (
                    station_name,
                    component_name,
                    100.0 * l2 / rms_reference,
                    100.0 * linf / peak_reference,
                )
            )
            rectangle = panel_rectangle(column, row)
            maximum = max(abs(point[0]) for point in relative_points)
            limit = max(5.0, 1.12 * maximum)
            ymin = min(sample[0] for sample in reference)
            ymax = max(sample[0] for sample in reference)
            bounds = -limit, limit, ymin, ymax
            panel_axes(
                elements,
                rectangle,
                bounds,
                f"x={station:.2f} m — {component_name}",
                "signed scale-normalized error (%)",
                column == 0,
            )
            points = coordinates(relative_points, bounds, rectangle)
            elements.append(
                f'<polyline points="{points}" fill="none" '
                f'stroke="{COLORS["simplefluid"]}" stroke-width="2.5"/>'
            )
            for point in points.split():
                x, y = point.split(",")
                elements.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{COLORS["simplefluid"]}"/>')
            elements.append(
                text_element(
                    (rectangle[0] + rectangle[2]) / 2,
                    rectangle[1] + 22,
                    f"relative L2={summary[-1][2]:.1f}%, relative Linf={summary[-1][3]:.1f}%",
                    size=13,
                    anchor="middle",
                )
            )
    output.write_text(
        document(
            "pitzDaily velocity-profile relative error",
            "Signed point error is normalized by max |OpenFOAM| in each station/component panel",
            elements,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openfoam-case", type=Path, required=True)
    parser.add_argument("--simplefluid-glob", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    profiles = collect_profiles(arguments.openfoam_case, arguments.simplefluid_glob)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    distribution_path = arguments.output_dir / "velocity_profile_distribution.svg"
    error_path = arguments.output_dir / "velocity_profile_relative_error.svg"
    render_distribution(distribution_path, profiles)
    summary = render_relative_error(error_path, profiles)

    print(f"distribution: {distribution_path}")
    print(f"relative error: {error_path}")
    for station, component, relative_l2, relative_linf in summary:
        print(
            f"{station} {component}: relative_l2={relative_l2:.12g}% "
            f"relative_linf={relative_linf:.12g}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
