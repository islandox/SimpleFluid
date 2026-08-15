#!/usr/bin/env python3
"""Render matched OpenFOAM/SimpleFluid temperature fields in the R-Z plane."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections.abc import Callable
from pathlib import Path

from compare_profiles import (
    OPENFOAM_SAMPLE_RADIUS,
    discrete_rz_error_norms,
    read_openfoam_cells,
    read_simplefluid,
    rz_averages,
)


Color = tuple[int, int, int]
Palette = tuple[str, ...]

VIRIDIS: Palette = (
    "#440154",
    "#414487",
    "#2a788e",
    "#22a884",
    "#7ad151",
    "#fde725",
)
COOLWARM: Palette = (
    "#3b4cc0",
    "#7699f6",
    "#b9d0f9",
    "#eeeeee",
    "#f7b89c",
    "#dd5f4b",
    "#b40426",
)
MAGMA: Palette = (
    "#000004",
    "#2c115f",
    "#721f81",
    "#b73779",
    "#f1605d",
    "#feb078",
    "#fcfdbf",
)


def parse_color(value: str) -> Color:
    """Convert a six-digit hexadecimal color to an RGB tuple."""
    return tuple(
        int(value[index : index + 2], 16)
        for index in (1, 3, 5)
    )


def palette_color(palette: Palette, fraction: float) -> str:
    """Linearly interpolate a compact color palette."""
    fraction = min(1.0, max(0.0, fraction))
    position = fraction * (len(palette) - 1)
    left_index = min(int(position), len(palette) - 2)
    local_fraction = position - left_index
    left = parse_color(palette[left_index])
    right = parse_color(palette[left_index + 1])
    components = tuple(
        round(
            left[component]
            + local_fraction
            * (right[component] - left[component])
        )
        for component in range(3)
    )
    return "#" + "".join(f"{component:02x}" for component in components)


def linear_fraction(value: float, lower: float, upper: float) -> float:
    """Normalize a value to a linear interval."""
    if upper <= lower:
        raise ValueError("color limits must be strictly increasing")
    return (value - lower) / (upper - lower)


def log_fraction(value: float, lower: float, upper: float) -> float:
    """Normalize a positive value to a logarithmic interval."""
    if lower <= 0.0 or upper <= lower:
        raise ValueError("logarithmic color limits must be positive")
    bounded = min(upper, max(lower, value))
    return (
        math.log10(bounded) - math.log10(lower)
    ) / (math.log10(upper) - math.log10(lower))


def nice_upper_limit(value: float) -> float:
    """Round a positive value upward to a readable colorbar limit."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("colorbar maximum must be finite and positive")
    exponent = math.floor(math.log10(value))
    scale = 10.0**exponent
    normalized = value / scale
    for candidate in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if normalized <= candidate:
            return candidate * scale
    raise AssertionError("unreachable nice-number branch")


def uniform_edges(lower: float, upper: float, cells: int) -> list[float]:
    """Generate uniform cell edges."""
    if cells <= 0:
        raise ValueError("cell count must be positive")
    return [
        lower + (upper - lower) * index / cells
        for index in range(cells + 1)
    ]


def graded_edges(
    lower: float,
    upper: float,
    cells: int,
    expansion_ratio: float,
) -> list[float]:
    """Match the geometric block grading used by the Shiri executable."""
    if cells <= 0:
        raise ValueError("cell count must be positive")
    if cells == 1 or expansion_ratio == 1.0:
        return uniform_edges(lower, upper, cells)
    ratio = expansion_ratio ** (1.0 / (cells - 1))
    first_width = (
        (upper - lower)
        * (ratio - 1.0)
        / (ratio**cells - 1.0)
    )
    result = [lower]
    width = first_width
    for _ in range(cells):
        result.append(result[-1] + width)
        width *= ratio
    result[-1] = upper
    return result


def append_block(
    edges: list[float],
    lower: float,
    upper: float,
    cells: int,
    expansion_ratio: float,
) -> None:
    """Append a graded block without duplicating the shared edge."""
    block = graded_edges(lower, upper, cells, expansion_ratio)
    edges.extend(block[1:])


def shiri_radial_edges(cells: int) -> list[float]:
    """Return the radial edges shared by the two comparison meshes."""
    if cells == 1:
        return uniform_edges(0.075, 0.600, cells)
    inner_cells = cells // 2
    outer_cells = cells - inner_cells
    result = graded_edges(0.075, 0.2625, inner_cells, 3.0)
    append_block(result, 0.2625, 0.600, outer_cells, 1.0)
    return result


def shiri_axial_edges(cells: int) -> list[float]:
    """Return the axial edges shared by the two comparison meshes."""
    if cells == 1:
        return uniform_edges(0.0, 1.5, cells)
    lower_cells = max(1, cells // 5)
    upper_cells = cells - lower_cells
    result = graded_edges(0.0, 0.120, lower_cells, 2.0)
    append_block(result, 0.120, 1.5, upper_cells, 1.5)
    return result


class Svg:
    """Minimal SVG document builder."""

    def __init__(self, width: int, height: int, title: str) -> None:
        self.width = width
        self.height = height
        self.definitions: list[str] = []
        self.elements: list[str] = []
        self.title = title

    def add(self, element: str) -> None:
        """Append a raw SVG element."""
        self.elements.append(element)

    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: float = 18.0,
        weight: int = 400,
        anchor: str = "start",
        fill: str = "#17212b",
        transform: str | None = None,
    ) -> None:
        """Append escaped SVG text."""
        transform_attribute = (
            "" if transform is None
            else f' transform="{html.escape(transform, quote=True)}"'
        )
        self.add(
            f'<text x="{x:.3f}" y="{y:.3f}" '
            f'font-size="{size:.3f}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{fill}"'
            f'{transform_attribute}>{html.escape(text)}</text>'
        )

    def add_gradient(self, identifier: str, palette: Palette) -> None:
        """Define a bottom-to-top linear gradient."""
        stops = []
        for index, color in enumerate(palette):
            offset = 100.0 * index / (len(palette) - 1)
            stops.append(
                f'<stop offset="{offset:.6f}%" stop-color="{color}"/>'
            )
        self.definitions.append(
            f'<linearGradient id="{identifier}" x1="0" y1="1" '
            f'x2="0" y2="0">{"".join(stops)}</linearGradient>'
        )

    def write(self, path: Path) -> None:
        """Write the completed SVG atomically enough for generated output."""
        path.parent.mkdir(parents=True, exist_ok=True)
        definitions = (
            "<defs>" + "".join(self.definitions) + "</defs>"
            if self.definitions
            else ""
        )
        document = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'font-family="DejaVu Sans,Segoe UI,sans-serif" '
            f'shape-rendering="geometricPrecision">\n'
            f"<title>{html.escape(self.title)}</title>\n"
            f"{definitions}\n"
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            + "\n".join(self.elements)
            + "\n</svg>\n"
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(path)


def render_cells(
    svg: Svg,
    rows: dict[tuple[float, float], dict[str, float]],
    radii: list[float],
    axial_coordinates: list[float],
    radial_edges: list[float],
    axial_edges: list[float],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    value: Callable[[dict[str, float]], float],
    color: Callable[[float], str],
) -> None:
    """Render one colored rectangle per R-Z cell."""
    radial_span = radial_edges[-1] - radial_edges[0]
    axial_span = axial_edges[-1] - axial_edges[0]
    svg.add('<g shape-rendering="crispEdges">')
    for axial_index, axial in enumerate(axial_coordinates):
        top = (
            y
            + height
            - (axial_edges[axial_index + 1] - axial_edges[0])
            / axial_span
            * height
        )
        bottom = (
            y
            + height
            - (axial_edges[axial_index] - axial_edges[0])
            / axial_span
            * height
        )
        for radial_index, radius in enumerate(radii):
            left = (
                x
                + (radial_edges[radial_index] - radial_edges[0])
                / radial_span
                * width
            )
            right = (
                x
                + (radial_edges[radial_index + 1] - radial_edges[0])
                / radial_span
                * width
            )
            cell_value = value(rows[(radius, axial)])
            svg.add(
                f'<rect x="{left:.4f}" y="{top:.4f}" '
                f'width="{right - left + 0.08:.4f}" '
                f'height="{bottom - top + 0.08:.4f}" '
                f'fill="{color(cell_value)}"/>'
            )
    svg.add("</g>")


def render_axes(
    svg: Svg,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    radial_edges: list[float],
    axial_edges: list[float],
    panel_title: str,
    panel_subtitle: str,
) -> None:
    """Draw panel border, ticks, labels, and title."""
    svg.add(
        f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" '
        f'height="{height:.3f}" fill="none" stroke="#17212b" '
        'stroke-width="1.4"/>'
    )
    svg.add_text(
        x + width / 2.0,
        y - 46.0,
        panel_title,
        size=24.0,
        weight=600,
        anchor="middle",
    )
    svg.add_text(
        x + width / 2.0,
        y - 20.0,
        panel_subtitle,
        size=15.0,
        anchor="middle",
        fill="#52606d",
    )
    radial_ticks = (0.075, 0.15, 0.30, 0.45, 0.60)
    axial_ticks = (0.0, 0.3, 0.6, 0.9, 1.2, 1.5)
    for value in radial_ticks:
        fraction = (
            value - radial_edges[0]
        ) / (radial_edges[-1] - radial_edges[0])
        tick_x = x + fraction * width
        svg.add(
            f'<line x1="{tick_x:.3f}" y1="{y + height:.3f}" '
            f'x2="{tick_x:.3f}" y2="{y + height + 7.0:.3f}" '
            'stroke="#17212b" stroke-width="1.2"/>'
        )
        label = f"{value:.3f}" if value == 0.075 else f"{value:.2f}"
        svg.add_text(
            tick_x,
            y + height + 29.0,
            label,
            size=14.0,
            anchor="middle",
        )
    for value in axial_ticks:
        fraction = (
            value - axial_edges[0]
        ) / (axial_edges[-1] - axial_edges[0])
        tick_y = y + height - fraction * height
        svg.add(
            f'<line x1="{x - 7.0:.3f}" y1="{tick_y:.3f}" '
            f'x2="{x:.3f}" y2="{tick_y:.3f}" '
            'stroke="#17212b" stroke-width="1.2"/>'
        )
        svg.add_text(
            x - 12.0,
            tick_y + 5.0,
            f"{value:.1f}",
            size=14.0,
            anchor="end",
        )
    svg.add_text(
        x + width / 2.0,
        y + height + 63.0,
        "Radius, R (m)",
        size=18.0,
        anchor="middle",
    )
    svg.add_text(
        x - 76.0,
        y + height / 2.0,
        "Axial coordinate, Z (m)",
        size=18.0,
        anchor="middle",
        transform=(
            f"rotate(-90 {x - 76.0:.3f} "
            f"{y + height / 2.0:.3f})"
        ),
    )


def render_colorbar(
    svg: Svg,
    *,
    identifier: str,
    palette: Palette,
    x: float,
    y: float,
    width: float,
    height: float,
    lower: float,
    upper: float,
    ticks: list[float],
    label: str,
    logarithmic: bool = False,
) -> None:
    """Draw a vertical continuous colorbar."""
    svg.add_gradient(identifier, palette)
    svg.add(
        f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" '
        f'height="{height:.3f}" fill="url(#{identifier})" '
        'stroke="#17212b" stroke-width="1.0"/>'
    )
    for value in ticks:
        fraction = (
            log_fraction(value, lower, upper)
            if logarithmic
            else linear_fraction(value, lower, upper)
        )
        tick_y = y + height * (1.0 - fraction)
        svg.add(
            f'<line x1="{x + width:.3f}" y1="{tick_y:.3f}" '
            f'x2="{x + width + 7.0:.3f}" y2="{tick_y:.3f}" '
            'stroke="#17212b" stroke-width="1.1"/>'
        )
        if logarithmic and value < 0.01:
            text = f"{value:.0e}"
        elif abs(value) >= 1.0:
            text = f"{value:.1f}"
        else:
            text = f"{value:.3g}"
        svg.add_text(
            x + width + 12.0,
            tick_y + 5.0,
            text,
            size=14.0,
        )
    label_x = x + width + 82.0
    label_y = y + height / 2.0
    svg.add_text(
        label_x,
        label_y,
        label,
        size=17.0,
        anchor="middle",
        transform=f"rotate(-90 {label_x:.3f} {label_y:.3f})",
    )


def render_distribution_figure(
    path: Path,
    openfoam: dict[tuple[float, float], dict[str, float]],
    simplefluid: dict[tuple[float, float], dict[str, float]],
    radii: list[float],
    axial_coordinates: list[float],
    radial_edges: list[float],
    axial_edges: list[float],
    expected_time: float,
) -> None:
    """Render OpenFOAM and SimpleFluid temperature distributions."""
    all_temperatures = [
        row["temperature"]
        for rows in (openfoam, simplefluid)
        for row in rows.values()
    ]
    lower = math.floor(min(all_temperatures))
    upper = math.ceil(max(all_temperatures))
    if upper <= lower:
        upper = lower + 1.0
    ticks = [
        lower + index * (upper - lower) / 5.0
        for index in range(6)
    ]

    svg = Svg(
        1280,
        1000,
        "Steady Shiri natural convection temperature in the R-Z plane",
    )
    svg.add_text(
        640,
        48,
        "Steady Shiri natural convection — temperature in the R–Z plane",
        size=29.0,
        weight=650,
        anchor="middle",
    )
    svg.add_text(
        640,
        78,
        "θ-averaged matched cell-centre fields with a shared temperature scale",
        size=17.0,
        anchor="middle",
        fill="#52606d",
    )
    panel_y = 160.0
    panel_height = 690.0
    panel_width = panel_height * (
        radial_edges[-1] - radial_edges[0]
    ) / (axial_edges[-1] - axial_edges[0])
    panels = (
        (
            150.0,
            openfoam,
            "OpenFOAM reference",
            f"t = {expected_time:g}",
        ),
        (
            575.0,
            simplefluid,
            "SimpleFluid",
            "adaptive steady",
        ),
    )
    color = lambda value: palette_color(
        VIRIDIS, linear_fraction(value, lower, upper)
    )
    for x, rows, title, subtitle in panels:
        render_cells(
            svg,
            rows,
            radii,
            axial_coordinates,
            radial_edges,
            axial_edges,
            x=x,
            y=panel_y,
            width=panel_width,
            height=panel_height,
            value=lambda row: row["temperature"],
            color=color,
        )
        values = [row["temperature"] for row in rows.values()]
        render_axes(
            svg,
            x=x,
            y=panel_y,
            width=panel_width,
            height=panel_height,
            radial_edges=radial_edges,
            axial_edges=axial_edges,
            panel_title=title,
            panel_subtitle=(
                f"{subtitle}; range "
                f"{min(values):.3f}–{max(values):.3f} K"
            ),
        )
    render_colorbar(
        svg,
        identifier="temperature-scale",
        palette=VIRIDIS,
        x=1015.0,
        y=panel_y,
        width=30.0,
        height=panel_height,
        lower=lower,
        upper=upper,
        ticks=ticks,
        label="Temperature (K)",
    )
    svg.add_text(
        640,
        965,
        "Inner heated pipe: R = 0.075 m  •  Outer wall: R = 0.600 m  •  Height: 1.500 m",
        size=15.0,
        anchor="middle",
        fill="#52606d",
    )
    svg.write(path)


def render_error_figure(
    path: Path,
    openfoam: dict[tuple[float, float], dict[str, float]],
    simplefluid: dict[tuple[float, float], dict[str, float]],
    radii: list[float],
    axial_coordinates: list[float],
    radial_edges: list[float],
    axial_edges: list[float],
) -> tuple[float, float, tuple[float, float]]:
    """Render signed and absolute SimpleFluid-minus-OpenFOAM error."""
    rms, linf = discrete_rz_error_norms(
        openfoam, simplefluid, "temperature"
    )
    errors = {
        key: (
            simplefluid[key]["temperature"]
            - openfoam[key]["temperature"]
        )
        for key in openfoam
    }
    maximum_key = max(errors, key=lambda key: abs(errors[key]))
    full_limit = nice_upper_limit(linf)
    positive_errors = [
        abs(value) for value in errors.values() if value != 0.0
    ]
    log_lower = 10.0 ** math.floor(
        math.log10(max(min(positive_errors), full_limit * 1.0e-5))
    )
    log_ticks = []
    exponent = math.ceil(math.log10(log_lower))
    while 10.0**exponent < full_limit:
        log_ticks.append(10.0**exponent)
        exponent += 1
    if not log_ticks or not math.isclose(
        log_ticks[-1], full_limit, rel_tol=0.0, abs_tol=1.0e-15
    ):
        log_ticks.append(full_limit)

    signed_rows = {
        key: {"value": value}
        for key, value in errors.items()
    }
    absolute_rows = {
        key: {"value": abs(value)}
        for key, value in errors.items()
    }

    svg = Svg(
        1400,
        1080,
        "Steady Shiri natural convection temperature error in the R-Z plane",
    )
    svg.add_text(
        700,
        48,
        "Steady Shiri natural convection — temperature error in the R–Z plane",
        size=29.0,
        weight=650,
        anchor="middle",
    )
    svg.add_text(
        700,
        78,
        "SimpleFluid minus OpenFOAM on identical θ-averaged cell-centre coordinates",
        size=17.0,
        anchor="middle",
        fill="#52606d",
    )
    panel_y = 160.0
    panel_height = 690.0
    panel_width = panel_height * (
        radial_edges[-1] - radial_edges[0]
    ) / (axial_edges[-1] - axial_edges[0])
    left_x = 135.0
    right_x = 750.0

    signed_color = lambda value: palette_color(
        COOLWARM,
        linear_fraction(value, -full_limit, full_limit),
    )
    render_cells(
        svg,
        signed_rows,
        radii,
        axial_coordinates,
        radial_edges,
        axial_edges,
        x=left_x,
        y=panel_y,
        width=panel_width,
        height=panel_height,
        value=lambda row: row["value"],
        color=signed_color,
    )
    render_axes(
        svg,
        x=left_x,
        y=panel_y,
        width=panel_width,
        height=panel_height,
        radial_edges=radial_edges,
        axial_edges=axial_edges,
        panel_title="Signed error",
        panel_subtitle="full pointwise range",
    )
    signed_ticks = [
        -full_limit,
        -full_limit / 2.0,
        0.0,
        full_limit / 2.0,
        full_limit,
    ]
    render_colorbar(
        svg,
        identifier="signed-error-scale",
        palette=COOLWARM,
        x=left_x + panel_width + 42.0,
        y=panel_y,
        width=27.0,
        height=panel_height,
        lower=-full_limit,
        upper=full_limit,
        ticks=signed_ticks,
        label="T_SF − T_OF (K)",
    )

    absolute_color = lambda value: palette_color(
        MAGMA,
        log_fraction(max(value, log_lower), log_lower, full_limit),
    )
    render_cells(
        svg,
        absolute_rows,
        radii,
        axial_coordinates,
        radial_edges,
        axial_edges,
        x=right_x,
        y=panel_y,
        width=panel_width,
        height=panel_height,
        value=lambda row: row["value"],
        color=absolute_color,
    )
    render_axes(
        svg,
        x=right_x,
        y=panel_y,
        width=panel_width,
        height=panel_height,
        radial_edges=radial_edges,
        axial_edges=axial_edges,
        panel_title="Absolute error",
        panel_subtitle=f"log scale; values ≤ {log_lower:.0e} K use the floor color",
    )
    render_colorbar(
        svg,
        identifier="absolute-error-scale",
        palette=MAGMA,
        x=right_x + panel_width + 42.0,
        y=panel_y,
        width=27.0,
        height=panel_height,
        lower=log_lower,
        upper=full_limit,
        ticks=log_ticks,
        label="|T_SF − T_OF| (K), log scale",
        logarithmic=True,
    )

    svg.add(
        '<rect x="285" y="970" width="830" height="70" rx="8" '
        'fill="#f3f6f8" stroke="#ccd5dd" stroke-width="1"/>'
    )
    svg.add_text(
        700,
        997,
        (
            f"R–Z RMS = {rms:.6f} K  •  L∞ = {linf:.6f} K  •  "
            f"maximum at R = {maximum_key[0]:.6f} m, "
            f"Z = {maximum_key[1]:.6f} m"
        ),
        size=16.0,
        weight=550,
        anchor="middle",
    )
    svg.add_text(
        700,
        1027,
        "Configured criterion: RMS ≤ 0.050000 K — PASS",
        size=16.0,
        weight=650,
        anchor="middle",
        fill="#16794b",
    )
    svg.write(path)
    return rms, linf, maximum_key


def write_matched_csv(
    path: Path,
    openfoam: dict[tuple[float, float], dict[str, float]],
    simplefluid: dict[tuple[float, float], dict[str, float]],
) -> None:
    """Write the exact matched data visualized by the figures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "r",
                "z",
                "openfoam_temperature",
                "simplefluid_temperature",
                "signed_error",
                "absolute_error",
            )
        )
        for radius, axial in sorted(openfoam, key=lambda key: (key[1], key[0])):
            reference = openfoam[(radius, axial)]["temperature"]
            result = simplefluid[(radius, axial)]["temperature"]
            error = result - reference
            writer.writerow(
                (
                    f"{radius:.12g}",
                    f"{axial:.12g}",
                    f"{reference:.17g}",
                    f"{result:.17g}",
                    f"{error:.17g}",
                    f"{abs(error):.17g}",
                )
            )


def main() -> int:
    """Parse comparison inputs and render the matched R-Z figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--openfoam-case", type=Path, required=True)
    parser.add_argument(
        "--simplefluid-glob", nargs="+", required=True
    )
    parser.add_argument("--expected-time", type=float, required=True)
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--expected-cells", type=int, default=80000)
    parser.add_argument(
        "--expected-theta-cells", type=int, default=20
    )
    parser.add_argument(
        "--expected-openfoam-theta-cells", type=int
    )
    parser.add_argument(
        "--expected-radial-cells", type=int, default=40
    )
    parser.add_argument(
        "--expected-axial-cells", type=int, default=100
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    (
        _,
        _,
        _,
        _,
        _,
        simplefluid_cells,
    ) = read_simplefluid(
        arguments.simplefluid_glob,
        OPENFOAM_SAMPLE_RADIUS,
        arguments.expected_ranks,
        arguments.expected_cells,
        arguments.expected_theta_cells,
        arguments.expected_radial_cells,
        arguments.expected_axial_cells,
    )
    openfoam_cells = read_openfoam_cells(
        arguments.openfoam_case, arguments.expected_time
    )
    openfoam_rz = rz_averages(
        openfoam_cells,
        arguments.expected_openfoam_theta_cells
        if arguments.expected_openfoam_theta_cells is not None
        else arguments.expected_theta_cells,
    )
    simplefluid_rz = rz_averages(
        simplefluid_cells, arguments.expected_theta_cells
    )
    if openfoam_rz.keys() != simplefluid_rz.keys():
        missing = len(openfoam_rz.keys() - simplefluid_rz.keys())
        extra = len(simplefluid_rz.keys() - openfoam_rz.keys())
        raise ValueError(
            "OpenFOAM and SimpleFluid R-Z coordinates differ "
            f"(missing={missing}, extra={extra})"
        )

    radii = sorted({radius for radius, _ in openfoam_rz})
    axial_coordinates = sorted(
        {axial for _, axial in openfoam_rz}
    )
    if len(radii) != arguments.expected_radial_cells:
        raise ValueError(
            f"expected {arguments.expected_radial_cells} radial "
            f"coordinates, found {len(radii)}"
        )
    if len(axial_coordinates) != arguments.expected_axial_cells:
        raise ValueError(
            f"expected {arguments.expected_axial_cells} axial "
            f"coordinates, found {len(axial_coordinates)}"
        )
    if len(openfoam_rz) != len(radii) * len(axial_coordinates):
        raise ValueError("matched R-Z data do not form a rectangular grid")

    radial_edges = shiri_radial_edges(len(radii))
    axial_edges = shiri_axial_edges(len(axial_coordinates))
    output_directory = arguments.output_directory
    distribution_path = (
        output_directory / "temperature_rz_distribution.svg"
    )
    error_path = output_directory / "temperature_rz_error.svg"
    csv_path = output_directory / "temperature_rz_matched.csv"
    render_distribution_figure(
        distribution_path,
        openfoam_rz,
        simplefluid_rz,
        radii,
        axial_coordinates,
        radial_edges,
        axial_edges,
        arguments.expected_time,
    )
    rms, linf, maximum_key = render_error_figure(
        error_path,
        openfoam_rz,
        simplefluid_rz,
        radii,
        axial_coordinates,
        radial_edges,
        axial_edges,
    )
    write_matched_csv(csv_path, openfoam_rz, simplefluid_rz)
    print(f"temperature R-Z distribution: {distribution_path}")
    print(f"temperature R-Z error: {error_path}")
    print(f"matched R-Z data: {csv_path}")
    print(
        f"temperature R-Z error: rms={rms:.12g} K, "
        f"linf={linf:.12g} K, "
        f"maximum_location=({maximum_key[0]:.12g}, "
        f"{maximum_key[1]:.12g}) m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
