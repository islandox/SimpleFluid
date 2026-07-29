#!/usr/bin/env python3
"""Compare and render the matched Gaussian-tank fields in the R-Z plane."""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import html
import json
import math
import re
import shutil
import subprocess
from pathlib import Path


RADIUS = 0.1
HEIGHT = 0.3
RADIAL_CELLS = 50
AXIAL_CELLS = 150
BOUNDARY_LAYER_COUNT = 5
BOUNDARY_LAYER_FIRST_HEIGHT = 1.0e-3
BOUNDARY_LAYER_GROWTH = 1.18
REFERENCE_TEMPERATURE = 300.0
TOTAL_POWER = 100.0
SECTOR_FRACTION = 5.0 / 360.0
GAUSSIAN_RADIAL_WIDTH = 0.03
GAUSSIAN_AXIAL_WIDTH = 0.075
GAUSSIAN_AXIAL_CENTER = 0.15

FIELDS = (
    "temperature",
    "ur",
    "utheta",
    "uz",
    "k",
    "omega",
    "nut",
    "alphat",
    "qdot_fission",
    "wall_y_plus",
)

VIRIDIS = (
    "#440154",
    "#414487",
    "#2a788e",
    "#22a884",
    "#7ad151",
    "#fde725",
)
COOLWARM = (
    "#3b4cc0",
    "#7699f6",
    "#b9d0f9",
    "#eeeeee",
    "#f7b89c",
    "#dd5f4b",
    "#b40426",
)
MAGMA = (
    "#000004",
    "#2c115f",
    "#721f81",
    "#b73779",
    "#f1605d",
    "#feb078",
    "#fcfdbf",
)


def graded_axis_edges(
    length: float,
    base_cells: int,
    lower_layers: int,
    upper_layers: int,
) -> list[float]:
    """Reproduce MeshFactory's wall-stack replacement convention."""
    if lower_layers + upper_layers >= base_cells:
        raise ValueError("boundary-layer counts overlap")

    def layer_widths() -> list[float]:
        return [
            BOUNDARY_LAYER_FIRST_HEIGHT * BOUNDARY_LAYER_GROWTH**index
            for index in range(BOUNDARY_LAYER_COUNT)
        ]

    lower_widths = layer_widths()[:lower_layers]
    upper_widths = layer_widths()[:upper_layers]
    interior_cells = base_cells - lower_layers - upper_layers
    interior_width = (
        length - sum(lower_widths) - sum(upper_widths)
    ) / interior_cells
    widths = (
        lower_widths
        + [interior_width] * interior_cells
        + list(reversed(upper_widths))
    )
    edges = [0.0]
    for width in widths:
        edges.append(edges[-1] + width)
    edges[-1] = length
    return edges


def radial_edges() -> list[float]:
    return graded_axis_edges(
        RADIUS, RADIAL_CELLS, 0, BOUNDARY_LAYER_COUNT
    )


def axial_edges() -> list[float]:
    return graded_axis_edges(
        HEIGHT,
        AXIAL_CELLS,
        BOUNDARY_LAYER_COUNT,
        BOUNDARY_LAYER_COUNT,
    )


def time_directory(case: Path, expected_time: float) -> Path:
    """Locate one reconstructed OpenFOAM time containing comparison fields."""
    required = ("C", "V", "T", "U", "k", "omega", "nut", "alphat")
    matches: list[Path] = []
    for path in case.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if (
            math.isclose(value, expected_time, rel_tol=0.0, abs_tol=1.0e-12)
            and all((path / name).is_file() for name in required)
        ):
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one OpenFOAM field directory at t={expected_time}, "
            f"found {len(matches)}"
        )
    return matches[0]


def read_foam_internal(
    path: Path,
    expected_kind: str,
    uniform_count: int | None = None,
) -> list[float] | list[tuple[float, float, float]]:
    """Read an ASCII OpenFOAM internal field."""
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"internalField\s+nonuniform\s+List<(scalar|vector)>\s+"
        r"([0-9]+)\s*\((.*?)\)\s*;",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        uniform_match = re.search(
            r"internalField\s+uniform\s+([^;]+)\s*;",
            text,
        )
        if (
            uniform_match is not None
            and expected_kind == "scalar"
            and uniform_count is not None
        ):
            value = float(uniform_match.group(1))
            return [value] * uniform_count
        raise ValueError(
            f"{path} does not contain a supported ASCII internal field"
        )
    kind = match.group(1)
    if kind != expected_kind:
        raise ValueError(
            f"{path} contains {kind}; expected {expected_kind}"
        )
    count = int(match.group(2))
    body = match.group(3)
    if kind == "scalar":
        values: list[float] | list[tuple[float, float, float]] = [
            float(token) for token in body.split()
        ]
    else:
        values = []
        for token in re.findall(r"\(([^()]*)\)", body):
            components = tuple(float(value) for value in token.split())
            if len(components) != 3:
                raise ValueError(f"{path} contains a malformed vector")
            values.append(components)
    if len(values) != count:
        raise ValueError(
            f"{path} declares {count} values but contains {len(values)}"
        )
    return values


def read_foam_boundary_scalar_max(path: Path) -> float:
    """Return the largest scalar boundary value in an OpenFOAM field."""
    text = path.read_text(encoding="utf-8")
    values: list[float] = []
    for match in re.finditer(
        r"value\s+nonuniform\s+List<scalar>\s+"
        r"([0-9]+)\s*\((.*?)\)\s*;",
        text,
        flags=re.DOTALL,
    ):
        declared = int(match.group(1))
        patch_values = [
            float(token) for token in match.group(2).split()
        ]
        if len(patch_values) != declared:
            raise ValueError(
                f"{path} declares {declared} boundary values but contains "
                f"{len(patch_values)}"
            )
        values.extend(patch_values)
    values.extend(
        float(match.group(1))
        for match in re.finditer(
            r"value\s+uniform\s+"
            r"([-+0-9.eE]+)\s*;",
            text,
        )
    )
    return max(values, default=0.0)


def gaussian_weight(radius: float, axial: float) -> float:
    return math.exp(
        -0.5
        * (
            (radius / GAUSSIAN_RADIAL_WIDTH) ** 2
            + ((axial - GAUSSIAN_AXIAL_CENTER) / GAUSSIAN_AXIAL_WIDTH)
            ** 2
        )
    )


def read_openfoam(
    case: Path, expected_time: float
) -> tuple[list[dict[str, float]], float]:
    """Read reconstructed OpenFOAM cells and reconstruct the exact source."""
    directory = time_directory(case, expected_time)
    centers = read_foam_internal(directory / "C", "vector")
    volumes = read_foam_internal(directory / "V", "scalar")
    velocity = read_foam_internal(directory / "U", "vector")
    scalars = {
        name: read_foam_internal(directory / filename, "scalar")
        for name, filename in (
            ("temperature", "T"),
            ("k", "k"),
            ("omega", "omega"),
            ("nut", "nut"),
            ("alphat", "alphat"),
        )
    }
    y_plus_path = directory / "yPlus"
    y_plus = (
        read_foam_internal(y_plus_path, "scalar", len(centers))
        if y_plus_path.is_file()
        else [0.0] * len(centers)
    )
    maximum_boundary_y_plus = (
        read_foam_boundary_scalar_max(y_plus_path)
        if y_plus_path.is_file()
        else 0.0
    )
    count = len(centers)
    if (
        len(volumes) != count
        or len(velocity) != count
        or len(y_plus) != count
        or any(len(values) != count for values in scalars.values())
    ):
        raise ValueError("OpenFOAM fields contain different cell counts")

    weights = [
        gaussian_weight(math.hypot(center[0], center[1]), center[2])
        for center in centers
    ]
    shape_integral = math.fsum(
        weight * volume for weight, volume in zip(weights, volumes)
    )
    power_density_scale = (
        TOTAL_POWER * SECTOR_FRACTION / shape_integral
    )

    rows: list[dict[str, float]] = []
    for index, center in enumerate(centers):
        x, y, z = center
        ux, uy, uz = velocity[index]
        radius = math.hypot(x, y)
        theta = math.atan2(y, x)
        cosine = math.cos(theta)
        sine = math.sin(theta)
        row = {
            "r": radius,
            "theta": theta,
            "z": z,
            "cell_volume": volumes[index],
            "temperature": scalars["temperature"][index],
            "ur": ux * cosine + uy * sine,
            "utheta": -ux * sine + uy * cosine,
            "uz": uz,
            "k": scalars["k"][index],
            "omega": scalars["omega"][index],
            "nut": scalars["nut"][index],
            "alphat": scalars["alphat"][index],
            "qdot_fission": power_density_scale * weights[index],
            "wall_y_plus": y_plus[index],
        }
        if any(not math.isfinite(value) for value in row.values()):
            raise ValueError(
                f"OpenFOAM cell {index} contains a non-finite value"
            )
        rows.append(row)
    return rows, maximum_boundary_y_plus


def read_simplefluid(
    patterns: list[str],
    expected_ranks: int | None,
    expected_time: float,
) -> list[dict[str, float]]:
    """Read consecutive rank-local SimpleFluid cell CSV files."""
    paths = sorted(
        {Path(name) for pattern in patterns for name in glob.glob(pattern)}
    )
    if not paths:
        raise FileNotFoundError("no SimpleFluid rank CSV files matched")
    rank_pattern = re.compile(r"_rank([0-9]+)[.]csv$")
    ranks: list[int] = []
    for path in paths:
        match = rank_pattern.search(path.name)
        if match is None:
            raise ValueError(f"cannot determine rank from {path}")
        ranks.append(int(match.group(1)))
    if sorted(ranks) != list(range(len(paths))):
        raise ValueError(
            f"SimpleFluid rank CSVs are not consecutive: {sorted(ranks)}"
        )
    if expected_ranks is not None and len(paths) != expected_ranks:
        raise ValueError(
            f"expected {expected_ranks} rank CSVs, found {len(paths)}"
        )

    required = {
        "time",
        "r",
        "theta",
        "z",
        "cell_volume",
        *FIELDS,
    }
    rows: list[dict[str, float]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"{path} has no header")
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"{path} is missing columns {sorted(missing)}"
                )
            for line_number, raw in enumerate(reader, start=2):
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError(f"{path}:{line_number} is malformed")
                try:
                    row = {
                        name: float(value)
                        for name, value in raw.items()
                        if name is not None and value is not None
                    }
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line_number} contains non-numeric data"
                    ) from error
                if not math.isclose(
                    row["time"],
                    expected_time,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError(
                        f"{path}:{line_number} is at t={row['time']}; "
                        f"expected {expected_time}"
                    )
                if any(not math.isfinite(value) for value in row.values()):
                    raise ValueError(
                        f"{path}:{line_number} contains non-finite data"
                    )
                rows.append(row)
    if not rows:
        raise ValueError("SimpleFluid CSV files contain no cells")
    return rows


def bin_index(value: float, edges: list[float], name: str) -> int:
    """Return a cell-bin index with a roundoff-sized endpoint tolerance."""
    tolerance = 1.0e-10 * max(1.0, abs(edges[-1]))
    if value < edges[0] - tolerance or value > edges[-1] + tolerance:
        raise ValueError(
            f"{name} coordinate {value} lies outside "
            f"[{edges[0]}, {edges[-1]}]"
        )
    index = bisect.bisect_right(edges, value) - 1
    return min(max(index, 0), len(edges) - 2)


def rz_volume_averages(
    rows: list[dict[str, float]],
    r_edges: list[float],
    z_edges: list[float],
) -> dict[tuple[int, int], dict[str, float]]:
    """Volume-average arbitrary 3-D cells into the target R-Z bins."""
    accumulators: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        key = (
            bin_index(row["r"], r_edges, "radial"),
            bin_index(row["z"], z_edges, "axial"),
        )
        accumulator = accumulators.setdefault(
            key,
            {
                "cell_volume": 0.0,
                "cell_count": 0.0,
                **{field: 0.0 for field in FIELDS},
            },
        )
        volume = row["cell_volume"]
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError("cell volumes must be finite and positive")
        accumulator["cell_volume"] += volume
        accumulator["cell_count"] += 1.0
        for field in FIELDS:
            accumulator[field] += volume * row[field]

    expected_keys = {
        (radial, axial)
        for radial in range(len(r_edges) - 1)
        for axial in range(len(z_edges) - 1)
    }
    missing = expected_keys.difference(accumulators)
    extra = accumulators.keys() - expected_keys
    if missing or extra:
        raise ValueError(
            f"R-Z aggregation is incomplete (missing={len(missing)}, "
            f"extra={len(extra)})"
        )

    result: dict[tuple[int, int], dict[str, float]] = {}
    for key, accumulator in accumulators.items():
        volume = accumulator["cell_volume"]
        result[key] = {
            "cell_volume": volume,
            "cell_count": accumulator["cell_count"],
            **{
                field: accumulator[field] / volume
                for field in FIELDS
            },
        }
    return result


def bin_weights(
    r_edges: list[float], z_edges: list[float]
) -> dict[tuple[int, int], float]:
    """Return full-cylinder physical volumes for the R-Z bins."""
    return {
        (radial, axial): (
            math.pi
            * (
                r_edges[radial + 1] ** 2
                - r_edges[radial] ** 2
            )
            * (z_edges[axial + 1] - z_edges[axial])
        )
        for radial in range(len(r_edges) - 1)
        for axial in range(len(z_edges) - 1)
    }


def weighted_percentile(
    values: list[float], weights: list[float], fraction: float
) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    target = fraction * math.fsum(weights)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


def error_statistics(
    reference: list[float],
    result: list[float],
    weights: list[float],
) -> dict[str, float]:
    total_weight = math.fsum(weights)
    errors = [value - target for target, value in zip(reference, result)]
    absolute = [abs(value) for value in errors]
    return {
        "bias": math.fsum(
            weight * error for weight, error in zip(weights, errors)
        )
        / total_weight,
        "mae": math.fsum(
            weight * error
            for weight, error in zip(weights, absolute)
        )
        / total_weight,
        "rms": math.sqrt(
            math.fsum(
                weight * error * error
                for weight, error in zip(weights, errors)
            )
            / total_weight
        ),
        "p99_absolute": weighted_percentile(absolute, weights, 0.99),
        "maximum_absolute": max(absolute),
    }


def logarithmic_ratio_statistics(
    reference: list[float],
    result: list[float],
    weights: list[float],
) -> dict[str, float]:
    floor = 1.0e-30
    ratios = [
        math.log10(max(value, floor) / max(target, floor))
        for target, value in zip(reference, result)
    ]
    total_weight = math.fsum(weights)
    mean = math.fsum(
        weight * ratio for weight, ratio in zip(weights, ratios)
    ) / total_weight
    rms = math.sqrt(
        math.fsum(
            weight * ratio * ratio
            for weight, ratio in zip(weights, ratios)
        )
        / total_weight
    )
    within_factor_two = math.fsum(
        weight
        for weight, ratio in zip(weights, ratios)
        if abs(ratio) <= math.log10(2.0)
    ) / total_weight
    return {
        "geometric_mean_ratio_simplefluid_over_openfoam": 10.0**mean,
        "rms_log10_ratio": rms,
        "maximum_absolute_log10_ratio": max(abs(value) for value in ratios),
        "volume_fraction_within_factor_two": within_factor_two,
    }


def parse_color(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))


def palette_color(
    palette: tuple[str, ...], fraction: float
) -> str:
    fraction = min(1.0, max(0.0, fraction))
    position = fraction * (len(palette) - 1)
    left_index = min(int(position), len(palette) - 2)
    local = position - left_index
    left = parse_color(palette[left_index])
    right = parse_color(palette[left_index + 1])
    channels = tuple(
        round(
            left[channel]
            + local * (right[channel] - left[channel])
        )
        for channel in range(3)
    )
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def nice_upper(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(value))
    scale = 10.0**exponent
    normalized = value / scale
    for candidate in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if normalized <= candidate:
            return candidate * scale
    raise AssertionError("unreachable")


def tick_label(value: float) -> str:
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude < 1.0e-3 or magnitude >= 1.0e3):
        return f"{value:.1e}"
    return f"{value:.4g}"


class Svg:
    def __init__(self, width: int, height: int, title: str) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.definitions: list[str] = []
        self.elements: list[str] = []

    def add(self, value: str) -> None:
        self.elements.append(value)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 17.0,
        anchor: str = "start",
        weight: int = 400,
        fill: str = "#17212b",
        transform: str | None = None,
    ) -> None:
        transform_attribute = (
            "" if transform is None
            else f' transform="{html.escape(transform, quote=True)}"'
        )
        self.add(
            f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size:.3f}" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'fill="{fill}"{transform_attribute}>'
            f"{html.escape(value)}</text>"
        )

    def gradient(
        self, identifier: str, palette: tuple[str, ...]
    ) -> None:
        stops = "".join(
            f'<stop offset="{100.0 * index / (len(palette) - 1):.6f}%" '
            f'stop-color="{color}"/>'
            for index, color in enumerate(palette)
        )
        self.definitions.append(
            f'<linearGradient id="{identifier}" x1="0" y1="1" '
            f'x2="0" y2="0">{stops}</linearGradient>'
        )

    def write(self, path: Path) -> None:
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
            'font-family="DejaVu Sans,Segoe UI,sans-serif" '
            'shape-rendering="geometricPrecision">\n'
            f"<title>{html.escape(self.title)}</title>\n"
            f"{definitions}\n"
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            + "\n".join(self.elements)
            + "\n</svg>\n"
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(path)


def render_grid(
    svg: Svg,
    values: dict[tuple[int, int], float],
    r_edges: list[float],
    z_edges: list[float],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    color,
) -> None:
    svg.add('<g shape-rendering="crispEdges">')
    for (radial, axial), value in values.items():
        left = x + r_edges[radial] / RADIUS * width
        right = x + r_edges[radial + 1] / RADIUS * width
        top = y + height - z_edges[axial + 1] / HEIGHT * height
        bottom = y + height - z_edges[axial] / HEIGHT * height
        svg.add(
            f'<rect x="{left:.4f}" y="{top:.4f}" '
            f'width="{right - left + 0.05:.4f}" '
            f'height="{bottom - top + 0.05:.4f}" '
            f'fill="{color(value)}"/>'
        )
    svg.add("</g>")


def render_axes(
    svg: Svg,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
) -> None:
    svg.add(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        'fill="none" stroke="#17212b" stroke-width="1.4"/>'
    )
    svg.text(
        x + width / 2,
        y - 45,
        title,
        size=23,
        weight=600,
        anchor="middle",
    )
    svg.text(
        x + width / 2,
        y - 20,
        subtitle,
        size=14,
        anchor="middle",
        fill="#52606d",
    )
    for radius in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        tick_x = x + radius / RADIUS * width
        svg.add(
            f'<line x1="{tick_x}" y1="{y + height}" '
            f'x2="{tick_x}" y2="{y + height + 7}" '
            'stroke="#17212b" stroke-width="1.1"/>'
        )
        svg.text(
            tick_x,
            y + height + 27,
            f"{radius:.2f}",
            size=13,
            anchor="middle",
        )
    for axial in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        tick_y = y + height - axial / HEIGHT * height
        svg.add(
            f'<line x1="{x - 7}" y1="{tick_y}" '
            f'x2="{x}" y2="{tick_y}" '
            'stroke="#17212b" stroke-width="1.1"/>'
        )
        svg.text(
            x - 12,
            tick_y + 5,
            f"{axial:.2f}",
            size=13,
            anchor="end",
        )
    svg.text(
        x + width / 2,
        y + height + 55,
        "R (m)",
        size=17,
        anchor="middle",
    )
    svg.text(
        x - 62,
        y + height / 2,
        "Z (m)",
        size=17,
        anchor="middle",
        transform=f"rotate(-90 {x - 62} {y + height / 2})",
    )


def render_colorbar(
    svg: Svg,
    identifier: str,
    palette: tuple[str, ...],
    *,
    x: float,
    y: float,
    height: float,
    lower: float,
    upper: float,
    label: str,
    logarithmic: bool = False,
) -> None:
    svg.gradient(identifier, palette)
    svg.add(
        f'<rect x="{x}" y="{y}" width="27" height="{height}" '
        f'fill="url(#{identifier})" stroke="#17212b" stroke-width="1"/>'
    )
    for index in range(5):
        fraction = index / 4
        tick_y = y + height * (1.0 - fraction)
        if logarithmic:
            value = 10.0 ** (
                math.log10(lower)
                + fraction * (math.log10(upper) - math.log10(lower))
            )
        else:
            value = lower + fraction * (upper - lower)
        svg.add(
            f'<line x1="{x + 27}" y1="{tick_y}" '
            f'x2="{x + 34}" y2="{tick_y}" '
            'stroke="#17212b" stroke-width="1"/>'
        )
        svg.text(
            x + 39,
            tick_y + 5,
            tick_label(value),
            size=12,
        )
    svg.text(
        x + 105,
        y + height / 2,
        label,
        size=15,
        anchor="middle",
        transform=f"rotate(-90 {x + 105} {y + height / 2})",
    )


def linear_color(
    palette: tuple[str, ...], lower: float, upper: float
):
    span = upper - lower
    if span <= 0.0:
        span = 1.0
    return lambda value: palette_color(palette, (value - lower) / span)


def logarithmic_color(
    palette: tuple[str, ...], lower: float, upper: float
):
    log_lower = math.log10(lower)
    span = math.log10(upper) - log_lower
    return lambda value: palette_color(
        palette,
        (math.log10(min(upper, max(lower, value))) - log_lower) / span,
    )


def render_distribution(
    path: Path,
    openfoam: dict[tuple[int, int], dict[str, float]],
    simplefluid: dict[tuple[int, int], dict[str, float]],
    r_edges: list[float],
    z_edges: list[float],
    *,
    value,
    quantity: str,
    units: str,
    expected_time: float,
) -> None:
    reference = {key: value(row) for key, row in openfoam.items()}
    result = {key: value(row) for key, row in simplefluid.items()}
    lower = min(0.0, min(reference.values()), min(result.values()))
    upper = nice_upper(
        max(max(reference.values()), max(result.values()))
    )
    if upper <= lower:
        upper = lower + 1.0
    svg = Svg(1220, 1120, f"{quantity} R-Z distribution")
    svg.text(
        610,
        52,
        f"Gaussian fissile-solution tank — {quantity}",
        size=29,
        weight=650,
        anchor="middle",
    )
    svg.text(
        610,
        82,
        f"t = {expected_time:g} s; 100 W full-tank Gaussian; SST k-omega",
        size=16,
        anchor="middle",
        fill="#52606d",
    )
    panel_y = 155
    panel_width = 275
    panel_height = 825
    for identifier, x, title, rows in (
        ("openfoam-scale", 110, "OpenFOAM", reference),
        ("simplefluid-scale", 650, "SimpleFluid", result),
    ):
        render_grid(
            svg,
            rows,
            r_edges,
            z_edges,
            x=x,
            y=panel_y,
            width=panel_width,
            height=panel_height,
            color=linear_color(VIRIDIS, lower, upper),
        )
        render_axes(
            svg,
            x=x,
            y=panel_y,
            width=panel_width,
            height=panel_height,
            title=title,
            subtitle="volume-averaged 50 × 150 R-Z bins",
        )
        render_colorbar(
            svg,
            identifier,
            VIRIDIS,
            x=x + panel_width + 35,
            y=panel_y,
            height=panel_height,
            lower=lower,
            upper=upper,
            label=f"{quantity} ({units})",
        )
    svg.write(path)


def render_error(
    path: Path,
    openfoam: dict[tuple[int, int], dict[str, float]],
    simplefluid: dict[tuple[int, int], dict[str, float]],
    r_edges: list[float],
    z_edges: list[float],
    *,
    value,
    quantity: str,
    units: str,
    statistics: dict[str, float],
) -> None:
    signed = {
        key: value(simplefluid[key]) - value(openfoam[key])
        for key in openfoam
    }
    absolute = {key: abs(error) for key, error in signed.items()}
    limit = nice_upper(max(absolute.values()))
    log_lower = max(limit * 1.0e-6, 1.0e-14)
    svg = Svg(1220, 1120, f"{quantity} R-Z error")
    svg.text(
        610,
        52,
        f"Gaussian fissile-solution tank — {quantity} error",
        size=29,
        weight=650,
        anchor="middle",
    )
    svg.text(
        610,
        82,
        "SimpleFluid minus OpenFOAM after azimuthal volume averaging",
        size=16,
        anchor="middle",
        fill="#52606d",
    )
    panel_y = 155
    panel_width = 275
    panel_height = 825
    render_grid(
        svg,
        signed,
        r_edges,
        z_edges,
        x=110,
        y=panel_y,
        width=panel_width,
        height=panel_height,
        color=linear_color(COOLWARM, -limit, limit),
    )
    render_axes(
        svg,
        x=110,
        y=panel_y,
        width=panel_width,
        height=panel_height,
        title="Signed error",
        subtitle="SimpleFluid − OpenFOAM",
    )
    render_colorbar(
        svg,
        "signed-error-scale",
        COOLWARM,
        x=420,
        y=panel_y,
        height=panel_height,
        lower=-limit,
        upper=limit,
        label=f"Delta {quantity} ({units})",
    )
    render_grid(
        svg,
        absolute,
        r_edges,
        z_edges,
        x=650,
        y=panel_y,
        width=panel_width,
        height=panel_height,
        color=logarithmic_color(MAGMA, log_lower, limit),
    )
    render_axes(
        svg,
        x=650,
        y=panel_y,
        width=panel_width,
        height=panel_height,
        title="Absolute error",
        subtitle=f"log scale; floor {tick_label(log_lower)} {units}",
    )
    render_colorbar(
        svg,
        "absolute-error-scale",
        MAGMA,
        x=960,
        y=panel_y,
        height=panel_height,
        lower=log_lower,
        upper=limit,
        label=f"|Delta {quantity}| ({units})",
        logarithmic=True,
    )
    svg.text(
        610,
        1080,
        (
            f"RMS = {statistics['rms']:.6g} {units}   "
            f"MAE = {statistics['mae']:.6g} {units}   "
            f"p99 = {statistics['p99_absolute']:.6g} {units}   "
            f"max = {statistics['maximum_absolute']:.6g} {units}"
        ),
        size=15,
        weight=550,
        anchor="middle",
    )
    svg.write(path)


def convert_svg(path: Path) -> list[Path]:
    """Create PNG and PDF copies when librsvg is available."""
    executable = shutil.which("rsvg-convert")
    if executable is None:
        return []
    outputs: list[Path] = []
    for output_format in ("png", "pdf"):
        output = path.with_suffix("." + output_format)
        subprocess.run(
            [
                executable,
                "--format",
                output_format,
                "--output",
                str(output),
                str(path),
            ],
            check=True,
        )
        outputs.append(output)
    return outputs


def write_matched_csv(
    path: Path,
    openfoam: dict[tuple[int, int], dict[str, float]],
    simplefluid: dict[tuple[int, int], dict[str, float]],
    r_edges: list[float],
    z_edges: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        columns = [
            "radial_index",
            "axial_index",
            "r_lower",
            "r_upper",
            "z_lower",
            "z_upper",
            "openfoam_bin_volume",
            "simplefluid_bin_volume",
            "simplefluid_cell_count",
        ]
        for field in FIELDS:
            columns.extend(
                (
                    f"openfoam_{field}",
                    f"simplefluid_{field}",
                    f"signed_error_{field}",
                )
            )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for radial, axial in sorted(
            openfoam, key=lambda key: (key[1], key[0])
        ):
            reference = openfoam[(radial, axial)]
            result = simplefluid[(radial, axial)]
            row = {
                "radial_index": radial,
                "axial_index": axial,
                "r_lower": r_edges[radial],
                "r_upper": r_edges[radial + 1],
                "z_lower": z_edges[axial],
                "z_upper": z_edges[axial + 1],
                "openfoam_bin_volume": reference["cell_volume"],
                "simplefluid_bin_volume": result["cell_volume"],
                "simplefluid_cell_count": int(result["cell_count"]),
            }
            for field in FIELDS:
                row[f"openfoam_{field}"] = reference[field]
                row[f"simplefluid_{field}"] = result[field]
                row[f"signed_error_{field}"] = (
                    result[field] - reference[field]
                )
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openfoam-case", type=Path, required=True)
    parser.add_argument("--simplefluid-glob", nargs="+", required=True)
    parser.add_argument("--expected-time", type=float, required=True)
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    r_edges = radial_edges()
    z_edges = axial_edges()
    openfoam_cells, openfoam_maximum_wall_y_plus = read_openfoam(
        arguments.openfoam_case, arguments.expected_time
    )
    simplefluid_cells = read_simplefluid(
        arguments.simplefluid_glob,
        arguments.expected_ranks,
        arguments.expected_time,
    )
    if len(openfoam_cells) != RADIAL_CELLS * AXIAL_CELLS:
        raise ValueError(
            f"expected {RADIAL_CELLS * AXIAL_CELLS} OpenFOAM cells, "
            f"found {len(openfoam_cells)}"
        )

    openfoam = rz_volume_averages(openfoam_cells, r_edges, z_edges)
    simplefluid = rz_volume_averages(
        simplefluid_cells, r_edges, z_edges
    )
    keys = sorted(openfoam, key=lambda key: (key[1], key[0]))
    physical_weights_by_bin = bin_weights(r_edges, z_edges)
    weights = [physical_weights_by_bin[key] for key in keys]

    def values(rows, field):
        return [rows[key][field] for key in keys]

    temperature_reference = values(openfoam, "temperature")
    temperature_result = values(simplefluid, "temperature")
    temperature_statistics = error_statistics(
        temperature_reference, temperature_result, weights
    )
    temperature_scale = max(
        max(abs(value - REFERENCE_TEMPERATURE) for value in temperature_reference),
        1.0e-30,
    )
    temperature_statistics["openfoam_peak_rise"] = max(
        temperature_reference
    ) - REFERENCE_TEMPERATURE
    temperature_statistics["simplefluid_peak_rise"] = max(
        temperature_result
    ) - REFERENCE_TEMPERATURE
    temperature_statistics["rms_over_openfoam_peak_rise"] = (
        temperature_statistics["rms"] / temperature_scale
    )

    speed_reference = [
        math.sqrt(
            openfoam[key]["ur"] ** 2
            + openfoam[key]["utheta"] ** 2
            + openfoam[key]["uz"] ** 2
        )
        for key in keys
    ]
    speed_result = [
        math.sqrt(
            simplefluid[key]["ur"] ** 2
            + simplefluid[key]["utheta"] ** 2
            + simplefluid[key]["uz"] ** 2
        )
        for key in keys
    ]
    speed_statistics = error_statistics(
        speed_reference, speed_result, weights
    )
    vector_errors = [
        math.sqrt(
            (simplefluid[key]["ur"] - openfoam[key]["ur"]) ** 2
            + (
                simplefluid[key]["utheta"]
                - openfoam[key]["utheta"]
            )
            ** 2
            + (simplefluid[key]["uz"] - openfoam[key]["uz"]) ** 2
        )
        for key in keys
    ]
    total_weight = math.fsum(weights)
    speed_statistics["vector_rms"] = math.sqrt(
        math.fsum(
            weight * error * error
            for weight, error in zip(weights, vector_errors)
        )
        / total_weight
    )
    speed_scale = max(max(speed_reference), 1.0e-30)
    speed_statistics["openfoam_peak_speed"] = max(speed_reference)
    speed_statistics["simplefluid_peak_speed"] = max(speed_result)
    speed_statistics["vector_rms_over_openfoam_peak_speed"] = (
        speed_statistics["vector_rms"] / speed_scale
    )

    source_statistics = error_statistics(
        values(openfoam, "qdot_fission"),
        values(simplefluid, "qdot_fission"),
        weights,
    )
    source_statistics["openfoam_full_equivalent_power"] = (
        math.fsum(
            row["qdot_fission"] * row["cell_volume"]
            for row in openfoam_cells
        )
        / SECTOR_FRACTION
    )
    source_statistics["simplefluid_integrated_power"] = math.fsum(
        row["qdot_fission"] * row["cell_volume"]
        for row in simplefluid_cells
    )
    source_statistics["openfoam_peak_power_density"] = max(
        values(openfoam, "qdot_fission")
    )
    source_statistics["simplefluid_peak_power_density"] = max(
        values(simplefluid, "qdot_fission")
    )
    source_statistics["rms_over_openfoam_peak_power_density"] = (
        source_statistics["rms"]
        / source_statistics["openfoam_peak_power_density"]
    )

    turbulence_statistics = {
        field: logarithmic_ratio_statistics(
            values(openfoam, field),
            values(simplefluid, field),
            weights,
        )
        for field in ("k", "omega", "nut")
    }

    summary = {
        "expected_time_s": arguments.expected_time,
        "mesh": {
            "radial_cells": RADIAL_CELLS,
            "axial_cells": AXIAL_CELLS,
            "openfoam_cells": len(openfoam_cells),
            "simplefluid_cells": len(simplefluid_cells),
            "nominal_rz_spacing_m": 0.002,
            "boundary_layer_count_per_wall": BOUNDARY_LAYER_COUNT,
            "boundary_layer_first_height_m": BOUNDARY_LAYER_FIRST_HEIGHT,
            "boundary_layer_growth_ratio": BOUNDARY_LAYER_GROWTH,
            "openfoam_full_equivalent_volume_m3": (
                math.fsum(row["cell_volume"] for row in openfoam_cells)
                / SECTOR_FRACTION
            ),
            "simplefluid_volume_m3": math.fsum(
                row["cell_volume"] for row in simplefluid_cells
            ),
            "analytic_volume_m3": math.pi * RADIUS**2 * HEIGHT,
        },
        "source": source_statistics,
        "temperature": temperature_statistics,
        "velocity_magnitude": speed_statistics,
        "sst_log_ratio": turbulence_statistics,
        "wall_y_plus": {
            "openfoam_maximum": openfoam_maximum_wall_y_plus,
            "simplefluid_maximum": max(
                row["wall_y_plus"] for row in simplefluid_cells
            ),
        },
    }

    output_directory = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "comparison_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    matched_path = output_directory / "rz_matched.csv"
    write_matched_csv(
        matched_path, openfoam, simplefluid, r_edges, z_edges
    )

    quantities = (
        (
            "temperature",
            lambda row: row["temperature"] - REFERENCE_TEMPERATURE,
            "Temperature rise",
            "K",
            temperature_statistics,
        ),
        (
            "velocity",
            lambda row: math.sqrt(
                row["ur"] ** 2
                + row["utheta"] ** 2
                + row["uz"] ** 2
            ),
            "Velocity magnitude",
            "m/s",
            speed_statistics,
        ),
        (
            "fission_source",
            lambda row: row["qdot_fission"],
            "Fission power density",
            "W/m3",
            source_statistics,
        ),
    )
    generated: list[Path] = []
    for stem, value, title, units, statistics in quantities:
        distribution_path = (
            output_directory / f"{stem}_rz_distribution.svg"
        )
        error_path = output_directory / f"{stem}_rz_error.svg"
        render_distribution(
            distribution_path,
            openfoam,
            simplefluid,
            r_edges,
            z_edges,
            value=value,
            quantity=title,
            units=units,
            expected_time=arguments.expected_time,
        )
        render_error(
            error_path,
            openfoam,
            simplefluid,
            r_edges,
            z_edges,
            value=value,
            quantity=title,
            units=units,
            statistics=statistics,
        )
        generated.extend((distribution_path, error_path))
        generated.extend(convert_svg(distribution_path))
        generated.extend(convert_svg(error_path))

    print(f"comparison summary: {summary_path}")
    print(f"matched R-Z data: {matched_path}")
    for path in generated:
        print(f"figure: {path}")
    print(
        "temperature: "
        f"rms={temperature_statistics['rms']:.12g} K, "
        "normalized_rms="
        f"{temperature_statistics['rms_over_openfoam_peak_rise']:.12g}"
    )
    print(
        "velocity: "
        f"vector_rms={speed_statistics['vector_rms']:.12g} m/s, "
        "normalized_vector_rms="
        f"{speed_statistics['vector_rms_over_openfoam_peak_speed']:.12g}"
    )
    print(
        "power: "
        "OpenFOAM_full_equivalent="
        f"{source_statistics['openfoam_full_equivalent_power']:.12g} W, "
        "SimpleFluid="
        f"{source_statistics['simplefluid_integrated_power']:.12g} W"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
