#!/usr/bin/env python3
"""Plot matched column-cell outputs as SVG, with optional PNG/PDF exports.

Uses the same SVG/librsvg workflow as the existing verification figures.
These figures are diagnostics; the case comparison remains the acceptance gate.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import shutil
import subprocess

from compare_verification import ComparisonError, load_manifest, read_samples, validate_manifest

FIELDS = {
    "temperature_K": ("Temperature", "K", 1e-12),
    "density_kg_m3": ("Liquid density", "kg/m³", 1e-12),
    "alpha_g": ("Gas volume fraction", "1", 1e-14),
    "speed_m_s": ("Liquid speed", "m/s", 1e-12),
}
COLUMNS = ["z_lower_m", "z_upper_m", "temperature_K", "density_kg_m3", "alpha_g",
           "ux_m_s", "uy_m_s", "uz_m_s"]
VIRIDIS = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"]
DIVERGING = ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"]
MASK = "#d4d8dd"


def relative_error(reference: float, actual: float, floor: float) -> float | None:
    """Signed percent error; undefined near a zero reference, including 0/0."""
    return None if abs(reference) <= floor else 100.0 * (actual - reference) / abs(reference)


def load_fields(manifest: dict, openfoam: Path, simplefluid: Path,
                not_before: float | None = None) -> tuple[dict, dict]:
    spatial = manifest.get("spatial_output", {})
    samples = spatial.get("cell_samples")
    planar_grid = spatial.get("layout") == "xz"
    columns = COLUMNS + (["x_lower_m", "x_upper_m"] if planar_grid else [])
    width = spatial.get("width_m")
    if (not isinstance(width, (int, float)) or isinstance(width, bool)
            or not math.isfinite(width) or width <= 0):
        raise ComparisonError("spatial_output.width_m must be positive and finite")
    contract = validate_manifest({**manifest, "expected_samples": samples, "conservation": {},
        "quantities": {key: {"units": "SI", "absolute_tolerance": 0, "relative_tolerance": 0}
                       for key in columns}})
    if openfoam.samefile(simplefluid):
        raise ComparisonError("spatial solver inputs must be distinct files")
    datasets = tuple(read_samples(path, contract, not_before) for path in (openfoam, simplefluid))
    for rows in datasets:
        for index, _ in enumerate(manifest["expected_times_s"]):
            groups = {}
            for sample in samples:
                row = rows[index, sample]
                bounds = (row["x_lower_m"], row["x_upper_m"]) if planar_grid else (0.0, width)
                groups.setdefault(bounds, []).append(row)
            previous_x, column_height = 0.0, None
            for (xmin, xmax), group in sorted(groups.items()):
                if xmax <= xmin or abs(xmin-previous_x) > 1e-9:
                    raise ComparisonError("spatial grid has a horizontal gap or overlap")
                previous_x = xmax
                previous = 0.0
                for row in sorted(group, key=lambda r: r["z_lower_m"]):
                    lower, upper = row["z_lower_m"], row["z_upper_m"]
                    if upper <= lower or abs(lower - previous) > 1e-9:
                        raise ComparisonError("spatial column has a gap, overlap, or invalid cell height")
                    previous = upper
                if column_height is not None and abs(previous-column_height) > 1e-9:
                    raise ComparisonError("spatial columns have inconsistent heights")
                column_height = previous
            if abs(previous_x-width) > 1e-9:
                raise ComparisonError("spatial grid does not cover its declared width")
            for sample in samples:
                row = rows[index, sample]
                lower, upper = row["z_lower_m"], row["z_upper_m"]
                if row["temperature_K"] <= 0 or row["density_kg_m3"] <= 0 or not 0 <= row["alpha_g"] <= 1:
                    raise ComparisonError("invalid temperature, liquid density, or gas volume fraction")
                row["speed_m_s"] = math.hypot(*(row[key] for key in ("ux_m_s", "uy_m_s", "uz_m_s")))
    for key, reference in datasets[0].items():
        actual = datasets[1][key]
        if abs(actual["time_s"] - reference["time_s"]) > contract["time_tolerance_s"]:
            raise ComparisonError("spatial solver times differ")
        extents = ["z_lower_m", "z_upper_m"] + (["x_lower_m", "x_upper_m"] if planar_grid else [])
        if any(abs(actual[k] - reference[k]) > 1e-9 for k in extents):
            raise ComparisonError("spatial solver cell extents differ")
    expected_mesh=manifest.get('mesh_edges_m')
    if expected_mesh:
        nx=len(expected_mesh['x'])-1
        for rows in datasets:
            heights={index:max(rows[index,s]['z_upper_m'] for s in samples)
                     for index in range(len(manifest['expected_times_s']))}
            for (index,sample),row in rows.items():
                ident=int(sample);ix,iz=ident%nx,ident//nx
                if ident<0 or iz>=len(expected_mesh['z'])-1:
                    raise ComparisonError('Cell ID outside shared reference mesh')
                scale=heights[index]/expected_mesh['z'][-1] if manifest['case']=='planarALE' else 1
                expected={'z_lower_m':expected_mesh['z'][iz]*scale,'z_upper_m':expected_mesh['z'][iz+1]*scale}
                if planar_grid:expected.update(x_lower_m=expected_mesh['x'][ix],x_upper_m=expected_mesh['x'][ix+1])
                if any(abs(row[k]-v)>1e-9 for k,v in expected.items()):
                    raise ComparisonError('Cell bounds differ from shared reference mesh')
    return datasets


def errors(reference: dict, actual: dict, fields: dict = FIELDS) -> dict:
    result = {}
    for key, (_, _, floor) in fields.items():
        delta = actual[key] - reference[key]
        absolute = abs(delta)
        relative = relative_error(reference[key], actual[key], floor)
        if key == "speed_m_s":
            # A speed-only difference could hide a reversed velocity vector.
            absolute = math.hypot(*(actual[k] - reference[k] for k in ("ux_m_s", "uy_m_s", "uz_m_s")))
            relative = None if reference[key] <= floor else 100 * absolute / reference[key]
        result[key] = {"difference": delta, "absolute_error": absolute, "relative_error_percent": relative}
    return result


def color(value: float | None, bounds: tuple[float, float], palette: list[str]) -> str:
    if value is None:
        return MASK
    fraction = min(1.0, max(0.0, (value - bounds[0]) / (bounds[1] - bounds[0])))
    position = fraction * (len(palette) - 1)
    index = min(int(position), len(palette) - 2)
    weight = position - index
    channels = [round(int(palette[index][j:j+2], 16) * (1-weight)
                      + int(palette[index+1][j:j+2], 16) * weight) for j in (1, 3, 5)]
    return "#" + "".join(f"{value:02x}" for value in channels)


def text(x: float, y: float, value: str, size: int = 16, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{html.escape(value)}</text>'


def rectangle(x: float, y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" shape-rendering="crispEdges"/>'


def limits(values: list[float], field: str, mode: str, fields: dict = FIELDS) -> tuple[float, float]:
    if mode == "relative":
        bound = max([1e-12, *(abs(value) for value in values)])
        return (0, bound) if field == "speed_m_s" else (-bound, bound)
    if mode == "absolute":
        return 0, max([fields[field][2], *values])
    lower, upper = min(values), max(values)
    if field in ("alpha_g", "speed_m_s", "k_m2_s2", "omega_1_s", "nut_m2_s", "wall_yplus"):
        return 0, max(upper, fields[field][2])
    if upper - lower <= 1e-10 * max(abs(lower), abs(upper), 1):
        margin = max(abs(lower), 1) * 1e-5
        return lower - margin, upper + margin
    return lower, upper


def render(manifest: dict, reference: dict, actual: dict, differences: dict,
           mode: str, history: bool, fields: dict = FIELDS) -> str:
    times = manifest["expected_times_s"]
    samples = manifest["spatial_output"]["cell_samples"]
    planar_grid = manifest["spatial_output"].get("layout") == "xz"
    if history and planar_grid:
        middle = manifest["spatial_output"]["width_m"]/2
        samples = [sample for sample in samples
                   if reference[0,sample]["x_lower_m"] <= middle < reference[0,sample]["x_upper_m"]]
    slice_bounds=manifest['spatial_output'].get('y_slice_bounds_m')
    slice_note=(f'Central y cell plane: {slice_bounds[0]:.5g}–{slice_bounds[1]:.5g} m. ' if slice_bounds else '')
    title = f"{manifest['case']} / {manifest['mode']} — "
    title += {"distribution": "field distributions", "relative": "relative errors (%)",
              "absolute": "absolute errors"}[mode]
    title += " / full history" if history else f" / t = {times[-1]:g} s"
    if manifest.get('validation_scope'):
        title += " / partial check"
    panels = 2 if mode == "distribution" else 1
    height = 170 + panels * 410
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1760" height="{height}" viewBox="0 0 1760 {height}">',
           '<style>text{font-family:DejaVu Sans,sans-serif;fill:#17212b}</style>',
           '<defs><marker id="arrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5" fill="#17212b"/></marker></defs>',
           rectangle(0, 0, 1760, height, "#f7f9fb"), text(35, 38, title, 25)]
    bubble = manifest["case"] == "dispersedBubbleFlow"
    convection = manifest["case"] == "bottomHeatedBubblyConvection"
    scope = ("Prescribed liquid velocity; isothermal water; transported bubble fraction."
             if bubble else "Bottom heat and H2 source; both solvers solve buoyant circulation from rest; cooled side/top walls."
             if convection else "Liquid only: gas fraction = 0. Velocity: solved SimpleFluid cells vs affine OpenFOAM kinematic reference.")
    svg.append(text(35, 68, manifest.get("field_context", scope), 16))
    svg.append(text(35, 94, "History: central x column; sample-centred time bins, no time interpolation."
                    if history and planar_grid else "History: sample-centred time bins, no time interpolation; height follows each mesh."
                    if history else slice_note+"Native x–z section; velocity components shown by arrows."
                    if planar_grid else "Native Cartesian x–z cell sections; one cell across x, uniform through y.", 15))
    selected = list(range(len(times))) if history else [len(times)-1]
    edges = [times[0], *[(a+b)/2 for a, b in zip(times, times[1:])], times[-1]]
    for field_index, (field, (label, unit, _)) in enumerate(fields.items()):
        keys = [(index, sample) for index in selected for sample in samples]
        if mode == "distribution":
            values = [rows[key][field] for rows in (reference, actual) for key in keys]
        else:
            name = "relative_error_percent" if mode == "relative" else "absolute_error"
            values = [differences[key][field][name] for key in keys if differences[key][field][name] is not None]
        bounds = limits(values, field, mode, fields)
        palette = (DIVERGING[2:] if field == "speed_m_s" else DIVERGING) if mode == "relative" else VIRIDIS
        for row_index in range(panels):
            rows = reference if row_index == 0 else actual
            left, top, width, plot_height = 85 + field_index*435, 165 + row_index*410, 305, 270
            prefix = ("OpenFOAM" if row_index == 0 else "SimpleFluid") + " · " if mode == "distribution" else ""
            title_label = "Velocity vector error" if field == "speed_m_s" and mode != "distribution" else label
            svg.append(text(left, top-32, prefix + title_label, 18))
            unit_label = unit if mode != "relative" else "%"
            if mode == "distribution" and all(rows[key][field] == 0 for key in keys):
                unit_label += " · identically zero"
            svg.append(text(left, top-10, unit_label, 14))
            x_max = times[-1] if history else manifest["spatial_output"]["width_m"]
            y_max = 1.0 if history else max(reference[key]["z_upper_m"] for key in keys)
            for index in selected:
                level = max(rows[index, sample]["z_upper_m"] for sample in samples)
                for sample in samples:
                    key = (index, sample)
                    cell = rows[key]
                    lo, hi = cell["z_lower_m"], cell["z_upper_m"]
                    if history:
                        lo, hi = lo/level, hi/level
                    x0, x1 = (edges[index], edges[index+1]) if history else (cell.get("x_lower_m",0),cell.get("x_upper_m",x_max))
                    value = cell[field] if mode == "distribution" else differences[key][field][name]
                    svg.append(rectangle(left+x0/x_max*width, top+(1-hi/y_max)*plot_height,
                        (x1-x0)/x_max*width+0.02, (hi-lo)/y_max*plot_height+0.02, color(value, bounds, palette)))
            if mode == "distribution" and field == "speed_m_s" and not history:
                scale = max(bounds[1], fields[field][2])
                if planar_grid:
                    centers_x=sorted({(rows[selected[-1],s]["x_lower_m"]+rows[selected[-1],s]["x_upper_m"])/2 for s in samples})
                    centers_z=sorted({(rows[selected[-1],s]["z_lower_m"]+rows[selected[-1],s]["z_upper_m"])/2 for s in samples})
                    selected_x={min(centers_x,key=lambda x:abs(x-(i+.5)*x_max/8)) for i in range(8)}
                    selected_z={min(centers_z,key=lambda z:abs(z-(i+.5)*y_max/10)) for i in range(10)}
                for offset, sample in enumerate(samples):
                    if not planar_grid and offset % max(1, len(samples)//10):
                        continue
                    cell = rows[selected[-1], sample]
                    if planar_grid:
                        center_x=(cell["x_lower_m"]+cell["x_upper_m"])/2
                        center_z=(cell["z_lower_m"]+cell["z_upper_m"])/2
                        if center_x not in selected_x or center_z not in selected_z:
                            continue
                    if cell["speed_m_s"] <= fields[field][2]:
                        continue
                    cx = left+(cell["x_lower_m"]+cell["x_upper_m"])/2/x_max*width if planar_grid else left+width/2
                    cy = top+(1-(cell["z_lower_m"]+cell["z_upper_m"])/2/y_max)*plot_height
                    dx, dy = 20*cell["ux_m_s"]/scale, -20*cell["uz_m_s"]/scale
                    svg.append(f'<line x1="{cx-dx/2}" y1="{cy-dy/2}" x2="{cx+dx/2}" y2="{cy+dy/2}" stroke="#17212b" stroke-width="1.4" marker-end="url(#arrow)"/>')
            svg.append(f'<rect x="{left}" y="{top}" width="{width}" height="{plot_height}" fill="none" stroke="#617080"/>')
            if convection and mode == "distribution" and not history:
                source = manifest["spatial_output"].get("source_region")
                if source:
                    sx=left+source["x_lower_m"]/x_max*width
                    sy=top+(1-source["z_upper_m"]/y_max)*plot_height
                    sw=(source["x_upper_m"]-source["x_lower_m"])/x_max*width
                    sh=(source["z_upper_m"]-source["z_lower_m"])/y_max*plot_height
                    svg.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" fill="none" stroke="#ff7900" stroke-width="2" stroke-dasharray="5 3"/>')
            for tick in range(5):
                fraction = tick/4
                svg.append(text(left-8, top+(1-fraction)*plot_height+5, f"{fraction*y_max:.5g}", 13, "end"))
                svg.append(text(left+fraction*width, top+plot_height+20, f"{fraction*x_max:.4g}", 13, "middle"))
            svg.append(text(left+width/2, top+plot_height+42, "Time (s)" if history else "x (m)", 15, "middle"))
            axis_x, axis_y = left-64, top+plot_height/2
            svg.append(f'<g transform="rotate(-90 {axis_x} {axis_y})">' +
                       text(axis_x, axis_y, "z/L(t)" if history else "z (m)", 13, "middle") + '</g>')
            bar_top = top + plot_height + 59
            for segment in range(100):
                value = bounds[0] + (bounds[1]-bounds[0])*(segment+.5)/100
                svg.append(rectangle(left+segment*width/100, bar_top, width/100+.1, 13, color(value, bounds, palette)))
            svg.append(text(left, bar_top+31, f"{bounds[0]:.6g}", 13))
            svg.append(text(left+width, bar_top+31, f"{bounds[1]:.6g}", 13, "end"))
            if mode == "relative":
                masked = sum(differences[key][field][name] is None for key in keys)
                svg.append(text(left, bar_top+52, f"Gray: undefined ({masked}/{len(keys)}); floor {fields[field][2]:g}", 12))
    footer = ("Relative error: 100(SF−OF)/|OF|; velocity: 100||U_SF−U_OF||/||U_OF||. Values at/below the stated reference floor are undefined."
              if mode == "relative" else "Resolved laminar thermal/gas buoyancy; prescribed bubble slip. This is a developing transient, not a steady-state claim." if convection
              else "Liquid velocity comparison is diagnostic; the OpenFOAM ALE reference does not solve momentum." if not bubble
              else "Velocity is the prescribed liquid carrier field (0.1 m/s); bubble rise includes an additional prescribed 0.4 m/s slip.")
    if mode == "distribution" and not history:
        footer += " Arrows: Ux, Uz."
        if convection:
            footer += " Orange box: heat/H2 source."
    svg.append(text(35, height-13, footer, 14))
    svg.append("</svg>")
    return "\n".join(svg)


def render_mesh(manifest: dict, reference: dict) -> str:
    width=manifest['spatial_output']['width_m']
    cells=[row for (index,_),row in reference.items() if index==0]
    declared=manifest.get('mesh_edges_m',{})
    x=declared.get('x') or sorted({value for row in cells for value in (row.get('x_lower_m',0),row.get('x_upper_m',width))})
    z=declared.get('z') or sorted({value for row in cells for value in (row['z_lower_m'],row['z_upper_m'])})
    height=z[-1];left,top,size=85,95,520
    elements=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="700">',
              '<style>text{font-family:DejaVu Sans,sans-serif;fill:#17212b}</style>',
              rectangle(0,0,1000,700,'#f7f9fb'),text(35,40,manifest['case']+' — shared reference mesh',23),
              text(35,67,'Actual cell boundaries at t = 0; OpenFOAM and SimpleFluid extents checked for agreement.',15),
              rectangle(left,top,size,size,'#ffffff')]
    for value in x:
        px=left+value/width*size
        elements.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{top+size}" stroke="#526170" stroke-width=".7"/>')
    for value in z:
        py=top+(1-value/height)*size
        elements.append(f'<line x1="{left}" y1="{py}" x2="{left+size}" y2="{py}" stroke="#526170" stroke-width=".7"/>')
    for i in range(5):
        elements.append(text(left-10,top+(1-i/4)*size+5,f'{height*i/4:.5g}',14,'end'))
        elements.append(text(left+size*i/4,top+size+24,f'{width*i/4:.5g}',14,'middle'))
    elements.extend([text(left+size/2,top+size+50,'x (m)',16,'middle'),
                     '<g transform="rotate(-90 22 355)">'+text(22,355,'z (m)',16,'middle')+'</g>'])
    dx=[b-a for a,b in zip(x,x[1:])];dz=[b-a for a,b in zip(z,z[1:])]
    notes=[f"{len(x)-1} × {len(declared.get('y',[0,1]))-1} × {len(z)-1} cells",f"Total: {manifest.get('mesh_statistics',{}).get('cells',len(cells))}",f'Slice cells: {len(cells)}',
           f'First Δx: {dx[0]:.6g} m',f'First Δz: {dz[0]:.6g} m',
           f'Min Δx: {min(dx):.6g} m',f'Min Δz: {min(dz):.6g} m',
           'No geometry interpolation.']
    for i,note in enumerate(notes):elements.append(text(650,145+40*i,note,17))
    elements.append('</svg>')
    return '\n'.join(elements)


def generate(manifest: dict, reference: dict, actual: dict, output: Path, formats: list[str],
             fields: dict = FIELDS) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    differences = {key: errors(reference[key], actual[key], fields) for key in reference}
    statistics = {"case": manifest["case"], "mode": manifest["mode"], "status": "diagnostic",
                  "matched_cells": len(reference), "fields": {}}
    if manifest.get('validation_scope'):
        statistics['validation_scope'] = manifest['validation_scope']
    for field, (label, unit, floor) in fields.items():
        records = [value[field] for value in differences.values()]
        relative = [r["relative_error_percent"] for r in records if r["relative_error_percent"] is not None]
        statistics["fields"][field] = {"label": label, "units": unit, "reference_floor": floor,
            "max_absolute_error": max(r["absolute_error"] for r in records),
            "max_absolute_relative_error_percent": max(map(abs, relative)) if relative else None,
            "undefined_relative_samples": len(records)-len(relative)}
    records = []
    for key in sorted(reference):
        record = {"time_s": reference[key]["time_s"], "sample": key[1]}
        for solver, rows in (("openfoam", reference), ("simplefluid", actual)):
            columns = list(dict.fromkeys([*COLUMNS, "speed_m_s", *fields])) + (["x_lower_m", "x_upper_m"] if "x_lower_m" in rows[key] else [])
            record.update({f"{solver}_{column}": rows[key][column] for column in columns})
        for field in fields:
            record.update({f"{field}_{metric}": value for metric, value in differences[key][field].items()})
        records.append(record)
    with (output/"matched_fields.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    converter = shutil.which("rsvg-convert")
    if any(fmt != "svg" for fmt in formats) and converter is None:
        raise ValueError("PNG/PDF output requires rsvg-convert; install librsvg or select --formats svg")
    plots = [("mesh", False), ("distribution", False), ("relative", False), ("absolute", False),
             ("distribution", True), ("relative", True)]
    gallery = ["<!doctype html><html><head><meta charset='utf-8'><title>Water verification fields</title>",
               "<style>body{font:16px sans-serif;margin:2em;color:#17212b}img{width:100%;max-width:1600px}p{max-width:1100px}</style></head><body>",
               f"<h1>{html.escape(manifest['case'])} / {html.escape(manifest['mode'])}</h1>",
               f"<p>{html.escape(manifest.get('validation_scope', 'Complete declared output history.'))}</p>",
               "<p>Matched cell data, not interpolated fields. Relative errors at or below the reference floors are undefined (gray), including zero/zero. Velocity errors use the full vector difference.</p>",
               (f"<p>{html.escape(manifest.get('field_context', 'SST field diagnostics; wall y+ is zero in interior cells.'))}</p>" if fields != FIELDS else
                "<p>Bubble carrier velocity is prescribed. ALE gas fraction is identically zero (liquid-only case); its velocity reference is affine mesh kinematics, not an OpenFOAM momentum solution. The velocity figures are diagnostic.</p>"),
               "<p><a href='matched_fields.csv'>Matched values and errors (CSV)</a> · <a href='statistics.json'>Statistics and denominator floors</a></p>"]
    if manifest["case"] == "bottomHeatedBubblyConvection":
        gallery[4] = "<p>Both solvers advance momentum, pressure, temperature and dilute gas transport. Bottom-localized heat and gas sources drive convection from rest. Histories show the central x column; all cells are retained in CSV.</p>"
    for mode, history in plots:
        stem = ("history_" if history else "") + mode
        svg = output/f"{stem}.svg"
        svg.write_text(render_mesh(manifest,reference) if mode=='mesh' else render(manifest, reference, actual, differences, mode, history, fields), encoding="utf-8")
        for fmt in formats:
            if fmt != "svg":
                subprocess.run([converter, "--format", fmt, "--output", str(output/f"{stem}.{fmt}"), str(svg)], check=True)
        gallery.append(f"<h2>{stem.replace('_', ' ').title()}</h2><p>" + " · ".join(
            f"<a href='{stem}.{fmt}'>{fmt.upper()}</a>" for fmt in dict.fromkeys(["svg", *formats])) + "</p>")
        gallery.append(f"<img src='{stem}.svg' alt='{stem}'>")
    (output/"statistics.json").write_text(json.dumps(statistics, indent=2, allow_nan=False)+"\n")
    (output/"index.html").write_text("\n".join([*gallery, "</body></html>"]), encoding="utf-8")
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--openfoam", required=True, type=Path)
    parser.add_argument("--simplefluid", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--not-before", type=float)
    parser.add_argument("--formats", nargs="+", choices=("svg", "png", "pdf"),
                        help="Default: SVG plus PNG/PDF when rsvg-convert is available")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    reference, actual = load_fields(manifest, args.openfoam, args.simplefluid, args.not_before)
    formats = args.formats or (["svg", "png", "pdf"] if shutil.which("rsvg-convert") else ["svg"])
    statistics = generate(manifest, reference, actual, args.output_directory, formats)
    print(json.dumps(statistics, indent=2, allow_nan=False))
    print(f"Figures: {args.output_directory / 'index.html'}")


if __name__ == "__main__":
    main()
