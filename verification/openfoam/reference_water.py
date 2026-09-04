#!/usr/bin/env python3
"""Read shared SI material inputs and optionally write an OpenFOAM dictionary.

The checked-in snapshot comes from export_reference_water.cc linked to the
optional SimpleFluid IF97 library. This converter only copies material inputs;
it neither calculates nor reads SimpleFluid solution fields. The SimpleFluid
comparison drivers verify all recorded coefficients against live IF97 calls.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


PROPERTY_KEYS = {
    "schema_version", "temperature_K", "absolute_pressure_Pa", "density_kg_m3",
    "specific_heat_capacity_J_kg_K", "dynamic_viscosity_Pa_s",
    "thermal_conductivity_W_m_K", "thermal_expansion_1_K", "surface_tension_N_m",
    "kinematic_viscosity_m2_s", "thermal_diffusivity_m2_s",
}


def read_reference_water(path: Path) -> dict[str, float]:
    """Reject missing, duplicate, unknown, nonfinite, or inconsistent inputs."""
    values: dict[str, float] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        tokens = raw.split("#", 1)[0].split()
        if not tokens:
            continue
        if len(tokens) != 2 or tokens[0] in values:
            raise ValueError(f"{path}:{line_number}: malformed or duplicate property")
        value = float(tokens[1])
        if not math.isfinite(value):
            raise ValueError(f"{path}:{line_number}: nonfinite property {tokens[0]}")
        values[tokens[0]] = value
    if set(values) != PROPERTY_KEYS:
        raise ValueError(f"{path}: invalid material keys: "
                         f"missing={sorted(PROPERTY_KEYS - values.keys())}, "
                         f"unexpected={sorted(values.keys() - PROPERTY_KEYS)}")
    if values["schema_version"] != 1:
        raise ValueError(f"{path}: expected schema_version 1")
    if any(value <= 0 for key, value in values.items() if key != "thermal_expansion_1_K"):
        raise ValueError(f"{path}: water material properties must be positive")
    derived = {
        "kinematic_viscosity_m2_s": values["dynamic_viscosity_Pa_s"] / values["density_kg_m3"],
        "thermal_diffusivity_m2_s": values["thermal_conductivity_W_m_K"]
        / (values["density_kg_m3"] * values["specific_heat_capacity_J_kg_K"]),
    }
    for key, expected in derived.items():
        if not math.isfinite(expected) or not math.isclose(values[key], expected, rel_tol=2e-9, abs_tol=0):
            raise ValueError(f"{path}: inconsistent derived material property {key}")
    return values


def write_openfoam_dictionary(properties: Path, output: Path) -> None:
    if properties.resolve() == output.resolve():
        raise ValueError("OpenFOAM dictionary output must not overwrite the material snapshot")
    values = read_reference_water(properties)
    text = ("// Shared IF97 reference material inputs; no solver solution fields.\n"
            "FoamFile { version 2.0; format ascii; class dictionary; object referenceWater; }\n")
    text += "".join(f"{key} {value:.17g};\n" for key, value in sorted(values.items()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--properties", type=Path,
                        default=Path(__file__).with_name("reference_water.properties"))
    parser.add_argument("--openfoam-dictionary", required=True, type=Path)
    args = parser.parse_args()
    write_openfoam_dictionary(args.properties, args.openfoam_dictionary)


if __name__ == "__main__":
    main()
