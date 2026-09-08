#!/usr/bin/env python3
"""Build a fresh OpenFOAM case from the same numeric inputs as SimpleFluid."""
import argparse
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reference_water import read_reference_water
from structured_mesh import write_openfoam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("steady", "transient"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=Path(__file__).with_name("mesh.dat"))
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    values = {}
    for raw in (directory / "reference.properties").read_text().splitlines():
        tokens = raw.split("#", 1)[0].split()
        if not tokens:
            continue
        if len(tokens) != 2 or tokens[0] in values:
            raise ValueError(f"Malformed or duplicate parameter: {raw}")
        value = float(tokens[1])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite parameter: {raw}")
        values[tokens[0]] = value
    water = read_reference_water(directory.parent / "reference_water.properties")
    if values.keys() & water.keys():
        raise ValueError("Problem parameters must not override the shared IF97 reference state")
    values.update(water)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output case must be empty: {args.output}")
    shutil.copytree(directory / "openfoam/template", args.output, dirs_exist_ok=True)
    (args.output / "constant").mkdir()
    (args.output / "0").mkdir()
    header = lambda name: f"FoamFile {{ version 2.0; format ascii; class dictionary; object {name}; }}\n"
    properties = header("verificationProperties") + f"mode {args.mode};\n"
    properties += "".join(f"{key} {value:.17g};\n" for key, value in values.items())
    (args.output / "constant/verificationProperties").write_text(properties)
    control = header("controlDict") + f"""
application dispersedBubbleReferenceFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {values[args.mode + '_end_time']:.17g};
deltaT {values['dt']:.17g};
writeControl timeStep;
writeInterval 1000000;
writeFormat ascii;
writePrecision 17;
timeFormat general;
timePrecision 12;
runTimeModifiable false;
"""
    (args.output / "system/controlDict").write_text(control)
    write_openfoam(args.mesh,args.output)


if __name__ == "__main__":
    main()
