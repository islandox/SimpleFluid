#!/usr/bin/env python3
"""Build a fresh OpenFOAM case from the same numeric inputs as SimpleFluid."""
import argparse
import math
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("steady", "transient"), required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    w, h, cells = values["width"], values["height"], values["cells"]
    if int(cells) != cells or cells < 2:
        raise ValueError("cells must be an integer >= 2")
    block = header("blockMeshDict") + f"""
scale 1;
vertices ((0 0 0) ({w} 0 0) ({w} {w} 0) (0 {w} 0)
          (0 0 {h}) ({w} 0 {h}) ({w} {w} {h}) (0 {w} {h}));
blocks (hex (0 1 2 3 4 5 6 7) (1 1 {int(cells)}) simpleGrading (1 1 1));
edges ();
boundary
(
    zmin {{ type wall; faces ((0 3 2 1)); }}
    zmax {{ type patch; faces ((4 5 6 7)); }}
    walls {{ type wall; faces ((0 1 5 4) (1 2 6 5) (2 3 7 6) (3 0 4 7)); }}
);
mergePatchPairs ();
"""
    (args.output / "system/blockMeshDict").write_text(block)


if __name__ == "__main__":
    main()
