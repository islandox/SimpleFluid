#!/bin/bash
set -e
case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
output=${1:-$(mktemp -d "${TMPDIR:-/tmp}/bottom-convection-of.XXXXXX")}
if ! command -v wmake >/dev/null 2>&1; then
    set +e
    . "${OPENFOAM_BASHRC:-/opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc}"
    set -e
fi
set -eu
mkdir -p "$output"
output=$(CDPATH= cd -- "$output" && pwd)
python3 "$case_dir/prepare_openfoam.py" --output "$output" --mesh "${SIMPLEFLUID_VERIFICATION_MESH:-$case_dir/mesh.dat}"
cp -R "$case_dir/solver" "$output/solver"
cp "$case_dir/../StructuredCaseMesh.H" "$output/solver/"
mkdir "$output/bin"
export FOAM_USER_APPBIN="$output/bin"
(cd "$output/solver" && wmake) > "$output/build.log" 2>&1
blockMesh -case "$output" > "$output/blockMesh.log" 2>&1
checkMesh -case "$output" > "$output/checkMesh.log" 2>&1
grep -q 'Mesh OK' "$output/checkMesh.log"
"$output/bin/bottomBubblyConvectionFoam" -case "$output" > "$output/solver.log" 2>&1
printf 'OpenFOAM results: %s\n' "$output"
