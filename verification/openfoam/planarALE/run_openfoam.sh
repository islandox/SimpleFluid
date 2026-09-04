#!/bin/bash
set -e
case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
mode=${1:-transient}
output=${2:-"$case_dir/results/openfoam-$mode"}
case "$mode" in steady|transient) ;; *) echo 'Mode must be steady or transient' >&2; exit 2 ;; esac
if ! command -v wmake >/dev/null 2>&1; then
    foam_bashrc=${OPENFOAM_BASHRC:-/opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc}
    if [ ! -f "$foam_bashrc" ]; then
        echo 'Source the OpenFOAM environment or set OPENFOAM_BASHRC.' >&2
        exit 2
    fi
    set +e
    . "$foam_bashrc"
    set -e
    command -v wmake >/dev/null
fi
set -eu
mkdir -p "$output"
output=$(CDPATH= cd -- "$output" && pwd)
cp -R "$case_dir/template/." "$output/"
mkdir -p "$output/constant" "$output/solver" "$output/bin"
python3 "$case_dir/../reference_water.py" \
    --properties "$case_dir/../reference_water.properties" \
    --openfoam-dictionary "$output/constant/referenceWater"
cp -R "$case_dir/solver/." "$output/solver/"
# Keep compilation, executable, and case output under the selected run directory.
export FOAM_USER_APPBIN="$output/bin"
(cd "$output/solver" && wmake) >"$output/build.log" 2>&1
blockMesh -case "$output" >"$output/blockMesh.log" 2>&1
checkMesh -case "$output" >"$output/checkMesh.log" 2>&1
if ! grep -q 'Mesh OK' "$output/checkMesh.log"; then
    cat "$output/checkMesh.log" >&2
    exit 1
fi
"$output/bin/planarALEBudgetFoam" -case "$output" -mode "$mode" >"$output/solver.log" 2>&1
printf 'OpenFOAM planarALE %s: %s/history.csv\n' "$mode" "$output"
