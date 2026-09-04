#!/usr/bin/env bash
set -eo pipefail
case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
mode=${1:-transient}
case "$mode" in steady|transient) ;; *) echo "Usage: $0 [steady|transient] [output-directory]" >&2; exit 2 ;; esac
[[ $# -le 2 ]] || { echo "Too many arguments" >&2; exit 2; }
output_dir=${2:-$(mktemp -d "${TMPDIR:-/tmp}/openfoam-bubbles-$mode.XXXXXX")}
mkdir -p "$output_dir"
output_dir=$(CDPATH= cd -- "$output_dir" && pwd)
if [[ -n "${OPENFOAM_BASHRC:-}" ]] || ! command -v wmake >/dev/null 2>&1; then
    foam_bashrc=${OPENFOAM_BASHRC:-/opt/OpenFOAM/OpenFOAM-v2606/etc/bashrc}
    [[ -r "$foam_bashrc" ]] || { echo "Set OPENFOAM_BASHRC to an OpenFOAM installation (tested v2606)" >&2; exit 2; }
    # OpenFOAM's optional dependency probes do not support errexit/nounset.
    set +e
    source "$foam_bashrc"
fi
set -eu
command -v wmake >/dev/null
command -v blockMesh >/dev/null
command -v checkMesh >/dev/null
python3 "$case_dir/prepare_openfoam.py" --mode "$mode" --output "$output_dir"
cp -R "$case_dir/openfoam/solver" "$output_dir/reference-solver"
mkdir "$output_dir/bin"
export FOAM_USER_APPBIN="$output_dir/bin"
(cd "$output_dir/reference-solver" && wmake) > "$output_dir/log.wmake" 2>&1
blockMesh -case "$output_dir" > "$output_dir/log.blockMesh" 2>&1
checkMesh -case "$output_dir" > "$output_dir/log.checkMesh" 2>&1
"$output_dir/bin/dispersedBubbleReferenceFoam" -case "$output_dir" > "$output_dir/log.solver" 2>&1
echo "OpenFOAM $mode dispersed bubble profiles: $output_dir/profiles.csv"
