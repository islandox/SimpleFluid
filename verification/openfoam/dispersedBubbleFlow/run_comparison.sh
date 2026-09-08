#!/bin/sh
set -eu
case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
mode=${1:-transient}
case "$mode" in steady|transient) ;; *) echo "Usage: $0 [steady|transient] [output-root]" >&2; exit 2 ;; esac
[ "$#" -le 2 ] || { echo "Too many arguments" >&2; exit 2; }
output_root=${2:-"${TMPDIR:-/tmp}/simplefluid-openfoam-verification"}
mkdir -p "$output_root"
output_root=$(CDPATH= cd -- "$output_root" && pwd)
run_dir=$(mktemp -d "$output_root/dispersedBubbleFlow-$mode.XXXXXX")
cp "${SIMPLEFLUID_VERIFICATION_MESH:-$case_dir/mesh.dat}" "$run_dir/mesh.dat"
export SIMPLEFLUID_VERIFICATION_MESH="$run_dir/mesh.dat"
python3 "$case_dir/../structured_mesh.py" --mesh "$SIMPLEFLUID_VERIFICATION_MESH" \
    --manifest "$case_dir/$mode.json" --output-manifest "$run_dir/manifest.json"
start_epoch=$(python3 -c 'import time; print(time.time())')
echo "Dispersed bubble $mode comparison run: $run_dir"
"$case_dir/run_openfoam.sh" "$mode" "$run_dir/openfoam"
"$case_dir/run_simplefluid.sh" "$mode" "$run_dir/simplefluid"
python3 "$case_dir/../compare_verification.py" \
    --manifest "$run_dir/manifest.json" \
    --openfoam "$run_dir/openfoam/profiles.csv" \
    --simplefluid "$run_dir/simplefluid/profiles.csv" \
    --report "$run_dir/comparison.json" --not-before "$start_epoch"
echo "Comparison report: $run_dir/comparison.json"
python3 "$case_dir/../plot_water_fields.py" \
    --manifest "$run_dir/manifest.json" \
    --openfoam "$run_dir/openfoam/fields.csv" \
    --simplefluid "$run_dir/simplefluid/fields.csv" \
    --not-before "$start_epoch" --output-directory "$run_dir/figures"
