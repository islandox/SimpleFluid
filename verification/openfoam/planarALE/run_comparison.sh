#!/bin/sh
set -eu
case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
mode=${1:-transient}
output_root=${2:-"$case_dir/results"}
case "$mode" in steady|transient) ;; *) echo 'Mode must be steady or transient' >&2; exit 2 ;; esac
mkdir -p "$output_root"
output_root=$(CDPATH= cd -- "$output_root" && pwd)
run_dir=$(mktemp -d "$output_root/$mode.XXXXXX")
cp "${SIMPLEFLUID_VERIFICATION_MESH:-$case_dir/mesh.dat}" "$run_dir/mesh.dat"
export SIMPLEFLUID_VERIFICATION_MESH="$run_dir/mesh.dat"
python3 "$case_dir/../structured_mesh.py" --mesh "$SIMPLEFLUID_VERIFICATION_MESH" \
    --manifest "$case_dir/$mode.json" --output-manifest "$run_dir/manifest.json"
started=$(python3 -c 'import time; print(time.time())')
printf 'Running planarALE %s comparison in %s\n' "$mode" "$run_dir"
"$case_dir/run_simplefluid.sh" "$mode" "$run_dir/simplefluid" >"$run_dir/simplefluid.log" 2>&1
"$case_dir/run_openfoam.sh" "$mode" "$run_dir/openfoam" >"$run_dir/openfoam.log" 2>&1
python3 "$case_dir/../compare_verification.py" \
    --manifest "$run_dir/manifest.json" \
    --openfoam "$run_dir/openfoam/history.csv" \
    --simplefluid "$run_dir/simplefluid/history.csv" \
    --not-before "$started" --report "$run_dir/comparison.json"
printf 'Comparison report: %s/comparison.json\n' "$run_dir"
python3 "$case_dir/../plot_water_fields.py" \
    --manifest "$run_dir/manifest.json" \
    --openfoam "$run_dir/openfoam/fields.csv" \
    --simplefluid "$run_dir/simplefluid/fields.csv" \
    --not-before "$started" --output-directory "$run_dir/figures"
