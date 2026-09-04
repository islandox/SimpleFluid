#!/bin/sh
set -eu
case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
mode=${1:-transient}
output_root=${2:-"$case_dir/results"}
case "$mode" in steady|transient) ;; *) echo 'Mode must be steady or transient' >&2; exit 2 ;; esac
mkdir -p "$output_root"
output_root=$(CDPATH= cd -- "$output_root" && pwd)
run_dir=$(mktemp -d "$output_root/$mode.XXXXXX")
started=$(python3 -c 'import time; print(time.time())')
printf 'Running planarALE %s comparison in %s\n' "$mode" "$run_dir"
"$case_dir/run_simplefluid.sh" "$mode" "$run_dir/simplefluid" >"$run_dir/simplefluid.log" 2>&1
"$case_dir/run_openfoam.sh" "$mode" "$run_dir/openfoam" >"$run_dir/openfoam.log" 2>&1
python3 "$case_dir/../compare_verification.py" \
    --manifest "$case_dir/$mode.json" \
    --openfoam "$run_dir/openfoam/history.csv" \
    --simplefluid "$run_dir/simplefluid/history.csv" \
    --not-before "$started" --report "$run_dir/comparison.json"
printf 'Comparison report: %s/comparison.json\n' "$run_dir"
