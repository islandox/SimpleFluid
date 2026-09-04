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
start_epoch=$(python3 -c 'import time; print(time.time())')
echo "Dispersed bubble $mode comparison run: $run_dir"
"$case_dir/run_openfoam.sh" "$mode" "$run_dir/openfoam"
"$case_dir/run_simplefluid.sh" "$mode" "$run_dir/simplefluid"
python3 "$case_dir/../compare_verification.py" \
    --manifest "$case_dir/$mode.json" \
    --openfoam "$run_dir/openfoam/profiles.csv" \
    --simplefluid "$run_dir/simplefluid/profiles.csv" \
    --report "$run_dir/comparison.json" --not-before "$start_epoch"
echo "Comparison report: $run_dir/comparison.json"
