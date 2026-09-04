#!/bin/sh
set -eu
case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$case_dir/../../.." && pwd)
mode=${1:-transient}
output=${2:-"$case_dir/results/simplefluid-$mode"}
case "$mode" in steady|transient) ;; *) echo 'Mode must be steady or transient' >&2; exit 2 ;; esac
. "$repo_dir/verification/environments.sh"
export_build_env "$repo_dir"
simplefluid_build_target planar_ale_comparison "${SIMPLEFLUID_BUILD_JOBS:-4}"
executable=$(simplefluid_executable planar_ale_comparison)
"$executable" --mode "$mode" --output "$output"
