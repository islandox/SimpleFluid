#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$case_dir/../../.." && pwd)
invocation_dir=$(pwd -P)
. "$repo_dir/verification/environments.sh"
export_build_env "$repo_dir"

build_jobs=${SIMPLEFLUID_BUILD_JOBS:-4}
output_dir=${SIMPLEFLUID_PROFILE_OUTPUT_DIR:-"$invocation_dir/profiles"}
case "$output_dir" in
    /*) ;;
    *) output_dir="$invocation_dir/$output_dir" ;;
esac

simplefluid_build_target testVerificationCases "$build_jobs"
executable="$(simplefluid_executable testVerificationCases)"

mkdir -p "$output_dir"
rm -f "$output_dir/simplefluid_lineX.csv" \
      "$output_dir/simplefluid_lineY.csv"
export SIMPLEFLUID_PROFILE_OUTPUT_DIR="$output_dir"

"$executable" \
    --gtest_filter=VerificationCasesTest.LidDrivenCavityRe1000
