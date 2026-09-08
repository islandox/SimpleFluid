#!/bin/sh
set -eu
case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$case_dir/../../.." && pwd)
output=${1:-$(mktemp -d "${TMPDIR:-/tmp}/bottom-convection-sf.XXXXXX")}
. "$repo_dir/verification/environments.sh"
export_build_env "$repo_dir"
simplefluid_build_target bottom_heated_bubbly_convection
executable=$(simplefluid_executable bottom_heated_bubbly_convection)
"$executable" --output "$output" --properties "$case_dir/reference.properties" \
    --water-properties "$case_dir/../reference_water.properties" \
    --mesh-file "${SIMPLEFLUID_VERIFICATION_MESH:-$case_dir/mesh.dat}"
