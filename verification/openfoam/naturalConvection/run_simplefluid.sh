#!/bin/sh
set -eu

case "${1:-6}" in
    ''|*[!0-9]*) echo "MPI rank count must be an integer from 1 to 6" >&2; exit 2 ;;
esac
np="${1:-6}"
if [ "$np" -lt 1 ] || [ "$np" -gt 6 ]; then
    echo "MPI rank count must be from 1 to 6" >&2
    exit 2
fi

case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$case_dir/../../.." && pwd)
invocation_dir=$(pwd -P)
build_config=${SIMPLEFLUID_BUILD_CONFIG:-RelWithDebInfo}
executable="$repo_dir/build/bin/$build_config/natural_convection_shiri"
output_dir=${SIMPLEFLUID_SHIRI_OUTPUT_DIR:-"$case_dir/profiles"}
case "$output_dir" in
    /*) ;;
    *) output_dir="$invocation_dir/$output_dir" ;;
esac

(
    cd "$repo_dir"
    cmake --build --preset "$build_config" -j "$np" \
        --target natural_convection_shiri
)
mkdir -p "$output_dir"
rm -f "$output_dir"/simplefluid_cells_rank*.csv \
      "$output_dir"/natural_convection_shiri*.vtu \
      "$output_dir"/natural_convection_shiri*.pvtu

cd "$output_dir"
export SIMPLEFLUID_SHIRI_OUTPUT_PREFIX="$output_dir/simplefluid_cells"
if [ "$np" -eq 1 ]; then
    "$executable"
else
    mpiexec -n "$np" "$executable"
fi
