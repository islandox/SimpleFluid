#!/bin/sh
set -eu

case "${1:-12}" in
    ''|*[!0-9]*)
        echo "MPI rank count must be an integer from 1 to 12" >&2
        exit 2
        ;;
esac
np="${1:-12}"
if [ "$np" -lt 1 ] || [ "$np" -gt 12 ]; then
    echo "MPI rank count must be from 1 to 12" >&2
    exit 2
fi

case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$case_dir/../../.." && pwd)
. "$repo_dir/verification/environments.sh"
export_build_env "$repo_dir"
simplefluid_build_target fissile_solution_tank_sst

output_dir=${SIMPLEFLUID_TANK_OUTPUT_DIR:-"$case_dir/profiles"}
case "$output_dir" in
    /*) ;;
    *) output_dir="$(pwd -P)/$output_dir" ;;
esac
mkdir -p "$output_dir"
rm -f "$output_dir"/simplefluid_cells_rank*.csv \
      "$output_dir"/fissile_solution_tank_sst*.vtu \
      "$output_dir"/fissile_solution_tank_sst*.pvtu
cd "$output_dir"
export SIMPLEFLUID_TANK_OUTPUT_PREFIX="$output_dir/simplefluid_cells"

if [ "$np" -eq 1 ]; then
    "$SIMPLEFLUID_BIN_DIR/fissile_solution_tank_sst"
else
    mpiexec --use-hwthread-cpus -n "$np" \
        "$SIMPLEFLUID_BIN_DIR/fissile_solution_tank_sst"
fi
