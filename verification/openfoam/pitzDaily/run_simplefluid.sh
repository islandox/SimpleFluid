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
. "$repo_dir/verification/environments.sh"
export_build_env "$repo_dir"
output_dir=${SIMPLEFLUID_PITZ_OUTPUT_DIR:-"$case_dir/profiles"}

simplefluid_build_target pitz_daily "$np"
executable="$(simplefluid_executable pitz_daily)"
mkdir -p "$output_dir"
rm -f "$output_dir"/simplefluid_cells_rank*.csv \
      "$output_dir"/pitz_daily*.vtu \
      "$output_dir"/pitz_daily*.pvtu

cd "$output_dir"
export SIMPLEFLUID_PITZ_OUTPUT_PREFIX="$output_dir/simplefluid_cells"
mpiexec -n "$np" "$executable"
