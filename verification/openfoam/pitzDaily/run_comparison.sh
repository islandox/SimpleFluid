#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
np=${1:-6}
acceptance_file=${SIMPLEFLUID_PITZ_ACCEPTANCE_FILE:-"$case_dir/reference/physical_acceptance.json"}

case "$np" in
    ''|*[!0-9]*) echo "MPI rank count must be an integer from 1 to 6" >&2; exit 2 ;;
esac
if [ "$np" -lt 1 ] || [ "$np" -gt 6 ]; then
    echo "MPI rank count must be from 1 to 6" >&2
    exit 2
fi

# Do not spend time on both solvers unless the selected physical limits carry
# authenticated provenance and exact executable settings. The four values are
# validated numeric fields, printed as a whitespace-delimited shell contract.
settings=$(python3 "$case_dir/compare_profiles.py" \
    --tolerances "$acceptance_file" \
    --required-scope physical_reference \
    --print-simplefluid-settings)
set -- $settings
if [ "$#" -ne 4 ]; then
    echo "Qualified pitzDaily settings must contain four values" >&2
    exit 2
fi
if [ "$np" -ne "$4" ]; then
    echo "Qualified pitzDaily reference requires $4 MPI ranks, not $np" >&2
    exit 2
fi
export SIMPLEFLUID_PITZ_MESH_DIVISOR=$1
export SIMPLEFLUID_PITZ_STEPS=$2
export SIMPLEFLUID_PITZ_DT=$3

"$case_dir/openfoam/Allrun" "$np"
"$case_dir/run_simplefluid.sh" "$np"
python3 "$case_dir/compare_profiles.py" \
    --openfoam-case "$case_dir/openfoam" \
    --simplefluid-glob "$case_dir/profiles/simplefluid_cells_rank*.csv" \
    --tolerances "$acceptance_file" \
    --required-scope physical_reference
