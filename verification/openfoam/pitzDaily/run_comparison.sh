#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
np=${1:-6}
acceptance_file=${SIMPLEFLUID_PITZ_ACCEPTANCE_FILE:-"$case_dir/reference/physical_acceptance.json"}
output_dir=${SIMPLEFLUID_PITZ_OUTPUT_DIR:-"$case_dir/profiles"}

case "$np" in
    ''|*[!0-9]*) echo "MPI rank count must be an integer from 1 to 6" >&2; exit 2 ;;
esac
if [ "$np" -lt 1 ] || [ "$np" -gt 6 ]; then
    echo "MPI rank count must be from 1 to 6" >&2
    exit 2
fi

# Do not spend time on both solvers unless the selected physical limits carry
# authenticated provenance and every executable setting. The values are
# validated numeric/boolean fields, printed as a whitespace-delimited shell
# contract in the same fixed order used below.
settings=$(python3 "$case_dir/compare_profiles.py" \
    --tolerances "$acceptance_file" \
    --required-scope physical_reference \
    --print-simplefluid-settings)
set -- $settings
if [ "$#" -ne 20 ]; then
    echo "Qualified pitzDaily settings must contain twenty values" >&2
    exit 2
fi
if [ "$np" -ne "$4" ]; then
    echo "Qualified pitzDaily reference requires $4 MPI ranks, not $np" >&2
    exit 2
fi
export SIMPLEFLUID_PITZ_MESH_DIVISOR=$1
export SIMPLEFLUID_PITZ_STEPS=$2
export SIMPLEFLUID_PITZ_DT=$3
export SIMPLEFLUID_PITZ_STEADY_STATE=$5
export SIMPLEFLUID_PITZ_LINEAR_TOLERANCE=$6
export SIMPLEFLUID_PITZ_STEADY_CONSECUTIVE_STEPS=$7
export SIMPLEFLUID_PITZ_STEADY_MIN_STEPS=$8
export SIMPLEFLUID_PITZ_STEADY_MAX_RETRIES=$9
export SIMPLEFLUID_PITZ_STEADY_REJECTION_RECOVERY_STEPS=${10}
export SIMPLEFLUID_PITZ_STEADY_TOLERANCE=${11}
export SIMPLEFLUID_PITZ_STEADY_MIN_DT=${12}
export SIMPLEFLUID_PITZ_STEADY_MAX_DT=${13}
export SIMPLEFLUID_PITZ_STEADY_TARGET_COURANT=${14}
export SIMPLEFLUID_PITZ_STEADY_DT_GROWTH=${15}
export SIMPLEFLUID_PITZ_STEADY_DT_REDUCTION=${16}
export SIMPLEFLUID_PITZ_STEADY_REJECTION_SAFETY=${17}
export SIMPLEFLUID_PITZ_STEADY_RELAXED_LINEAR_TOLERANCE=${18}
export SIMPLEFLUID_PITZ_STEADY_FULL_ACCURACY_UPDATE_RATIO=${19}
export SIMPLEFLUID_PITZ_STEADY_PROGRESS_INTERVAL=${20}

mkdir -p "$output_dir"
output_dir=$(CDPATH= cd -- "$output_dir" && pwd)
export SIMPLEFLUID_PITZ_OUTPUT_DIR=$output_dir

"$case_dir/openfoam/Allrun" "$np"
"$case_dir/run_simplefluid.sh" "$np"
python3 "$case_dir/compare_profiles.py" \
    --openfoam-case "$case_dir/openfoam" \
    --simplefluid-glob "$output_dir/simplefluid_cells_rank*.csv" \
    --tolerances "$acceptance_file" \
    --required-scope physical_reference
