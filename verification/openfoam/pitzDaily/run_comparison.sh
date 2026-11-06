#!/bin/sh
set -eu

# The existing acceptance manifest contract pins the historical algorithms.
# Diagnostic target-study overrides are retained separately, never ambient.
export SIMPLEFLUID_PITZ_TRANSPORT_BACKEND=gmres
export SIMPLEFLUID_PITZ_PRESSURE_MODE=gmres-muelu

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
# validated numeric/boolean/enum fields, printed as a whitespace-delimited shell
# contract in the same fixed order used below.
settings=$(python3 "$case_dir/compare_profiles.py" \
    --tolerances "$acceptance_file" \
    --required-scope physical_reference \
    --print-simplefluid-settings)
set -- $settings
if [ "$#" -ne 25 ]; then
    echo "Qualified pitzDaily settings must contain twenty-five values" >&2
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
export SIMPLEFLUID_PITZ_TRANSPORT_PRECONDITIONER=${21}
export SIMPLEFLUID_PITZ_MODIFIED_PRESSURE=${22}
export SIMPLEFLUID_PITZ_SAMPLING_METHOD=${23}
export SIMPLEFLUID_PITZ_OPENFOAM_OUTLET_STRESS=${24}
export SIMPLEFLUID_PITZ_MESH_MULTIPLIER=${25}

mkdir -p "$output_dir"
output_dir=$(CDPATH= cd -- "$output_dir" && pwd)
export SIMPLEFLUID_PITZ_OUTPUT_DIR=$output_dir

# Acceptance uses the immutable data that the manifest actually qualifies.
# Regeneration is a separate evidence-collection operation.
"$case_dir/run_simplefluid.sh" "$np"
sampling_method=$SIMPLEFLUID_PITZ_SAMPLING_METHOD
if [ "$sampling_method" = "openfoam-cellPoint-matched-mesh-v1" ]; then
    sampling_dir="$output_dir/common-sampling-$(date +%s)-$$"
    python3 "$case_dir/common_sampling.py" --source-case "$case_dir/openfoam" \
        --cells "$output_dir/simplefluid_cells_rank*.csv" --divisor "$SIMPLEFLUID_PITZ_MESH_DIVISOR" --output "$sampling_dir" --mesh-multiplier "$SIMPLEFLUID_PITZ_MESH_MULTIPLIER"
    set -- --simplefluid-profile-case "$sampling_dir"
else
    set --
fi
python3 "$case_dir/compare_profiles.py" "$@" \
    --use-retained-reference \
    --simplefluid-glob "$output_dir/simplefluid_cells_rank*.csv" \
    --tolerances "$acceptance_file" \
    --required-scope physical_reference
