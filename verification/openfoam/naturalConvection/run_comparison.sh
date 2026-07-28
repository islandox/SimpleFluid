#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
invocation_dir=$(pwd -P)
np=${1:-6}
if [ "${SIMPLEFLUID_SHIRI_NR+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_NTHETA+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_NZ+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEPS+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_DT+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_RESTART_PREFIX+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEADY_STATE+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEADY_TOLERANCE+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEADY_MIN_STEPS+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEADY_CONSECUTIVE_STEPS+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEADY_MAX_RETRIES+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEADY_REJECTION_RECOVERY_STEPS+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEADY_REJECTION_SAFETY+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEADY_MIN_DT+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEADY_MAX_DT+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEADY_TARGET_COURANT+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEADY_DT_GROWTH+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_STEADY_DT_REDUCTION+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_ENFORCE_AXISYMMETRY+x}" = x ] \
    || [ "${SIMPLEFLUID_SHIRI_PRESSURE_VELOCITY_COUPLING+x}" = x ]; then
    echo "Mesh/time/restart/steady overrides are only supported by run_simplefluid.sh;" >&2
    echo "the matched comparison requires the default 40x20x100, t=0.4 setup." >&2
    exit 2
fi
output_dir=${SIMPLEFLUID_SHIRI_OUTPUT_DIR:-"$case_dir/profiles"}
case "$output_dir" in
    /*) ;;
    *) output_dir="$invocation_dir/$output_dir" ;;
esac

"$case_dir/openfoam/Allrun" "$np"
"$case_dir/run_simplefluid.sh" "$np"
python3 "$case_dir/compare_profiles.py" \
    --openfoam-case "$case_dir/openfoam" \
    --simplefluid-glob "$output_dir"/simplefluid_cells_rank\*.csv \
    --expected-time 0.4 \
    --expected-ranks "$np" \
    --expected-cells 80000 \
    --expected-radial-cells 40 \
    --expected-theta-cells 20 \
    --expected-axial-cells 100
