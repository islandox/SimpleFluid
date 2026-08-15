#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
np=${1:-12}

if [ "${SIMPLEFLUID_TANK_RZ_SPACING+x}" = x ] \
    || [ "${SIMPLEFLUID_TANK_CIRCUMFERENTIAL_SPACING+x}" = x ] \
    || [ "${SIMPLEFLUID_TANK_BOUNDARY_LAYER_COUNT+x}" = x ] \
    || [ "${SIMPLEFLUID_TANK_BOUNDARY_LAYER_FIRST_HEIGHT+x}" = x ] \
    || [ "${SIMPLEFLUID_TANK_BOUNDARY_LAYER_GROWTH+x}" = x ] \
    || [ "${SIMPLEFLUID_TANK_STEPS+x}" = x ] \
    || [ "${SIMPLEFLUID_TANK_DT+x}" = x ]; then
    echo "Mesh and time overrides are only supported by run_simplefluid.sh;" >&2
    echo "the matched comparison requires the default 50x150, t=120 s setup." >&2
    exit 2
fi

output_dir=${SIMPLEFLUID_TANK_OUTPUT_DIR:-"$case_dir/profiles"}
case "$output_dir" in
    /*) ;;
    *) output_dir="$(pwd -P)/$output_dir" ;;
esac

"$case_dir/openfoam/Allrun" "$np"
"$case_dir/run_simplefluid.sh" "$np"
python3 "$case_dir/plot_rz_comparison.py" \
    --openfoam-case "$case_dir/openfoam" \
    --simplefluid-glob "$output_dir/simplefluid_cells_rank*.csv" \
    --expected-time 120 \
    --expected-ranks "$np" \
    --output-directory "$output_dir/rz"
