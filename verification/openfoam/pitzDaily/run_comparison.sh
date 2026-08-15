#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
np=${1:-6}

"$case_dir/openfoam/Allrun" "$np"
"$case_dir/run_simplefluid.sh" "$np"
python3 "$case_dir/compare_profiles.py" \
    --openfoam-case "$case_dir/openfoam" \
    --simplefluid-glob "$case_dir"/profiles/simplefluid_cells_rank\*.csv
