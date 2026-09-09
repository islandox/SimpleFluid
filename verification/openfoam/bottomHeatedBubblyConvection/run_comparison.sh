#!/bin/sh
set -eu
case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
output_root=${1:-"$case_dir/results"}
mkdir -p "$output_root"
output_root=$(CDPATH= cd -- "$output_root" && pwd)
run=$(mktemp -d "$output_root/transient.XXXXXX")
cp "${SIMPLEFLUID_VERIFICATION_MESH:-$case_dir/mesh.dat}" "$run/mesh.dat"
export SIMPLEFLUID_VERIFICATION_MESH="$run/mesh.dat"
python3 "$case_dir/../structured_mesh.py" --mesh "$SIMPLEFLUID_VERIFICATION_MESH" \
    --manifest "$case_dir/transient.json" --output-manifest "$run/manifest.json"
started=$(python3 -c 'import time; print(time.time())')
printf 'Bottom-heated bubbly convection comparison: %s\n' "$run"
"$case_dir/run_openfoam.sh" "$run/openfoam" > "$run/openfoam.log" 2>&1
"$case_dir/run_simplefluid.sh" "$run/simplefluid" > "$run/simplefluid.log" 2>&1
# Keep the field figures even when a numerical comparison exceeds its limits.
comparison_status=0
python3 "$case_dir/../compare_verification.py" --manifest "$run/manifest.json" \
    --openfoam "$run/openfoam/history.csv" --simplefluid "$run/simplefluid/history.csv" \
    --not-before "$started" --report "$run/comparison.json" || comparison_status=$?
python3 "$case_dir/../plot_water_fields.py" --manifest "$run/manifest.json" \
    --openfoam "$run/openfoam/fields.csv" --simplefluid "$run/simplefluid/fields.csv" \
    --not-before "$started" --output-directory "$run/figures"
printf 'Comparison: %s/comparison.json\nFigures: %s/figures/index.html\n' "$run" "$run"

python3 "$case_dir/../plot_sst_fields.py" --manifest "$run/manifest.json" \
    --openfoam-directory "$run/openfoam" --simplefluid-directory "$run/simplefluid" \
    --not-before "$started" --output-directory "$run/sst-figures"
exit "$comparison_status"
