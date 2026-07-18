#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$case_dir/../../.." && pwd)
invocation_dir=$(pwd -P)
build_config=${SIMPLEFLUID_BUILD_CONFIG:-RelWithDebInfo}
build_jobs=${SIMPLEFLUID_BUILD_JOBS:-4}
executable="$repo_dir/build/bin/$build_config/natural_convection_shiri"
profile_dir=${SIMPLEFLUID_SHIRI_PROFILE_DIR:-"$case_dir/profiles/callgrind"}
case "$profile_dir" in
    /*) ;;
    *) profile_dir="$invocation_dir/$profile_dir" ;;
esac
profile_data="$profile_dir/callgrind.out"
profile_report="$profile_dir/callgrind.annotated.txt"

if ! command -v valgrind >/dev/null 2>&1; then
    echo "Valgrind with Callgrind is required to profile this case." >&2
    exit 1
fi
if ! command -v callgrind_annotate >/dev/null 2>&1; then
    echo "callgrind_annotate is required to generate the text report." >&2
    exit 1
fi

case "$build_jobs" in
    ''|*[!0-9]*) echo "SIMPLEFLUID_BUILD_JOBS must be a positive integer." >&2; exit 2 ;;
esac
if [ "$build_jobs" -lt 1 ]; then
    echo "SIMPLEFLUID_BUILD_JOBS must be a positive integer." >&2
    exit 2
fi

(
    cd "$repo_dir"
    cmake --build --preset "$build_config" -j "$build_jobs" \
        --target natural_convection_shiri
)

mkdir -p "$profile_dir"
rm -f "$profile_data" "$profile_report" \
      "$profile_dir/simplefluid_cells_rank0.csv" \
      "$profile_dir/natural_convection_shiri.vtu"

# Keep the default profiling workload short but large enough to exercise
# repeated pressure corrections. Callers can override every case parameter.
export SIMPLEFLUID_SHIRI_NR=${SIMPLEFLUID_SHIRI_NR:-6}
export SIMPLEFLUID_SHIRI_NTHETA=${SIMPLEFLUID_SHIRI_NTHETA:-3}
export SIMPLEFLUID_SHIRI_NZ=${SIMPLEFLUID_SHIRI_NZ:-12}
export SIMPLEFLUID_SHIRI_STEPS=${SIMPLEFLUID_SHIRI_STEPS:-2}
export SIMPLEFLUID_SHIRI_DT=${SIMPLEFLUID_SHIRI_DT:-0.002}
export SIMPLEFLUID_SHIRI_OUTPUT_PREFIX="$profile_dir/simplefluid_cells"

cd "$profile_dir"
valgrind --tool=callgrind \
    --collect-atstart=no \
    '--toggle-collect=*BoussinesqSolver*step*' \
    --callgrind-out-file="$profile_data" \
    "$executable"

callgrind_annotate \
    --inclusive=yes \
    --auto=no \
    --threshold=90 \
    "$profile_data" > "$profile_report"

echo "Callgrind data: $profile_data"
echo "Annotated report: $profile_report"
