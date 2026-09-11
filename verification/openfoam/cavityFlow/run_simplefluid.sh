#!/bin/sh
set -eu

usage()
{
    cat <<'EOF'
Usage: run_simplefluid.sh [--coupling piso|nox] [--reynolds 100|1000]

Defaults: PISO (or SIMPLEFLUID_CAVITY_COUPLING), Re=1000.
NOX requires the selected build tree to be configured with
SIMPLEFLUID_ENABLE_NOX=ON. The launcher does not enable it automatically.
EOF
}

launcher_error()
{
    printf 'SimpleFluid cavity: %s\n' "$*" >&2
    exit 2
}

coupling=${SIMPLEFLUID_CAVITY_COUPLING:-piso}
reynolds=1000
while [ "$#" -gt 0 ]; do
    case "$1" in
        --coupling)
            [ "$#" -ge 2 ] || launcher_error '--coupling requires piso or nox.'
            coupling=$2
            shift 2
            ;;
        --reynolds)
            [ "$#" -ge 2 ] || launcher_error '--reynolds requires 100 or 1000.'
            reynolds=$2
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            launcher_error "unknown argument '$1'. See --help."
            ;;
    esac
done
case "$coupling" in
    piso|nox) ;;
    *) launcher_error "coupling must be piso or nox (got '$coupling')." ;;
esac
case "$reynolds" in
    100|1000) ;;
    *) launcher_error "Reynolds number must be 100 or 1000 (got '$reynolds')." ;;
esac
if [ "$coupling" = piso ]; then
    [ -z "${SIMPLEFLUID_CAVITY_COUPLED_OPERATOR+x}${SIMPLEFLUID_CAVITY_COUPLED_WORKSPACE+x}" ] || \
        launcher_error 'coupled operator/workspace environment controls require --coupling nox.'
else
    case "${SIMPLEFLUID_CAVITY_COUPLED_OPERATOR-assembled}" in
        assembled|block_composite) ;;
        *) launcher_error 'SIMPLEFLUID_CAVITY_COUPLED_OPERATOR must be assembled or block_composite.' ;;
    esac
    case "${SIMPLEFLUID_CAVITY_COUPLED_WORKSPACE-cached_products}" in
        cached_products|streamed_products) ;;
        *) launcher_error 'SIMPLEFLUID_CAVITY_COUPLED_WORKSPACE must be cached_products or streamed_products.' ;;
    esac
fi

case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$case_dir/../../.." && pwd)
invocation_dir=$(pwd -P)
. "$repo_dir/verification/environments.sh"
export_build_env "$repo_dir"

test_name="VerificationCasesTest.LidDrivenCavityRe${reynolds}"
if [ "$coupling" = nox ]; then
    nox_enabled=$(simplefluid_cache_value \
        "$SIMPLEFLUID_BUILD_DIR/CMakeCache.txt" SIMPLEFLUID_ENABLE_NOX) || nox_enabled=
    case "$nox_enabled" in
        ON|TRUE|YES|Y|1) ;;
        *)
            launcher_error "NOX requires SIMPLEFLUID_ENABLE_NOX=ON in '$SIMPLEFLUID_BUILD_DIR/CMakeCache.txt'; configure that build tree explicitly first."
            ;;
    esac
    test_name="${test_name}Nox"
fi

build_jobs=${SIMPLEFLUID_BUILD_JOBS:-4}
output_dir=${SIMPLEFLUID_PROFILE_OUTPUT_DIR:-"$invocation_dir/profiles"}
case "$output_dir" in
    /*) ;;
    *) output_dir="$invocation_dir/$output_dir" ;;
esac

simplefluid_build_target testVerificationCases "$build_jobs"
executable="$(simplefluid_executable testVerificationCases)"

mkdir -p "$output_dir"
rm -f "$output_dir/simplefluid_lineX.csv" \
      "$output_dir/simplefluid_lineY.csv"
export SIMPLEFLUID_PROFILE_OUTPUT_DIR="$output_dir"

"$executable" --gtest_filter="$test_name"

for profile in simplefluid_lineX.csv simplefluid_lineY.csv; do
    [ -s "$output_dir/$profile" ] || launcher_error \
        "'$test_name' did not produce '$output_dir/$profile'; a skipped or missing test is not a successful profile run."
done
