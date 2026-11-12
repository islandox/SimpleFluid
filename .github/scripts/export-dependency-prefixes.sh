#!/usr/bin/env bash
set -euo pipefail

# Export the dependency prefixes used to configure, link and run SimpleFluid.
# The install trees come either from setup-trilinos-source in the same job or
# from the dependencies artifact unpacked into _deps by the build job.
deps="${GITHUB_WORKSPACE:?}/_deps"
partitioners="$deps/partitioners_install"
trilinos="$deps/trilinos_install"

for prefix in "$trilinos" "$partitioners"; do
    if [[ ! -d "$prefix" ]]; then
        printf 'missing dependency prefix: %s\n' "$prefix" >&2
        exit 1
    fi
done

if [[ "${RUNNER_OS:-}" == "macOS" ]]; then
    brew_prefix="$(brew --prefix)"
    hdf5_prefix="$(brew --prefix hdf5-mpi)"
    netcdf_prefix="$deps/netcdf_install"
    printf 'CMAKE_PREFIX_PATH=%s:%s:%s:%s:%s:%s\n' \
        "$partitioners" "$trilinos" "$hdf5_prefix" "$netcdf_prefix" "$brew_prefix" \
        "${CMAKE_PREFIX_PATH:-}" >> "$GITHUB_ENV"
    printf 'DYLD_FALLBACK_LIBRARY_PATH=%s:%s:%s:%s:%s:%s\n' \
        "$partitioners/lib" "$trilinos/lib" "$hdf5_prefix/lib" "$netcdf_prefix/lib" \
        "$brew_prefix/lib" "${DYLD_FALLBACK_LIBRARY_PATH:-}" >> "$GITHUB_ENV"
else
    printf 'CMAKE_PREFIX_PATH=%s:%s:%s\n' \
        "$partitioners" "$trilinos" "${CMAKE_PREFIX_PATH:-}" >> "$GITHUB_ENV"
    printf 'LD_LIBRARY_PATH=%s:%s\n' \
        "$partitioners/lib" "${LD_LIBRARY_PATH:-}" >> "$GITHUB_ENV"
fi

printf 'Trilinos_DIR=%s/lib/cmake/Trilinos\n' "$trilinos" >> "$GITHUB_ENV"
