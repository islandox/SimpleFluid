#!/bin/sh

# Shared SimpleFluid build environment for verification launchers.
#
# Source this file, then call:
#
#   export_build_env /path/to/SimpleFluid
#   simplefluid_build_target target [jobs]
#
# The default toolchain is GCC with RelWithDebInfo. Override it with:
#
#   SIMPLEFLUID_COMPILER=LLVM
#   SIMPLEFLUID_BUILD_CONFIG=Debug
#
# SIMPLEFLUID_BUILD_DIR selects an already configured non-preset build tree.
# When it is unset, the compiler-specific CMake configure/build presets are
# used and a missing build tree is configured automatically.
# SIMPLEFLUID_CONFIGURE_PRESET and SIMPLEFLUID_BUILD_PRESET can override the
# derived preset names. SIMPLEFLUID_BUILD_JOBS controls the default parallel
# build width, and SIMPLEFLUID_ENV_QUIET=1 suppresses the resolved summary.

simplefluid_environment_error()
{
    printf 'SimpleFluid environment: %s\n' "$*" >&2
}

simplefluid_normalize_compiler()
{
    case "${1:-GCC}" in
        GCC|gcc|GNU|gnu)
            printf '%s\n' "GCC"
            ;;
        LLVM|llvm|Clang|clang)
            printf '%s\n' "LLVM"
            ;;
        *)
            simplefluid_environment_error \
                "SIMPLEFLUID_COMPILER must be GCC or LLVM (got '$1')."
            return 2
            ;;
    esac
}

simplefluid_validate_build_config()
{
    case "$1" in
        Debug|Release|RelWithDebInfo)
            ;;
        *)
            simplefluid_environment_error \
                "SIMPLEFLUID_BUILD_CONFIG must be Debug, Release, or RelWithDebInfo (got '$1')."
            return 2
            ;;
    esac
}

simplefluid_validate_jobs()
{
    case "$1" in
        ''|*[!0-9]*)
            simplefluid_environment_error \
                "the build job count must be a positive integer (got '$1')."
            return 2
            ;;
    esac
    if [ "$1" -lt 1 ]; then
        simplefluid_environment_error \
            "the build job count must be a positive integer (got '$1')."
        return 2
    fi
}

simplefluid_prepend_path()
{
    local variable_name="$1"
    local directory="$2"
    local current_value
    local new_value

    [ -d "$directory" ] || return 0
    eval "current_value=\${${variable_name}-}"
    case ":$current_value:" in
        *":$directory:"*)
            return 0
            ;;
    esac

    new_value="$directory${current_value:+:$current_value}"
    export "$variable_name=$new_value"
}

simplefluid_cache_value()
{
    local cmake_cache="$1"
    local variable_name="$2"
    local line

    [ -f "$cmake_cache" ] || return 1
    while IFS= read -r line; do
        case "$line" in
            "$variable_name":*=*)
                printf '%s\n' "${line#*=}"
                return 0
                ;;
        esac
    done < "$cmake_cache"
    return 1
}

# Print configured CMake build directories below a repository root.
detect_build_dirs()
{
    local root_dir="${1:-.}"
    local physical_root
    local cmake_cache
    local directory
    local source_directory

    physical_root="$(cd "$root_dir" 2>/dev/null && pwd -P)" || return

    # Keep the search bounded to the supported top-level build layouts. In
    # particular, do not descend into CMakeFiles or dependency build trees,
    # which can contain their own CMakeCache.txt files.
    for cmake_cache in \
        "$root_dir"/build/CMakeCache.txt \
        "$root_dir"/build-*/CMakeCache.txt \
        "$root_dir"/_build/CMakeCache.txt \
        "$root_dir"/cmake-build-*/CMakeCache.txt \
        "$root_dir"/out/CMakeCache.txt \
        "$root_dir"/dist/CMakeCache.txt \
        "$root_dir"/build/*/CMakeCache.txt; do
        [ -f "$cmake_cache" ] || continue
        directory="${cmake_cache%/CMakeCache.txt}"
        case "$directory" in
            "$root_dir"/build/CMakeFiles|"$root_dir"/build/_deps)
                continue
                ;;
        esac

        source_directory="$(
            simplefluid_cache_value "$cmake_cache" CMAKE_HOME_DIRECTORY
        )" || continue
        source_directory="$(
            cd "$source_directory" 2>/dev/null && pwd -P
        )" || continue
        [ "$source_directory" = "$physical_root" ] || continue
        printf '%s\n' "$directory"
    done
}

# Select the most recently configured build tree. Verification launchers use
# deterministic compiler-specific presets; this helper remains useful for
# interactive inspection and explicitly managed build trees.
select_latest_build_dir()
{
    local root_dir="${1:-.}"
    local directory
    local cmake_cache
    local latest_directory=""
    local latest_cache=""

    detect_build_dirs "$root_dir" | (
        while IFS= read -r directory; do
            cmake_cache="$directory/CMakeCache.txt"
            [ -f "$cmake_cache" ] || continue
            if [ -z "$latest_cache" ] || [ -n "$(
                find "$cmake_cache" -newer "$latest_cache" -print \
                    2>/dev/null
            )" ]; then
                latest_directory="$directory"
                latest_cache="$cmake_cache"
            fi
        done

        [ -n "$latest_directory" ] || exit 1
        printf '%s\n' "$latest_directory"
    )
}

simplefluid_export_cache_paths()
{
    local cmake_cache="$1"
    local line
    local key
    local typed_value
    local value

    [ -f "$cmake_cache" ] || return 0
    while IFS= read -r line; do
        case "$line" in
            ''|\#*|//*)
                continue
                ;;
            *:*=*)
                ;;
            *)
                continue
                ;;
        esac

        key="${line%%:*}"
        typed_value="${line#*:}"
        value="${typed_value#*=}"
        [ -n "$key" ] && [ "$typed_value" != "$value" ] || continue

        case "$key" in
            CMAKE_INSTALL_PREFIX|CMAKE_MODULE_PATH|BOOST_ROOT|\
            Python_ROOT_DIR|Python3_ROOT_DIR|Qt5_DIR|Qt6_DIR)
                [ -n "$value" ] || continue
                export "$key=$value"
                ;;
        esac
    done < "$cmake_cache"
}

# Resolve and export the compiler-specific preset, build tree, and
# configuration-specific runtime paths used by verification scripts.
export_build_env()
{
    local root_dir="${1:-.}"
    local compiler
    local build_config="${SIMPLEFLUID_BUILD_CONFIG:-RelWithDebInfo}"
    local configured_build_dir="${SIMPLEFLUID_BUILD_DIR:-}"
    local build_dir
    local build_suffix
    local configuration_types=""
    local site_packages=""

    root_dir="$(cd "$root_dir" && pwd)" || {
        simplefluid_environment_error \
            "repository root '$root_dir' does not exist."
        return 1
    }
    compiler="$(simplefluid_normalize_compiler \
        "${SIMPLEFLUID_COMPILER:-GCC}")" || return
    simplefluid_validate_build_config "$build_config" || return

    case "$compiler" in
        GCC)
            build_suffix="gcc"
            ;;
        LLVM)
            build_suffix="llvm"
            ;;
    esac

    if [ -n "$configured_build_dir" ]; then
        if [ "${configured_build_dir#/}" = "$configured_build_dir" ]; then
            configured_build_dir="$root_dir/$configured_build_dir"
        fi
        build_dir="${configured_build_dir%/}"
        SIMPLEFLUID_BUILD_MODE="directory"
    else
        build_dir="$root_dir/build/$build_suffix"
        SIMPLEFLUID_BUILD_MODE="preset"
    fi

    SIMPLEFLUID_REPO_DIR="$root_dir"
    SIMPLEFLUID_COMPILER="$compiler"
    SIMPLEFLUID_BUILD_CONFIG="$build_config"
    SIMPLEFLUID_CONFIGURE_PRESET="${SIMPLEFLUID_CONFIGURE_PRESET:-${compiler}-ninja-multi}"
    SIMPLEFLUID_BUILD_PRESET="${SIMPLEFLUID_BUILD_PRESET:-${compiler}-${build_config}}"
    SIMPLEFLUID_BUILD_DIR="$build_dir"
    if [ -f "$build_dir/CMakeCache.txt" ]; then
        configuration_types="$(
            simplefluid_cache_value \
                "$build_dir/CMakeCache.txt" CMAKE_CONFIGURATION_TYPES
        )" || configuration_types=""
    elif [ "$SIMPLEFLUID_BUILD_MODE" = "preset" ]; then
        # The repository presets use a multi-configuration generator. Retain
        # their expected layout before the build tree has been configured.
        configuration_types="$build_config"
    fi
    if [ -n "$configuration_types" ]; then
        SIMPLEFLUID_BIN_DIR="$build_dir/bin/$build_config"
        SIMPLEFLUID_LIB_DIR="$build_dir/lib/$build_config"
    else
        SIMPLEFLUID_BIN_DIR="$build_dir/bin"
        SIMPLEFLUID_LIB_DIR="$build_dir/lib"
    fi

    export SIMPLEFLUID_REPO_DIR
    export SIMPLEFLUID_COMPILER
    export SIMPLEFLUID_BUILD_CONFIG
    export SIMPLEFLUID_CONFIGURE_PRESET
    export SIMPLEFLUID_BUILD_PRESET
    export SIMPLEFLUID_BUILD_MODE
    export SIMPLEFLUID_BUILD_DIR
    export SIMPLEFLUID_BIN_DIR
    export SIMPLEFLUID_LIB_DIR

    simplefluid_prepend_path PATH "$SIMPLEFLUID_BIN_DIR"
    simplefluid_prepend_path PATH "$build_dir/bin"
    simplefluid_prepend_path LD_LIBRARY_PATH "$SIMPLEFLUID_LIB_DIR"
    simplefluid_prepend_path LD_LIBRARY_PATH "$build_dir/lib"
    simplefluid_prepend_path LD_LIBRARY_PATH "$build_dir/lib64/$build_config"
    simplefluid_prepend_path LD_LIBRARY_PATH "$build_dir/lib64"
    simplefluid_prepend_path PKG_CONFIG_PATH \
        "$build_dir/lib/$build_config/pkgconfig"
    simplefluid_prepend_path PKG_CONFIG_PATH "$build_dir/lib/pkgconfig"
    simplefluid_prepend_path CMAKE_PREFIX_PATH "$build_dir"

    if [ -d "$build_dir/lib" ]; then
        site_packages="$(
            find "$build_dir/lib" -type d -name site-packages \
                -print -quit 2>/dev/null
        )"
    fi
    if [ -n "$site_packages" ]; then
        simplefluid_prepend_path PYTHONPATH "$site_packages"
    fi

    simplefluid_export_cache_paths "$build_dir/CMakeCache.txt"

    if [ "${SIMPLEFLUID_ENV_QUIET:-0}" != "1" ]; then
        printf 'SimpleFluid build environment:\n'
        printf '  compiler: %s\n' "$SIMPLEFLUID_COMPILER"
        printf '  configuration: %s\n' "$SIMPLEFLUID_BUILD_CONFIG"
        printf '  build directory: %s\n' "$SIMPLEFLUID_BUILD_DIR"
        if [ "$SIMPLEFLUID_BUILD_MODE" = "preset" ]; then
            printf '  configure preset: %s\n' \
                "$SIMPLEFLUID_CONFIGURE_PRESET"
            printf '  build preset: %s\n' "$SIMPLEFLUID_BUILD_PRESET"
        else
            printf '  build mode: configured directory\n'
        fi
    fi
}

simplefluid_configure_build()
{
    [ -n "${SIMPLEFLUID_BUILD_DIR:-}" ] || {
        simplefluid_environment_error \
            "call export_build_env before configuring or building."
        return 2
    }
    [ -f "$SIMPLEFLUID_BUILD_DIR/CMakeCache.txt" ] && return 0

    if [ "$SIMPLEFLUID_BUILD_MODE" = "directory" ]; then
        simplefluid_environment_error \
            "custom build directory '$SIMPLEFLUID_BUILD_DIR' is not configured."
        return 1
    fi

    (
        cd "$SIMPLEFLUID_REPO_DIR"
        cmake --preset "$SIMPLEFLUID_CONFIGURE_PRESET"
    )
}

simplefluid_build_target()
{
    local target="${1:-}"
    local jobs="${2:-${SIMPLEFLUID_BUILD_JOBS:-4}}"

    [ -n "$target" ] || {
        simplefluid_environment_error \
            "simplefluid_build_target requires a CMake target."
        return 2
    }
    simplefluid_validate_jobs "$jobs" || return
    simplefluid_configure_build || return

    if [ "$SIMPLEFLUID_BUILD_MODE" = "preset" ]; then
        (
            cd "$SIMPLEFLUID_REPO_DIR"
            cmake --build --preset "$SIMPLEFLUID_BUILD_PRESET" \
                --parallel "$jobs" --target "$target"
        )
    else
        cmake --build "$SIMPLEFLUID_BUILD_DIR" \
            --config "$SIMPLEFLUID_BUILD_CONFIG" \
            --parallel "$jobs" --target "$target"
    fi
}

simplefluid_executable()
{
    local target="${1:-}"
    local executable

    [ -n "$target" ] || {
        simplefluid_environment_error \
            "simplefluid_executable requires a target name."
        return 2
    }
    executable="$SIMPLEFLUID_BIN_DIR/$target"
    [ -x "$executable" ] || {
        simplefluid_environment_error \
            "expected executable '$executable' does not exist."
        return 1
    }
    printf '%s\n' "$executable"
}
