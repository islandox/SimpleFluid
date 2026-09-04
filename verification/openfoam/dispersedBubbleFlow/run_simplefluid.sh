#!/bin/sh
set -eu
case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$case_dir/../../.." && pwd)
mode=${1:-transient}
case "$mode" in steady|transient) ;; *) echo "Usage: $0 [steady|transient] [output-directory]" >&2; exit 2 ;; esac
[ "$#" -le 2 ] || { echo "Too many arguments" >&2; exit 2; }
output_dir=${2:-$(mktemp -d "${TMPDIR:-/tmp}/simplefluid-bubbles-$mode.XXXXXX")}
mkdir -p "$output_dir"
output_dir=$(CDPATH= cd -- "$output_dir" && pwd)
. "$repo_dir/verification/environments.sh"
export_build_env "$repo_dir"
simplefluid_build_target dispersed_bubble_verification
executable=$(simplefluid_executable dispersed_bubble_verification)
"$executable" --mode "$mode" --output "$output_dir" --parameters "$case_dir/reference.properties"
