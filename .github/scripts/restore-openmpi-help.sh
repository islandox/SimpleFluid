#!/usr/bin/env bash
set -euo pipefail

# Ubuntu 26.04 packages PRRTE help under its development documentation, while
# Open MPI looks under the relocated library prefix (open-mpi/ompi#13886).
help_target="/usr/lib/$(gcc -dumpmachine)/prrte3/share/prte"
if [[ -r "$help_target/help-prun.txt" ]]; then
    exit 0
fi

help_source=/usr/share/doc/libprrte-dev
if [[ ! -r "$help_source/help-prun.txt" && ! -r "$help_source/help-prun.txt.gz" ]]; then
    printf '%s\n' 'path-include=/usr/share/doc/libprrte-dev/help-*' |
        sudo tee /etc/dpkg/dpkg.cfg.d/simplefluid-prrte-help >/dev/null
    sudo apt-get install -y --no-install-recommends --reinstall libprrte-dev
fi

sudo mkdir -p "$help_target"
shopt -s nullglob
for help_file in "$help_source"/help-*.txt "$help_source"/help-*.txt.gz; do
    help_name=${help_file##*/}
    if [[ "$help_name" == *.gz ]]; then
        gzip -cd "$help_file" | sudo tee "$help_target/${help_name%.gz}" >/dev/null
    else
        sudo cp "$help_file" "$help_target/$help_name"
    fi
done
test -r "$help_target/help-prun.txt"
