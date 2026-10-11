#!/usr/bin/env python3
"""Use a portable x86-64 baseline in the pinned METIS/ParMETIS GCC builds."""

import sys
from pathlib import Path

native = 'set(GK_COPTIONS "${GK_COPTIONS} -march=native")'
portable = 'set(GK_COPTIONS "${GK_COPTIONS} -march=x86-64 -mtune=generic")'
for filename in sys.argv[1:]:
    path = Path(filename)
    source = path.read_text()
    if source.count(native) != 1:
        raise SystemExit(f"Unexpected partitioner compiler configuration: {path}")
    path.write_text(source.replace(native, portable))
