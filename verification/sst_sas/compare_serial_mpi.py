#!/usr/bin/env python3
"""Compare the small transient activation fixture in serial and on two ranks.

Usage: python3 verification/sst_sas/compare_serial_mpi.py build/gcc/bin/Debug/sst_sas_activation
Requires a functioning MPI launcher. Relative tolerance 1e-9, absolute 1e-11.
This compares partition consistency, not OpenFOAM or turbulence physics.
"""
import math
import subprocess
import sys


def run(command):
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    rows = [line for line in output.splitlines() if line.startswith('sas_result ')]
    if len(rows) != 1:
        raise RuntimeError(f'Expected one reduced result from {command}:\n{output}')
    return list(map(float, rows[0].split()[1:]))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    serial = run([sys.argv[1]])
    parallel = run(['mpiexec', '-n', '2', sys.argv[1]])
    names = 'active_fraction cap_fraction min_source max_source volume_mean volume_integral omega_integral'.split()
    if len(serial) != len(names) or len(parallel) != len(names):
        raise SystemExit('Unexpected result width')
    for name, a, b in zip(names, serial, parallel):
        print(f'{name}: serial={a:.17g} two_rank={b:.17g}')
        if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-11):
            raise SystemExit(f'{name} partition comparison failed')
    if serial[3] <= 1e-6:
        raise SystemExit('The SAS source must be nonzero')
    print('Serial/two-rank transient activation agrees within rtol=1e-9, atol=1e-11.')
