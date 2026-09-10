#!/usr/bin/env python3
"""Independent Decimal evaluation of the v13 SAS algebra with SST-1994 inputs.

Run --check to compare the checked-in table used by testSSTSASSource; default
prints CSV. No OpenFOAM runtime is needed. This is a source-algebra comparison,
not a whole-flow OpenFOAM result. Decimal precision is 60 digits.
"""
import argparse
import csv
import io
from decimal import Decimal as D, getcontext
from pathlib import Path

getcontext().prec = 60
cases = [
    ('0', '.2', '2', '4', '1', '0', '0', '.1', '.001'),
    ('1', '.2', '2', '4', '100', '0', '0', '.1', '.001'),
    ('.35', '.2', '2', '4', '1', '.02', '.1', '.1', '.001'),
    ('.7', '.2', '2', '4', '1', '2', '10', '.1', '.001'),
    ('.25', '1', '1', '100', '100', '0', '0', '.1', '1'),
    ('1', '.2', '2', '0', '1', '.01', '0', '.1', '.001'),
]


def table():
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow('F1 k omega S2 H grad_k grad_omega delta dt L Lvk_flow Lvk_grid Lvk delta_out production damping Q_raw Q_applied grid_active cap_active'.split())
    for row in cases:
        f, k, w, s2, h, gk, gw, delta, dt = map(D, row)
        beta_star, kappa, zeta, cs = map(D, ('.09', '.41', '3.51', '.11'))
        beta = f*D('.075') + (1-f)*D('.0828')
        gamma1 = D('.075')/beta_star - D('.5')*kappa**2/beta_star.sqrt()
        gamma2 = D('.0828')/beta_star - D('.856')*kappa**2/beta_star.sqrt()
        gamma = f*gamma1 + (1-f)*gamma2
        length = k.sqrt()/beta_star.sqrt().sqrt()/w
        flow = kappa*s2.sqrt()/h
        grid = cs*delta*(kappa*zeta/(beta/beta_star-gamma)).sqrt()
        lvk = max(flow, grid)
        production = zeta*kappa*s2*(length/lvk)**2
        damping = 2*D(2)/(D(2)/3)*k*max((gw/w)**2, (gk/k)**2)
        raw = max(production-damping, D(0))
        cap = w/(D('.1')*dt)
        values = [length, flow, grid, lvk, delta, production, damping, raw, min(raw, cap), int(grid>flow), int(raw>cap)]
        writer.writerow(list(row) + [format(float(v), '.17g') for v in values])
    return output.getvalue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = table()
    if args.check:
        reference = Path(__file__).resolve().parents[2] / 'src/equations/unitTests/sst_sas_reference.csv'
        if reference.read_text() != result:
            raise SystemExit('Pointwise reference table mismatch')
        print('Six independent SAS reference rows match.')
    else:
        print(result, end='')
