#!/usr/bin/env python3
"""Compare completed Q5 per-rank XML against serial physical fields.

History channels are numerical error bounds, not physical fields: report their
cross-decomposition differences, while the C++ fixture verifies their migration
and adaptation contracts. Pressure comparison removes the arbitrary gauge.
"""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def cases(directory):
    for path in sorted(directory.rglob('rank-*.xml')):
        for test in ET.parse(path).findall('.//testcase'):
            if test.get('classname') != 'GeneralVoFFaceTransactions':
                continue
            properties = {p.get('name'): p.get('value') for p in test.findall('./properties/property')}
            yield path, test, properties


def fields(properties):
    rows = [[float(x) for x in row.split(',') if x]
            for row in properties['final_physical_fields'].split(';') if row]
    if not rows or any(len(row) != 14 or not all(math.isfinite(x) for x in row) for row in rows):
        raise ValueError('Absent, malformed or nonfinite physical field rows')
    if any(row[13] <= 0 or min(row[3], row[4], row[11], row[12]) < 0 for row in rows):
        raise ValueError('Invalid phase, history or bulk volume')
    volume = math.fsum(row[13] for row in rows)
    gauge = math.fsum(row[8] * row[13] for row in rows) / volume
    for row in rows:
        row[8] -= gauge
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    references = {}
    for path, test, properties in cases(args.serial):
        if test.findall('failure') or test.findall('skipped'):
            continue
        if int(properties.get('actual_ranks', 0)) != 1 or abs(float(properties['accepted_time']) - .02) > 1e-16:
            continue
        references[properties['configured_backend']] = fields(properties)
    if set(references) != {'DonorCell', 'GeometricFace'}:
        raise ValueError('Both complete successful serial backend references are required')
    results = []
    for directory in args.candidate:
        for path, test, properties in cases(directory):
            result = {'xml': str(path), 'case': test.get('name'), 'passed': False}
            try:
                if test.findall('failure') or test.findall('skipped'):
                    raise ValueError('Candidate test did not pass')
                if abs(float(properties['accepted_time']) - .02) > 1e-16 or int(properties['accepted_steps']) != 10:
                    raise ValueError('Candidate did not complete the original ten-step .02 endpoint')
                rows = fields(properties)
                reference = references[properties['configured_backend']]
                if len(rows) != len(reference):
                    raise ValueError('Candidate and serial mesh sizes differ')
                remaining = set(range(len(rows)))
                maximum = [0.] * 14
                for expected in reference:
                    matches = [i for i in remaining
                               if max(abs(rows[i][a] - expected[a]) for a in range(3)) <= 1e-14]
                    if len(matches) != 1:
                        raise ValueError('Physical cell center has no unique counterpart')
                    i = matches[0]
                    remaining.remove(i)
                    row = rows[i]
                    for a in range(14):
                        delta = abs(row[a] - expected[a])
                        maximum[a] = max(maximum[a], delta)
                        if a in (11, 12):
                            continue
                        tolerance = (1e-14 if a < 3 else 2e-11 * expected[13] if a in (3, 4, 13)
                                     else 1e-9 if a in (5, 6, 7)
                                     else 1e-8 * max(1., abs(expected[a])) if a == 8
                                     else 2e-11 if a == 9 else 1e-12)
                        if delta > tolerance:
                            raise ValueError(f'Physical field column {a}: difference {delta} exceeds {tolerance}')
                result.update(passed=True, cells=len(rows), ranks=int(properties['actual_ranks']),
                              rank=int(path.stem.split('-')[-1]),
                              empty_owners=int(properties['asserted_empty_owners']),
                              maximum_absolute_differences=maximum)
            except (KeyError, ValueError) as error:
                result['error'] = str(error)
            results.append(result)
    coverage = {}
    for result in results:
        if result['passed']:
            coverage.setdefault((result['ranks'], result['case']), set()).add(result['rank'])
    expected = [(ranks, backend + 'MovingMixed' + layout + 'RefineCoarsenRepartition')
                for ranks in (2, 4) for backend in ('DonorCell', 'GeometricFace')
                for layout in (('Spatial',) if ranks == 2 else ('Spatial', 'EmptyOwners'))]
    missing = [f'{ranks}:{case}' for ranks, case in expected
               if coverage.get((ranks, case), set()) != set(range(ranks))]
    report = {'policy': 'verification/vof/algebraic_flow_manifest.json Q5.decomposition_comparison',
              'results': results, 'missing_rank_coverage': missing,
              'all_passed': bool(results) and not missing and all(r['passed'] for r in results)}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'comparisons': len(results), 'all_passed': report['all_passed']}))
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
