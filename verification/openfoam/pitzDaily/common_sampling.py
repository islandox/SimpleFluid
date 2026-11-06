#!/usr/bin/env python3
"""Sample both solvers with OpenFOAM cellPoint on geometrically matched meshes.

This changes representation only. It never solves or adjusts a velocity field.
The geometry tolerance below is in metres, not a physical error limit.
"""
import argparse
import csv
import glob
import itertools
import math
import os
from pathlib import Path
import re
import shutil

import collect_evidence as evidence
import compare_profiles as comparator
import evidence_checks

GEOMETRY_TOLERANCE_M = 1e-9
METHOD = 'openfoam-cellPoint-matched-mesh-v1'


def read_vectors(path):
    text = path.read_text()
    match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;', text, re.S)
    if not match:
        raise ValueError(f'expected nonuniform vector field: {path}')
    values = [float(v) for v in re.findall(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', match[2])]
    if len(values) != int(match[1]) * 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(f'invalid vector field: {path}')
    return [tuple(values[i:i+3]) for i in range(0, len(values), 3)]


def match_vectors(centres, rows, tolerance=GEOMETRY_TOLERANCE_M):
    """Require a bijection, including spanwise position; never nearest-map fields."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('geometry tolerance must be finite and positive')
    buckets = {}
    for i, point in enumerate(centres):
        if len(point) != 3 or not all(math.isfinite(v) for v in point):
            raise ValueError('invalid centre')
        buckets.setdefault(tuple(math.floor(v / tolerance) for v in point), []).append(i)
    values, used, maximum = [None] * len(centres), set(), 0.0
    for row in rows:
        point = tuple(float(row[k]) for k in ('x', 'y', 'z'))
        velocity = tuple(float(row[k]) for k in ('ux', 'uy', 'uz'))
        if not all(math.isfinite(v) for v in (*point, *velocity)):
            raise ValueError('nonfinite cell data')
        key = tuple(math.floor(v / tolerance) for v in point)
        matches = []
        for offset in itertools.product((-1, 0, 1), repeat=3):
            for i in buckets.get(tuple(key[d] + offset[d] for d in range(3)), ()):
                distance = math.dist(point, centres[i])
                if distance <= tolerance:
                    matches.append((i, distance))
        if len(matches) != 1 or matches[0][0] in used:
            raise ValueError('cell coordinates do not match mesh bijectively')
        i, distance = matches[0]
        values[i] = velocity
        used.add(i)
        maximum = max(maximum, distance)
    if len(used) != len(centres):
        raise ValueError('cell data does not cover complete mesh')
    return values, maximum


def coarsen_blocks(text, divisor, mesh_multiplier=1):
    if type(divisor) is not int or divisor < 1:
        raise ValueError('mesh divisor must be a positive integer')
    if type(mesh_multiplier) is not int or mesh_multiplier not in (1,2):
        raise ValueError('mesh_multiplier must be integer 1 or 2')
    count = 0
    def replace(match):
        nonlocal count
        count += 1
        nx, ny = (int(match[k]) for k in (2, 3))
        return f'{match[1]}({(nx * mesh_multiplier + divisor - 1)//divisor} {(ny * mesh_multiplier + divisor - 1)//divisor} 1)'
    result = re.sub(r'(hex\s*\([^)]*\)\s*)\((\d+)\s+(\d+)\s+1\)', replace, text)
    if count != 5:
        raise ValueError('expected exactly five pitzDaily mesh blocks')
    return result


def sample(source_case, paths, divisor, destination, mesh_multiplier=1):
    if os.environ.get('WM_PROJECT_VERSION') != 'v2606':
        raise ValueError('load OpenFOAM.com v2606')
    evidence_checks.validate_profile_control(
        (source_case / 'system/controlDict').read_text(), comparator.STATIONS)
    # The sampler also supports velocity-only self-calibration inputs.
    # Physical collection separately requires all transported fields.
    evidence_checks.cell_quality(paths)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('0', 'constant', 'system'):
        shutil.copytree(source_case / name, destination / name,
                        ignore=shutil.ignore_patterns('polyMesh'))
    block = destination / 'system/blockMeshDict'
    block.write_text(coarsen_blocks(block.read_text(), divisor, mesh_multiplier))
    control = destination / 'system/controlDict'
    control.write_text(control.read_text().replace('#includeFunc streamlines', '').replace('writePrecision      12;', 'writePrecision      17;'))
    commands = {}
    def run(name, args):
        commands[name] = evidence.run_command(args, destination, dict(os.environ), destination / f'log.{name}')
        evidence.write_json(destination / 'sampling-commands.json', commands)
        if commands[name]['returncode']:
            raise ValueError(f'common sampling command failed: {name}')
    run('blockMesh', ['blockMesh'])
    run('checkMesh', ['checkMesh'])
    run('centres', ['postProcess', '-func', 'writeCellCentres', '-time', '0'])
    centres = read_vectors(destination / '0/C')
    rows = []
    for path in paths:
        with path.open() as stream:
            rows.extend(csv.DictReader(stream))
    values, maximum = match_vectors(centres, rows)
    target = destination / '0/U'
    text = target.read_text()
    internal = 'internalField nonuniform List<vector>\n' + str(len(values)) + '\n(\n'
    internal += '\n'.join('(' + ' '.join(format(v, '.17g') for v in row) + ')' for row in values) + '\n);'
    text, count = re.subn(r'internalField\s+uniform\s*\([^)]*\)\s*;', lambda _: internal, text)
    if count != 1:
        raise ValueError('expected one uniform initial velocity field')
    target.write_text(text)
    run('profiles', ['postProcess', '-time', '0', '-fields', '(U)'])
    for path in comparator.latest_common_profiles(destination).values():
        if len(comparator.read_openfoam(path)) != 129:
            raise ValueError('common sampler did not retain all 129 station points')
    tool = Path(shutil.which('postProcess')).resolve()
    metadata = {'method': METHOD, 'divisor': divisor, 'mesh_multiplier': mesh_multiplier, 'matched_cells': len(values),
                'maximum_coordinate_distance_m': maximum, 'geometry_tolerance_m': GEOMETRY_TOLERANCE_M,
                'input_sha256': {str(p.resolve()): evidence.sha256(p) for p in paths},
                'postProcess': {'path': str(tool), 'sha256': evidence.sha256(tool)},
                'velocity_boundary_source_sha256': evidence.sha256(source_case / '0/U'),
                'interpretation': 'Identical official interpolation and velocity boundary conditions; no CFD solve or error adjustment.'}
    evidence.write_json(destination / 'sampling.json', metadata)
    return metadata


def profile_metrics(reference_case, result_case):
    refs = comparator.latest_common_profiles(reference_case)
    actuals = comparator.latest_common_profiles(result_case)
    return profile_metrics_from_paths(refs, actuals)


def profile_metrics_from_paths(refs, actuals):
    """Use explicit authenticated references with the same established norms."""
    if set(refs) != set(comparator.STATIONS) or set(actuals) != set(comparator.STATIONS):
        raise ValueError('every physical station requires an explicit profile')
    result = {}
    for station, x in comparator.STATIONS.items():
        ref = comparator.read_openfoam(refs[station])
        actual = comparator.read_openfoam(actuals[station])
        if len(ref) != 129 or len(actual) != 129 or any(abs(a[0]-b[0]) > 1e-12 for a,b in zip(ref,actual)):
            raise ValueError('common profiles must cover identical 129-point grids')
        # The same uniform grid can print at different decimal precision.
        # Align only after checking every coordinate; never extrapolate fields.
        actual = [(r[0], a[1], a[2]) for r, a in zip(ref, actual)]
        result[station] = {'x_m': x, 'samples': len(actual), 'method': METHOD}
        for component, column in comparator.COMPONENTS.items():
            l2, linf = comparator.error_norms(ref, actual, column)
            result[station][component] = {'l2': l2, 'linf': linf}
    return result


def verify_sampling(source_case, paths, divisor, destination, required, mesh_multiplier=1):
    """Recheck raw-field binding without requiring OpenFOAM at replay time."""
    metadata = evidence.read_json(required(destination / 'sampling.json'))
    evidence_checks.validate_profile_control(
        (source_case / 'system/controlDict').read_text(), comparator.STATIONS)
    if metadata.get('geometry_tolerance_m') != GEOMETRY_TOLERANCE_M:
        raise ValueError('mismatched sampling geometry tolerance')
    if metadata['method'] != METHOD or metadata['divisor'] != divisor or metadata.get('mesh_multiplier',1) != mesh_multiplier:
        raise ValueError('mismatched sampling method or mesh divisor')
    # Input names are historical absolute paths; authenticate by complete rank
    # basename set and digest so relocation does not weaken the binding.
    recorded = {Path(k).name: v for k,v in metadata['input_sha256'].items()}
    if len(recorded) != len(paths) or recorded != {p.name: evidence.sha256(p) for p in paths}:
        raise ValueError('sampling inputs do not match retained rank files')
    commands = evidence.read_json(required(destination / 'sampling-commands.json'))
    if set(commands) != {'blockMesh', 'checkMesh', 'centres', 'profiles'} or any(r['returncode'] != 0 for r in commands.values()):
        raise ValueError('missing or failed sampling command')
    for name in commands:
        required(destination / f'log.{name}')
    block = required(destination / 'system/blockMeshDict')
    if block.read_text() != coarsen_blocks((source_case / 'system/blockMeshDict').read_text(), divisor, mesh_multiplier):
        raise ValueError('sampling mesh configuration differs from source')
    control = required(destination / 'system/controlDict')
    expected_control = (source_case / 'system/controlDict').read_text().replace('#includeFunc streamlines', '').replace('writePrecision      12;', 'writePrecision      17;')
    if control.read_text() != expected_control:
        raise ValueError('sampling controls differ from source')
    source_u = (source_case / '0/U').read_text()
    actual_u = required(destination / '0/U').read_text()
    boundary = lambda text: text[text.index('boundaryField'):]
    if boundary(source_u) != boundary(actual_u) or metadata['velocity_boundary_source_sha256'] != evidence.sha256(source_case / '0/U'):
        raise ValueError('sampling velocity boundary conditions differ')
    rows = []
    for path in paths:
        with path.open() as stream:
            rows.extend(csv.DictReader(stream))
    centres = read_vectors(required(destination / '0/C'))
    values, maximum = match_vectors(centres, rows)
    if values != read_vectors(destination / '0/U') or metadata['matched_cells'] != len(values) or metadata['maximum_coordinate_distance_m'] != maximum:
        raise ValueError('sampled velocity field differs from raw cell values')
    for path in comparator.latest_common_profiles(destination).values():
        required(path)
        if len(comparator.read_openfoam(path)) != 129:
            raise ValueError('incomplete common sampled profile')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case', required=True, type=Path)
    parser.add_argument('--cells', required=True, nargs='+')
    parser.add_argument('--divisor', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mesh-multiplier', type=int, default=1)
    args = parser.parse_args()
    paths = sorted({Path(p) for pattern in args.cells for p in glob.glob(pattern)})
    if not paths:
        parser.error('no cell CSV files matched')
    sample(args.source_case, paths, args.divisor, args.output, args.mesh_multiplier)


if __name__ == '__main__':
    main()
