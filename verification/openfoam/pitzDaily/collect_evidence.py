#!/usr/bin/env python3
"""Collect reproducible pitzDaily evidence; never assign physical tolerances.

Only the existing comparator/acceptance launcher can validate a reviewed,
qualified manifest. A successful collection is deliberately still pending.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import compare_profiles as comparator
import evidence_checks

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# Same order and names as run_comparison.sh's executable contract.
ENV_NAMES = dict(zip((
    'mesh_divisor', 'steps', 'dt_s', 'steady_state', 'linear_tolerance',
    'steady_consecutive_steps', 'steady_min_steps', 'steady_max_retries',
    'steady_rejection_recovery_steps', 'steady_tolerance', 'steady_min_dt_s',
    'steady_max_dt_s', 'steady_target_courant', 'steady_dt_growth_factor',
    'steady_dt_reduction_factor', 'steady_rejection_safety_factor',
    'steady_relaxed_linear_tolerance', 'steady_full_accuracy_update_ratio',
    'steady_progress_interval'), (
    'MESH_DIVISOR', 'STEPS', 'DT', 'STEADY_STATE', 'LINEAR_TOLERANCE',
    'STEADY_CONSECUTIVE_STEPS', 'STEADY_MIN_STEPS', 'STEADY_MAX_RETRIES',
    'STEADY_REJECTION_RECOVERY_STEPS', 'STEADY_TOLERANCE', 'STEADY_MIN_DT',
    'STEADY_MAX_DT', 'STEADY_TARGET_COURANT', 'STEADY_DT_GROWTH',
    'STEADY_DT_REDUCTION', 'STEADY_REJECTION_SAFETY',
    'STEADY_RELAXED_LINEAR_TOLERANCE', 'STEADY_FULL_ACCURACY_UPDATE_RATIO',
    'STEADY_PROGRESS_INTERVAL')))
ROLES = ('baseline', 'stationarity', 'mesh', 'timestep')
ENV_NAMES['transport_preconditioner'] = 'TRANSPORT_PRECONDITIONER'
ENV_NAMES['modified_pressure'] = 'MODIFIED_PRESSURE'
ENV_NAMES['openfoam_outlet_stress'] = 'OPENFOAM_OUTLET_STRESS'
ENV_NAMES['mesh_multiplier'] = 'MESH_MULTIPLIER'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('evidence paths must be nonempty relative paths')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('evidence path escapes bundle')
    return path


def validate_plan(plan):
    if plan.get('schema_version') != 1 or set(plan.get('runs', {})) != set(ROLES):
        raise ValueError('plan requires schema_version 1 and exactly four run roles')
    if plan.get('sampling_method', 'nearest-layer-v1') not in ('nearest-layer-v1', 'openfoam-cellPoint-matched-mesh-v1'):
        raise ValueError('unsupported sampling method')
    if plan.get('reference_field_controls') is not None:
        import reference_field_gate
        reference_field_gate.validate_controls(plan['reference_field_controls'])
    ranks = plan.get('mpi_ranks')
    if type(ranks) is not int or not 1 <= ranks <= 6:
        raise ValueError('mpi_ranks must be an integer from 1 to 6')
    if plan.get('build') is None or set(plan['build']) != {'compiler', 'config'}:
        raise ValueError('build requires compiler and config')
    if plan['build']['compiler'] not in ('GCC', 'LLVM') or plan['build']['config'] not in ('Debug', 'Release', 'RelWithDebInfo'):
        raise ValueError('unsupported build compiler/config')
    integer_keys = {'mesh_multiplier', 'mesh_divisor', 'steps', 'steady_consecutive_steps',
                    'steady_min_steps', 'steady_max_retries',
                    'steady_rejection_recovery_steps', 'steady_progress_interval'}
    for role, run in plan['runs'].items():
        # Historical bundles used mechanical pressure. Never infer parity.
        run.setdefault('mesh_multiplier', 1)
        if run['mesh_multiplier'] not in (1,2) or type(run['mesh_multiplier']) is not int:
            raise ValueError('mesh_multiplier must be integer 1 or 2')
        run.setdefault('modified_pressure', False)
        run.setdefault('openfoam_outlet_stress', False)
        if set(run) != set(ENV_NAMES):
            raise ValueError(f'{role}: supply every executable setting, no extra keys')
        for key, value in run.items():
            if key == 'transport_preconditioner':
                if not isinstance(value, str) or value not in ('none', 'ilut'):
                    raise ValueError('transport_preconditioner must be none or ilut')
                continue
            if key in ('steady_state', 'modified_pressure', 'openfoam_outlet_stress'):
                if type(value) is not bool:
                    raise ValueError('steady_state must be boolean')
                continue
            minimum = 0 if key in ('steady_max_retries', 'steady_rejection_recovery_steps') else 1
            if key in integer_keys:
                if type(value) is not int or value < minimum:
                    raise ValueError(f'{role}.{key}: invalid integer')
            elif type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{role}.{key}: require finite positive number')
        if not run['steady_min_dt_s'] <= run['dt_s'] <= run['steady_max_dt_s']:
            raise ValueError('dt_s must lie inside adaptive timestep bounds')
        if run['steady_min_steps'] > run['steps'] or run['steady_consecutive_steps'] > run['steps']:
            raise ValueError('step budget is smaller than convergence window')
        if run['steady_dt_growth_factor'] <= 1 or run['steady_dt_reduction_factor'] >= 1 or run['steady_rejection_safety_factor'] >= 1:
            raise ValueError('invalid adaptive growth/reduction/safety factors')
        if run['steady_relaxed_linear_tolerance'] < run['linear_tolerance'] or run['steady_full_accuracy_update_ratio'] <= 1:
            raise ValueError('invalid adaptive linear tolerance schedule')
    base = plan['runs']['baseline']
    if base['mesh_divisor'] != 1:
        raise ValueError('baseline must use tutorial mesh divisor 1')
    for role, allowed in (
        ('stationarity', {'steps', 'steady_min_steps'}),
        ('mesh', {'mesh_divisor'}),
        ('timestep', {'dt_s', 'steps', 'steady_min_dt_s', 'steady_max_dt_s', 'steady_target_courant'}),
    ):
        other = plan['runs'][role]
        if any(base[k] != other[k] for k in base.keys() - allowed):
            raise ValueError(f'{role}: unmatched simulation settings')
    mesh, dt, later = (plan['runs'][r] for r in ('mesh', 'timestep', 'stationarity'))
    if mesh['mesh_divisor'] != 2 or dt['dt_s'] >= base['dt_s']:
        raise ValueError('require divisor-2 mesh and smaller timestep')
    if later['steps'] <= base['steps']:
        raise ValueError('stationarity requires a larger step budget')
    if base['steady_state']:
        if later['steady_min_steps'] <= base['steps']:
            raise ValueError('stationarity minimum steps must exceed baseline budget')
        if dt['steady_max_dt_s'] >= base['steady_max_dt_s']:
            raise ValueError('adaptive timestep refinement must reduce maximum dt too')
        if dt['steady_target_courant'] > base['steady_target_courant']:
            raise ValueError('adaptive timestep refinement cannot increase target Courant number')
    else:
        if dt['steady_target_courant'] != base['steady_target_courant']:
            raise ValueError('fixed timestep refinement must preserve adaptive controls')
        if not math.isclose(base['steps'] * base['dt_s'], dt['steps'] * dt['dt_s'], rel_tol=1e-12):
            raise ValueError('fixed timestep refinement must end at the same physical time')
    return plan


def run_command(command, cwd, env, log):
    start = datetime.now(timezone.utc).isoformat()
    with log.open('wb') as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    return {'argv': command, 'cwd': str(cwd), 'started_utc': start,
            'finished_utc': datetime.now(timezone.utc).isoformat(),
            'returncode': result.returncode}


def output(command, cwd=REPO):
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def diagnostics(text, kind):
    """Extract observations, not a numerical convergence qualification."""
    if kind == 'openfoam':
        converged = re.findall(r'SIMPLE solution converged in ([0-9]+) iterations', text)
        times = re.findall(r'^Time = ([^\s]+)', text, re.M)
        residuals = re.findall(r'Solving for (\w+), Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations ([0-9]+)', text)
        return {'residual_control_reported': bool(converged),
                'converged_iteration': int(converged[-1]) if converged else None,
                'last_time': times[-1] if times else None,
                'residuals': [dict(zip(('field', 'initial', 'final', 'iterations'), r)) for r in residuals]}
    summaries = re.findall(r'^pitzDaily: .*?divisor=(\d+), steps=(\d+), t=([^,]+), MPI ranks=(\d+)', text, re.M)
    steady = re.findall(r'^steady_state_search: reached=(yes|no)(.*)$', text, re.M)
    preconditioners = re.findall(r'^transport_linear_solver: .*?preconditioner=(\w+)\s*$', text, re.M)
    return {'summary': dict(zip(('mesh_divisor', 'steps', 'time_s', 'mpi_ranks'), summaries[-1])) if summaries else None,
            'transport_preconditioner': preconditioners[-1] if preconditioners else None,
            'mesh_multiplier': int((re.findall(r'^pitzDaily: .*?mesh_multiplier=(\d+)', text, re.M) or ['1'])[-1]),
            'velocity_neumann': (re.findall(r'^velocity_neumann: (zeroGradient_stress|traction_free)$', text, re.M) or [None])[-1],
            'pressure_convention': (re.findall(r'^pressure_convention: (modified|mechanical)$', text, re.M) or [None])[-1],
            'reported_unconverged_steps': re.findall(r'^step=(\d+)/\d+ .*?converged=no\b', text, re.M),
            'steady_reported': bool(steady and steady[-1][0] == 'yes'),
            'steady_diagnostics': steady[-1][1] if steady else None}


def clean_environment():
    # Preserve loaded OpenFOAM/MPI library paths; discard ambient run/build knobs.
    return {k: v for k, v in os.environ.items() if not k.startswith('SIMPLEFLUID_')}


def retain_build_provenance(destination, plan, required=False):
    build = REPO / 'build' / plan['build']['compiler'].lower()
    cache = build / 'CMakeCache.txt'
    binary = build / 'bin' / plan['build']['config'] / 'pitz_daily'
    if required or cache.is_file():
        shutil.copy2(cache, destination / 'CMakeCache.txt')
    if required or binary.is_file():
        write_json(destination / 'binary.json', {
            'path': str(binary), 'sha256': sha256(binary),
            'scope': ('completed solver launchers' if required else
                      'available after a failed launcher attempt; consult logs to establish whether the executable ran')})


def collect(plan_path, destination):
    if destination.exists():
        raise FileExistsError('evidence destination already exists')
    try:
        return _collect(plan_path, destination)
    except Exception as error:
        if destination.is_dir():
            failure = {
                'status': 'failed', 'error': str(error),
                'qualification': 'pending_external_qualification'}
            # A failed solve can still have valid build provenance. Capture
            # available artifacts without claiming a failed build executed it.
            try:
                commands_path = destination / 'commands.json'
                if commands_path.is_file() and set(read_json(commands_path)) & set(ROLES):
                    retain_build_provenance(destination, read_json(destination / 'plan.json'))
            except (OSError, ValueError, KeyError, TypeError) as build_error:
                failure['build_provenance_error'] = str(build_error)
            write_json(destination / 'collection_failure.json', failure)
            write_json(destination / 'checksums.json', {
                p.relative_to(destination).as_posix(): sha256(p)
                for p in sorted(destination.rglob('*'))
                if p.is_file() and p.name != 'checksums.json'})
        raise


def _collect(plan_path, destination):
    plan = validate_plan(read_json(plan_path))
    destination = destination.resolve()
    # No reuse or overwrite: stale rank files must never become fresh evidence.
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'plan.json', plan)
    env = clean_environment()
    if env.get('WM_PROJECT_VERSION') != 'v2606':
        raise ValueError('load OpenFOAM.com v2606 before collecting')
    provenance = {'revision': output(['git', 'rev-parse', 'HEAD']),
                  'git_status': output(['git', 'status', '--porcelain']),
                  'platform': platform.platform(), 'python': sys.version,
                  'openfoam_environment': {k: env.get(k) for k in ('WM_PROJECT', 'WM_PROJECT_VERSION', 'WM_OPTIONS', 'WM_PROJECT_DIR')}}
    write_json(destination / 'provenance.json', provenance)
    (destination / 'source.diff').write_text(output(['git', 'diff', 'HEAD', '--binary']), encoding='utf-8')
    # Retain exact workflow and solver source even for an uncommitted implementation.
    source = destination / 'source'
    source.mkdir()
    for name, path in [('pitz_daily.cc', REPO / 'src/examples/pitz_daily.cc'),
                       ('environments.sh', REPO / 'verification/environments.sh'),
                       ('run_simplefluid.sh', HERE / 'run_simplefluid.sh'),
                       ('compare_profiles.py', HERE / 'compare_profiles.py'),
                       ('collect_evidence.py', Path(__file__)),
                       ('TimeStepperOptions.hh', REPO / 'src/equations/TimeStepperOptions.hh'),
                       ('IncompressibleIsothermalSolver.tcc', REPO / 'src/solvers/IncompressibleIsothermalSolver.tcc'),
                       ('common_sampling.py', HERE / 'common_sampling.py'),
                       ('IncompressibleMomentumEquation.tcc', REPO / 'src/equations/IncompressibleMomentumEquation.tcc'),
                       ('FieldStoredTransportSystem.hh', REPO / 'src/FVM/details/FieldStoredTransportSystem.hh'),
                       ('TransportSystem.hh', REPO / 'src/FVM/TransportSystem.hh'),
                       ('evidence_checks.py', HERE / 'evidence_checks.py'),
                       ('replay_validation.py', HERE / 'replay_validation.py'),
                       ('reference_field_gate.py', HERE / 'reference_field_gate.py')]:
        shutil.copy2(path, source / name)
    tools = {}
    for name, args in (('simpleFoam', ['-help']), ('mpiexec', ['--version']),
                       ('cmake', ['--version']),
                       ('g++' if plan['build']['compiler'] == 'GCC' else 'clang++', ['--version'])):
        executable = shutil.which(name)
        if executable is None:
            raise ValueError(f'missing required tool: {name}')
        log = destination / f'tool-{name}.log'
        tools[name] = {'path': executable, 'sha256': sha256(Path(executable)),
                       'probe': run_command([executable] + args, REPO, env, log)}
    write_json(destination / 'tools.json', tools)
    of = destination / 'openfoam'
    of.mkdir()
    for name in ('0', 'constant', 'system'):
        shutil.copytree(HERE / 'openfoam' / name, of / name,
                        ignore=shutil.ignore_patterns('polyMesh'))
    for name in ('Allrun', 'Allclean'):
        shutil.copy2(HERE / 'openfoam' / name, of / name)
    commands = {}
    # Write command receipts after every invocation, including unsuccessful runs.
    commands['openfoam'] = run_command(['sh', str(of / 'Allrun'), str(plan['mpi_ranks'])], REPO, env, destination / 'openfoam-launcher.log')
    write_json(destination / 'commands.json', commands)
    if commands['openfoam']['returncode']:
        raise ValueError('OpenFOAM failed; logs retained, no candidate manifest generated')
    if plan.get('reference_field_controls') is not None:
        import reference_field_gate
        write_json(destination / 'reference-field-controls.json',
                   plan['reference_field_controls'])
        reference_field_gate.run(of, plan['mpi_ranks'],
                                 destination / 'reference-field-controls.json')
    env.update(SIMPLEFLUID_COMPILER=plan['build']['compiler'], SIMPLEFLUID_BUILD_CONFIG=plan['build']['config'])
    for role in ROLES:
        run_env = dict(env)
        run_env.update({'SIMPLEFLUID_PITZ_' + ENV_NAMES[k]: str(int(v)) if isinstance(v, bool) else str(v)
                        for k, v in plan['runs'][role].items()})
        run_env['SIMPLEFLUID_PITZ_OUTPUT_DIR'] = str(destination / role)
        commands[role] = run_command(['sh', str(HERE / 'run_simplefluid.sh'), str(plan['mpi_ranks'])], REPO, run_env, destination / f'{role}.log')
        write_json(destination / 'commands.json', commands)
        if commands[role]['returncode']:
            raise ValueError(f'{role} failed; logs retained, no candidate manifest generated')
    retain_build_provenance(destination, plan, required=True)
    retain_profiles(destination, plan)
    if plan.get('sampling_method') == 'openfoam-cellPoint-matched-mesh-v1':
        import common_sampling
        for role in ROLES:
            common_sampling.sample(of,
                [destination / role / f'simplefluid_cells_rank{rank}.csv' for rank in range(plan['mpi_ranks'])],
                plan['runs'][role]['mesh_divisor'], destination / role / 'common-sampling',
                mesh_multiplier=plan['runs'][role]['mesh_multiplier'])
    checksums = {p.relative_to(destination).as_posix(): sha256(p)
                 for p in sorted(destination.rglob('*')) if p.is_file()}
    write_json(destination / 'checksums.json', checksums)
    report = verify(destination)
    write_json(destination / 'measurements.json', report)
    checksums['measurements.json'] = sha256(destination / 'measurements.json')
    write_json(destination / 'checksums.json', checksums)
    draft(destination, report)
    return report


def retain_profiles(root, plan):
    """Keep the actual sampled SimpleFluid layers alongside the raw rank CSVs."""
    for role in ROLES:
        cells = comparator.read_simplefluid([
            str(root / role / f'simplefluid_cells_rank{rank}.csv')
            for rank in range(plan['mpi_ranks'])])
        directory = root / role / 'profiles'
        directory.mkdir()
        for station, x in comparator.STATIONS.items():
            actual_x, rows = comparator.simplefluid_profile(cells, x)
            with (directory / f'{station}.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(('x', 'y', 'ux', 'uy'))
                writer.writerows((actual_x, *row) for row in rows)


def profile_deltas(reference_cells, result_cells):
    result = {}
    for station, x in comparator.STATIONS.items():
        ref_x, ref = comparator.simplefluid_profile(reference_cells, x)
        actual_x, actual = comparator.simplefluid_profile(result_cells, x)
        # Meshes can have different outer cell-centre positions: retain and
        # report the common-domain fraction instead of extrapolating.
        common = [r for r in actual if ref[0][0] <= r[0] <= ref[-1][0]]
        metrics = {'reference_x_m': ref_x, 'result_x_m': actual_x,
                   'samples': len(common), 'total_result_samples': len(actual),
                   'common_span_fraction': (common[-1][0] - common[0][0]) / (ref[-1][0] - ref[0][0]) if len(common) >= 2 else 0}
        for component, index in comparator.COMPONENTS.items():
            l2, linf = comparator.error_norms(ref, common, index)
            if not math.isfinite(l2) or not math.isfinite(linf):
                raise ValueError('non-finite refinement metric')
            metrics[component] = {'l2': l2, 'linf': linf}
        result[station] = metrics
    return result


def verify(root, replay_sampling=False):
    root = root.resolve()
    checksums = read_json(root / 'checksums.json')
    if (root / 'collection_failure.json').exists():
        raise ValueError('collection failed; retained files are diagnostic only')
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError('missing evidence checksums')
    for name, digest in checksums.items():
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or sha256(inside(root, name)) != digest:
            raise ValueError(f'SHA-256 mismatch: {name}')

    def required(name):
        if name not in checksums:
            raise ValueError(f'unauthenticated required evidence: {name}')
        return inside(root, name)

    plan = validate_plan(read_json(required('plan.json')))
    if replay_sampling and plan.get('sampling_method') != 'openfoam-cellPoint-matched-mesh-v1':
        raise ValueError('official sampler replay requires a common-sampling bundle')
    provenance = read_json(required('provenance.json'))
    if not re.fullmatch('[0-9a-f]{40}', provenance['revision']):
        raise ValueError('invalid source revision')
    if provenance['openfoam_environment']['WM_PROJECT_VERSION'] != 'v2606':
        raise ValueError('incorrect OpenFOAM version')
    for name in ('tools.json', 'binary.json', 'CMakeCache.txt', 'source.diff',
                 'source/pitz_daily.cc', 'source/environments.sh',
                 'source/run_simplefluid.sh', 'source/compare_profiles.py', 'source/collect_evidence.py'):
        required(name)
    commands = read_json(required('commands.json'))
    if set(commands) != {'openfoam', *ROLES} or any(type(r.get('returncode')) is not int or r['returncode'] != 0 for r in commands.values()):
        raise ValueError('missing/failed solver command')
    reference_field_proof = None
    if plan.get('reference_field_controls') is not None:
        import reference_field_gate
        controls = read_json(required('reference-field-controls.json'))
        if controls != plan['reference_field_controls']:
            raise ValueError('reference field controls differ from declared plan')
        reference_field_proof = reference_field_gate.verify(
            root / 'openfoam', plan['mpi_ranks'], controls,
            lambda path: required(path.relative_to(root).as_posix()))
    of_diagnostics = diagnostics(required('openfoam/log.solver').read_text(encoding='utf-8', errors='replace'), 'openfoam')
    if not of_diagnostics['residual_control_reported']:
        raise ValueError('OpenFOAM did not report residual-control convergence')
    if not of_diagnostics['residuals'] or any(
            not math.isfinite(float(r[k])) or float(r[k]) < 0
            for r in of_diagnostics['residuals'] for k in ('initial', 'final')):
        raise ValueError('OpenFOAM residual history is empty or non-finite')
    of_profiles = comparator.latest_common_profiles(root / 'openfoam')
    if {p.parent.name for p in of_profiles.values()} != {of_diagnostics['last_time']}:
        raise ValueError('OpenFOAM selected profile time differs from final solver time')
    for p in of_profiles.values():
        required(p.relative_to(root).as_posix())
    for path in ('system/controlDict', 'system/fvSolution', 'system/fvSchemes', 'system/blockMeshDict',
                 'constant/transportProperties', 'constant/turbulenceProperties', '0/U', '0/k', '0/epsilon', '0/p', '0/nut'):
        required('openfoam/' + path)
    if any(run['openfoam_outlet_stress'] for run in plan['runs'].values()):
        for name in ('IncompressibleMomentumEquation.tcc', 'FieldStoredTransportSystem.hh', 'TransportSystem.hh'):
            required('source/' + name)
    cells, sf_diagnostics, field_quality = {}, {}, {}
    for role in ROLES:
        paths = [required(f'{role}/simplefluid_cells_rank{rank}.csv') for rank in range(plan['mpi_ranks'])]
        actual = set((root / role).glob('simplefluid_cells_rank*.csv'))
        if actual != set(paths):
            raise ValueError(f'{role}: unexpected rank CSV set')
        field_quality[role] = evidence_checks.cell_quality(
            paths, require_physical_fields=plan.get('sampling_method') == 'openfoam-cellPoint-matched-mesh-v1')
        diag = diagnostics(required(f'{role}.log').read_text(encoding='utf-8', errors='replace'), 'simplefluid')
        summary = diag['summary']
        run = plan['runs'][role]
        if diag['mesh_multiplier'] != run['mesh_multiplier']:
            raise ValueError(f'{role}: mismatched mesh multiplier')
        expected_outlet = 'zeroGradient_stress' if run['openfoam_outlet_stress'] else 'traction_free'
        if diag['velocity_neumann'] != expected_outlet and (run['openfoam_outlet_stress'] or diag['velocity_neumann'] is not None):
            raise ValueError(f'{role}: missing or mismatched velocity outlet stress declaration')
        if run['modified_pressure'] and diag['pressure_convention'] != 'modified':
            raise ValueError(f'{role}: missing or mismatched modified pressure declaration')
        if not run['modified_pressure'] and diag['pressure_convention'] not in (None, 'mechanical'):
            raise ValueError(f'{role}: mismatched mechanical pressure declaration')
        if diag['transport_preconditioner'] != run['transport_preconditioner']:
            raise ValueError(f'{role}: missing or mismatched transport preconditioner')
        if diag['reported_unconverged_steps']:
            raise ValueError(f'{role}: solver reported unconverged steps')
        if summary is None or int(summary['mpi_ranks']) != plan['mpi_ranks'] or int(summary['mesh_divisor']) != run['mesh_divisor']:
            raise ValueError(f'{role}: missing or mismatched solver summary')
        final_time = float(summary['time_s'])
        if not math.isfinite(final_time) or final_time <= 0:
            raise ValueError(f'{role}: invalid final time')
        if run['steady_state'] and not diag['steady_reported']:
            raise ValueError(f'{role}: steady-state criterion was not reached')
        if run['steady_state']:
            values = dict(re.findall(r'(\w+)=([^,\s]+)', diag['steady_diagnostics']))
            final_values = {k: float(values[k]) for k in (
                'final_update_rate', 'final_max_Co', 'final_dt', 'last_transport_linear_tolerance')}
            if (not all(math.isfinite(v) and v >= 0 for v in final_values.values())
                    or final_values['final_update_rate'] > run['steady_tolerance']
                    or final_values['last_transport_linear_tolerance'] > run['linear_tolerance']
                    or not run['steady_min_dt_s'] <= final_values['final_dt'] <= run['steady_max_dt_s']
                    or not run['steady_min_steps'] <= int(summary['steps']) <= run['steps']
                    or int(values['steps']) != int(summary['steps'])):
                raise ValueError(f'{role}: inconsistent final steady diagnostics')
            diag['final_dt_s'] = final_values['final_dt']
        if not run['steady_state'] and (int(summary['steps']) != run['steps'] or not math.isclose(final_time, run['steps'] * run['dt_s'], rel_tol=1e-5)):
            raise ValueError(f'{role}: fixed run did not reach requested endpoint')
        cells[role] = comparator.read_simplefluid([str(p) for p in paths])
        for station, x in comparator.STATIONS.items():
            profile = required(f'{role}/profiles/{station}.csv')
            actual_x, rows = comparator.simplefluid_profile(cells[role], x)
            retained = comparator.read_simplefluid([str(profile)])
            if [(r['x'], r['y'], r['ux'], r['uy']) for r in retained] != [(actual_x, *r) for r in rows]:
                raise ValueError('retained extracted profile differs from raw rank CSVs')
        sf_diagnostics[role] = diag
    if float(sf_diagnostics['stationarity']['summary']['time_s']) <= float(sf_diagnostics['baseline']['summary']['time_s']):
        raise ValueError('stationarity sample must be later than baseline')
    if (plan['runs']['baseline']['steady_state']
            and sf_diagnostics['timestep']['final_dt_s'] >= sf_diagnostics['baseline']['final_dt_s']):
        raise ValueError('adaptive timestep refinement did not realize a smaller final timestep; reduce the Courant target and cap')
    measurements = comparator.collect_metrics(root / 'openfoam', [str(root / 'baseline/simplefluid_cells_rank*.csv')])
    for metrics in measurements.values():
        metrics['openfoam_path'] = metrics['openfoam_path'].relative_to(root).as_posix()
    refinements = {role: profile_deltas(cells['baseline'], cells[role]) for role in ROLES[1:]}
    legacy = None
    if plan.get('sampling_method') == 'openfoam-cellPoint-matched-mesh-v1':
        import common_sampling
        required('source/common_sampling.py')
        required('source/TimeStepperOptions.hh')
        required('source/IncompressibleIsothermalSolver.tcc')
        for role in ROLES:
            directory = root / role / 'common-sampling'
            common_sampling.verify_sampling(root / 'openfoam',
                [root / role / f'simplefluid_cells_rank{rank}.csv' for rank in range(plan['mpi_ranks'])],
                plan['runs'][role]['mesh_divisor'], directory,
                lambda path: required(path.relative_to(root).as_posix()),
                mesh_multiplier=plan['runs'][role]['mesh_multiplier'])
            if replay_sampling:
                import replay_validation
                replay_validation.replay_profiles(root / 'openfoam',
                    [root / role / f'simplefluid_cells_rank{rank}.csv' for rank in range(plan['mpi_ranks'])],
                    plan['runs'][role]['mesh_divisor'], directory,
                    mesh_multiplier=plan['runs'][role]['mesh_multiplier'])
        legacy = {'candidate_errors': measurements, 'refinement_deltas': refinements,
                  'interpretation': 'Legacy nearest-layer diagnostics; use common-sampler metrics for this study.'}
        measurements = common_sampling.profile_metrics(root / 'openfoam', root / 'baseline/common-sampling')
        for station, metric in measurements.items():
            metric['openfoam_path'] = of_profiles[station].relative_to(root).as_posix()
        refinements = {role: common_sampling.profile_metrics(root / 'baseline/common-sampling', root / role / 'common-sampling') for role in ROLES[1:]}
    return {'status': 'pending_external_qualification', 'units': 'm/s',
            'sampling_method': plan.get('sampling_method', 'nearest-layer-v1'),
            'field_quality': field_quality,
            'reference_field_stationarity': reference_field_proof,
            'legacy_diagnostics': legacy,
            'diagnostics': {'openfoam': of_diagnostics, 'simplefluid': sf_diagnostics},
            'candidate_errors': measurements,
            'refinement_deltas': refinements,
            'review_required': ['Check matched physics, boundaries, schemes and compiler/build provenance.',
                                'Assess residual histories, stationarity, spatial coverage and refinement deltas.',
                                'Justify each tolerance from refinement and repeatability; hashes do not certify physical accuracy.']}


def draft(root, report):
    document = read_json(HERE / 'reference/physical_acceptance.json')
    document['qualification']['status'] = 'pending_external_qualification'
    document['qualification']['source_files_sha256'] = read_json(root / 'checksums.json')
    document['qualification']['reason'] = 'Collected candidate evidence requires independent CFD qualification and justified limits.'
    definition = document['reference_definition']
    definition['sampling_method'] = read_json(root / 'plan.json').get('sampling_method', 'nearest-layer-v1')
    definition['simplefluid_source_revision'] = read_json(root / 'provenance.json')['revision']
    definition['openfoam_profile_files'] = {s: m['openfoam_path'] for s, m in report['candidate_errors'].items()}
    plan = read_json(root / 'plan.json')
    definition['simplefluid_run'] = copy.deepcopy(plan['runs']['baseline'])
    definition['simplefluid_run'].update(mpi_ranks=plan['mpi_ranks'], rank_csv_files=[f'baseline/simplefluid_cells_rank{r}.csv' for r in range(plan['mpi_ranks'])])
    # Measured errors are retained separately. Never turn observations into limits.
    document['stations'] = {}
    document['evidence_measurements'] = 'measurements.json'
    write_json(root / 'candidate_acceptance.json', document)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    collect_parser = sub.add_parser('collect')
    collect_parser.add_argument('--plan', type=Path, required=True)
    collect_parser.add_argument('--output', type=Path, required=True)
    verify_parser = sub.add_parser('verify')
    verify_parser.add_argument('bundle', type=Path)
    verify_parser.add_argument('--replay-sampling', action='store_true',
                               help='independently regenerate common profiles using the recorded OpenFOAM tool')
    template_parser = sub.add_parser('template')
    template_parser.add_argument('--output', type=Path, required=True)
    try:
        args = parser.parse_args(argv)
        if args.command == 'collect':
            collect(args.plan, args.output)
        elif args.command == 'verify':
            print(json.dumps(verify(args.bundle, replay_sampling=args.replay_sampling), indent=2, allow_nan=False))
        else:
            template = read_json(HERE / 'reference/physical_acceptance.json')['reference_definition']['simplefluid_run']
            template.setdefault('mesh_multiplier', 1)
            template.setdefault('modified_pressure', False)
            template.setdefault('openfoam_outlet_stress', False)
            template = {k: template[k] for k in ENV_NAMES}
            # Retained evidence must not silently choose a new study's budget.
            template['steps'] = None
            runs = {r: copy.deepcopy(template) for r in ROLES}
            runs['mesh']['mesh_divisor'] = 2
            with args.output.open('x', encoding='utf-8') as stream:
                json.dump({'schema_version': 1, 'mpi_ranks': None,
                           'reference_field_controls': None,
                           'build': {'compiler': 'GCC', 'config': 'RelWithDebInfo'},
                           'runs': runs}, stream, indent=2, allow_nan=False)
                stream.write('\n')
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f'evidence collection/verification failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
