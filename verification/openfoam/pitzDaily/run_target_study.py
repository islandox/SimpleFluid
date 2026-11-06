"""Run one retained 48,900-cell diagnostic using the existing launchers.

A completed run remains pending physical qualification. Outputs are never reused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import time
import os
from pathlib import Path
import shutil
import sys

import collect_evidence as e
import common_sampling as sampling
import evidence_checks
import target_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--min-steps', type=int, default=20)
    parser.add_argument('--dt', type=float, default=1e-4)
    parser.add_argument('--transport', choices=('gmres', 'bicgstab'), default='bicgstab')
    parser.add_argument('--pressure', choices=('gmres-muelu', 'cg-symmetric-muelu'),
                        default='cg-symmetric-muelu')
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if e.output(['git', 'status', '--porcelain']):
        raise ValueError('freeze a clean source revision before collecting evidence')
    run = e.read_json(args.settings)
    if (set(run) != set(e.ENV_NAMES) or run['mesh_multiplier'] != 2
            or run['mesh_divisor'] != 1 or not run['modified_pressure']
            or not run['openfoam_outlet_stress'] or run['transport_preconditioner'] != 'ilut'):
        raise ValueError('require the complete matched 48900-cell target configuration')
    if args.steps < args.min_steps or args.min_steps < 1 or args.dt < run['steady_min_dt_s'] * 20:
        raise ValueError('invalid target time-step or minimum/maximum step budget')
    run.update(steps=args.steps, steady_min_steps=args.min_steps,
               dt_s=args.dt/20, steady_max_dt_s=args.dt)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, destination/'run-script.py')
    settings = dict(run, transport_backend=args.transport, pressure_mode=args.pressure)
    e.write_json(destination/'settings.json', settings)
    env = e.clean_environment()
    for key, value in run.items():
        env['SIMPLEFLUID_PITZ_' + e.ENV_NAMES[key]] = (
            str(int(value)) if isinstance(value, bool) else str(value))
    env.update(SIMPLEFLUID_PITZ_TRANSPORT_BACKEND=args.transport,
               SIMPLEFLUID_PITZ_PRESSURE_MODE=args.pressure,
               SIMPLEFLUID_COMPILER='GCC', SIMPLEFLUID_BUILD_CONFIG='RelWithDebInfo',
               SIMPLEFLUID_PITZ_OUTPUT_DIR=str(destination/'result'),
               PRTE_MCA_hwloc_default_binding_policy='none')
    e.write_json(destination/'provenance.json', {
        'revision': e.output(['git', 'rev-parse', 'HEAD']),
        'git_status': e.output(['git', 'status', '--porcelain']),
        'source_diff': e.output(['git', 'diff', '--binary']),
        'environment': {k: v for k, v in env.items()
                        if k.startswith(('SIMPLEFLUID_', 'WM_', 'PRTE_'))}})
    command = ['taskset', '-c', os.environ.get('PITZ_CPUS', '0,2,4'),
               'sh', str(e.HERE/'run_simplefluid.sh'), '3']
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(e.run_command, command, e.REPO, env, destination/'solver.log')
        while not pending.done():
            target_study.capture_runtime(destination.parent)
            time.sleep(10)
        receipt = pending.result()
    e.write_json(destination/'command.json', receipt)
    e.retain_build_provenance(destination,
                              {'build': {'compiler': 'GCC', 'config': 'RelWithDebInfo'}},
                              required=True)
    diagnostics = e.diagnostics((destination/'solver.log').read_text(errors='replace'),
                                'simplefluid')
    e.write_json(destination/'diagnostics.json', diagnostics)
    success = receipt['returncode'] == 0 and diagnostics['steady_reported']
    if success:
        fields = sorted((destination/'result').glob('simplefluid_cells_rank*.csv'))
        sampling.sample(e.HERE/'openfoam', fields, 1,
                        destination/'common-sampling', mesh_multiplier=2)
        e.write_json(destination/'field-quality.json',
                     evidence_checks.cell_quality(fields, require_physical_fields=True))
    e.write_json(destination/'checksums.json', {
        p.relative_to(destination).as_posix(): e.sha256(p)
        for p in sorted(destination.rglob('*'))
        if p.is_file() and p.name != 'checksums.json'})
    if success:
        target_study.verify(destination)
    print(f'returncode={receipt["returncode"]}; steady_reported={diagnostics["steady_reported"]}',
          flush=True)
    if not args.probe and not success:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
