#!/usr/bin/env python3
"""Run one explicitly selected preregistered Q2 GTest selection and retain evidence.

Example:
  python3 verification/vof/run_algebraic_motion.py \
    --executable build/gcc/bin/Debug/testGeneralVoFMotionAccuracy \
    --output build/gcc/verification/q2/planar-uniform-c50 \
    --filter '*GeometricFace_PlanarPatch_Uniform_C50'

MPI qualification should use the registered CTest launchers when available;
--launcher permits an explicitly specified matching launcher for this opt-in
campaign and records its complete arguments. Never launches multiple runs.
"""
import argparse
import datetime
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--filter', required=True, help='Explicit GTest filter; no automatic campaign expansion')
    parser.add_argument('--phase-cleanup', action='store_true', help='Enable GeometricFace tiny-phase cleanup; DonorCell remains raw')
    parser.add_argument('--timeout', type=float, default=1800, help='Wall limit per selected GTest invocation; a timeout is not a qualification pass')
    parser.add_argument('--expected-ranks', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--launcher', nargs=argparse.REMAINDER, default=[], help='Final option, e.g. mpiexec -n 2')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    executable = args.executable.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = root / 'verification/vof/algebraic_motion_manifest.json'
    (output / manifest.name).write_bytes(manifest.read_bytes())
    source_paths = [p for p in subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=root, text=True).splitlines()
        if (root / p).is_file() and (p.startswith('src/') or p.startswith('verification/vof/'))]
    (output / 'working-tree.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=root))
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=root, text=True)
    with tarfile.open(output / 'untracked-source.tar.gz', 'w:gz') as archive:
        for path in untracked.splitlines():
            if (root / path).is_file():
                archive.add(root / path, arcname=path)
    metadata = {
        'utc_started': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'executable': str(executable), 'executable_sha256': digest(executable),
        'source_sha256': {p: digest(root / p) for p in source_paths},
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
        'configured_backend_and_actual_family': 'Recorded per accepted test run in XML/stdout',
        'requested_gtest_filter': args.filter,
        'phase_cleanup_scope': 'Requested switch applies to GeometricFace only; DonorCell remains raw',
    }
    command = [*args.launcher, sys.executable, str(Path(__file__).resolve()),
               '--rank-worker', str(executable), str(output), args.filter]
    metadata['expected_ranks'] = args.expected_ranks
    metadata['phase_cleanup_requested'] = args.phase_cleanup or os.environ.get('SIMPLEFLUID_QUALIFICATION_PHASE_CLEANUP') == '1'
    metadata['wall_timeout_seconds'] = args.timeout
    for parent in executable.parents:
        cache = parent / 'CMakeCache.txt'
        if cache.is_file():
            metadata['cmake_cache'] = str(cache)
            metadata['cmake_cache_sha256'] = digest(cache)
            (output / 'CMakeCache.txt').write_bytes(cache.read_bytes())
            break
    metadata['shared_library_sha256'] = {}
    for parent in executable.parents:
        library_dir = parent / 'lib' / executable.parent.name
        if library_dir.is_dir():
            metadata['shared_library_sha256'] = {str(p): digest(p) for p in library_dir.glob('*.so')}
            break
    metadata['command'] = command
    (output / 'invocation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    env = dict(os.environ, SIMPLEFLUID_RUN_ALGEBRAIC_MOTION='1',
               SIMPLEFLUID_QUALIFICATION_ARTIFACT_DIR=str(output / 'failing-states'))
    env['SIMPLEFLUID_QUALIFICATION_PHASE_CLEANUP'] = '1' if metadata['phase_cleanup_requested'] else '0'
    if args.expected_ranks > 1:
        env['PRTE_MCA_state_base_error_non_zero_exit'] = '0'
    with (output / 'run.log').open('w') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        metadata['pid'] = process.pid
        (output / 'invocation.json').write_text(json.dumps(metadata, indent=2) + '\n')
        try:
            metadata['returncode'] = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            metadata['returncode'] = 124
            metadata['wall_timeout'] = True
    xml_files = sorted(output.glob('rank-*.xml'))
    xml_counts = {}
    for xml in xml_files:
        document = ET.parse(xml).getroot()
        xml_counts[xml.name] = {key: int(document.get(key, '0'))
                                for key in ('tests', 'failures', 'disabled', 'errors')}
        xml_counts[xml.name]['skipped'] = len(document.findall('.//skipped'))
    metadata['rank_xml_counts'] = xml_counts
    complete = (len(xml_counts) == args.expected_ranks and
                all(v['tests'] > 0 and not any(v[k] for k in
                    ('failures', 'errors', 'skipped', 'disabled')) for v in xml_counts.values()))
    metadata['complete_rank_xml_without_skips'] = complete
    metadata['utc_finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (output / 'invocation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(output)
    return metadata['returncode'] if metadata['returncode'] else (0 if complete else 1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--rank-worker':
        executable, destination, selection = sys.argv[2:]
        rank = next((os.environ[key] for key in
                     ('OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'PMIX_RANK', 'SLURM_PROCID')
                     if key in os.environ), '0')
        os.execv(executable, [executable, '--gtest_filter=' + selection,
                             '--gtest_output=xml:' + str(Path(destination) / ('rank-' + rank + '.xml'))])
    sys.exit(main())
