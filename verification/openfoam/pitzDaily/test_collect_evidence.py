#!/usr/bin/env python3
"""Synthetic collection tests; these make no physical qualification claim."""
import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import collect_evidence as evidence
import compare_profiles as comparator
from test_compare_profiles import qualified_simplefluid_run


def plan():
    base = qualified_simplefluid_run()
    for key in ('rank_csv_files', 'mpi_ranks'):
        base.pop(key)
    base.update(steady_state=False, steps=100, dt_s=1e-5)
    runs = {role: copy.deepcopy(base) for role in evidence.ROLES}
    runs['stationarity']['steps'] = 200
    runs['mesh']['mesh_divisor'] = 2
    runs['timestep'].update(steps=200, dt_s=5e-6)
    return {'schema_version': 1, 'mpi_ranks': 1,
            'build': {'compiler': 'GCC', 'config': 'RelWithDebInfo'}, 'runs': runs}


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bundle(self):
        """Construct minimal fake receipts around the existing synthetic data."""
        root = self.root
        settings = plan()
        evidence.write_json(root / 'plan.json', settings)
        evidence.write_json(root / 'provenance.json', {
            'revision': 'a' * 40, 'git_status': 'synthetic test only',
            'openfoam_environment': {'WM_PROJECT_VERSION': 'v2606'}})
        evidence.write_json(root / 'commands.json', {
            role: {'returncode': 0} for role in ('openfoam', *evidence.ROLES)})
        for name in ('tools.json', 'binary.json', 'CMakeCache.txt', 'source.diff',
                     'source/pitz_daily.cc', 'source/environments.sh',
                     'source/run_simplefluid.sh', 'source/compare_profiles.py',
                     'source/collect_evidence.py'):
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('{}', encoding='utf-8')
        shutil.copytree(evidence.HERE / 'openfoam', root / 'openfoam')
        shutil.copytree(evidence.HERE / 'fixtures/openfoam/postProcessing', root / 'openfoam/postProcessing')
        (root / 'openfoam/log.solver').write_text(
            'Time = 2000\nSolving for Ux, Initial residual = 1e-5, Final residual = 1e-7, No Iterations 2\nSIMPLE solution converged in 2000 iterations\n', encoding='utf-8')
        for role, run in settings['runs'].items():
            (root / role).mkdir()
            shutil.copy2(evidence.HERE / 'fixtures/simplefluid_cells_rank0.csv', root / role)
            (root / f'{role}.log').write_text(
                'transport_linear_solver: backend=gmres preconditioner=none\n'
                f"pitzDaily: 6 rank-zero owned cells, divisor={run['mesh_divisor']}, steps={run['steps']}, t={run['steps'] * run['dt_s']}, MPI ranks=1\n", encoding='utf-8')
        evidence.retain_profiles(root, settings)
        self.rehash()
        return root

    def rehash(self):
        evidence.write_json(self.root / 'checksums.json', {
            p.relative_to(self.root).as_posix(): evidence.sha256(p)
            for p in self.root.rglob('*') if p.is_file() and p.name != 'checksums.json'})

    def test_valid_plan_and_refinement_contract(self):
        evidence.validate_plan(plan())
        for role, key, value in [('mesh', 'mesh_divisor', 4),
                                  ('mesh', 'linear_tolerance', 1e-6),
                                  ('timestep', 'steps', 100),
                                  ('timestep', 'steady_target_courant', 0.4),
                                  ('stationarity', 'steps', 100),
                                  ('baseline', 'dt_s', float('nan')),
                                  ('baseline', 'steps', True)]:
            bad = plan()
            bad['runs'][role][key] = value
            with self.subTest(role=role, key=key), self.assertRaises(ValueError):
                evidence.validate_plan(bad)

    def test_preconditioner_is_explicit_and_consistent_across_study(self):
        settings = plan()
        for run in settings['runs'].values():
            run['transport_preconditioner'] = 'ilut'
        evidence.validate_plan(settings)
        settings['runs']['mesh']['transport_preconditioner'] = 'none'
        with self.assertRaisesRegex(ValueError, 'unmatched simulation settings'):
            evidence.validate_plan(settings)
        settings['runs']['mesh']['transport_preconditioner'] = 'ilut; unsafe'
        with self.assertRaisesRegex(ValueError, 'must be none or ilut'):
            evidence.validate_plan(settings)

    def test_logged_preconditioner_must_match_declared_run(self):
        root = self.bundle()
        log = root / 'baseline.log'
        log.write_text(log.read_text().replace('preconditioner=none', 'preconditioner=ilut'))
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'mismatched transport preconditioner'):
            evidence.verify(root)

    def test_logged_pressure_and_outlet_contracts_are_required(self):
        root = self.bundle()
        settings = evidence.read_json(root / 'plan.json')
        for run in settings['runs'].values():
            run['modified_pressure'] = True
            run['openfoam_outlet_stress'] = True
        evidence.write_json(root / 'plan.json', settings)
        for name in ('IncompressibleMomentumEquation.tcc', 'FieldStoredTransportSystem.hh', 'TransportSystem.hh'):
            (root / 'source' / name).write_text('synthetic source fixture')
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'velocity outlet stress declaration'):
            evidence.verify(root)
        for role in evidence.ROLES:
            log = root / f'{role}.log'
            log.write_text('velocity_neumann: zeroGradient_stress\n' + log.read_text())
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'modified pressure declaration'):
            evidence.verify(root)
        for role in evidence.ROLES:
            log = root / f'{role}.log'
            log.write_text('pressure_convention: modified\n' + log.read_text())
        self.rehash()
        self.assertEqual(evidence.verify(root)['status'], 'pending_external_qualification')

    def test_missing_setting_and_adaptive_refinement(self):
        bad = plan()
        del bad['runs']['baseline']['linear_tolerance']
        with self.assertRaises(ValueError):
            evidence.validate_plan(bad)
        adaptive = plan()
        for run in adaptive['runs'].values():
            run['steady_state'] = True
        with self.assertRaisesRegex(ValueError, 'minimum steps'):
            evidence.validate_plan(adaptive)
        adaptive['runs']['stationarity']['steady_min_steps'] = 101
        with self.assertRaisesRegex(ValueError, 'maximum dt'):
            evidence.validate_plan(adaptive)
        adaptive['runs']['timestep']['steady_max_dt_s'] /= 2
        evidence.validate_plan(adaptive)
        adaptive['runs']['timestep']['steady_target_courant'] /= 2
        evidence.validate_plan(adaptive)
        adaptive['runs']['timestep']['steady_target_courant'] = 1.6
        with self.assertRaisesRegex(ValueError, 'cannot increase target Courant'):
            evidence.validate_plan(adaptive)

    def test_recomputed_metrics_and_pending_draft(self):
        root = self.bundle()
        report = evidence.verify(root)
        original = comparator.collect_metrics(evidence.HERE / 'fixtures/openfoam',
                                             [str(evidence.HERE / 'fixtures/simplefluid_cells_rank0.csv')])
        self.assertEqual(report['candidate_errors']['x050']['ux'], original['x050']['ux'])
        self.assertEqual(report['refinement_deltas']['mesh']['x100']['ux']['linf'], 0)
        evidence.draft(root, report)
        candidate = evidence.read_json(root / 'candidate_acceptance.json')
        self.assertEqual(candidate['stations'], {})
        self.assertEqual(candidate['qualification']['status'], 'pending_external_qualification')
        with self.assertRaisesRegex(comparator.ToleranceManifestError, 'not qualified'):
            comparator.load_tolerance_manifest(root / 'candidate_acceptance.json', 'physical_reference')

    def test_tamper_and_missing_checksum_rejected(self):
        root = self.bundle()
        path = root / 'baseline/simplefluid_cells_rank0.csv'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            evidence.verify(root)
        self.rehash()
        hashes = evidence.read_json(root / 'checksums.json')
        del hashes['baseline/simplefluid_cells_rank0.csv']
        evidence.write_json(root / 'checksums.json', hashes)
        with self.assertRaisesRegex(ValueError, 'unauthenticated'):
            evidence.verify(root)

    def test_path_escape_and_missing_rank(self):
        root = self.bundle()
        with self.assertRaises(ValueError):
            evidence.inside(root, '../outside')
        with self.assertRaises(ValueError):
            evidence.inside(root, str(root / 'plan.json'))
        (root / 'baseline/simplefluid_cells_rank1.csv').write_text('extra', encoding='utf-8')
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'rank CSV set'):
            evidence.verify(root)

    def test_exit_zero_is_not_convergence(self):
        root = self.bundle()
        (root / 'openfoam/log.solver').write_text('Time = 2000\nEnd\n', encoding='utf-8')
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'residual-control'):
            evidence.verify(root)

    def test_failed_command_and_mismatched_endpoint(self):
        root = self.bundle()
        commands = evidence.read_json(root / 'commands.json')
        commands['mesh']['returncode'] = 1
        evidence.write_json(root / 'commands.json', commands)
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'failed solver'):
            evidence.verify(root)
        commands['mesh']['returncode'] = 0
        evidence.write_json(root / 'commands.json', commands)
        (root / 'baseline.log').write_text('transport_linear_solver: backend=gmres preconditioner=none\npitzDaily: 6 rank-zero owned cells, divisor=1, steps=1, t=0.001, MPI ranks=1\n', encoding='utf-8')
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'requested endpoint'):
            evidence.verify(root)

    def test_unconverged_steps_rejected_even_with_successful_exit(self):
        root = self.bundle()
        with (root / 'baseline.log').open('a') as stream:
            stream.write('step=1/100 time=1e-5 converged=no nonlinear=1\n')
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'unconverged steps'):
            evidence.verify(root)

    def test_steady_failure_and_profile_time_mismatch(self):
        root = self.bundle()
        log = root / 'openfoam/log.solver'
        log.write_text(log.read_text().replace('Time = 2000', 'Time = 2001'))
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'profile time'):
            evidence.verify(root)
        self.assertFalse(evidence.diagnostics('steady_state_search: reached=no, steps=5', 'simplefluid')['steady_reported'])

    def test_ambient_controls_removed_and_no_overwrite(self):
        with patch.dict('os.environ', {'SIMPLEFLUID_PITZ_DT': '0.5', 'SIMPLEFLUID_BUILD_DIR': '/wrong'}):
            self.assertFalse(any(k.startswith('SIMPLEFLUID_') for k in evidence.clean_environment()))
        settings = self.root / 'settings.json'
        evidence.write_json(settings, plan())
        with self.assertRaises(FileExistsError):
            evidence.collect(settings, self.root)

    def test_failed_collection_retains_authenticated_diagnostics(self):
        settings = self.root / 'plan.json'
        evidence.write_json(settings, plan())
        destination = self.root / 'failed'
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'load OpenFOAM'):
                evidence.collect(settings, destination)
        self.assertEqual(evidence.read_json(destination / 'collection_failure.json')['status'], 'failed')
        self.assertEqual(evidence.read_json(destination / 'checksums.json')['plan.json'], evidence.sha256(destination / 'plan.json'))
        self.assertFalse((destination / 'candidate_acceptance.json').exists())
        with self.assertRaisesRegex(ValueError, 'collection failed'):
            evidence.verify(destination)

    def test_command_failure_receipt_and_log(self):
        log = self.root / 'command.log'
        receipt = evidence.run_command([sys.executable, '-c',
            'print("diagnostic"); raise SystemExit(7)'], self.root, evidence.clean_environment(), log)
        self.assertEqual(receipt['returncode'], 7)
        self.assertIn('diagnostic', log.read_text())
        self.assertIn('started_utc', receipt)

    def test_failed_solver_keeps_build_provenance_without_claiming_execution(self):
        settings = self.root / 'plan.json'
        evidence.write_json(settings, plan())
        repo = self.root / 'repo'
        build = repo / 'build/gcc'
        binary = build / 'bin/RelWithDebInfo/pitz_daily'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'synthetic executable bytes for failure-provenance test')
        (build / 'CMakeCache.txt').write_text('synthetic build cache\n')
        destination = self.root / 'failed-solver'

        def failed_collect(plan_path, output):
            output.mkdir()
            evidence.write_json(output / 'plan.json', plan())
            evidence.write_json(output / 'commands.json', {'baseline': {'returncode': 1}})
            raise ValueError('synthetic launcher failure')

        with patch.object(evidence, 'REPO', repo), patch.object(evidence, '_collect', failed_collect):
            with self.assertRaisesRegex(ValueError, 'launcher failure'):
                evidence.collect(settings, destination)
        record = evidence.read_json(destination / 'binary.json')
        self.assertEqual(record['sha256'], evidence.sha256(binary))
        self.assertIn('available after a failed launcher', record['scope'])
        self.assertIn('CMakeCache.txt', evidence.read_json(destination / 'checksums.json'))
        with self.assertRaisesRegex(ValueError, 'collection failed'):
            evidence.verify(destination)

    def test_template_never_supplies_numerical_limits(self):
        path = self.root / 'template.json'
        self.assertEqual(evidence.main(['template', '--output', str(path)]), 0)
        template = evidence.read_json(path)
        self.assertIsNone(template['runs']['baseline']['steps'])
        self.assertEqual(template['runs']['mesh']['mesh_divisor'], 2)
        with self.assertRaises(ValueError):
            evidence.validate_plan(template)

    def test_nonfinite_residual_and_extracted_profile_mismatch(self):
        root = self.bundle()
        path = root / 'openfoam/log.solver'
        original = path.read_text()
        path.write_text(original.replace('1e-7', 'nan'))
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            evidence.verify(root)
        path.write_text(original)
        profile = root / 'baseline/profiles/x050.csv'
        profile.write_text('x,y,ux,uy\n0.05,0,1,0\n0.05,0.01,1,0\n')
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'differs from raw'):
            evidence.verify(root)

    def test_adaptive_final_diagnostics_checked_against_run_contract(self):
        root = self.bundle()
        settings = plan()
        for run in settings['runs'].values():
            run['steady_state'] = True
        settings['runs']['stationarity']['steady_min_steps'] = 101
        settings['runs']['timestep']['steady_max_dt_s'] /= 2
        evidence.write_json(root / 'plan.json', settings)
        for role, run in settings['runs'].items():
            with (root / f'{role}.log').open('a') as stream:
                stream.write(f"steady_state_search: reached=yes, rejected_steps=0, steps={run['steps']}, final_update_rate=0.00001, final_max_Co=0.1, final_dt={run['dt_s']}, last_transport_linear_tolerance=1e-9\n")
        self.rehash()
        self.assertEqual(evidence.verify(root)['status'], 'pending_external_qualification')
        log = root / 'baseline.log'
        log.write_text(log.read_text().replace('final_update_rate=0.00001', 'final_update_rate=nan'))
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'inconsistent final steady'):
            evidence.verify(root)

    def test_adaptive_refinement_requires_realized_smaller_timestep(self):
        root = self.bundle()
        settings = plan()
        for run in settings['runs'].values():
            run['steady_state'] = True
        settings['runs']['stationarity']['steady_min_steps'] = 101
        settings['runs']['timestep']['steady_max_dt_s'] /= 2
        settings['runs']['timestep']['steady_target_courant'] /= 2
        evidence.write_json(root / 'plan.json', settings)
        for role, run in settings['runs'].items():
            with (root / f'{role}.log').open('a') as stream:
                # The configured cap decreased but was inactive: both final
                # timesteps remained equal, so this is not temporal refinement.
                stream.write(f"steady_state_search: reached=yes, rejected_steps=0, steps={run['steps']}, final_update_rate=0.00001, final_max_Co=0.1, final_dt=1e-5, last_transport_linear_tolerance=1e-9\n")
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'did not realize a smaller final timestep'):
            evidence.verify(root)


if __name__ == '__main__':
    unittest.main()
