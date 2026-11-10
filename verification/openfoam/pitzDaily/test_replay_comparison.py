"""Independent failure-evidence tests without a solver or MPI runtime."""
import copy
import csv
import json
import math
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from replay_comparison import compare_replays, format_comparison, integrated_row_comparison


DENSITY_COLUMNS = ('momentum_x', 'momentum_y', 'momentum_z', 'continuity', 'k', 'secondary')


def summary(densities=(0.0, 0.0), scales=(2.0, 2.0)):
    # Two independently specified cell volumes are 1 and 3.
    return dict(volume_rms=math.sqrt((densities[0] ** 2 + 3 * densities[1] ** 2) / 4),
                max_density=max(abs(value) for value in densities),
                signed_sum=densities[0] + 3 * densities[1],
                scale=math.sqrt((scales[0] ** 2 + 3 * scales[1] ** 2) / 4))


def diagnostics():
    return dict(momentum=[summary() for _ in range(3)], k=summary(), secondary=summary(),
                reconstructed_continuity=summary(),
                residual_density_units=dict(momentum='m/s^2', k='m^2/s^3',
                                            secondary='m^2/s^4', continuity='1/s'))


def cell(geometry_id, algebraic_id, x, volume):
    row = dict(cell_geometry_gid=geometry_id, cell_id=algebraic_id, x=x, y=0.0, z=0.0, volume=volume)
    for column in DENSITY_COLUMNS:
        row[column] = 0.0
        row[column + '_algebraic_scale_density'] = 2.0
    return row


class ReplayComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.serial = self.root / 'serial'
        self.parallel = self.root / 'parallel'
        self.serial.mkdir()
        self.parallel.mkdir()
        self.a = diagnostics()
        self.b = diagnostics()
        self.serial_cells = [cell(41, 0, 0.0, 1.0), cell(42, 1, 1.0, 3.0)]
        # Geometry identity is preserved while algebraic IDs and rank order differ.
        self.parallel_cells = [cell(42, 900, 1.0, 3.0), cell(41, 700, 0.0, 1.0)]

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, geometry=True, scales=True, blank_geometry=False, integrated=False):
        for directory, data in ((self.serial, self.a), (self.parallel, self.b)):
            (directory / 'diagnostics_step0.json').write_text(json.dumps(data))
        columns = ['cell_id', 'x', 'y', 'z', 'volume', *DENSITY_COLUMNS]
        if geometry:
            columns.append('cell_geometry_gid')
        if scales:
            columns.extend(column + '_algebraic_scale_density' for column in DENSITY_COLUMNS)
        if integrated:
            for column in DENSITY_COLUMNS:
                columns.extend(column + suffix for suffix in (
                    '_integrated_residual', '_integrated_algebraic_scale',
                    '_accumulation_terms', '_accumulation_eligible'))
        groups = [(self.serial, 0, self.serial_cells)]
        groups.extend((self.parallel, rank, [row]) for rank, row in enumerate(self.parallel_cells))
        for directory, rank, rows in groups:
            with (directory / f'diagnostics_step0_cells_rank{rank}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
                writer.writeheader()
                for original in rows:
                    row = dict(original)
                    if blank_geometry:
                        row['cell_geometry_gid'] = ''
                    writer.writerow(row)

    def fail_z(self):
        self.a['momentum'][2] = summary(scales=(2.0, 4.0))
        self.b['momentum'][2] = summary((0.0, 4e-10), (3.0, 5.0))
        self.serial_cells[0]['momentum_z_algebraic_scale_density'] = 2.0
        self.serial_cells[1]['momentum_z_algebraic_scale_density'] = 4.0
        self.parallel_cells[0]['momentum_z'] = 4e-10
        self.parallel_cells[0]['momentum_z_algebraic_scale_density'] = 5.0
        self.parallel_cells[1]['momentum_z_algebraic_scale_density'] = 3.0

    def compare(self):
        return compare_replays(self.serial, self.parallel)

    def test_passing_summaries_require_no_cell_files(self):
        (self.serial / 'diagnostics_step0.json').write_text(json.dumps(self.a))
        (self.parallel / 'diagnostics_step0.json').write_text(json.dumps(self.b))
        report = self.compare()
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['comparisons']), 18)
        self.assertEqual(report['failures'], [])
        self.assertEqual(report['cell_evidence']['status'], 'not_requested')

    def test_requested_summary_and_cell_details_use_geometry_identity(self):
        self.fail_z()
        self.write()
        report = self.compare()
        self.assertFalse(report['passed'])
        self.assertEqual(len(report['comparisons']), 18)
        self.assertEqual({f['norm'] for f in report['failures']},
                         {'volume_rms', 'max_density', 'signed_sum'})
        self.assertEqual(report['cell_evidence']['matching'], 'cell_geometry_gid')
        failure = next(f for f in report['failures'] if f['norm'] == 'volume_rms')
        self.assertEqual(failure['component'], 'momentum_z')
        self.assertEqual(failure['serial_value'], 0.0)
        self.assertAlmostEqual(failure['parallel_value'], math.sqrt(3.0) * 2e-10, delta=1e-25)
        self.assertEqual(failure['absolute_difference'], failure['parallel_value'])
        self.assertEqual(failure['serial_algebraic_scale'], math.sqrt(13.0))
        self.assertEqual(failure['parallel_algebraic_scale'], math.sqrt(21.0))
        self.assertEqual(failure['algebraic_scale_kind'], 'algebraic_absolute_row_terms_density_rms')
        self.assertEqual(failure['serial_algebraic_scale_units'], 'm/s^2')
        worst = failure['worst_cell_difference']
        self.assertEqual(worst['cell_geometry_gid'], 42)
        self.assertEqual((worst['serial_cell_id'], worst['parallel_cell_id']), (1, 900))
        self.assertEqual((worst['serial_volume'], worst['parallel_volume']), (3.0, 3.0))
        self.assertEqual((worst['serial_value'], worst['parallel_value']), (0.0, 4e-10))
        self.assertEqual(worst['absolute_difference'], 4e-10)
        self.assertEqual(worst['serial_algebraic_scale_density'], 4.0)
        self.assertEqual(worst['parallel_algebraic_scale_density'], 5.0)
        text = format_comparison(report)
        for token in ['momentum_z', 'volume_rms', 'serial=', 'parallel=', 'absolute_difference=',
                      'serial_algebraic_scale=', 'parallel_algebraic_scale=', 'cell_geometry_gid',
                      'serial_volume', 'parallel_volume', 'integrated residual units']:
            self.assertIn(token, text)
        self.assertEqual(json.loads(json.dumps(report))['failures'][0]['component'], 'momentum_z')

    def test_all_eighteen_failures_are_accumulated(self):
        for key in ['k', 'secondary', 'reconstructed_continuity']:
            self.b[key] = summary((1.0, 2.0))
        self.b['momentum'] = [summary((1.0, 2.0)) for _ in range(3)]
        for row in self.parallel_cells:
            value = 2.0 if row['cell_geometry_gid'] == 42 else 1.0
            for column in DENSITY_COLUMNS:
                row[column] = value
        self.write()
        report = self.compare()
        self.assertEqual(len(report['failures']), 18)
        self.assertEqual(report['cell_evidence']['status'], 'available')

    def test_original_absolute_tolerance_boundary(self):
        self.b['momentum'][2]['volume_rms'] = 1e-10
        self.write()
        self.assertTrue(self.compare()['passed'])
        self.b['momentum'][2]['volume_rms'] = math.nextafter(1e-10, math.inf)
        self.write()
        report = self.compare()
        self.assertFalse(report['passed'])
        self.assertEqual(len(report['failures']), 1)
        self.assertEqual(report['failures'][0]['allowed_difference'], 1e-10)

    def test_original_relative_tolerance_is_retained(self):
        self.a['momentum'][0]['volume_rms'] = 1e12
        self.b['momentum'][0]['volume_rms'] = 1e12 + 50
        self.write()
        self.assertTrue(self.compare()['passed'])
        self.b['momentum'][0]['volume_rms'] = 1e12 + 200
        self.write()
        self.assertFalse(self.compare()['passed'])

    def test_cell_evidence_adds_no_acceptance_gate(self):
        self.parallel_cells[0]['momentum_z'] = 100.0
        self.write()
        report = self.compare()
        self.assertTrue(report['passed'])
        self.assertEqual(report['cell_evidence']['status'], 'not_requested')

    def test_legacy_and_blank_geometry_ids_are_explicitly_unavailable(self):
        self.fail_z()
        for geometry, blank in [(False, False), (True, True)]:
            with self.subTest(geometry_column=geometry, blank=blank):
                self.write(geometry=geometry, scales=False, blank_geometry=blank)
                report = self.compare()
                self.assertEqual(report['cell_evidence']['matching'], 'coordinate_key')
                self.assertFalse(report['cell_evidence']['geometry_id_available'])
                worst = report['failures'][0]['worst_cell_difference']
                self.assertIsNone(worst['cell_geometry_gid'])
                self.assertIsNone(worst['serial_algebraic_scale_density'])
                self.assertIsNone(worst['parallel_algebraic_scale_density'])
                self.assertEqual((worst['serial_cell_id'], worst['parallel_cell_id']), (1, 900))
                self.assertIn('Cell algebraic scales unavailable', format_comparison(report))

    def test_malformed_cells_preserve_original_failures(self):
        self.fail_z()
        original = copy.deepcopy(self.parallel_cells)
        cases = ['duplicate_id', 'different_id_set', 'different_coordinates', 'different_volume', 'mixed_geometry']
        for problem in cases:
            with self.subTest(problem=problem):
                self.parallel_cells = copy.deepcopy(original)
                if problem == 'duplicate_id':
                    self.parallel_cells[1]['cell_geometry_gid'] = 42
                elif problem == 'different_id_set':
                    self.parallel_cells[1]['cell_geometry_gid'] = 99
                elif problem == 'different_coordinates':
                    self.parallel_cells[1]['x'] = 0.1
                elif problem == 'different_volume':
                    self.parallel_cells[0]['volume'] = 6.0
                else:
                    self.parallel_cells[1]['cell_geometry_gid'] = ''
                self.write()
                report = self.compare()
                self.assertFalse(report['passed'])
                self.assertEqual(len(report['failures']), 3)
                self.assertEqual(report['cell_evidence']['status'], 'unavailable')
                self.assertIn('evidence_error', report)
                self.assertTrue(all('worst_cell_difference' not in f for f in report['failures']))
                self.assertIn('momentum_z volume_rms', format_comparison(report))

    def test_late_evidence_error_does_not_publish_partial_details(self):
        self.fail_z()
        self.b['k'] = summary((1.0, 1.0))
        for row in self.parallel_cells:
            row['k'] = 1.0
            row['momentum_z'] = ''
        self.write()
        report = self.compare()
        self.assertEqual(len(report['failures']), 6)
        self.assertIn('momentum_z', report['evidence_error'])
        self.assertTrue(all('worst_cell_difference' not in f for f in report['failures']))

    def test_maximum_locations_are_separate_from_largest_cell_difference(self):
        self.a['momentum'][2] = summary((10.0, 9.0))
        self.b['momentum'][2] = summary((9.0, 1.0))
        self.serial_cells[0]['momentum_z'] = 10.0
        self.serial_cells[1]['momentum_z'] = 9.0
        self.parallel_cells[0]['momentum_z'] = 1.0
        self.parallel_cells[1]['momentum_z'] = 9.0
        self.write()
        failure = next(f for f in self.compare()['failures'] if f['norm'] == 'max_density')
        self.assertEqual(failure['worst_cell_difference']['cell_geometry_gid'], 42)
        self.assertEqual(failure['serial_argmax']['cell_geometry_gid'], 41)
        self.assertEqual(failure['parallel_argmax']['cell_geometry_gid'], 41)
        self.assertEqual(failure['serial_argmax']['value'], 10.0)
        self.assertEqual(failure['parallel_argmax']['value'], 9.0)

    def test_legacy_coordinate_key_matches_half_away_from_zero(self):
        self.fail_z()
        self.serial_cells[0]['x'] = 0.5e-12
        self.parallel_cells[1]['x'] = 0.6e-12
        self.serial_cells[1]['x'] = -0.5e-12
        self.parallel_cells[0]['x'] = -0.6e-12
        self.write(geometry=False)
        report = self.compare()
        self.assertEqual(report['cell_evidence']['status'], 'available')
        self.assertEqual(report['failures'][0]['worst_cell_difference']['serial_cell_id'], 1)


    def test_direct_integrated_data_is_checked_even_when_summaries_pass(self):
        metadata = evaluation()
        self.a['momentum'][2]['residual_evaluation'] = metadata
        self.b['momentum'][2]['residual_evaluation'] = copy.deepcopy(metadata)
        for row in self.serial_cells + self.parallel_cells:
            row['momentum_z_integrated_residual'] = 0.0
            row['momentum_z_integrated_algebraic_scale'] = 1.0
            row['momentum_z_accumulation_terms'] = 1
            row['momentum_z_accumulation_eligible'] = 1
        self.parallel_cells[0]['momentum_z_integrated_residual'] = 0.25
        # Deliberately leave densities/global summaries at zero to catch reconstruction.
        self.write(integrated=True)
        report = self.compare()
        self.assertTrue(report['passed'])
        self.assertEqual(report['failures'], [])
        check = report['integrated_residual_check']
        self.assertEqual(check['matched_cell_count'], 2)
        cells = check['components']['momentum_z']['cells']
        worst = next(row for row in cells if row['cell_geometry_gid'] == 42)
        self.assertEqual(worst['absolute_difference'], 0.25)
        self.assertEqual(worst['status'], 'exceeds')
        self.assertEqual(worst['serial_cell_id'], 1)
        self.assertEqual(worst['parallel_cell_id'], 900)
        self.assertEqual(check['components']['momentum_z']['counts']['exceeds'], 1)
        self.assertIn('upstream_unassessed', format_comparison(report))

    def test_legacy_exports_have_an_unavailable_integrated_bound(self):
        self.fail_z()
        self.write()
        report = self.compare()
        self.assertFalse(report['passed'])
        self.assertEqual(report['integrated_residual_check']['status'], 'unavailable')
        self.assertEqual(len(report['failures']), 3)


def evaluation(digits=53):
    return dict(kind='crs_row_dot_minus_rhs', scalar_radix=2, scalar_digits=digits,
                unit_roundoff=math.ldexp(1.0, -digits), round_to_nearest=True,
                arithmetic_assumptions=dict(iec559_binary=True, fast_math_disabled=True,
                                            underflow_checks='normal_or_zero_intermediates'))


def integrated_cell(residual, scale, terms, volume=1.0, geometry_id=41):
    return dict(cell_geometry_gid=geometry_id, cell_id=7, volume=volume,
                integrated_residuals={'momentum_z': residual},
                integrated_scales={'momentum_z': scale},
                accumulation_terms={'momentum_z': terms},
                accumulation_eligible={'momentum_z': '1'})


class IntegratedRowComparisonTests(unittest.TestCase):
    def compare(self, a, b, metadata=None, parallel_metadata=None):
        metadata = evaluation() if metadata is None else metadata
        parallel_metadata = metadata if parallel_metadata is None else parallel_metadata
        return integrated_row_comparison(a, b, 'momentum_z', metadata, parallel_metadata)

    def test_hand_bound_uses_each_operation_count_and_stored_maximum_scale(self):
        a = integrated_cell(0.0, 2.0, '1')
        b = integrated_cell(math.ldexp(1.0, -51), 3.0, '2')
        report = self.compare(a, b)
        u = Fraction(1, 2 ** 53)
        # Independently expand the two beta terms for k=2 and k=4.
        expected_coefficient = Fraction(2, 1) / (1 - 4 * u) + Fraction(4, 1) / (1 - 8 * u)
        expected_bound = expected_coefficient * u * 3
        self.assertEqual(report['status'], 'within')
        self.assertEqual(report['serial_operation_count'], 2)
        self.assertEqual(report['parallel_operation_count'], 4)
        self.assertEqual(report['accumulation_coefficient'], float(expected_coefficient))
        self.assertEqual(report['accumulation_bound'], float(expected_bound))
        self.assertEqual(report['signed_difference'], -math.ldexp(1.0, -51))
        self.assertEqual(report['tau'], 0.0)
        self.assertEqual(report['tau_status'], 'upstream_unassessed')

    def test_zero_term_rows_have_no_artificial_absolute_floor(self):
        a = integrated_cell(0.0, 1.0, 0)
        b = integrated_cell(0.0, 1.0, 0)
        self.assertEqual(self.compare(a, b)['status'], 'within')
        b['integrated_residuals']['momentum_z'] = 1e-9
        report = self.compare(a, b)
        self.assertEqual(report['status'], 'exceeds')
        self.assertEqual(report['accumulation_bound'], 0.0)
        self.assertEqual(report['required_tau_diagnostic'], 1e-9)
        self.assertEqual(report['tau'], 0.0)

    def test_exact_boundary_with_real_binary32_precision(self):
        # m=2^21 and u=2^-24 give beta=1/2 on each side, so bound=s=1 exactly.
        metadata = evaluation(24)
        a = integrated_cell(0.0, 1.0, 2 ** 21)
        for difference, expected in [(1.0 - 2 ** -24, 'within'),
                                     (1.0, 'within'), (1.0 + 2 ** -23, 'exceeds')]:
            with self.subTest(difference=difference):
                b = integrated_cell(difference, 1.0, 2 ** 21)
                report = self.compare(a, b, metadata)
                self.assertEqual(report['status'], expected)
                self.assertEqual(report['accumulation_bound'], 1.0)
                self.assertEqual(report['unit_roundoff'], 2 ** -24)

    def test_actual_precision_changes_the_envelope(self):
        a = integrated_cell(0.0, 1.0, 1)
        b = integrated_cell(2 ** -24, 1.0, 1)
        self.assertEqual(self.compare(a, b, evaluation(24))['status'], 'within')
        self.assertEqual(self.compare(a, b, evaluation(53))['status'], 'exceeds')

    def test_cancellation_uses_row_term_scale_instead_of_residual_size(self):
        a = integrated_cell(0.0, 1e16, 10)
        b = integrated_cell(0.25, 2e16, 10)
        report = self.compare(a, b)
        self.assertEqual(report['status'], 'within')
        self.assertGreater(report['accumulation_bound'], 80.0)
        self.assertEqual(report['absolute_difference'], 0.25)

    def test_volume_division_does_not_enter_the_check(self):
        a = integrated_cell(0.0, 1.0, 1, volume=1e-30)
        b = integrated_cell(1e-10, 1.0, 1, volume=1e-30)
        small = self.compare(a, b)
        a['volume'] = b['volume'] = 1e30
        large = self.compare(a, b)
        for key in ['status', 'signed_difference', 'accumulation_bound', 'required_tau_diagnostic']:
            self.assertEqual(small[key], large[key])
        self.assertEqual(small['status'], 'exceeds')

    def test_power_of_two_scaling_preserves_classification(self):
        for factor in [2 ** -100, 1.0, 2 ** 100]:
            with self.subTest(factor=factor):
                a = integrated_cell(0.0, factor, 2)
                b = integrated_cell(math.ldexp(factor, -52), factor, 3)
                self.assertEqual(self.compare(a, b)['status'], 'within')

    def test_invalid_arithmetic_metadata_is_explicitly_unavailable(self):
        a, b = integrated_cell(0.0, 1.0, 1), integrated_cell(0.0, 1.0, 1)
        variants = []
        for key, value in [('scalar_radix', 10), ('scalar_digits', 64),
                           ('round_to_nearest', False), ('unit_roundoff', 2 ** -52),
                           ('kind', 'caller_supplied_integrated')]:
            metadata = evaluation()
            metadata[key] = value
            variants.append(metadata)
        for key, value in [('iec559_binary', False), ('fast_math_disabled', False),
                           ('underflow_checks', 'unverified')]:
            metadata = evaluation()
            metadata['arithmetic_assumptions'][key] = value
            variants.append(metadata)
        for metadata in variants:
            with self.subTest(metadata=metadata):
                report = self.compare(a, b, metadata)
                self.assertEqual(report['status'], 'unavailable')
                self.assertIsNone(report['within_accumulation_envelope'])
                self.assertTrue(report['unavailable_reason'])
        self.assertEqual(self.compare(a, b, evaluation(), evaluation(24))['status'], 'unavailable')

    def test_invalid_rows_and_large_operation_budget_are_unavailable(self):
        original = integrated_cell(0.0, 1.0, 1)
        cases = [('integrated_scales', -1.0), ('integrated_residuals', math.inf),
                 ('accumulation_terms', '-1'), ('accumulation_terms', '1.5'),
                 ('accumulation_terms', True), ('accumulation_terms', 2 ** 51),
                 ('accumulation_eligible', '0')]
        for mapping, value in cases:
            with self.subTest(mapping=mapping, value=value):
                other = copy.deepcopy(original)
                other[mapping]['momentum_z'] = value
                self.assertEqual(self.compare(original, other)['status'], 'unavailable')
        other = copy.deepcopy(original)
        other['cell_geometry_gid'] = None
        self.assertEqual(self.compare(original, other)['status'], 'unavailable')

    def test_large_difference_is_classified_without_invalid_json_numbers(self):
        report = self.compare(integrated_cell(1e308, 1e308, 1),
                              integrated_cell(-1e308, 1e308, 1))
        self.assertEqual(report['status'], 'exceeds')
        self.assertIn('absolute_difference', report['numeric_range_unavailable_fields'])
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
