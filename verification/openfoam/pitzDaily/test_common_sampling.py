import copy
import math
import unittest

import common_sampling as sampling
import collect_evidence as evidence
from test_collect_evidence import plan


class CommonSamplingTests(unittest.TestCase):
    def test_bijection_preserves_values_despite_rank_order(self):
        centres = [(0., 0., 0.), (1., 0., 0.)]
        rows = [dict(x=1., y=0., z=0., ux=7., uy=-2., uz=0.),
                dict(x=0., y=0., z=0., ux=3., uy=1., uz=0.)]
        values, distance = sampling.match_vectors(centres, rows)
        self.assertEqual(values, [(3., 1., 0.), (7., -2., 0.)])
        self.assertEqual(distance, 0)
        for bad in (rows[:1], rows + rows[:1], [dict(rows[0], x=1.01), rows[1]],
                    [dict(rows[0], ux=math.nan), rows[1]], [dict(rows[0], z=.001), rows[1]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sampling.match_vectors(centres, bad)

    def test_ambiguous_mesh_cannot_silently_average(self):
        row = dict(x=0., y=0., z=0., ux=1., uy=0., uz=0.)
        with self.assertRaisesRegex(ValueError, 'bijectively'):
            sampling.match_vectors([(0., 0., 0.), (0., 0., 0.)], [row, row])

    def test_coarse_mesh_uses_exact_same_ceiling_counts_as_native(self):
        text = '\n'.join('hex (0 1 2 3 4 5 6 7) (' + s + ')' for s in
                         ['18 30 1', '180 27 1', '180 30 1', '25 27 1', '25 30 1'])
        result = sampling.coarsen_blocks(text, 4)
        for counts in ['(5 8 1)', '(45 7 1)', '(45 8 1)', '(7 7 1)', '(7 8 1)']:
            self.assertIn(counts, result)
        with self.assertRaises(ValueError):
            sampling.coarsen_blocks(text, 0)
        with self.assertRaises(ValueError):
            sampling.coarsen_blocks('hex (0 1 2) (2 3 1)', 2)

    def test_pressure_convention_must_be_boolean_and_consistent(self):
        settings = plan()
        for run in settings['runs'].values():
            run['modified_pressure'] = True
            run['openfoam_outlet_stress'] = True
        evidence.validate_plan(settings)
        settings['runs']['mesh']['modified_pressure'] = False
        with self.assertRaisesRegex(ValueError, 'unmatched simulation settings'):
            evidence.validate_plan(settings)
        settings['runs']['mesh']['modified_pressure'] = 'true'
        with self.assertRaisesRegex(ValueError, 'boolean'):
            evidence.validate_plan(settings)

    def test_outlet_stress_setting_is_not_a_free_mesh_parameter(self):
        settings = plan()
        settings['runs']['mesh']['openfoam_outlet_stress'] = True
        with self.assertRaisesRegex(ValueError, 'unmatched simulation settings'):
            evidence.validate_plan(settings)
        settings['runs']['mesh']['openfoam_outlet_stress'] = 1
        with self.assertRaisesRegex(ValueError, 'boolean'):
            evidence.validate_plan(settings)

    def test_finer_mesh_and_invalid_scale(self):
        block=(sampling.evidence.HERE / 'openfoam/system/blockMeshDict').read_text()
        fine=sampling.coarsen_blocks(block, 1, 2)
        for counts in ('(36 60 1)', '(360 54 1)', '(360 60 1)', '(50 54 1)', '(50 60 1)'):
            self.assertIn(counts, fine)
        for scale in (0, 3, True, 1.5):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                sampling.coarsen_blocks(block,1,scale)
            settings=plan()
            settings['runs']['baseline']['mesh_multiplier']=scale
            with self.assertRaises(ValueError):
                evidence.validate_plan(settings)

    def test_historical_mechanical_pressure_is_never_inferred_modified(self):
        settings = plan()
        for run in settings['runs'].values():
            run.pop('modified_pressure', None)
        result = evidence.validate_plan(settings)
        self.assertTrue(all(not r['modified_pressure'] for r in result['runs'].values()))


if __name__ == '__main__':
    unittest.main()
