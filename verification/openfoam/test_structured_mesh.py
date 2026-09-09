#!/usr/bin/env python3
"""Boundary-layer coordinate/physical-source invariants for matched fixtures."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from structured_mesh import read_mesh,mesh_statistics,write_openfoam,with_mesh_manifest
from check_structured_mesh import check_points

ROOT=Path(__file__).resolve().parent


class StructuredMeshTests(unittest.TestCase):
    def test_convection_layers_preserve_geometry_source_and_smooth_spacing(self):
        for name,count,layers in [('mesh.dat',790560,6),('mesh_fine.dat',1761760,8)]:
            with self.subTest(mesh=name):
                edges=read_mesh(ROOT/'bottomHeatedBubblyConvection'/name)
                stats=mesh_statistics(edges)
                self.assertEqual(stats['cells'],count)
                self.assertAlmostEqual(stats['volume_m3'],8e-4,15)
                old=read_mesh(ROOT/'bottomHeatedBubblyConvection'/name.replace('.dat','_scale1.dat'))
                for axis in 'xyz':
                    widths=[b-a for a,b in zip(edges[axis],edges[axis][1:])]
                    original=[b-a for a,b in zip(old[axis],old[axis][1:])]
                    for width in widths:self.assertLess(min(abs(width-v) for v in original),1e-13)
                for axis in ['x','z']:
                    values=edges[axis];d=[b-a for a,b in zip(values,values[1:])]
                    for a,b in zip(d[:layers],d[1:layers]): self.assertAlmostEqual(b/a,1.25,12)
                    self.assertLess(stats[axis]['max_adjacent_ratio'],1.5)
                    for a,b in zip(values,reversed(values)): self.assertAlmostEqual(a+b,.2,14)
                for plane in [.2/3,.4/3]:self.assertTrue(any(abs(x-plane)<1e-14 for x in edges['x']))
                self.assertTrue(any(abs(z-.025)<1e-14 for z in edges['z']))
                source_volume=sum((b-a)*(d-c)*.02 for a,b in zip(edges['x'],edges['x'][1:])
                                  for c,d in zip(edges['z'],edges['z'][1:])
                                  if .2/3<(a+b)/2<.4/3 and (c+d)/2<.025)
                self.assertAlmostEqual(source_volume*4e6,400/3,9)

    def test_axial_cases_have_mirrored_graded_layers(self):
        for name,count in [('dispersedBubbleFlow',40400),('planarALE',8400)]:
            edges=read_mesh(ROOT/name/'mesh.dat')
            self.assertEqual(mesh_statistics(edges)['cells'],count)
            self.assertEqual(len(edges['x']),11)
            self.assertEqual(len(edges['y']),11)
            old=read_mesh(ROOT/name/'mesh_scale1.dat')
            for axis in 'xyz':
                original=[b-a for a,b in zip(old[axis],old[axis][1:])]
                for a,b in zip(edges[axis],edges[axis][1:]):
                    self.assertLess(min(abs(b-a-width) for width in original),1e-13)
            self.assertLess(edges['z'][1],edges['z'][2]-edges['z'][1])
            for a,b in zip(edges['z'],reversed(edges['z'])):self.assertAlmostEqual(a+b,10,12)

    def test_openfoam_and_manifest_use_actual_fine_cell_count(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);case=ROOT/'bottomHeatedBubblyConvection'
            edges=write_openfoam(case/'mesh_fine.dat',out)
            self.assertEqual((out/'system/blockMeshDict').read_text().count('hex ('),1)
            self.assertIn('(364 10 484)',(out/'system/blockMeshDict').read_text())
            self.assertEqual(read_mesh(out/'mesh.dat'),edges)
            with_mesh_manifest(case/'transient.json',case/'mesh_fine.dat',out/'manifest.json')
            data=json.loads((out/'manifest.json').read_text())
            self.assertEqual(data['expected_samples'],['global'])
            self.assertEqual(len(data['spatial_output']['cell_samples']),176176)
            self.assertEqual(data['mesh_statistics']['cells'],1761760)
            self.assertEqual(data['mesh_edges_m'],edges)
            self.assertEqual(data['spatial_output']['y_slice_index'],5)

    def test_manifest_scales_global_budgets_but_not_cell_gcl_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);case=ROOT/'planarALE'
            with_mesh_manifest(case/'transient.json',case/'mesh_scale1.dat',out/'small.json')
            small=json.loads((out/'small.json').read_text())
            self.assertAlmostEqual(small['conservation']['energy_balance_residual_J']['absolute_tolerance'],5e-5)
            self.assertEqual(small['conservation']['gcl_residual_m3_per_s']['absolute_tolerance'],2e-11)
            with_mesh_manifest(out/'small.json',case/'mesh.dat',out/'large.json')
            with_mesh_manifest(out/'large.json',case/'mesh.dat',out/'again.json')
            self.assertEqual(json.loads((out/'large.json').read_text()),json.loads((out/'again.json').read_text()))
            large=json.loads((out/'large.json').read_text())
            self.assertAlmostEqual(large['conservation']['energy_balance_residual_J']['absolute_tolerance'],.05)
            self.assertEqual(large['conservation']['gcl_residual_m3_per_s']['absolute_tolerance'],2e-11)

    def test_coordinate_checker_covers_all_y_planes_and_rejects_bad_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mesh=root/'mesh.dat'
            mesh.write_text('x 0 .5 1\ny 0 1 2\nz 0 .25 1\n')
            path=root/'constant/polyMesh/points';path.parent.mkdir(parents=True)
            points=[f'({x} {y} {z})' for z in [0,.25,1] for y in [0,1,2] for x in [0,.5,1]]
            path.write_text('27\n(\n'+'\n'.join(points)+'\n)\n')
            self.assertEqual(check_points(mesh,root)['points'],27)
            for bad in ['(nan 0 0)','(.1 0 0)',points[1]]:
                path.write_text('27\n(\n'+'\n'.join([bad,*points[1:]])+'\n)\n')
                with self.assertRaises(ValueError):check_points(mesh,root)

    def test_duplicate_or_nonincreasing_edges_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'mesh.dat'
            for text in ['x 0 1\nx 0 1\ny 0 1\nz 0 1\n','x 0 .5 .5 1\ny 0 1\nz 0 1\n','x 0 nan\ny 0 1\nz 0 1\n']:
                p.write_text(text)
                with self.assertRaises(ValueError):read_mesh(p)


if __name__=='__main__':unittest.main()
