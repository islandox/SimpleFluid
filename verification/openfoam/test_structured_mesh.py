#!/usr/bin/env python3
"""Boundary-layer coordinate/physical-source invariants for matched fixtures."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from structured_mesh import read_mesh,mesh_statistics,write_openfoam,with_mesh_manifest

ROOT=Path(__file__).resolve().parent


class StructuredMeshTests(unittest.TestCase):
    def test_convection_layers_preserve_geometry_source_and_smooth_spacing(self):
        for name,count,layers in [('mesh.dat',1008,6),('mesh_fine.dat',2080,8)]:
            with self.subTest(mesh=name):
                edges=read_mesh(ROOT/'bottomHeatedBubblyConvection'/name)
                stats=mesh_statistics(edges)
                self.assertEqual(stats['cells'],count)
                self.assertAlmostEqual(stats['volume_m3'],8e-7,18)
                for axis in ['x','z']:
                    values=edges[axis];d=[b-a for a,b in zip(values,values[1:])]
                    for a,b in zip(d[:layers],d[1:layers]): self.assertAlmostEqual(b/a,1.25,12)
                    self.assertLess(stats[axis]['max_adjacent_ratio'],1.5)
                    for a,b in zip(values,reversed(values)): self.assertAlmostEqual(a+b,.02,14)
                for plane in [.02/3,.04/3]:self.assertTrue(any(abs(x-plane)<1e-14 for x in edges['x']))
                self.assertTrue(any(abs(z-.0025)<1e-14 for z in edges['z']))
                source_volume=sum((b-a)*(d-c)*.002 for a,b in zip(edges['x'],edges['x'][1:])
                                  for c,d in zip(edges['z'],edges['z'][1:])
                                  if .02/3<(a+b)/2<.04/3 and (c+d)/2<.0025)
                self.assertAlmostEqual(source_volume*4e6,2/15,13)

    def test_axial_cases_have_mirrored_graded_layers(self):
        for name,count in [('dispersedBubbleFlow',44),('planarALE',12)]:
            edges=read_mesh(ROOT/name/'mesh.dat')
            self.assertEqual(mesh_statistics(edges)['cells'],count)
            self.assertEqual(len(edges['x']),2)
            self.assertLess(edges['z'][1],edges['z'][2]-edges['z'][1])
            for a,b in zip(edges['z'],reversed(edges['z'])):self.assertAlmostEqual(a+b,1,14)

    def test_openfoam_and_manifest_use_actual_fine_cell_count(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);case=ROOT/'bottomHeatedBubblyConvection'
            edges=write_openfoam(case/'mesh_fine.dat',out)
            self.assertEqual((out/'system/blockMeshDict').read_text().count('hex ('),2080)
            self.assertEqual(read_mesh(out/'mesh.dat'),edges)
            with_mesh_manifest(case/'transient.json',case/'mesh_fine.dat',out/'manifest.json')
            data=json.loads((out/'manifest.json').read_text())
            self.assertEqual(data['expected_samples'],['global'])
            self.assertEqual(len(data['spatial_output']['cell_samples']),2080)
            self.assertEqual(data['mesh_edges_m'],edges)

    def test_duplicate_or_nonincreasing_edges_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'mesh.dat'
            for text in ['x 0 1\nx 0 1\ny 0 1\nz 0 1\n','x 0 .5 .5 1\ny 0 1\nz 0 1\n','x 0 nan\ny 0 1\nz 0 1\n']:
                p.write_text(text)
                with self.assertRaises(ValueError):read_mesh(p)


if __name__=='__main__':unittest.main()
