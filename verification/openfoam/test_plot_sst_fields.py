#!/usr/bin/env python3
import csv
from pathlib import Path
import tempfile
import unittest
from compare_verification import ComparisonError
from plot_sst_fields import SST_FIELDS, load_sst
from plot_water_fields import generate

class SSTFiguresTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.manifest={'schema_version':1,'case':'bottomHeatedBubblyConvection','mode':'transient',
                       'expected_times_s':[0,1],'expected_samples':['global'],
                       'quantities':{'temperature_K':{'units':'K','absolute_tolerance':0,'relative_tolerance':0}},
                       'conservation':{},'spatial_output':{'layout':'xz','width_m':1,'cell_samples':['0','1']}}
        for solver in ['of','sf']:
            path=self.root/solver;path.mkdir()
            with (path/'fields.csv').open('w') as f:
                f.write('time_s,sample,x_lower_m,x_upper_m,z_lower_m,z_upper_m,temperature_K,density_kg_m3,alpha_g,ux_m_s,uy_m_s,uz_m_s\n')
                for t in [0,1]:
                    for i in [0,1]:f.write(f'{t},{i},0,1,{i/2},{(i+1)/2},300,997,0,0,0,0\n')
            self.rows=[f'{t},{i},1e-8,1,1e-8,0.25,{i/2}\n' for t in [0,1] for i in [0,1]]
            self.write_sst(path,self.rows)
    @staticmethod
    def write_sst(path,rows):
        (path/'turbulence.csv').write_text('time_s,sample,k_m2_s2,omega_1_s,nut_m2_s,wall_distance_m,wall_yplus\n'+''.join(rows))
    def test_scalar_gallery_exports_sst_values_and_masks_zero_yplus(self):
        a,b=load_sst(self.manifest,self.root/'of',self.root/'sf')
        stats=generate(self.manifest,a,b,self.root/'plots',['svg'],SST_FIELDS)
        self.assertEqual(stats['fields']['wall_yplus']['undefined_relative_samples'],2)
        with (self.root/'plots/matched_fields.csv').open() as f:
            row=next(csv.DictReader(f));self.assertEqual(float(row['openfoam_k_m2_s2']),1e-8)
        self.assertIn('Turbulent kinetic energy',(self.root/'plots/distribution.svg').read_text())
    def test_bad_sst_state_and_incomplete_coverage_fail(self):
        for rows in [self.rows[:-1],[self.rows[0].replace('1e-8','-1e-8',1),*self.rows[1:]],
                     [self.rows[0].replace(',0.25,',',0,'),*self.rows[1:]]]:
            with self.subTest(rows=rows):
                self.write_sst(self.root/'sf',rows)
                with self.assertRaises(ComparisonError):load_sst(self.manifest,self.root/'of',self.root/'sf')

if __name__=='__main__':unittest.main()
