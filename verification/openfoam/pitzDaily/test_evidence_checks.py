import csv
from pathlib import Path
import tempfile
import unittest
import evidence_checks as checks


class EvidenceQualityTests(unittest.TestCase):
    def test_full_physical_fields_and_invalid_turbulence(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cells.csv'
            def write(updates=None,fields=checks.FIELDS):
                row=dict(zip(checks.FIELDS,[.1,.01,0.,10.,.2,0.,-1.,.375,14.855,.00085]))
                row.update(updates or {})
                with path.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields)
                    writer.writeheader()
                    writer.writerow({key:row[key] for key in fields})
            write()
            self.assertEqual(checks.cell_quality([path],True)['minimum']['pressure'],-1.)
            for field,value in [('epsilon',0.),('k',-1.),('nu_t',-1e-9),('pressure',float('nan')),('uz',float('inf'))]:
                write({field:value})
                with self.subTest(field=field),self.assertRaises(ValueError):
                    checks.cell_quality([path],True)
            write(fields=('x','y','ux','uy'))
            self.assertEqual(checks.cell_quality([path])['cells'],1)
            with self.assertRaises(ValueError):
                checks.cell_quality([path],True)

    def test_exact_station_configuration_and_duplicate_dictionaries(self):
        stations={'x050':.05,'x100':.1,'x200':.2}
        text='profiles { type sets; setFormat raw; interpolationScheme cellPoint; fields (U); sets ('
        for name,x in stations.items():
            text+=f'{name} {{ type uniform; axis y; start ({x} -.0253 0); end ({x} .0253 0); nPoints 129; }}'
        text+='); }'
        self.assertTrue(checks.validate_profile_control(text,stations))
        for bad in [text.replace('0.05 -.0253','0.06 -.0253'),text.replace('129','128',1),
                    text.replace('cellPoint','cell'),text+' profiles {}',text.replace('fields (U)','fields (U p)')]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                checks.validate_profile_control(bad,stations)


if __name__=='__main__':
    unittest.main()
