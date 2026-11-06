"""Synthetic protocol tests; no physical accuracy is inferred."""
import json
from pathlib import Path
import tempfile
import unittest
import reference_field_gate as gate


class ReferenceFieldGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.controls=dict(schema_version=1,probe_steps=4,required_consecutive_steps=3,
                           scaled_update_tolerance=1e-7,burnin_steps=10,maximum_burnin_cycles=0)

    def field(self,time,name,values):
        folder=self.root/str(time)
        folder.mkdir(exist_ok=True)
        vector=name=='U'
        rows=['('+ ' '.join(map(str,v))+')' for v in values] if vector else list(map(str,values))
        (folder/name).write_text('internalField nonuniform List<'+('vector' if vector else 'scalar')+'>\n'+str(len(values))+'\n(\n'+'\n'.join(rows)+'\n);\n')

    def fixture(self,epsilon_cycle=False):
        for time in range(6):
            for name in gate.FIELDS:
                values=[(1.,0.,0.),(2.,0.,0.)] if name=='U' else [1.,2.]
                if epsilon_cycle and name=='epsilon':
                    values=[1.+.01*(time%2),2.]
                self.field(time,name,values)

    def test_epsilon_cycle_rejects_even_when_velocity_is_unchanged(self):
        self.fixture(True)
        result=gate.window_summary(self.root,1,0,self.controls)
        self.assertFalse(result['passed'])
        self.assertEqual(result['fields']['U']['maximum_final_window_change'],0)

    def test_valid_window_and_initial_transient(self):
        self.fixture()
        self.field(0,'U',[(0.,0.,0.),(0.,0.,0.)])
        self.assertTrue(gate.window_summary(self.root,1,0,self.controls)['passed'])
        self.field(3,'epsilon',[-1.,2.])
        with self.assertRaisesRegex(ValueError,'nonpositive'):
            gate.window_summary(self.root,1,0,self.controls)

    def test_raw_endpoint_and_stored_summary_are_recomputed(self):
        self.fixture()
        result=gate.window_summary(self.root,1,0,self.controls)
        folder=self.root/'reference-diagnostics'
        folder.mkdir()
        (folder/'final.log').write_text('Time = 5\nSIMPLE solution converged in 5 iterations\n')
        summary=dict(status='passed_numerical_field_stationarity',controls=self.controls,ranks=1,
                     windows=[result],accepted_window_index=0,final_time=5,commands=[])
        (folder/'summary.json').write_text(json.dumps(summary))
        gate.verify(self.root,1,self.controls)
        self.field(2,'epsilon',[1.1,2.])
        with self.assertRaisesRegex(ValueError,'does not reproduce'):
            gate.verify(self.root,1,self.controls)

    def test_controls_and_field_types_fail_closed(self):
        for updates in ({'probe_steps':True},{'scaled_update_tolerance':float('nan')},
                        {'required_consecutive_steps':5}):
            with self.subTest(updates=updates),self.assertRaises(ValueError):
                gate.validate_controls({**self.controls,**updates})
        self.fixture()
        (self.root/'0/U').write_text('internalField nonuniform List<scalar> 2 (1 2);')
        with self.assertRaisesRegex(ValueError,'field type'):
            gate.window_summary(self.root,1,0,self.controls)


if __name__=='__main__':
    unittest.main()
