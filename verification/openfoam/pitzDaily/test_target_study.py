import copy
import importlib.util
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import target_study as t

SETTINGS=dict(transport_backend='bicgstab',transport_preconditioner='ilut',pressure_mode='cg-symmetric-muelu',steady_min_steps=20,steps=6000,steady_consecutive_steps=20,steady_tolerance=1e-4,linear_tolerance=1e-9,steady_min_dt_s=1e-7,steady_max_dt_s=1e-4)
LOG='''velocity_neumann: zeroGradient_stress
pressure_convention: modified
pressure_linear_solver: backend=cg preconditioner=MueLuSymmetric tolerance=1e-09
transport_linear_solver: backend=bicgstab preconditioner=ilut
steady_step=4162/6000 time=4.106218e-01 dt=1.000000e-04 next_dt=1.000000e-04 steady_window=20/20 steady=yes linear_converged=yes krylov_iterations=90 linear_tolerance=9e-10
pitzDaily: 16300 rank-zero owned cells, divisor=1, steps=4162, t=0.4106218, MPI ranks=3, mesh_multiplier=2
steady_state_search: reached=yes, rejected_steps=0, steps=4162, final_update_rate=9.42852e-5, final_max_Co=4, final_dt=0.0001, last_transport_linear_tolerance=1e-9
'''
class TargetAudit(unittest.TestCase):
 def test_converged_endpoint(self):
  self.assertEqual(t.endpoint(LOG,SETTINGS)['endpoint_values']['final_dt'],1e-4)
 def test_mismatched_solver(self):
  for a,b in [('backend=bicgstab','backend=gmres'),('MueLuSymmetric','MueLu'),('backend=cg','backend=gmres')]:
   with self.subTest(a=a),self.assertRaises(ValueError):t.endpoint(LOG.replace(a,b),SETTINGS)
 def test_nonconvergence(self):
  for a,b in [('reached=yes','reached=no'),('linear_converged=yes','linear_converged=no'),('steady_window=20/20','steady_window=19/20')]:
   with self.subTest(a=a),self.assertRaises(ValueError):t.endpoint(LOG.replace(a,b),SETTINGS)
 def test_invalid_endpoint(self):
  for a,b in [('final_update_rate=9.42852e-5','final_update_rate=nan'),('final_dt=0.0001','final_dt=0.0002'),('linear_tolerance=9e-10','linear_tolerance=2e-9'),('last_transport_linear_tolerance=1e-9','last_transport_linear_tolerance=1e-6')]:
   with self.subTest(a=a),self.assertRaises(ValueError):t.endpoint(LOG.replace(a,b),SETTINGS)
 def test_later_time_is_required(self):
  settings=dict(SETTINGS,steady_min_steps=6100,steps=8000)
  with self.assertRaises(ValueError):t.endpoint(LOG,settings)
 def test_wrong_grid_or_pressure(self):
  for a,b in [('mesh_multiplier=2','mesh_multiplier=1'),('pressure_convention: modified','pressure_convention: mechanical'),('zeroGradient_stress','traction_free')]:
   with self.subTest(a=a),self.assertRaises(ValueError):t.endpoint(LOG.replace(a,b),SETTINGS)
 def test_missing_hash_index_fails(self):
  import tempfile
  with tempfile.TemporaryDirectory() as directory,self.assertRaises(FileNotFoundError):t.verify(directory)
 def test_corrupted_authenticated_file_fails(self):
  import tempfile,json
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory);(p/'solver.log').write_text(LOG);(p/'checksums.json').write_text(json.dumps({'solver.log':'0'*64}))
   with self.assertRaisesRegex(ValueError,'changed evidence'):t.verify(p)
class MatchingStudies(unittest.TestCase):
 def setUp(self):
  self.base=dict(SETTINGS,dt_s=5e-6,mesh_multiplier=2,modified_pressure=True)
 def test_axes(self):
  t.matching_settings(self.base,dict(self.base,steps=8000,steady_min_steps=6100),'stationarity')
  t.matching_settings(self.base,dict(self.base,steps=12000,dt_s=2.5e-6,steady_max_dt_s=5e-5),'half')
 def test_rejects_unrelated_change(self):
  for key,value in [('mesh_multiplier',1),('modified_pressure',False),('transport_backend','gmres')]:
   with self.subTest(key=key),self.assertRaises(ValueError):
    t.matching_settings(self.base,dict(self.base,steps=8000,steady_min_steps=6100,**{key:value}),'stationarity')
 def test_rejects_early_stationarity(self):
  with self.assertRaises(ValueError):t.matching_settings(self.base,dict(self.base,steady_min_steps=5999),'stationarity')
 def test_rejects_false_refinement(self):
  with self.assertRaises(ValueError):t.matching_settings(self.base,dict(self.base,steps=12000),'half')
if __name__=='__main__':unittest.main()

