"""Focused authentication tests: no solver or external data required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('native_diagnostics',Path(__file__).with_name('run_native_diagnostics.py'))
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)

class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'result').mkdir()
        data={'settings.json':dict(mesh_divisor=1,mesh_multiplier=2,modified_pressure=True,openfoam_outlet_stress=True),
              'diagnostics.json':dict(steady_reported=True,reported_unconverged_steps=[],steady_diagnostics='final_dt=2.5e-05'),
              'provenance.json':{},'binary.json':{},'command.json':{}}
        for name,value in data.items():(self.root/name).write_text(json.dumps(value))
        for rank in range(3):(self.root/f'result/simplefluid_cells_rank{rank}.csv').write_text('fixture\n')
        self.reindex()

    def tearDown(self):self.temp.cleanup()

    def reindex(self):
        native.save(self.root/'checksums.json',{str(p.relative_to(self.root)):native.digest(p)
            for p in self.root.rglob('*') if p.is_file() and p.name!='checksums.json'})

    def test_provenance_excludes_mutable_data_mapping(self):
        code=self.root/'libcode.so';data=self.root/'mpi-shared-memory'
        code.write_text('code fixture');data.write_text('data fixture')
        text=f"100-200 r-xp 0 00:00 1 {code}\n200-300 rw-p 0 00:00 2 {data}\n"
        self.assertEqual(native.executable_mapping_paths(text),{str(code)})

    def test_provenance_handles_path_spaces_and_anonymous_mapping(self):
        code=self.root/'code with spaces.so';code.write_text('fixture')
        text=f"100-200 r-xp 0 00:00 1 {code}\n200-300 rw-p 0 00:00 0 [heap]\n"
        self.assertEqual(native.executable_mapping_paths(text),{str(code)})

    def make_portable_subset(self):
        (self.root/'unneeded-output.vtu').write_text('duplicated output fixture')
        self.reindex()
        original=self.root/'original-checksums.json'
        original.write_bytes((self.root/'checksums.json').read_bytes())
        data=json.loads(original.read_text());del data['unneeded-output.vtu']
        (self.root/'unneeded-output.vtu').unlink()
        native.save(self.root/'checksums.json',data)
        native.save(self.root/'input_scope.json',dict(kind='native-replay-input-subset-v1',
            original_checksum_manifest_sha256=native.digest(original)))

    def test_portable_subset_authenticates_against_original_index(self):
        self.make_portable_subset()
        self.assertIsNotNone(native.authenticate(self.root)['original_index_sha256'])

    def test_portable_subset_rejects_altered_original_index(self):
        self.make_portable_subset()
        (self.root/'original-checksums.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Original checksum manifest changed'):native.authenticate(self.root)

    def test_portable_subset_rejects_rehashed_changed_input(self):
        self.make_portable_subset()
        raw=self.root/'result/simplefluid_cells_rank0.csv';raw.write_text('changed')
        data=json.loads((self.root/'checksums.json').read_text());data['result/simplefluid_cells_rank0.csv']=native.digest(raw)
        native.save(self.root/'checksums.json',data)
        with self.assertRaisesRegex(ValueError,'Subset is not authenticated'):native.authenticate(self.root)

    def test_authenticates_declared_endpoint(self):
        self.assertEqual(native.authenticate(self.root)['dt'],2.5e-5)

    def test_rejects_changed_raw_fields(self):
        (self.root/'result/simplefluid_cells_rank0.csv').write_text('modified\n')
        with self.assertRaisesRegex(ValueError,'Checksum mismatch'):native.authenticate(self.root)

    def test_rejects_unindexed_required_file(self):
        data=json.loads((self.root/'checksums.json').read_text());del data['settings.json']
        native.save(self.root/'checksums.json',data)
        with self.assertRaisesRegex(ValueError,'not authenticated'):native.authenticate(self.root)

    def test_rejects_unconverged_endpoint(self):
        data=json.loads((self.root/'diagnostics.json').read_text());data['steady_reported']=False
        native.save(self.root/'diagnostics.json',data);self.reindex()
        with self.assertRaisesRegex(ValueError,'historical solver checks'):native.authenticate(self.root)

    def test_rejects_path_escape(self):
        data=json.loads((self.root/'checksums.json').read_text());data['../outside']='0'*64
        native.save(self.root/'checksums.json',data)
        with self.assertRaisesRegex(ValueError,'escapes'):native.authenticate(self.root)

    def test_rejects_pressure_convention_change(self):
        data=json.loads((self.root/'settings.json').read_text());data['modified_pressure']=False
        native.save(self.root/'settings.json',data);self.reindex()
        with self.assertRaisesRegex(ValueError,'mismatch'):native.authenticate(self.root)

if __name__=='__main__':unittest.main()
