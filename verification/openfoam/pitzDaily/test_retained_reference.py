"""Synthetic reference-binding tests, independent of physical accuracy."""
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import compare_profiles as comparison
from test_compare_profiles import qualified_reference_definition

HERE=Path(__file__).resolve().parent


class RetainedReferenceTests(unittest.TestCase):
    def setup_manifest(self,root):
        shutil.copytree(HERE/'fixtures',root/'fixture')
        folder=root/'fixture'
        manifest=folder/'tolerances.json'
        value=json.loads(manifest.read_text())
        value['qualification']['scope']='physical_reference'
        value['reference_definition']=qualified_reference_definition()
        # Relocate the authenticated role files away from the live case tree.
        for station,old in value['reference_definition']['openfoam_profile_files'].items():
            source=folder/old
            target=folder/'approved'/source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source,target)
            relative=target.relative_to(folder).as_posix()
            value['reference_definition']['openfoam_profile_files'][station]=relative
            value['qualification']['source_files_sha256'][relative]=hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(value))
        return folder,manifest

    def test_exact_authenticated_role_files_are_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,manifest=self.setup_manifest(Path(temporary))
            with redirect_stdout(io.StringIO()):
                result=comparison.main(['--tolerances',str(manifest),'--required-scope','physical_reference',
                                        '--use-retained-reference','--simplefluid-glob',str(folder/'simplefluid_cells_rank*.csv')])
            self.assertEqual(result,0)
            # Corruption is rejected before any comparison, even with a live
            # diagnostic tree that still contains the original good profiles.
            path=folder/'approved/x050_U.xy'
            path.write_bytes(path.read_bytes()+b'\n')
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                result=comparison.main(['--tolerances',str(manifest),'--use-retained-reference',
                                        '--simplefluid-glob',str(folder/'simplefluid_cells_rank*.csv')])
            self.assertEqual(result,2)

    def test_reference_sources_cannot_be_ambiguous(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,manifest=self.setup_manifest(Path(temporary))
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as result:
                comparison.main(['--tolerances',str(manifest),'--use-retained-reference',
                                 '--openfoam-case',str(folder/'openfoam'),
                                 '--simplefluid-glob',str(folder/'simplefluid_cells_rank*.csv')])
            self.assertEqual(result.exception.code,2)

    def test_retained_mode_requires_qualified_physical_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,manifest=self.setup_manifest(Path(temporary))
            value=json.loads(manifest.read_text())
            value['qualification']['scope']='fixture_only'
            manifest.write_text(json.dumps(value))
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                result=comparison.main(['--tolerances',str(manifest),'--use-retained-reference',
                                        '--simplefluid-glob',str(folder/'simplefluid_cells_rank*.csv')])
            self.assertEqual(result,2)


if __name__=='__main__':
    unittest.main()
