import ast,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from drpa.workspace import PACKAGE,SOURCE,inventory,prepare,verify_sources,verify_workspace,environment

class PackagingTests(unittest.TestCase):
    def test_inventory_and_upstream_provenance(self):
        verified=verify_sources();self.assertEqual(verified['python_source_files'],128)
        for row in inventory()['files']:
            if row['source_relative_path'].startswith('VoxTell/'):
                self.assertEqual(row['source_sha256'],row['distributed_sha256'])
    def test_all_python_sources_parse(self):
        for p in PACKAGE.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_guard_blocks_direct_historical_execution(self):
        env=os.environ.copy();env.pop('DRPA_PREPARED_WORKSPACE',None)
        result=subprocess.run([sys.executable,str(SOURCE/'convergence_continuous_seed20260809/validation_v2/experiment_v2.py'),'--help'],env=env,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('prepared copies', (SOURCE/'convergence_continuous_seed20260809/validation_v2/experiment_v2.py').read_text())
        self.assertIn('Use python -m drpa',result.stderr)
    def test_relocated_workspace_and_source_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'relocated source'
            result=prepare(dest);self.assertTrue(result['missing_convergence_assets'])
            self.assertEqual(verify_workspace(dest),dest.resolve())
            code=(dest/'drpa8_full_data_train.py').read_text()
            self.assertIn(str(dest.resolve()),code);self.assertNotIn('__DRPA_WORKSPACE__',code)
            self.assertEqual(environment(dest)['DRPA_PREPARED_WORKSPACE'],str(dest.resolve()))
            with self.assertRaises(FileExistsError):prepare(dest)
            with (dest/'drpa8_full_data_train.py').open('a') as f:f.write('\n# deliberate test modification\n')
            with self.assertRaises(RuntimeError):verify_workspace(dest)
    def test_asset_links_are_private_and_cannot_replace_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);asset=root/'private.csv';asset.write_text('synthetic header\n')
            conf=root/'assets.json';conf.write_text(json.dumps({'links':{'private_inputs/train.csv':str(asset)}}))
            dest=root/'workspace';prepare(dest,conf)
            self.assertTrue((dest/'private_inputs/train.csv').is_symlink())
            self.assertTrue(json.loads((dest/'DRPA_WORKSPACE.json').read_text())['contains_private_paths'])
            conf.write_text(json.dumps({'links':{'drpa8_full_data_train.py':str(asset)}}))
            with self.assertRaises(ValueError):prepare(root/'bad',conf)
            conf.write_text(json.dumps({'links':{'../outside.csv':str(asset)}}))
            with self.assertRaises(ValueError):prepare(root/'traversal',conf)
            conf.write_text(json.dumps({'links':{'inputs':str(root),'inputs/nested.csv':str(asset)}}))
            with self.assertRaises(ValueError):prepare(root/'overlap',conf)
    def test_cli_does_not_launch_with_missing_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'workspace';prepare(dest)
            result=subprocess.run([sys.executable,'-m','drpa','convergence','--workspace',str(dest)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('assets missing',result.stderr)
            self.assertFalse((dest/'convergence_continuous_seed20260809/queue_state.json').exists())

if __name__=='__main__':unittest.main()
