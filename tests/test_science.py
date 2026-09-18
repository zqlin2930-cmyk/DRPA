"""Optional CPU-only checks of collected modules in a relocated workspace."""
import os,subprocess,sys,tempfile,unittest
from pathlib import Path
from drpa.workspace import prepare,environment

@unittest.skipUnless(os.environ.get('DRPA_RUN_SCIENCE_TESTS')=='1','Set DRPA_RUN_SCIENCE_TESTS=1 with research dependencies installed')
class ScienceTests(unittest.TestCase):
    def test_imports_adapter_gradients_metric_equivalence_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'workspace';prepare(root);env=environment(root)
            code=r'''
import sys,pathlib,numpy as np,torch
root=pathlib.Path(sys.argv[1]);sys.path[:0]=[str(root/'quality_audit'/p) for p in ['voxtell_mtl_drpa8_pilot','voxtell_mtl_peft','voxtell_mtl_b1_bilateral_crop','voxtell_mtl_peft_pilot','voxtell_mtl_b3_decoder_capacity_upper_bound']]+[str(root/'convergence_continuous_seed20260809/validation_v2')]
from drpa8_wrapper import LowRankWeightUpdate
import evaluate_drpa8 as old
import fast_validation as fast
adapter=LowRankWeightUpdate((12,16),4,8.0)
base=torch.randn(12,16);initial=adapter(base)
assert torch.equal(initial,base)
initial.square().sum().backward()
assert adapter.lora_B.grad is not None and torch.count_nonzero(adapter.lora_B.grad)>0
assert base.grad is None
rng=np.random.default_rng(42);zero=np.zeros((13,15,17),dtype=bool);one=zero.copy();one[0,0,0]=True
pairs=[(zero,zero),(zero,one),(one,zero),(one,one),(np.ones_like(zero),np.ones_like(zero))]
pairs += [(rng.random(zero.shape)>.96,rng.random(zero.shape)>.96) for _ in range(8)]
for spacing in [(1,1,1),(.7,1.2,2.5)]:
 for a,b in pairs:
  expected=np.array([old.hd95(a,b,spacing),old.surface_dice(a,b,spacing)])
  actual=np.array(fast.boundary_metrics(a,b,spacing,old))
  assert np.array_equal(expected,actual,equal_nan=True)
assert not torch.cuda.is_initialized(),'CPU checks must not initialize CUDA'
print('PASS: adapter initialization/gradient and26 distance-metric edge cases; CUDA unused')
'''
            result=subprocess.run([sys.executable,'-c',code,str(root)],env=env,cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            command=[sys.executable,'-m','drpa','run','--workspace',str(root),'convergence_continuous_seed20260809/validation_v2/experiment_v2.py','--','--help']
            result=subprocess.run(command,env=env,cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('--model',result.stdout)
    @unittest.skipUnless(sys.platform=='linux','Signal/process checks require Linux')
    def test_inherited_plateau_lifecycle_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'workspace';prepare(root)
            result=subprocess.run([sys.executable,'-m','drpa','run','--workspace',str(root),
                'convergence_continuous_seed20260809/plateau_v3/test_policy.py'],cwd=root,env=environment(root),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('Ran 10 tests',result.stderr)

if __name__=='__main__':unittest.main()
