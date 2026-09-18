
# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('bench',Path(__file__).with_name('benchmark_x3.py'))
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class TestSummary(unittest.TestCase):
    def test_three_run_sd_and_artifact_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for rep in range(1,4):
              for model in b.MODELS:
                run=out/f'repeat{rep}_{model.replace("-","_")}';run.mkdir()
                metrics=['peak_allocated_gib','peak_reserved_gib','mean_sec_per_step','median_sec_per_step','samples_per_sec','end_to_end_sec_per_step',
                  'end_to_end_samples_per_sec','forward_sec','loss_sec','backward_sec','optimizer_sec','data_sec','h2d_sec','gpu_util_pct','median_gpu_util_pct','cpu_util_pct']
                r=dict(status='PASS',model=model,repeat=rep,configured_trainable_params=b.PARAMS[model],**{k:float(rep) for k in metrics})
                b.atomic(run/'result.json',r);b.atomic(out/f'{run.name}.exit.json',dict(exit_code=0))
                pd.DataFrame(dict(step=np.arange(1,121),phase=['warmup']*20+['measured']*100,loss=[1.]*120)).to_csv(run/'step_metrics.csv',index=False)
            b.summarize(out)
            s=pd.read_csv(out/'BENCHMARK_X3_SUMMARY.csv');r=s[s.metric=='mean_sec_per_step']
            self.assertTrue((r['mean']==2).all());self.assertTrue((r.sd==1).all());self.assertTrue((r.n_runs==3).all())
            self.assertEqual(len(pd.read_csv(out/'BENCHMARK_RAW.csv')),1440)
            b.atomic(out/'repeat1_B1.exit.json',dict(exit_code=2))
            with self.assertRaises(AssertionError):b.summarize(out)

if __name__=='__main__':unittest.main()
