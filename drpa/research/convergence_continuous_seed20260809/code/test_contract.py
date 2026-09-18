"""Small synthetic contract tests; no formal trajectory or result files written."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import unittest,random
import numpy as np
import pandas as pd
import torch
import experiment as e
import report as r

class Contract(unittest.TestCase):
    def test_loss_matches_original_fullft_and_gradient(self):
        z=torch.linspace(-4,4,192).reshape(1,1,4,6,8).requires_grad_()
        y=(torch.arange(192).reshape_as(z)%3==0).float()
        total,dice,bce=e.shared.loss_fp32(z,y);old=e.gate.canonical_loss(z,y)
        self.assertTrue(torch.equal(total,old));self.assertTrue(torch.equal(total,dice+bce))
        self.assertTrue(torch.equal(torch.autograd.grad(total,z,retain_graph=True)[0],torch.autograd.grad(old,z)[0]))
    def test_rng_roundtrip(self):
        state=e.rng_state();a=(random.random(),np.random.random(),torch.rand(5));e.restore_rng(state)
        b=(random.random(),np.random.random(),torch.rand(5));self.assertEqual(a[:2],b[:2]);self.assertTrue(torch.equal(a[2],b[2]))
    def test_order_and_full_epochs(self):
        f=pd.DataFrame({'case_id':[str(i) for i in range(971)]});rows=list(e.order_rows(f))
        self.assertEqual(len(rows),12000);self.assertEqual(rows[-1][0],12000)
        for ep in range(1,13):
            got=[x[3] for x in rows if x[1]==ep];self.assertEqual(got,np.random.default_rng(e.SEED+ep).permutation(971).tolist())
    def test_signed_changes_and_relative_units(self):
        data=[]
        for s in e.STEPS:
            data.append({'model':'test','step':s,'mean_dice':.8+s/120000,'hd95_mm':2-s/12000,'surface_dice_2mm':.9+s/240000})
        changes=r.changes(pd.DataFrame(data));d=changes[(changes.step==12000)&(changes.metric=='mean_dice')].iloc[0]
        self.assertAlmostEqual(d.absolute_change,.05);self.assertAlmostEqual(d.absolute_change_percentage_points,5)
        self.assertAlmostEqual(d.relative_change,.05/.85)
    def test_optimizer_counter_rejects_reset(self):
        p=torch.nn.Parameter(torch.tensor(1.));o=torch.optim.AdamW([p]);(p*p).backward();o.step()
        self.assertEqual(e.optimizer_audit(o,1)['optimizer_step_min'],1)
        with self.assertRaises(AssertionError):e.optimizer_audit(o,2)
    def test_ptid_aggregation_does_not_pool_participants_as_seeds(self):
        rows=[]
        for visit in range(247):
            subject=visit%85
            for prompt in e.shared.PROMPTS:
                rows.append({'case_id':str(visit),'ptid':str(subject),'prompt':prompt,'structure':prompt.split(' ',1)[1],
                             'dice':subject/100,'hd95_mm':1.,'surface_dice_2mm':.95,
                             'false_positive_volume_ml':.1,'false_negative_volume_ml':.2})
        raw=pd.DataFrame(rows);summary,ptid=e.aggregate_validation(raw,'DRPA',0)
        self.assertEqual(len(ptid),85);self.assertEqual(ptid.n_visits.sum(),247)
        self.assertAlmostEqual(summary['mean_dice'],raw.dice.mean())
        self.assertAlmostEqual(ptid.loc[ptid.ptid=='84','dice'].iloc[0],.84)

if __name__=='__main__':unittest.main()
