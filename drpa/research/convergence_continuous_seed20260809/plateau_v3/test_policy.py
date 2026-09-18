"""Isolated policy, process-lifecycle and variable-endpoint report tests; no real run writes."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import csv,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from decimal import Decimal
from policy import POLICY,atomic_json,decision,sha,rows
import controller

def pair(step=9000,dice='0',surface='0',hd='0'):
    a={'step':step-1500,'mean_dice':'0.85','surface_dice_2mm':'0.97','hd95_mm':'1.5','hd95_undefined_roi_count':0}
    b={**a,'step':step,'mean_dice':str(Decimal(a['mean_dice'])+Decimal(dice)/100),
       'surface_dice_2mm':str(Decimal(a['surface_dice_2mm'])+Decimal(surface)/100),
       'hd95_mm':str(Decimal(a['hd95_mm'])-Decimal(hd))}
    return a,b

def csv_write(path, data):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def fixture(root,model='DRPA',end=9000):
    import torch
    labels={'DRPA':'DRPA-8','PDFT':'PD-FT','FullFT':'VoxTell-FullFT'};label=labels[model]
    out=root/model;out.mkdir();(out/'checkpoints').mkdir();vals=[];cps=[];cont=[];raw=[];ptids=[]
    for step in range(0,end+1,1500):
        # 9000 stops at plateau;10500 run improves9000 then plateaus;12000 keeps improving.
        value=Decimal('.85')
        if end>=10500 and step>=9000:value+=Decimal('.002')
        if end==12000 and step>=10500:value+=Decimal('.002')
        if end==12000 and step>=12000:value+=Decimal('.002')
        row={'model':label,'seed':20260809,'step':step,'mean_dice':str(value),'hd95_mm':1.5,'surface_dice_2mm':.97,
             'fp_ml':.2,'fn_ml':.3,'hd95_undefined_roi_count':0};vals.append(row)
        cp=out/'checkpoints'/f'step_{step:05d}.pt';cp.write_bytes(f'ISOLATED TEST CHECKPOINT {model} {step}'.encode())
        cps.append({'global_step':step,'path':str(cp),'sha256':sha(cp)})
        cont.append({'global_step':step,'optimizer_state_entries':int(step>0),'optimizer_step_min':step,'optimizer_step_max':step,
           'optimizer_state_finite':True,'optimizer_id':123,'model_id':456,'optimizer_resets':0,'scheduler_resets':0,'rng_restored_after_validation':True})
        for case in range(247):
            for prompt in range(8):
                raw.append({'step':step,'case_id':f'p{case%85}_v{case}','ptid':f'p{case%85}','prompt':str(prompt),
                   'dice':str(value),'hd95_mm':1.5,'surface_dice_2mm':.97,'false_positive_volume_ml':.2,'false_negative_volume_ml':.3})
        ptids.extend({'step':step,'ptid':f'p{i}'} for i in range(85))
    train=[{'model':label,'seed':20260809,'global_step':i,'samples_seen':i,'total_loss':.3,'dice_loss':.2,'bce_loss':.1,
            'gradient_norm':1,'wall_clock_time_sec':i,'update_wall_time_sec':1,'nan_inf':0} for i in range(1,end+1)]
    for name,data in [('train_curve_raw.csv',train),('val_curve.csv',vals),('checkpoint_manifest.csv',cps),
                      ('continuity.csv',cont),('val_roi_records.csv',raw),('val_ptid_records.csv',ptids)]:csv_write(out/name,data)
    torch.save({'global_step':end,'scheduler':None,'model_checkpoint':cps[-1]['path'],
                'optimizer':{'state':{0:{'step':torch.tensor(float(end))}}}},out/'latest_optimizer_rng.pt')
    atomic_json(out/'progress.json',{'status':'ENDPOINT_COMPLETE','step':end,'wall_clock_time_sec':end,'updated_unix':time.time()})
    atomic_json(out/'config.json',{'model':label,'seed':20260809,'initialization':'ISOLATED_TEST','initialization_sha256':'TEST'})
    atomic_json(out/'effective_protocol_v3.json',{'policy':POLICY})
    return out

class PolicyTests(unittest.TestCase):
    def test_flat_and_small_changes(self):
        for gains in [('0','0','0'),('.099','.049','.049'),('-.099','-.049','-.049')]:
            d=decision(*pair(dice=gains[0],surface=gains[1],hd=gains[2]));self.assertTrue(d['stop']);self.assertTrue(d['practical_plateau'])
    def test_strict_positive_boundaries(self):
        for k,v in [('dice','.10'),('surface','.05'),('hd','.05')]:
            d=decision(*pair(**{k:v}));self.assertFalse(d['stop']);self.assertFalse(d['practical_plateau'])
    def test_negative_boundaries_not_plateau(self):
        for k,v in [('dice','-.10'),('surface','-.05'),('hd','-.05')]:
            d=decision(*pair(**{k:v}));self.assertFalse(d['stop']);self.assertTrue(d['material_regressions'])
    def test_10500_meaningful_improvement(self):
        self.assertFalse(decision(*pair(10500,dice='.10'))['stop'])
    def test_10500_deterioration_and_conflict(self):
        self.assertEqual(decision(*pair(10500,dice='-.10'))['reason'],'material_deterioration')
        d=decision(*pair(10500,dice='.20',hd='-.06'));self.assertTrue(d['stop']);self.assertEqual(d['reason'],'metric_conflict')
    def test_cap_is_not_automatic_plateau(self):
        d=decision(*pair(12000,dice='.20'));self.assertTrue(d['stop']);self.assertFalse(d['practical_plateau'])
    def test_undefined_and_wrong_intervals(self):
        for step in [9000,10500,12000]:
            a,b=pair(step);b['mean_dice']='nan';d=decision(a,b)
            self.assertEqual(d['stop'],step==12000);self.assertFalse(d['practical_plateau'])
            a,b=pair(step);b['hd95_undefined_roi_count']=1;self.assertFalse(decision(a,b)['practical_plateau'])
        a,b=pair();a['step']=6000
        with self.assertRaises(AssertionError):decision(a,b)

class IntegrationTests(unittest.TestCase):
    def test_endpoint_rejects_missing_update(self):
        with tempfile.TemporaryDirectory(prefix='plateau_test_') as tmp:
            out=fixture(Path(tmp));self.assertEqual(controller.verify_endpoint(out,9000)['global_step'],9000)
            data=rows(out/'train_curve_raw.csv');del data[500];csv_write(out/'train_curve_raw.csv',data)
            with self.assertRaises(AssertionError):controller.verify_endpoint(out,9000)
    def test_real_process_stop_at_verified_endpoint(self):
        with tempfile.TemporaryDirectory(prefix='plateau_test_') as tmp:
            root=Path(tmp);out=fixture(root);atomic_json(root/'queue_revision_v3.json',{'start_unix':time.time()})
            proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
            try:
                done=controller.monitor(root,'DRPA',proc.pid,os.getpid(),proc)
                self.assertEqual(done['final_step'],9000);self.assertEqual(proc.returncode,-15)
                self.assertEqual(controller.last_training_step(out),9000)
                self.assertEqual(json.loads((out/'policy_stop_audit.json').read_text())['status'],'POLICY_STOP_COMPLETE')
            finally:
                if proc.poll() is None:proc.kill();proc.wait()
    def test_report_variable_endpoints_and_decisions(self):
        from policy import record_decision
        import report_v3
        with tempfile.TemporaryDirectory(prefix='plateau_test_') as tmp:
            root=Path(tmp)
            for model,end in [('DRPA',9000),('PDFT',10500),('FullFT',12000)]:
                out=fixture(root,model,end);values={int(r['step']):r for r in rows(out/'val_curve.csv')}
                for step in [s for s in [9000,10500,12000] if s<=end]:last=record_decision(out,values[step-1500],values[step])
                atomic_json(out/'complete.json',{'final_step':end,'optimizer_reset_count':0,'stop_reason':last['reason'],
                       'practical_plateau':last['practical_plateau'],'wall_clock_time_sec':end})
                if end<12000:atomic_json(out/'policy_stop_audit.json',{'status':'POLICY_STOP_COMPLETE','last_committed_step':end})
            train,val,audits=report_v3.check_inputs(root)
            self.assertEqual(len(train),31500);self.assertEqual(len(val),24)
            change=report_v3.changes(val)
            self.assertEqual(set(change[change.model=='DRPA-8'].step),{7500,9000})
            # Exercise final text/CSV generation without expensive scientific figure rendering.
            report_v3.figures=lambda *args:None
            report_v3.main(root)
            self.assertTrue((root/'summary/common_budget_6000_9000.csv').exists())
            self.assertIn('budget_cap_without_plateau',(root/'summary/stopping_summary.csv').read_text())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if result.wasSuccessful():
        root=Path(__file__).resolve().parent
        atomic_json(root/'TEST_RESULTS.json',{'status':'PASS','tests_run':result.testsRun,'unix':time.time(),
          'scope':'isolated synthetic fixtures; real signal lifecycle; no scientific outputs written',
          'tested_source_hashes':{p.name:sha(p) for p in root.glob('*.py')}})
    sys.exit(0 if result.wasSuccessful() else 1)
