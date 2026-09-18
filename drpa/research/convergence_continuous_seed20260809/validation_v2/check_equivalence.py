"""Real-prediction and edge-case equality gate; writes no formal result CSV."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path('__DRPA_WORKSPACE__/convergence_continuous_seed20260809')
sys.path.insert(0,str(ROOT/'code'))
import experiment as e
import fast_validation as fast

def assert_equal(a,b,columns):
    a=a.sort_values(['case_id','prompt']).reset_index(drop=True);b=b.sort_values(['case_id','prompt']).reset_index(drop=True)
    assert a[['case_id','prompt']].equals(b[['case_id','prompt']])
    differences={}
    for c in columns:
        x=a[c].to_numpy(dtype=float);y=b[c].to_numpy(dtype=float)
        assert np.array_equal(x,y,equal_nan=True),f'Metric mismatch: {c}'
        finite=np.isfinite(x)&np.isfinite(y);differences[c]=float(np.abs(x[finite]-y[finite]).max(initial=0))
    return differences

def main():
    dest=ROOT/'validation_v2';e.shared.seed_all();rng=np.random.default_rng(20260809)
    edge=0
    for spacing in [(1.,1.,1.),(.7,1.2,2.5)]:
        pairs=[]
        zero=np.zeros((15,19,23),dtype=bool);one=zero.copy();one[0,0,0]=True
        pairs.extend([(zero,zero),(zero,one),(one,zero),(one,one),(np.ones_like(zero),np.ones_like(zero))])
        for _ in range(8):pairs.append((rng.random(zero.shape)>.97,rng.random(zero.shape)>.97))
        for a,b in pairs:
            h,s=fast.boundary_metrics(a,b,spacing,e.shared.evaluator)
            reference=np.array([e.shared.evaluator.hd95(a,b,spacing),e.shared.evaluator.surface_dice(a,b,spacing)])
            assert np.array_equal(np.array([h,s]),reference,equal_nan=True);edge+=1
    checkpoint=ROOT/'DRPA/checkpoints/step_03000.pt'
    wrapper,_,_=e.make_model('DRPA');wrapper.load_checkpoint(str(checkpoint));wrapper.model.eval()
    cache=e.shared.TextEmbeddingCache(e.shared.BANK,e.shared.MODEL,e.shared.CACHE)
    all_records=e.shared.records(e.manifests('DRPA')[1]);indices=[0,123,246]
    records=[all_records[i] for i in indices]
    ds=e.shared.BilateralGroupedPatchDataset(records,e.shared.load_crop_spec(e.shared.PILOT/'crop_spec.json'),cache_cases=True)
    original=e.shared.evaluator.infer;predictions={}
    for i,rec in enumerate(records):
        predictions[rec.case_id]=original(wrapper,ds[i],cache)
        assert all(np.isfinite(a).all() for a in predictions[rec.case_id].values())
    e.shared.evaluator.infer=lambda wrapper,item,cache:predictions[item['case_id']]
    tick=time.perf_counter();reference=e.formal.evaluate_fullft('EQUALITY_ONLY',wrapper,records,ds,cache,e.shared.evaluator)
    original_seconds=time.perf_counter()-tick
    tick=time.perf_counter();serial=fast.evaluate('EQUALITY_ONLY',wrapper,records,ds,cache,e.shared.evaluator,workers=1)
    serial_seconds=time.perf_counter()-tick
    tick=time.perf_counter();parallel=fast.evaluate('EQUALITY_ONLY',wrapper,records,ds,cache,e.shared.evaluator,workers=4)
    parallel_seconds=time.perf_counter()-tick
    difference=assert_equal(reference[reference.lcc==0],serial,fast.REQUIRED)
    assert_equal(serial,parallel,fast.REQUIRED)
    reference.to_csv(dest/'equivalence_reference_only.csv',index=False);parallel.to_csv(dest/'equivalence_optimized_only.csv',index=False)
    e.shared.evaluator.infer=original
    bench_records=[all_records[i] for i in np.linspace(0,246,8,dtype=int)]
    bench_ds=e.shared.BilateralGroupedPatchDataset(bench_records,e.shared.load_crop_spec(e.shared.PILOT/'crop_spec.json'),cache_cases=True)
    tick=time.perf_counter();fast.evaluate('TIMING_ONLY',wrapper,bench_records,bench_ds,cache,e.shared.evaluator,workers=4)
    end_to_end=time.perf_counter()-tick
    report={'status':'PASS','version':fast.VERSION,'checkpoint':str(checkpoint),'checkpoint_sha256':e.sha(checkpoint),
            'real_visits':len(records),'real_roi_pairs':len(records)*8,'edge_cases':edge,'required_metrics':fast.REQUIRED,
            'maximum_absolute_difference':difference,'comparison':'exact array equality, matching NaN locations',
            'reference_full_evaluator_seconds_identical_predictions':original_seconds,
            'optimized_serial_seconds_identical_predictions':serial_seconds,
            'optimized_threads4_seconds_identical_predictions':parallel_seconds,
            'end_to_end_8visits_seconds_including_actual_inference':end_to_end,
            'projected_247visit_seconds_not_a_full_run_measurement':end_to_end/8*247,
            'frozen_evaluator_sha256':e.sha(Path(e.shared.evaluator.__file__)),
            'optimized_source_sha256':e.sha(Path(fast.__file__))}
    e.atomic_json(dest/'EQUIVALENCE_GATE.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
