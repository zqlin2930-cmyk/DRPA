"""Metric-equivalent LCC0 validation with shared distances and bounded CPU threads."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,time
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

VERSION='raw_shared_edt_threads4_v2'
REQUIRED=['dice','hd95_mm','surface_dice_2mm','false_positive_volume_ml','false_negative_volume_ml','pred_volume_ml','gt_volume_ml','empty_mask']

def boundary_metrics(a,b,spacing,evaluator):
    if not a.any() or not b.any():return float('nan'),0.0
    a,b=evaluator.crop_pair(a,b)
    sa,sb=evaluator.surface(a),evaluator.surface(b)
    da=ndimage.distance_transform_edt(~sb,sampling=spacing)[sa]
    db=ndimage.distance_transform_edt(~sa,sampling=spacing)[sb]
    hd=float(np.percentile(np.r_[da,db],95))
    sd=float((np.count_nonzero(da<=2.0)+np.count_nonzero(db<=2.0))/(len(da)+len(db)))
    return hd,sd

def case_metrics(name,record,item,probabilities,evaluator):
    start=time.perf_counter()
    _,label,affine,_=evaluator.load_official_reader_case(record)
    spacing=tuple(float(x) for x in nib.affines.voxel_sizes(affine))
    voxel_ml=abs(float(np.linalg.det(affine[:3,:3])))/1000.0
    rows=[]
    for prompt,patch_probability in probabilities.items():
        probability=evaluator.restore(patch_probability,item,label.shape,affine)
        mask=probability>=0.5
        reference=label==evaluator.ROI_LABEL_IDS[prompt]
        hd,sd=boundary_metrics(mask,reference,spacing,evaluator)
        rows.append({'condition':name,'lcc':0,'case_id':record.case_id,'ptid':record.case_id.split('_')[0],
                     'prompt':prompt,'structure':prompt.split(' ',1)[1],'side':'left' if prompt.startswith('left') else 'right',
                     'dice':evaluator.dice(mask,reference),'hd95_mm':hd,'surface_dice_2mm':sd,
                     'connected_components_raw':None,'connected_components':None,
                     'false_positive_volume_ml':float((mask & ~reference).sum()*voxel_ml),
                     'false_negative_volume_ml':float((~mask & reference).sum()*voxel_ml),
                     'max_false_positive_distance_mm':None,
                     'pred_volume_ml':float(mask.sum()*voxel_ml),'gt_volume_ml':float(reference.sum()*voxel_ml),
                     'empty_mask':int(not mask.any())})
    return rows,time.perf_counter()-start

def evaluate(name,wrapper,validation_records,validation_dataset,cache,evaluator,workers=4):
    start=time.perf_counter();rows=[];pending=deque();worker_seconds=[];gpu_seconds=0.
    def collect():
        record,future=pending.popleft();chunk,elapsed=future.result();rows.extend(chunk);worker_seconds.append(elapsed)
        print(f'validated {name} {record.case_id}',flush=True)
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='validation_metrics') as executor:
        for index,record in enumerate(validation_records):
            item=validation_dataset[index]
            tick=time.perf_counter();probabilities=evaluator.infer(wrapper,item,cache);gpu_seconds+=time.perf_counter()-tick
            for array in probabilities.values():
                if not np.isfinite(array).all():raise FloatingPointError('Nonfinite validation probability')
            meta={'patch_affine':item['patch_affine'],'preprocess_meta':item['preprocess_meta']}
            pending.append((record,executor.submit(case_metrics,name,record,meta,probabilities,evaluator)))
            if len(pending)>=workers:collect()
            del item,probabilities
        while pending:collect()
    frame=pd.DataFrame(rows)
    assert len(frame)==len(validation_records)*8 and not frame.duplicated(['case_id','prompt']).any()
    record={'condition':name,'version':VERSION,'visits':len(validation_records),'roi_rows':len(frame),'workers':workers,
            'wall_clock_sec':time.perf_counter()-start,'inference_and_h2d_sec':gpu_seconds,
            'sum_cpu_case_task_sec':sum(worker_seconds),'max_pending_cases':workers,
            'omitted_optional_diagnostics':['LCC1','connected_components_raw','connected_components','max_false_positive_distance_mm'],
            'required_metrics_unchanged':True,'unix':time.time()}
    print('VALIDATION_V2 '+json.dumps(record),flush=True)
    root=Path('__DRPA_WORKSPACE__/convergence_continuous_seed20260809/validation_v2')
    if root.exists():
        with (root/'validation_timings.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    return frame
