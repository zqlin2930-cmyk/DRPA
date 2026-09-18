"""Readonly participant-paired addendum; writes only new report artifacts."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import json
import numpy as np
import pandas as pd
from run_fullft_fewshot import OUT,TARGET,PARENT,SEED,guarded,sha,atomic_csv,atomic_json


def main():
    guarded()
    gate=json.loads((TARGET/'MODEL_COMPLETE.json').read_text())
    assert gate['steps']==500 and gate['query_subjects']==15 and gate['roi_rows']==120
    assert sha(TARGET/'QUERY_METRICS.csv')==gate['metrics_sha256']
    parent_gate=json.loads((PARENT/'COMPLETE.json').read_text())
    assert parent_gate['status']=='OASIS_FEWSHOT_ADAPTATION_COMPLETE'
    old_path=PARENT/'OASIS_FEWSHOT_QUERY_METRICS.csv'
    assert sha(old_path)==parent_gate['output_hashes'][old_path.name]
    old=pd.read_csv(old_path); assert len(old)==480
    frames=[old,pd.read_csv(OUT/'QUERY_BEFORE_FROM_FROZEN.csv'),pd.read_csv(TARGET/'QUERY_METRICS.csv')]
    full=pd.concat(frames,ignore_index=True)
    ids=sorted(pd.read_csv(PARENT/'query_manifest.csv').case_id)
    assert len(full)==720 and set(full.case_id)==set(ids) and len(ids)==15
    assert not full.duplicated(['model','phase','case_id','prompt']).any()
    assert full.groupby(['model','phase']).size().eq(120).all()
    assert np.isfinite(full.dice).all() and full.lcc.eq(0).all()
    scopes={'All8':list(full.structure.unique()),'HippAmy':['hippocampus','amygdala'],
            'ECPHG':['entorhinal cortex','parahippocampal gyrus']}
    scopes.update({s:[s] for s in full.structure.unique()})
    metrics=['dice','hd95_mm','surface_dice_2mm','false_positive_volume_ml','false_negative_volume_ml',
             'pred_volume_ml','gt_volume_ml','volume_error_ml_signed','volume_error_ml_absolute','connected_components']
    summary=[]; paired=[]
    idx=np.random.default_rng(SEED).integers(0,15,size=(10000,15))
    for scope,structures in scopes.items():
        q=full[full.structure.isin(structures)]
        for (model,phase),part in q.groupby(['model','phase']):
            subject=part.groupby('case_id')[metrics].mean().reindex(ids)
            for metric in metrics:
                summary.append(dict(scope=scope,model=model,phase=phase,metric=metric,mean=subject[metric].mean(),
                                    valid_subjects=int(subject[metric].notna().sum()),
                                    valid_roi_rows=int(part[metric].notna().sum()),expected_subjects=15))
        table=q.groupby(['case_id','model','phase']).dice.mean().unstack(['model','phase']).reindex(ids)
        fa=table[('FullFT','after')].to_numpy(); fb=table[('FullFT','before')].to_numpy()
        effects={'FullFT_after-before':fa-fb,'FullFT_after-DRPA_after':fa-table[('DRPA8','after')].to_numpy(),
                 'FullFT_after-B3_after':fa-table[('B3Canonical','after')].to_numpy()}
        for comparison,delta in effects.items():
            assert np.isfinite(delta).all()
            samples=delta[idx].mean(axis=1)
            paired.append(dict(scope=scope,comparison=comparison,n_subjects=15,mean_difference=delta.mean(),
                               ci95_low=np.quantile(samples,.025),ci95_high=np.quantile(samples,.975),
                               positive_subjects=int((delta>0).sum()),negative_subjects=int((delta<0).sum()),
                               ties=int((delta==0).sum()),bootstrap_draws=10000,seed=SEED))
    summary=pd.DataFrame(summary); paired=pd.DataFrame(paired)
    atomic_csv(OUT/'OASIS_THREE_MODEL_FEWSHOT_QUERY_METRICS.csv',full)
    atomic_csv(OUT/'OASIS_THREE_MODEL_FEWSHOT_SUMMARY.csv',summary)
    atomic_csv(OUT/'OASIS_FULLFT_FEWSHOT_PAIRED_BOOTSTRAP.csv',paired)
    runtime=json.loads((TARGET/'RUNTIME_CONFIG.json').read_text())
    training=pd.read_csv(TARGET/'training.csv')
    assert training.step.tolist()==list(range(1,501)) and training.finite.all()
    report=['# FullFT OASIS Five-subject Adaptation Addendum','', '## Material Passport','',
            '- Status: OASIS_FULLFT_FEWSHOT_ADAPTATION_COMPLETE.',
            '- New execution: FullFT only,500updates,15queryparticipants,120uniquequeryROIrows.',
            '- DRPA/B3 and all frozen-before results are reused with hash verification; none rerun.',
            '', '## Protocol','',
            '- FullFT initialized from canonical ADNI100@6000;440029541unique configured trainable parameters;Qwen frozen.',
            '- Same5support/15query,seed20260809 andstored500-updateorder as completedDRPA/B3.',
            '- FreshAdamW LR1e-5,wd1e-5,betas(.9,.999),eps1e-8,no scheduler,clip1.0,batch1.',
            '- StrictFP32 training;no autocast/GradScaler/TF32;8promptFP32Dice+BCE/8 backward,one update/case.',
            '- Native NIfTI→originalVoxTellreader→reference-assisted192³crop→case normalization. nnU-Netv1stage0 NOT used.',
            '- Original groupedFP16forward/FP32sigmoid query evaluation,threshold>=0.5,LCC0.',
            '- Hardware: '+runtime['gpu']+';different from DRPA/B3 4090D and historicalPRO6000 benchmark. No cross-GPU efficiency claims.',
            '', '## Participant-equal Mean Dice','',
            summary[(summary.metric=='dice') & summary.scope.isin(['All8','HippAmy','ECPHG'])].to_markdown(index=False),
            '', '## FullFT paired differences,95% CI','',paired.to_markdown(index=False),
            '', '## Runtime','',
            f"- Completed500updates;mean recorded update time={training.sec.mean():.4f}s (checkpoint writes excluded).",
            f"- Peak allocated={training.peak_allocated_GiB.max():.4f}GiB;reserved={training.peak_reserved_GiB.max():.4f}GiB.",
            '', '## Scope and limitations','',
            '- Comparator was added after DRPA/B3 outcomes;not retroactively preregistered. No query-driven LR/checkpoint/split tuning.',
            '- Fifteen participants are adaptation-held-out with previous frozen-evaluation exposure,not pristine independent external testing.',
            '- GT-assisted crop is retained;not MRI-only deployment. Annotation-protocol shift and different initialADNItraining histories remain.',
            '- Subject bootstrap includes15participants,not120independentROIs or720independentobservations. CIs are conditional on one support/query split and one training seed.',
            '- Undefined distance metrics remainNA with validn. No invented results or all-metric-superiority assumption.',
            '- All original checkpoints,rawdata,manifests and existing reports remain immutable. Stop;no extra models/seeds/updates authorized.', '']
    path=OUT/'OASIS_FULLFT_FEWSHOT_FINAL_REPORT.md'
    if path.exists(): raise RuntimeError('Report already exists; no silent overwrite')
    path.write_text('\n'.join(report))
    atomic_json(OUT/'COMPLETE.json',dict(status='OASIS_FULLFT_FEWSHOT_ADAPTATION_COMPLETE',new_models=1,steps=500,
                query_subjects=15,new_after_rows=120,combined_rows=720,missing=0,duplicate=0,
                outputs={p.name:sha(p) for p in OUT.glob('OASIS_*') if p.is_file()}))


if __name__=='__main__': main()
