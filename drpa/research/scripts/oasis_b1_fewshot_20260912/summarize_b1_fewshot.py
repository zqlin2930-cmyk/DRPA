"""Append B1 to immutable completed results; participant-level descriptive CIs."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import json
import numpy as np
import pandas as pd
from run_b1_fewshot import OUT,TARGET,FULLFT,SEED,guarded,sha,atomic_csv,atomic_json


def main():
    guarded()
    gate=json.loads((TARGET/'MODEL_COMPLETE.json').read_text())
    assert gate['steps']==500 and gate['query_subjects']==15 and gate['roi_rows']==120
    assert sha(TARGET/'QUERY_METRICS.csv')==gate['metrics_sha256']
    parent_gate=json.loads((FULLFT/'COMPLETE.json').read_text())
    previous=FULLFT/'OASIS_THREE_MODEL_FEWSHOT_QUERY_METRICS.csv'
    assert sha(previous)==parent_gate['outputs'][previous.name]
    previous_rows=pd.read_csv(previous); assert len(previous_rows)==720
    rows=pd.concat([previous_rows,pd.read_csv(OUT/'QUERY_BEFORE_FROM_FROZEN.csv'),pd.read_csv(TARGET/'QUERY_METRICS.csv')],ignore_index=True)
    ids=sorted(pd.read_csv(OUT/'query_manifest.csv').case_id)
    assert len(rows)==960 and len(ids)==15 and set(rows.case_id)==set(ids)
    assert set(rows.model)=={'B1','DRPA8','B3Canonical','FullFT'}
    assert not rows.duplicated(['model','phase','case_id','prompt']).any()
    assert rows.groupby(['model','phase']).size().eq(120).all()
    assert rows.groupby(['model','phase','case_id']).size().eq(8).all()
    assert rows.lcc.eq(0).all() and np.isfinite(rows.dice).all()
    scopes={'All8':list(rows.structure.unique()),'HippAmy':['hippocampus','amygdala'],
            'ECPHG':['entorhinal cortex','parahippocampal gyrus']}
    scopes.update({s:[s] for s in rows.structure.unique()})
    metrics=['dice','hd95_mm','surface_dice_2mm','false_positive_volume_ml','false_negative_volume_ml',
             'pred_volume_ml','gt_volume_ml','volume_error_ml_signed','volume_error_ml_absolute','connected_components']
    summary=[]; paired=[]
    indices=np.random.default_rng(SEED).integers(0,15,size=(10000,15))
    for scope,structures in scopes.items():
        part=rows[rows.structure.isin(structures)]
        for (model,phase),frame in part.groupby(['model','phase']):
            subjects=frame.groupby('case_id')[metrics].mean().reindex(ids)
            for metric in metrics:
                summary.append(dict(scope=scope,model=model,phase=phase,metric=metric,mean=subjects[metric].mean(),
                                    valid_subjects=int(subjects[metric].notna().sum()),valid_roi_rows=int(frame[metric].notna().sum()),expected_subjects=15))
        table=part.groupby(['case_id','model','phase']).dice.mean().unstack(['model','phase']).reindex(ids)
        after=table[('B1','after')].to_numpy()
        effects={'B1_after-before':after-table[('B1','before')].to_numpy()}
        effects.update({f'{m}_after-B1_after':table[(m,'after')].to_numpy()-after for m in ['DRPA8','B3Canonical','FullFT']})
        for comparison,delta in effects.items():
            assert np.isfinite(delta).all()
            sampled=delta[indices].mean(axis=1)
            paired.append(dict(scope=scope,comparison=comparison,n_subjects=15,mean_difference=delta.mean(),
                               ci95_low=np.quantile(sampled,.025),ci95_high=np.quantile(sampled,.975),
                               positive_subjects=int((delta>0).sum()),negative_subjects=int((delta<0).sum()),ties=int((delta==0).sum()),
                               bootstrap_draws=10000,seed=SEED,CI_type='pointwise_percentile_exploratory'))
    summary=pd.DataFrame(summary); paired=pd.DataFrame(paired)
    atomic_csv(OUT/'OASIS_FOUR_MODEL_FEWSHOT_QUERY_METRICS.csv',rows)
    atomic_csv(OUT/'OASIS_FOUR_MODEL_FEWSHOT_SUMMARY.csv',summary)
    atomic_csv(OUT/'OASIS_B1_FEWSHOT_PAIRED_BOOTSTRAP.csv',paired)
    training=pd.read_csv(TARGET/'training.csv')
    assert training.step.tolist()==list(range(1,501)) and training.finite.all()
    runtime=json.loads((TARGET/'RUNTIME_CONFIG.json').read_text())
    lines=['# OASIS B1 Matched Few-shot Addendum','', '## Material Passport','',
           '- Origin: experiment-agent run; user-authorized B1 comparator.',
           '- Verification: completed artifacts and input hashes verified; scientific findings descriptive, no independent replication.',
           '- Status: OASIS_B1_FEWSHOT_ADAPTATION_COMPLETE.',
           '', '## Protocol','',
           '- B1 canonical ADNI100@6000;294912unique cross-attention LoRA parameters,rank4/alpha8/dropout.05. Other parameters andQwenfrozen.',
           '- Exact same5support/15query,seed20260809 and500-updateorder as completedDRPA/B3/FullFT.',
           '- FreshAdamW1e-5,wd1e-5,betas(.9,.999),eps1e-8,no scheduler,clip1.0,batch1,500updates.',
           '- StrictFP32training,noautocast/GradScaler/TF32;8promptDice+BCE/8backward,oneupdate/case. Canonicalmodel.eval()/LoRAparametrizations.train() preserved.',
           '- NativeNIfTI→originalVoxTellreader→GT-assisted192³crop→case normalization;nnU-Netv1stage0NOTused.',
           '- OriginalgroupedFP16queryforward/FP32sigmoid,threshold>=.5,LCC0;finalstep500 only.',
           '- Before results and other3models reused;no reruns. Fourmodels×2phases×15subjects×8ROIs=960uniqueROIrows.',
           '', '## Participant-equal Mean Dice','',
           summary[(summary.metric=='dice') & summary.scope.isin(['All8','HippAmy','ECPHG'])].to_markdown(index=False),
           '', '## Participant-paired Dice changes (pointwise95%CI)','',paired.to_markdown(index=False),
           '', '## Runtime','',
           f"- Hardware:{runtime['gpu']};mean recorded update time:{training.sec.mean():.4f}s (checkpoint writes excluded).",
           f"- Peakallocated:{training.peak_allocated_GiB.max():.4f}GiB;reserved:{training.peak_reserved_GiB.max():.4f}GiB.",
           '', '## Limitations / statistical integrity','',
           '- B1 added after other outcomes;not retroactively preregistered. No result-based split/LR/checkpoint selection.',
           '- One split and one adaptation seed. Query15 previously exposed to frozen evaluation;not pristine independent external test.',
           '- GT-assisted crop remains;not MRI-only deployment. Annotation protocol and initial ADNI histories differ.',
           '- Fifteen participants are bootstrap units,not120correlatedROIs or960rows. Pointwise CIs,28descriptive comparisons,not multiplicity-adjusted simultaneous claims.',
           '- No cross-GPU compute-efficiency conclusion or claim that larger models necessarily perform better.',
           '- Undefined metric values remainNA;valid counts are reported.',
           '', '### Fallacy scan (11/11 considered)','',
           '- Simpson: overall and anatomical scopes shown separately; no universal ROI claim.',
           '- Ecological: unit is participant; no voxel-level or patient-outcome inference.',
           '- Berkson/selection: fixed20participantdataset and5/15split limit generalization.',
           '- Collider: no post-outcome selection, covariate control or excluded failure cases.',
           '- Base rate: Dice/volume metrics,not diagnostic PPV orclinical prevalence claims.',
           '- Regression to mean: no extreme-score subject selection; prior score exposure disclosed.',
           '- Survivorship: all15querysubjects/120B1ROIrows required;failure cannot be omitted.',
           '- Look elsewhere: all28fixed contrasts reported;pointwise exploratoryCIsnotconfirmatorytests.',
           '- Forking paths: added comparator disclosed;frozen inputs/order/LR/steps/noquerytuning.',
           '- Causality: no claim about clinical causal efficacy or model capacity as sole cause.',
           '- Reverse causality: no clinical causal interpretation;fixed before/afteralgorithmiccomparison.',
           '', 'Stop after this comparator;no additional model/seed/update authorized.','']
    report=OUT/'OASIS_B1_FEWSHOT_FINAL_REPORT.md'
    if report.exists(): raise RuntimeError('Report exists; do not overwrite')
    report.write_text('\n'.join(lines))
    atomic_json(OUT/'COMPLETE.json',dict(status='OASIS_B1_FEWSHOT_ADAPTATION_COMPLETE',new_models=1,steps=500,
                query_subjects=15,new_after_rows=120,combined_rows=960,missing=0,duplicate=0,
                outputs={p.name:sha(p) for p in OUT.glob('OASIS_*') if p.is_file()}))


if __name__=='__main__': main()
