"""Aggregate only this experiment's CSVs; never read historical endpoints."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STEPS=list(range(0,12001,1500))
MODELS={'DRPA':'DRPA-8','PDFT':'PD-FT','FullFT':'VoxTell-FullFT'}
COLORS={'DRPA-8':'#35759A','PD-FT':'#C58B3E','VoxTell-FullFT':'#86639A'}
METRICS=['mean_dice','hd95_mm','surface_dice_2mm']

def changes(val):
    rows=[]
    for model,f in val.groupby('model',sort=False):
        base=f.set_index('step').loc[6000]
        for step in [7500,9000,10500,12000]:
            target=f.set_index('step').loc[step]
            for metric in METRICS:
                b=float(base[metric]);t=float(target[metric]);delta=t-b
                rows.append({'model':model,'seed':20260809,'reference_step':6000,'step':step,'metric':metric,
                             'value_at_6000':b,'value_at_step':t,'absolute_change':delta,
                             'relative_change':delta/b if b!=0 else np.nan,
                             'relative_change_percent':100*delta/b if b!=0 else np.nan,
                             'absolute_change_unit':'mm' if metric=='hd95_mm' else 'fraction',
                             'absolute_change_percentage_points':100*delta if metric!='hd95_mm' else np.nan,
                             'relative_change_defined':b!=0})
    return pd.DataFrame(rows)

def check_inputs(out):
    trains=[];vals=[];audits=[]
    for key,model in MODELS.items():
        d=out/key;cfg=json.loads((d/'config.json').read_text());done=json.loads((d/'complete.json').read_text())
        t=pd.read_csv(d/'train_curve_raw.csv');v=pd.read_csv(d/'val_curve.csv');c=pd.read_csv(d/'checkpoint_manifest.csv')
        raw=pd.read_csv(d/'val_roi_records.csv');ptid=pd.read_csv(d/'val_ptid_records.csv');continuity=pd.read_csv(d/'continuity.csv')
        assert t.global_step.tolist()==list(range(1,12001)),f'{model}: missing/duplicate training updates'
        assert (t.samples_seen==t.global_step).all()
        assert v.step.tolist()==STEPS and c.global_step.tolist()==STEPS and continuity.global_step.tolist()==STEPS
        assert set(t.model)=={model} and set(v.model)=={model} and set(t.seed)=={20260809}
        assert not raw.duplicated(['step','case_id','prompt']).any()
        assert not ptid.duplicated(['step','ptid']).any()
        val_cases=None
        for step in STEPS:
            r=raw[raw.step==step];p=ptid[ptid.step==step]
            assert len(r)==1976 and r.ptid.nunique()==85 and len(p)==85
            cases=set(r.case_id);assert val_cases is None or cases==val_cases;val_cases=cases
            vr=v[v.step==step].iloc[0]
            for field,source in [('mean_dice','dice'),('hd95_mm','hd95_mm'),('surface_dice_2mm','surface_dice_2mm'),('fp_ml','false_positive_volume_ml'),('fn_ml','false_negative_volume_ml')]:
                assert np.isclose(vr[field],r[source].mean(),atol=1e-12,equal_nan=True)
        if continuity.optimizer_id.nunique()==1:
            assert continuity.model_id.nunique()==1
        else:
            bridge=json.loads((d/'exact_resume_bridge.json').read_text())
            assert bridge['status']=='EXACT_STATE_RESTORED' and bridge['optimizer_state_equal'] and bridge['model_parameters_equal'] and bridge['rng_equal']
            assert bridge['optimizer_resets']==0 and bridge['no_logged_steps_replayed']
            s=bridge['resume_step']
            assert (continuity.loc[continuity.global_step<=s,'optimizer_id']==bridge['old_optimizer_id']).all()
            assert (continuity.loc[continuity.global_step>s,'optimizer_id']==bridge['new_optimizer_id']).all()
            assert (continuity.loc[continuity.global_step<=s,'model_id']==bridge['old_model_id']).all()
            assert (continuity.loc[continuity.global_step>s,'model_id']==bridge['new_model_id']).all()
            gate=json.loads((out/'validation_v2/RESUME_EQUIVALENCE_GATE.json').read_text())
            assert gate['status']=='PASS' and gate['exact_next_update_equal']
            first=json.loads((d/'first_resumed_update_verification.json').read_text())
            assert first['status']=='PASS' and first['all_losses_weights_optimizer_rng_exact'] and first['step']==s+1
        assert (continuity.optimizer_step_min==continuity.global_step).all() and (continuity.optimizer_step_max==continuity.global_step).all()
        assert continuity.optimizer_resets.sum()==0 and done['optimizer_reset_count']==0 and done['final_step']==12000
        numeric=t[['total_loss','dice_loss','bce_loss','gradient_norm','wall_clock_time_sec']].to_numpy()
        assert np.isfinite(numeric).all() and (t.nan_inf==0).all()
        assert np.allclose(t.total_loss,t.dice_loss+t.bce_loss,atol=1e-7)
        for record in c.itertuples():
            digest=hashlib.sha256()
            with open(record.path,'rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
            assert digest.hexdigest()==record.sha256
        audits.append({'model':model,'config':cfg,'complete':done,'checkpoints':c.to_dict('records'),
                       'undefined_hd95_roi_total':int(raw.hd95_mm.isna().sum()),
                       'nonfinite_validation_values':int((~np.isfinite(raw[['dice','hd95_mm','surface_dice_2mm','false_positive_volume_ml','false_negative_volume_ml']].to_numpy())).sum())})
        trains.append(t);vals.append(v)
    return pd.concat(trains,ignore_index=True),pd.concat(vals,ignore_index=True),audits

def figures(train,val,dest):
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    def save(fig,name):
        for suffix in ('png','pdf','svg'):fig.savefig(dest/f'{name}.{suffix}',dpi=400,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,3.8),layout='constrained')
    for model,color in COLORS.items():
        f=train[train.model==model].sort_values('global_step')
        ax.plot(f.global_step,f.total_loss,color=color,alpha=.1,lw=.35)
        ax.plot(f.global_step,f.total_loss.rolling(100,min_periods=1).mean(),color=color,lw=1.3,label=model)
    ax.set(xlabel='Optimizer updates',ylabel='Total training loss',xlim=(0,12000),ylim=(0,None));ax.legend()
    save(fig,'fig_train_loss')
    for field,name,label,factor in [('mean_dice','dice','Mean Dice (%)',100),('hd95_mm','hd95','HD95 (mm)',1),('surface_dice_2mm','surface_dice','Surface Dice@2mm (%)',100)]:
        for zoom in (False,True):
            fig,ax=plt.subplots(figsize=(6.4,3.8),layout='constrained')
            for model,color in COLORS.items():
                f=val[val.model==model].sort_values('step')
                ax.plot(f.step,f[field]*factor,'o-',color=color,lw=1.3,ms=4,label=model)
            ax.axvline(6000,ls='--',color='.5',lw=.8)
            ax.text(.5,1.035,'Primary matched-budget endpoint',transform=ax.transAxes,ha='center',fontsize=8)
            ax.set(xlabel='Optimizer updates',ylabel=label,xlim=(0,12000),xticks=[0,3000,6000,9000,12000])
            if not zoom:
                ax.set_ylim(0,100 if factor==100 else max(1,val[field].max()*1.08))
            else:
                values=val.loc[val.step>=1500,field]*factor
                span=max(values.max()-values.min(),.2 if factor==100 else .02)
                ax.set_ylim(max(0,values.min()-span*.15),values.max()+span*.15)
                ax.text(.02,.03,'Zoomed scale; step-0 value may be outside view',transform=ax.transAxes,fontsize=7)
            ax.legend(loc='best');save(fig,'fig_val_'+name+('_zoomed' if zoom else ''))
    (dest/'FIGURE_METADATA.json').write_text(json.dumps({'archetype':'quantitative line charts','claim':'Measured single-seed training and validation trajectories, with no assumed outcome',
         'source':'summary CSVs from this fresh experiment only','dimensions_inches':[6.4,3.8],'formats':['PNG','PDF','SVG'],
         'training_smoothing':'trailing arithmetic rolling mean, window100 updates, min_periods1; raw shown faintly and retained in CSV',
         'validation_smoothing':'none; straight line segments only between actual prespecified endpoints',
         'error_bars':'none: one training seed; no seed SD or participant-as-seed uncertainty',
         'full_range':'Dice and Surface Dice0–100%; HD95 starts0 and includes all endpoints',
         'zoomed':'range determined by measured steps>=1500; full-range counterparts always retained','colors':COLORS},indent=2))
    (dest/'FIGURE_CAPTIONS.md').write_text('Training loss: raw per-update values (faint lines) and trailing100-update arithmetic means (solid lines; min_periods=1). Smoothing affects visualization only.\n\nValidation: nine measured checkpoints per model on the same247 visits/85 participants. Mean Dice and Surface Dice are shown as percentages, HD95 in mm. Dashed line: primary matched6,000-update endpoint. Lines connect measured nodes only. One training seed20260809; no across-seed uncertainty estimate. Overall summaries follow the original visit-by-ROI macro aggregation; undefined HD95 counts are retained in val_curve.csv.\n')

def main(out):
    train,val,audits=check_inputs(out);summary=out/'summary';summary.mkdir(exist_ok=True)
    train.to_csv(summary/'train_curve_all_models.csv',index=False);val.to_csv(summary/'val_curve_all_models.csv',index=False)
    val.to_csv(summary/'convergence_endpoints.csv',index=False);delta=changes(val);delta.to_csv(summary/'convergence_changes_from_6000.csv',index=False)
    lines=['# CONVERGENCE AUDIT','', 'Fresh single-seed experiment; acceptance evaluated from actual artifacts.']
    fullpass=True
    for a in audits:
        c=a['config'];d=a['complete'];cp=a['checkpoints'];ok=a['nonfinite_validation_values']==0;fullpass &=ok
        lines += ['',f"## {a['model']}",f"- Seed: {c['seed']}; train337 PTIDs/971 visits; validation85 PTIDs/247 visits; overlap0.",
            f"- Initialization: `{c['initialization']}`; SHA256 `{c['initialization_sha256']}`.",
            f"- Final: `{cp[-1]['path']}`; SHA256 `{cp[-1]['sha256']}`.",
            f"- Batch1; AdamW betas{c['betas']}, eps{c['eps']}; LR{c['learning_rates']}; weight decay{c['weight_decay']}; clip{c['gradient_clip']}.",
            f"- Precision: {c['precision']}; activation checkpointing: {c['activation_checkpointing']}.",
            '- Start0; final12000; optimizer continuous yes (matching AdamW counters; any process transition requires an exact model/optimizer/RNG restoration bridge and next-update equality gate); optimizer resets0; scheduler absent throughout (resets0).',
            f'- Completed checkpoints: {STEPS}; missing0; duplicate0.',
            f"- NaN/Inf training0; nonfinite validation metric cells{a['nonfinite_validation_values']}; undefined HD95 ROI records{a['undefined_hd95_roi_total']}.",
            f"- Wall-clock: {d['wall_clock_time_sec']:.1f} seconds.",f"- Acceptance: {'PASS' if ok else 'FAIL: undefined/nonfinite validation metrics; no imputation'}."]
    lines += ['', 'Validation computation revision: optimized raw LCC0 metrics use the original inference/geometry and equivalent shared distance transforms. Optional LCC/component/max-FP diagnostics are intentionally not computed after revision; blank optional columns do not represent numerical failures. See validation_v2/EQUIVALENCE_GATE.json and DEPLOYMENT_AUDIT.json.',f"Overall acceptance: {'PASS' if fullpass else 'FAIL; inspect undefined metric counts'}."]
    (out/'CONVERGENCE_AUDIT.md').write_text('\n'.join(lines)+'\n')
    report=['# Continuous convergence sensitivity analysis','', '仅一个训练seed（20260809）。以下结论仅依据本次新CSV；不代表跨seed稳定性。','',
            '| 模型 | Dice6000→12000 (%) | ΔDice (百分点) | HD95变化(mm) | Surface Dice变化(百分点) |',
            '|---|---:|---:|---:|---:|']
    gains={};plateaus={}
    for model in MODELS.values():
        f=val[val.model==model].set_index('step');a=f.loc[6000];b=f.loc[12000];gains[model]=float(b.mean_dice-a.mean_dice)
        report.append(f'|{model}|{a.mean_dice*100:.4f}→{b.mean_dice*100:.4f}|{gains[model]*100:+.4f}|{b.hd95_mm-a.hd95_mm:+.6f}|{(b.surface_dice_2mm-a.surface_dice_2mm)*100:+.4f}|')
        candidates=[s for s in STEPS if s<=9000 and f.loc[s:,'mean_dice'].max()-f.loc[s:,'mean_dice'].min()<=.001]
        plateaus[model]=min(candidates) if candidates else None
    report += ['', '来源：summary/val_curve_all_models.csv（step6000与12000）；绝对/相对变化详见summary/convergence_changes_from_6000.csv。',
               f"6000步后最大Dice变化：{max(gains,key=gains.get)}，{max(gains.values())*100:+.4f}个百分点。"]
    for model in MODELS.values():
        f=val[val.model==model].set_index('step');a=f.loc[6000];b=f.loc[12000]
        consistent=np.sign(b.mean_dice-a.mean_dice)==np.sign(a.hd95_mm-b.hd95_mm)==np.sign(b.surface_dice_2mm-a.surface_dice_2mm)
        report += [f"{model}：HD95 {a.hd95_mm:.6f}→{b.hd95_mm:.6f} mm（{'改善' if b.hd95_mm<a.hd95_mm else '恶化' if b.hd95_mm>a.hd95_mm else '不变'}）；Surface Dice {a.surface_dice_2mm*100:.4f}→{b.surface_dice_2mm*100:.4f}%。6000→12000的三指标改善方向{'一致' if consistent else '不完全一致'}。"]
    ranks={s:val[val.step==s].sort_values('mean_dice',ascending=False).model.tolist() for s in (6000,12000)}
    report += ['',f"6000步排序：{' > '.join(ranks[6000])}；12000步排序：{' > '.join(ranks[12000])}。排序{'保持' if ranks[6000]==ranks[12000] else '改变'}。",
       '', '描述性平台筛查：要求某节点起所有后续Mean Dice的范围≤0.1个百分点，且至少还有两个实测后续节点。该阈值仅用于描述，不是统计等效检验，也不决定训练或checkpoint选择。',
       f'各模型候选平台起点：{plateaus}（None表示截至12000步未满足）。']
    qualifying={m:s for m,s in plateaus.items() if s is not None}
    if qualifying:
        earliest=min(qualifying.values());report.append(f"按上述描述性规则，最早满足的是{', '.join(m for m,s in qualifying.items() if s==earliest)}，从{earliest}步起。")
    else:report.append('按上述规则无法确定任何模型已经进入平台。')
    report += ['', '主结果仍应表述为“matched 6,000-update comparison”。有限节点、单一seed与训练上限不能证明所有模型已完全收敛；新增结果属于single-seed convergence sensitivity analysis。',
               '本报告不提供跨seed mean±SD，不将247 visits或85 participants解释为独立训练重复。未读取locked test或历史endpoint。']
    if not fullpass:report.append('审计未完全通过：存在未定义验证指标；不将其置零，完整计数见CONVERGENCE_AUDIT.md。')
    (out/'CONVERGENCE_REPORT.md').write_text('\n'.join(report)+'\n')
    figures(train,val,out/'figures')
    (out/'REPORT_COMPLETE.json').write_text(json.dumps({'complete':True,'acceptance_pass':fullpass,'models':list(MODELS.values())},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args().out)
