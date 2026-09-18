"""Aggregate only this experiment's CSVs; never read historical endpoints."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from policy import POLICY, decision, rows
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
        for step in sorted(int(s) for s in f.step if s>6000):
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
        final_step=int(done['final_step']); expected_steps=[s for s in STEPS if s<=final_step]
        assert final_step in (9000,10500,12000)
        assert json.loads((d/'effective_protocol_v3.json').read_text())['policy']==POLICY
        by_step={int(r['step']):r for r in rows(d/'val_curve.csv')}
        for endpoint in [s for s in (9000,10500,12000) if s<=final_step]:
            computed=decision(by_step[endpoint-1500],by_step[endpoint])
            saved=json.loads((d/'stopping_decisions'/f'step_{endpoint:05d}.json').read_text())
            assert all(saved[k]==v for k,v in computed.items())
            assert computed['stop']==(endpoint==final_step)
        assert done['stop_reason']==computed['reason'] and done['practical_plateau']==computed['practical_plateau']
        if final_step<12000:
            stop_audit=json.loads((d/'policy_stop_audit.json').read_text())
            assert stop_audit['status']=='POLICY_STOP_COMPLETE' and stop_audit['last_committed_step']==final_step
        assert t.global_step.tolist()==list(range(1,final_step+1)),f'{model}: missing/duplicate training updates'
        assert (t.samples_seen==t.global_step).all()
        assert v.step.tolist()==expected_steps and c.global_step.tolist()==expected_steps and continuity.global_step.tolist()==expected_steps
        assert set(t.model)=={model} and set(v.model)=={model} and set(t.seed)=={20260809}
        assert not raw.duplicated(['step','case_id','prompt']).any()
        assert not ptid.duplicated(['step','ptid']).any()
        val_cases=None
        for step in expected_steps:
            r=raw[raw.step==step];p=ptid[ptid.step==step]
            assert len(r)==1976 and r.ptid.nunique()==85 and len(p)==85
            cases=set(r.case_id);assert val_cases is None or cases==val_cases;val_cases=cases
            vr=v[v.step==step].iloc[0]
            for field,source in [('mean_dice','dice'),('hd95_mm','hd95_mm'),('surface_dice_2mm','surface_dice_2mm'),('fp_ml','false_positive_volume_ml'),('fn_ml','false_negative_volume_ml')]:
                assert np.isclose(vr[field],r[source].mean(),atol=1e-12,equal_nan=True)
        if not (d/'exact_resume_bridge.json').exists():
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
        assert continuity.optimizer_resets.sum()==0 and done['optimizer_reset_count']==0 and done['final_step']==final_step
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
         'validation_smoothing':'none; connect measured nodes only; stop at actual endpoint, never pad missing optional checkpoints',
         'error_bars':'none: one training seed; no seed SD or participant-as-seed uncertainty',
         'full_range':'Dice and Surface Dice0–100%; HD95 starts0 and includes all endpoints',
         'zoomed':'range determined by measured steps>=1500; full-range counterparts always retained','colors':COLORS},indent=2))
    (dest/'FIGURE_CAPTIONS.md').write_text('Training loss: raw per-update values (faint lines) and trailing100-update arithmetic means (solid lines; min_periods=1). Smoothing affects visualization only.\n\nValidation: Seven mandatory checkpoints (0 through9000), with10500/12000 conditional on a shared validation stopping rule, on the same247 visits/85 participants. Different stopping endpoints are shown without extrapolation. Mean Dice and Surface Dice are shown as percentages, HD95 in mm. Dashed line: primary matched6,000-update endpoint. Lines connect measured nodes only. One training seed20260809; no across-seed uncertainty estimate. Overall summaries follow the original visit-by-ROI macro aggregation; undefined HD95 counts are retained in val_curve.csv.\n')

def main(out):
    train,val,audits=check_inputs(out);summary=out/'summary';summary.mkdir(exist_ok=True)
    train.to_csv(summary/'train_curve_all_models.csv',index=False);val.to_csv(summary/'val_curve_all_models.csv',index=False)
    val.to_csv(summary/'convergence_endpoints.csv',index=False)
    changes(val).to_csv(summary/'convergence_changes_from_6000.csv',index=False)
    val[val.step.isin([6000,9000])].to_csv(summary/'common_budget_6000_9000.csv',index=False)
    stopping=[];all_decisions=[];fullpass=True
    audit_lines=['# CONVERGENCE AUDIT','', 'Prospectively amended before any stopping-decision endpoint. Original fixed12000 protocol retained for provenance; effective_protocol_v3.json supersedes stopping requirements.']
    report=['# Practical plateau sensitivity analysis','',
      '单一训练seed20260809。三模型必跑至9000，10500与12000为条件延长；采用同一验证集及同一停止规则。平台只表示相邻1500步的实用变化小于阈值，不代表统计等效或永久收敛。',
      '', '| 模型 | 停止步数 | 原因 | 确认平台 | Dice6000→停止点 (%) | ΔDice (pp) | HD95改善 (mm) | ΔSurface Dice (pp) |',
      '|---|---:|---|---|---:|---:|---:|---:|']
    for a in audits:
        model=a['model'];done=a['complete'];cfg=a['config'];end=done['final_step'];key=next(k for k,v in MODELS.items() if v==model)
        f=val[val.model==model].set_index('step');base=f.loc[6000];last=f.loc[end]
        ok=a['nonfinite_validation_values']==0;fullpass &=ok
        elapsed=train.loc[train.model==model,'update_wall_time_sec'].sum()
        stopping.append({'model':model,'stop_step':end,'stop_reason':done['stop_reason'],'practical_plateau':done['practical_plateau'],
          'plateau_observed_step':end if done['practical_plateau'] else None,'completed_endpoints':len(f),
          'optimizer_updates':end,'sum_update_wall_time_sec':elapsed,'operational_wall_time_sec':done['wall_clock_time_sec'],
          'mean_dice_at_stop':last.mean_dice,'hd95_mm_at_stop':last.hd95_mm,'surface_dice_at_stop':last.surface_dice_2mm,
          'plateau_not_confirmed':not done['practical_plateau']})
        report.append(f"|{model}|{end}|{done['stop_reason']}|{'是' if done['practical_plateau'] else '否'}|{base.mean_dice*100:.4f}→{last.mean_dice*100:.4f}|{(last.mean_dice-base.mean_dice)*100:+.4f}|{base.hd95_mm-last.hd95_mm:+.6f}|{(last.surface_dice_2mm-base.surface_dice_2mm)*100:+.4f}|")
        for p in sorted((out/key/'stopping_decisions').glob('step_*.json')):
            d=json.loads(p.read_text());all_decisions.append({'model':model,**{k:v for k,v in d.items() if k not in ['gains','previous_metrics','current_metrics']},**(d['gains'] or {})})
        audit_lines += ['',f'## {model}',f"- Seed{cfg['seed']}; train337 PTIDs/971 visits; validation85 PTIDs/247 visits; overlap0.",
          f"- Initialization: {cfg['initialization']}; SHA256 {cfg['initialization_sha256']}.",
          f"- Final checkpoint: {a['checkpoints'][-1]['path']}; SHA256 {a['checkpoints'][-1]['sha256']}.",
          f"- Measured endpoints: {f.index.tolist()}; all mandatory endpoints present; optional absent endpoints are not missing data errors.",
          f"- Final updates{end}; reason {done['stop_reason']}; plateau {done['practical_plateau']}; decisions independently recomputed from unrounded CSV values.",
          '- Optimizer counters continuous; resets0. Any earlier process restoration checked via exact-state bridge and next-update equality proof.',
          f"- Nonfinite metric cells {a['nonfinite_validation_values']}; undefined HD95 ROI count {a['undefined_hd95_roi_total']}.",
          '- Optional LCC/component/max-FP diagnostics intentionally absent after validation_v2; mandatory metrics unchanged.',
          f"- Audit acceptance: {'PASS' if ok else 'FAIL: undefined/nonfinite required metrics'}." ]
    pd.DataFrame(stopping).to_csv(summary/'stopping_summary.csv',index=False)
    pd.DataFrame(all_decisions).to_csv(summary/'stopping_decisions_all_models.csv',index=False)
    for step in (6000,9000):
        rank=val[val.step==step].sort_values('mean_dice',ascending=False).model.tolist()
        report += ['',f"共同{step}步Mean Dice排序：{' > '.join(rank)}。"]
    report += ['', '主表保留matched6000-update比较；共同9000步提供相同优化长度的补充比较。各自停止点用于描述达到规定平台所需的优化长度，不能将不同停止预算下的最终性能差异解释为同预算优势。',
      '停止原因须区分：practical_plateau、material_deterioration、metric_conflict、budget_cap_without_plateau，以及未解决指标问题。达到12000上限不等于已达到平台；未确认平台者不参与“谁最早达到平台”的排名。',
      'Dice与Surface Dice阈值分别0.10、0.05个百分点，HD95改善阈值0.05mm；相邻端点全部低于阈值且无同量级反向恶化才称平台。边界值按严格小于规则处理。阈值是本研究操作性定义，不是公认临床最小重要差异。',
      '10500步继续至12000要求至少一项达到实质改善阈值且无明显恶化。未定义指标不能被判为平台；延长至上限并记录审计失败。',
      '主要效率尺度为optimizer updates。DRPA早期使用旧版验证并经历过精确状态恢复，运行总时间包含不一致的工程开销，不可直接作三模型训练效率优势证据。summary/stopping_summary.csv另列训练更新耗时与运行总时间。',
      '相邻单个区间可受波动影响；本轮不提供跨seed稳健性或统计等效结论。使用同一验证集作停止判断会产生选择适应性；未使用locked test。',
      '协议是在训练已开始、停止判断节点结果尚未产生时修订，见plateau_v3/PROTOCOL_AMENDMENT.md及ADOPTION_REQUEST.json；不得写为实验开始前预注册。',
      '曲线只连接实际测量端点，不填补或外推未执行的10500/12000结果。']
    if not fullpass:report.append('必需指标存在未定义或非有限值，审计未完全通过；保留原值与计数。')
    (out/'CONVERGENCE_AUDIT.md').write_text('\n'.join(audit_lines)+'\n')
    (out/'CONVERGENCE_REPORT.md').write_text('\n'.join(report)+'\n')
    figures(train,val,out/'figures')
    (out/'REPORT_COMPLETE.json').write_text(json.dumps({'complete':True,'acceptance_pass':fullpass,'policy_version':POLICY['version'],'models':list(MODELS.values())},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args().out)
