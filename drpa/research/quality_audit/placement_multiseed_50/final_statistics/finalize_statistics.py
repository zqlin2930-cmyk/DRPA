#!/usr/bin/env python3
"""Read-only Placement multi-seed statistics; no model loading or inference."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

SEEDS=[20260809,3407,2026]
VARIANTS={'b1':'B1','b1_projection':'B1 + Projection','b1_decoder':'B1 + Decoder','drpa':'DRPA'}
PARAMS={'b1':294912,'b1_projection':737536,'b1_decoder':10527072,'drpa':10969696}
EFFECTS={'Projection - B1':('b1_projection','b1'),'Decoder - B1':('b1_decoder','b1'),'DRPA - Decoder':('drpa','b1_decoder')}
METRICS={
 'mean_dice':('overall','dice'), 'hipp_dice':('hippocampus','dice'),
 'ec_dice':('entorhinal cortex','dice'), 'phg_dice':('parahippocampal gyrus','dice'),
 'amy_dice':('amygdala','dice'), 'hd95_mm':('overall','hd95_mm'),
 'surface_dice_2mm':('overall','surface_dice_2mm'), 'fp_volume_ml':('overall','false_positive_volume_ml')}
BOOTSTRAP_SEED=20260907
BOOTSTRAP_DRAWS=10000

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bootstrap(diff,indices):
    diff=np.asarray(diff,dtype=np.float64)
    assert diff.ndim==1 and len(diff)==indices.shape[1] and np.isfinite(diff).all()
    samples=diff[indices].mean(axis=1)
    return float(diff.mean()),*map(float,np.quantile(samples,[.025,.975]))

def self_test():
    idx=np.random.default_rng(BOOTSTRAP_SEED).integers(0,85,size=(10000,85))
    a=bootstrap(np.full(85,.125),idx)
    assert np.allclose(a,[.125,.125,.125])
    a=bootstrap(np.arange(85)/85,idx);b=bootstrap(-np.arange(85)/85,idx)
    assert np.allclose([a[0],a[1],a[2]],[-b[0],-b[2],-b[1]])
    # Establish that PTID aggregation does not weight long-follow-up subjects more.
    f=pd.DataFrame({'ptid':['A','A','A','B'],'effect':[1.,1.,1.,0.]})
    assert f.effect.mean()==.75 and f.groupby('ptid').effect.mean().mean()==.5

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--integrity',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    self_test();a.out.mkdir(parents=True,exist_ok=True)
    ck=pd.read_csv(a.integrity/'CHECKPOINT_INTEGRITY.csv').set_index('run')
    prov=pd.read_csv(a.integrity/'SOURCE_ARTIFACT_MANIFEST.csv')
    assert len(ck)==12 and ck.readable.all() and ck.checkpoint_config_match.all() and (ck.nonfinite_tensors==0).all()
    integrity=[];runs=[];summary=[];frames={};ptidframes={};orders={};configref=None;refkeys=None;configs={}
    for seed in SEEDS:
      for variant in VARIANTS:
        run=f'seed{seed}_{variant}';p=a.raw/run
        for file in ['validation_rows_step_06000.csv','validation_summary_step_06000.csv','run_summary.json','config.json','training_dynamics.csv','parameter_summary.csv']:
            source=prov[(prov.run==run)&(prov.file==file)];assert len(source)==1
            assert sha(p/file)==source.iloc[0].sha256,('source hash mismatch',run,file)
        cfg=json.loads((p/'config.json').read_text());configs[run]=cfg
        result=json.loads((p/'run_summary.json').read_text())
        for key,value in dict(seed=seed,condition=variant,max_optimizer_steps=6000,train_visits=461,train_ptids=169,val_visits=247,val_ptids=85,batch_size=1,
                amp_forward=True,autocast_dtype='float16',grad_scaler=True,loss='FP32 Dice+BCE',lcc_main=0,trainable_parameters=PARAMS[variant]).items():
            assert cfg[key]==value,(run,key,cfg.get(key))
        excluded={'condition','seed','parameter_groups','trainable_parameters'}
        shared={k:v for k,v in cfg.items() if k not in excluded}
        if configref is None:configref=shared
        assert shared==configref,('protocol mismatch',run)
        assert sum(cfg['parameter_groups'].values())==PARAMS[variant]
        assert result['status']=='COMPLETE' and result['steps']==6000
        cmeta=json.loads(ck.loc[run,'metadata']);assert cmeta['global_step']==6000 and cmeta['condition']==variant
        assert ck.loc[run,'model_state_parameters']==PARAMS[variant]
        assert not json.loads(ck.loc[run,'active_failure_artifacts'])
        dynamics=pd.read_csv(p/'training_dynamics.csv')
        assert len(dynamics)==6000 and np.array_equal(dynamics.step.to_numpy(),np.arange(1,6001))
        nums=dynamics[['loss','pre_clip_grad_norm','post_clip_grad_norm','scaler_before','scaler_after']].to_numpy(float)
        assert np.isfinite(nums).all()
        scaler_drops=int((dynamics.scaler_after<dynamics.scaler_before).sum())
        assert scaler_drops==0,('possible skipped AMP steps',run,scaler_drops)
        sequence=dynamics[['step','epoch','case_id']].reset_index(drop=True)
        if seed not in orders:orders[seed]=sequence
        assert sequence.equals(orders[seed]),('sample sequence mismatch',run)
        raw=pd.read_csv(p/'validation_rows_step_06000.csv');assert set(raw.lcc)<= {0,1} and 0 in set(raw.lcc),run
        assert set(raw.condition)=={f'PLACEMENT_MULTI_{variant}@step06000'}
        f=raw[raw.lcc==cfg['lcc_main']].copy();keys=['ptid','case_id','prompt'];dups=int(f.duplicated(keys).sum());assert dups==0
        assert len(f)==1976 and f.case_id.nunique()==247 and f.ptid.nunique()==85
        assert (f.groupby('case_id').size()==8).all() and (f.groupby('case_id').ptid.nunique()==1).all()
        assert (f.groupby(['case_id','structure']).size()==2).all()
        index=f.set_index(keys).sort_index().index
        if refkeys is None:refkeys=index
        assert index.equals(refkeys),('pair key mismatch',run)
        for col in ['dice','hd95_mm','surface_dice_2mm','false_positive_volume_ml']:
            assert np.isfinite(f[col]).all(),('nonfinite metric',run,col)
        assert f.dice.between(0,1).all() and f.surface_dice_2mm.between(0,1).all() and (f.false_positive_volume_ml>=0).all()
        frames[(seed,variant)]=f
        original=pd.read_csv(p/'validation_summary_step_06000.csv')
        original=original[original.lcc==cfg['lcc_main']].set_index('scope')
        rm=dict(seed=seed,variant=VARIANTS[variant],variant_id=variant,trainable_parameters=PARAMS[variant]);patient={}
        for metric,(scope,col) in METRICS.items():
            part=f if scope=='overall' else f[f.structure==scope]
            mean=float(part[col].mean());assert abs(mean-float(original.loc[scope,col]))<1e-10
            # Equal ROI/side weighting within visit, equal visit weighting within PTID.
            byvisit=part.groupby(['ptid','case_id'])[col].mean()
            pp=byvisit.groupby('ptid').mean().sort_index();assert len(pp)==85
            patient[metric]=pp;rm[metric]=mean
            summary.append(dict(row_type='per_seed_variant',variant=VARIANTS[variant],effect='',seed=seed,metric=metric,mean=mean,sd=np.nan,n_seeds=1,aggregation='visit_ROI_macro',positive_seeds=np.nan,negative_seeds=np.nan,tie_seeds=np.nan))
        ptidframes[(seed,variant)]=pd.DataFrame(patient)
        assert np.isfinite(ptidframes[(seed,variant)].to_numpy()).all()
        runs.append(rm)
        integrity.append(dict(run=run,steps=6000,training_rows=6000,validation_rows=1976,visits=247,ptids=85,duplicate=0,failed=0,finite_metrics=True,
            scaler_drop_events=scaler_drops,within_seed_order_match=True,checkpoint_readable=True,checkpoint_sha256=ck.loc[run,'sha256'],
            post_completion_shutdown_warning=bool(ck.loc[run,'post_completion_shutdown_warning']),status='PASS'))
    runframe=pd.DataFrame(runs);effects=[];paired=[];ptiddeltas=[]
    for variant in VARIANTS.values():
      for metric in METRICS:
        vals=runframe.loc[runframe.variant==variant,metric]
        summary.append(dict(row_type='three_seed_variant',variant=variant,effect='',seed='',metric=metric,mean=vals.mean(),sd=vals.std(ddof=1),n_seeds=3,aggregation='visit_ROI_macro_then_seed_mean',positive_seeds=np.nan,negative_seeds=np.nan,tie_seeds=np.nan))
    for name,(v,b) in EFFECTS.items():
      for metric in METRICS:
        vals=[]
        for seed in SEEDS:
            rv=runframe[(runframe.seed==seed)&(runframe.variant_id==v)].iloc[0]
            rb=runframe[(runframe.seed==seed)&(runframe.variant_id==b)].iloc[0]
            d=float(rv[metric]-rb[metric]);vals.append(d)
            summary.append(dict(row_type='per_seed_effect',variant='',effect=name,seed=seed,metric=metric,mean=d,sd=np.nan,n_seeds=1,aggregation='visit_ROI_macro_difference',positive_seeds=int(d>0),negative_seeds=int(d<0),tie_seeds=int(d==0)))
            effects.append(dict(effect=name,seed=seed,metric=metric,visit_macro_difference=d))
        summary.append(dict(row_type='three_seed_effect',variant='',effect=name,seed='',metric=metric,mean=np.mean(vals),sd=np.std(vals,ddof=1),n_seeds=3,aggregation='visit_ROI_macro_difference_then_seed_mean',positive_seeds=sum(d>0 for d in vals),negative_seeds=sum(d<0 for d in vals),tie_seeds=sum(d==0 for d in vals)))
      for seed in SEEDS:
        vf=ptidframes[(seed,v)];bf=ptidframes[(seed,b)];assert vf.index.equals(bf.index)
        delta=vf-bf;indices=np.random.default_rng(BOOTSTRAP_SEED).integers(0,85,size=(BOOTSTRAP_DRAWS,85))
        for metric in METRICS:
            d=delta[metric].to_numpy();mean,lo,hi=bootstrap(d,indices)
            paired.append(dict(seed=seed,effect=name,metric=metric,n_ptids=85,mean_difference=mean,ci95_low=lo,ci95_high=hi,
                bootstrap_draws=BOOTSTRAP_DRAWS,bootstrap_seed=BOOTSTRAP_SEED,ci_method='paired_PTID_percentile',
                aggregation='equal_PTID_mean_of_within_PTID_visit_ROI_means',positive_ptid_fraction=float((d>0).mean()),negative_ptid_fraction=float((d<0).mean()),tie_ptid_fraction=float((d==0).mean())))
        for ptid,row in delta.iterrows():ptiddeltas.append(dict(seed=seed,effect=name,ptid=ptid,**row.to_dict()))
    summary=pd.DataFrame(summary);paired=pd.DataFrame(paired);effects=pd.DataFrame(effects)
    core=effects[effects.metric=='mean_dice'].pivot(index='seed',columns='effect',values='visit_macro_difference')
    decoder_dominates=bool((core['Decoder - B1']>core['Projection - B1']).all() and (core['Decoder - B1']>0).all())
    complement=paired[(paired.effect=='DRPA - Decoder')&(paired.metric=='mean_dice')]
    reproducible=bool((core['DRPA - Decoder']>0).all() and (complement.ci95_low>0).all())
    letter='A' if decoder_dominates and reproducible else ('B' if decoder_dominates else 'C')
    conclusions={'A':'Decoder adaptation provides the dominant specialization gain, while projection adaptation provides a smaller but reproducible complementary improvement.',
        'B':'Decoder adaptation provides the dominant gain, while projection provides only marginal/unstable additional benefit.',
        'C':'Placement effects are not robust across seeds.'}
    summary.to_csv(a.out/'PLACEMENT_MULTI_SEED_SUMMARY.csv',index=False,float_format='%.15g')
    paired.to_csv(a.out/'PLACEMENT_MULTI_SEED_PAIRED_BOOTSTRAP.csv',index=False,float_format='%.15g')
    runframe.to_csv(a.out/'PER_SEED_METRICS.csv',index=False,float_format='%.15g')
    effects.to_csv(a.out/'PER_SEED_EFFECTS.csv',index=False,float_format='%.15g')
    pd.DataFrame(ptiddeltas).to_csv(a.out/'PTID_PAIRED_EFFECTS.csv',index=False,float_format='%.15g')
    pd.DataFrame(integrity).to_csv(a.out/'RUN_INTEGRITY_AUDIT.csv',index=False)
    patientrows=[]
    for (seed,v),f in ptidframes.items():
        for ptid,row in f.iterrows():patientrows.append(dict(seed=seed,variant=VARIANTS[v],ptid=ptid,**row.to_dict()))
    pd.DataFrame(patientrows).to_csv(a.out/'PTID_METRICS.csv',index=False,float_format='%.15g')
    def ms(variant,metric):
        row=summary[(summary.row_type=='three_seed_variant')&(summary.variant==variant)&(summary.metric==metric)].iloc[0]
        return f'{row["mean"]:.6f} ± {row.sd:.6f}'
    report=['# Placement multi-seed @50%：最终统计与冻结结论','',
        '**状态：PLACEMENT_MULTI_SEED_FROZEN**',f'','## 1. 最终结论',f'',f'**{letter}. {conclusions[letter]}**','',
        '结论限定于本次50%数据、三个固定seed、AMP-forward＋FP32 loss的自然模块贡献实验；不是参数匹配的placement superiority证明。','',
        '## 2. 完整性与协议','',
        '- 12/12 runs；每组6000条连续唯一训练step、6000/6000updates；247/247visits、85/85PTIDs、1976/1976 canonical LCC=0 ROI records。原始CSV同时保留LCC=1诊断分支，不计入本次主统计，也不算重复病例。',
        '- 全部run的样本/ROI配对键一致；duplicate=0，failed=0；验证指标和训练loss/梯度均finite；GradScaler下降事件=0。',
        '- 12个step6000 checkpoint均在CPU可读，model/optimizer/GradScaler浮点tensor均finite；checkpoint config与run config一致，参数数量核对通过。',
        '- 相同seed内四组6000步实际case/epoch序列完全相同；各组manifest hash及共享配置一致。',
        '- Training：169PTIDs/461visits；validation：85PTIDs/247visits；seeds=20260809,3407,2026；batch1；AdamW；AMP FP16 forward＋FP32 Dice+BCE＋GradScaler；LCC0。',
        '- B1=294,912；+Projection=737,536；+Decoder=10,527,072；DRPA=10,969,696。',
        f'- {sum(r["post_completion_shutdown_warning"] for r in integrity)}/12组存在**产物完成后的DataLoader关闭警告**，已单独保留。failed=0指最终有效run失败数为0，不代表全程没有运行警告。',
        '- seed20260809 Decoder曾在5993发生epoch-boundary停顿；已有诊断记载其自主恢复完成，无重启拼接。不能用该旧诊断的pending状态替代现有完整终点产物。','',
        '## 3. 三seed汇总（mean ± sample SD）','',
        '每个seed先按247visits×8ROI等权汇总；左右结构均权。然后对三个seed计算mean与SD(ddof=1)。SD描述这三个训练seed的离散，不是标准误或CI。FP单位mL，HD95单位mm。','',
        '| Variant | Mean Dice | Hipp | EC | PHG | Amy | HD95 mm | Surface Dice@2mm | FP mL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for variant in VARIANTS.values():report.append('| '+variant+' | '+' | '.join(ms(variant,m) for m in METRICS)+' |')
    report+=['','## 4. 核心Mean Dice effects（visit-macro口径）','','| Effect | seed20260809 | seed3407 | seed2026 | 3-seed mean ± SD | 正方向一致性 |','|---|---:|---:|---:|---:|---:|']
    for name in EFFECTS:
        row=summary[(summary.row_type=='three_seed_effect')&(summary.effect==name)&(summary.metric=='mean_dice')].iloc[0]
        ds=[float(core.loc[s,name]) for s in SEEDS]
        report.append('| '+name+' | '+' | '.join(f'{v:+.6f}' for v in ds)+f' | {row["mean"]:+.6f} ± {row.sd:.6f} | {int(row.positive_seeds)}/3 |')
    report+=['','其余ROI/边界/FP effects及各seed数据保存在SUMMARY和PER_SEED_EFFECTS.csv中。','',
        '## 5. 每个seed独立的PTID paired bootstrap','',
        f'每PTID内部先平均每visit的ROI指标，再对该PTID全部visit平均。对应variant−baseline按同一85PTID成对相减，重采样85PTIDs、有放回{BOOTSTRAP_DRAWS}次，percentile2.5%–97.5%。统计RNG={BOOTSTRAP_SEED}，与训练seed不同，沿用既有Placement分析脚本的统计种子和次数。',
        '**没有把3×85=255条当作独立患者；不跨seed池化。** 以下是PTID等权差值，因此可能不同于上一节visit等权差值。所有CI为逐项、未做多重比较校正的pointwise CI，条件于对应seed/checkpoint，不估计训练seed总体不确定性。','',
        '| Seed | Effect | PTID mean difference | 95% CI |','|---:|---|---:|---:|']
    for seed in SEEDS:
      for name in EFFECTS:
        row=paired[(paired.seed==seed)&(paired.effect==name)&(paired.metric=='mean_dice')].iloc[0]
        report.append(f'| {seed} | {name} | {row.mean_difference:+.6f} | [{row.ci95_low:+.6f}, {row.ci95_high:+.6f}] |')
    report+=['','完整paired CSV包含3effects×3seeds×8metrics=72个分别计算的区间，每项n_PTID=85。','',
        '## 6. 解释边界','',
        '- “dominant”指当前自然可训练范围下观察到的更大增益，不说明每个参数的因果效率，也不是parameter-matched比较。',
        '- DRPA−Decoder隔离在已有decoder adaptation基础上加入projection的增量；不能仅用Projection−B1证明组合中的互补收益。',
        '- 互补结论主要基于Dice；不是所有指标全面占优。DRPA相比Decoder的三seed平均FP为0.299607 vs 0.296439 mL，略高，且逐seed FP变化方向不一致；不得宣称稳定降低FP。',
        '- 三个seed不足以强推任意未来训练seed都获益；主要证据为本次3/3方向一致性及每seed内PTID配对区间。',
        '- 参考标签是MALPEM自动silver reference；验证cohort是既有内部模型选择/评估队列，不是完全独立外部人工GT测试。',
        '- 本轮只读重聚合与CPU参数完整性检查；未训练、未forward、未重新评估、未修改原始rows/checkpoint/config/manifest。','',
        '## 7. 产物与复现','',
        '- PLACEMENT_MULTI_SEED_SUMMARY.csv：逐seed与三seed汇总、全部8metrics及三个核心effects；row_type和aggregation明确区分口径。',
        '- PLACEMENT_MULTI_SEED_PAIRED_BOOTSTRAP.csv：每seed、每comparison、每metric的PTID配对均值/95%CI。',
        '- RUN_INTEGRITY_AUDIT.csv、CHECKPOINT_INTEGRITY.csv、SOURCE_ARTIFACT_MANIFEST.csv：完整性和SHA256溯源。',
        '- PER_SEED_METRICS.csv、PER_SEED_EFFECTS.csv、PTID_METRICS.csv、PTID_PAIRED_EFFECTS.csv：可复算的中间统计。',
        '- finalize_statistics.py：给定--raw（原始run目录）、--integrity（完整性CSV目录）、--out（新目录）即可复算。',
        f'- NumPy={np.__version__}; pandas={pd.__version__}; bootstrap={BOOTSTRAP_DRAWS}; RNG={BOOTSTRAP_SEED}; SD ddof=1。','']
    (a.out/'PLACEMENT_MULTI_SEED_FINAL_REPORT.md').write_text('\n'.join(report))
    freeze=dict(status='PLACEMENT_MULTI_SEED_FROZEN',conclusion=letter,conclusion_text=conclusions[letter],runs=12,seeds=SEEDS,
        validation_visits_per_run=247,ptids_per_run=85,duplicate=0,failed=0,bootstrap_draws=BOOTSTRAP_DRAWS,bootstrap_rng_seed=BOOTSTRAP_SEED,
        seed_pooling=False,descriptive_aggregation='visit_ROI_macro',paired_aggregation='PTID_macro',sample_sd_ddof=1)
    (a.out/'PLACEMENT_MULTI_SEED_FREEZE.json').write_text(json.dumps(freeze,indent=2))
    print(json.dumps(freeze,indent=2));print(runframe.to_string(index=False));print(paired[paired.metric=='mean_dice'].to_string(index=False))

if __name__=='__main__':main()
