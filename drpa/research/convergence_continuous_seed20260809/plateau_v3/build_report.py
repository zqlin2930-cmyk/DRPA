from pathlib import Path
import difflib
R=Path(__file__).resolve().parent
old=(R.parent/'validation_v2/report_v2.py').read_text()
prefix=old[:old.index('def main(out):')]
def repl(a,b):
 global prefix
 assert prefix.count(a)==1,a
 prefix=prefix.replace(a,b)
repl('import matplotlib\n','from policy import POLICY, decision, rows\nimport matplotlib\n')
repl('for step in [7500,9000,10500,12000]:','for step in sorted(int(s) for s in f.step if s>6000):')
repl("assert t.global_step.tolist()==list(range(1,12001)),f'{model}: missing/duplicate training updates'", "final_step=int(done['final_step']); expected_steps=[s for s in STEPS if s<=final_step]\n        assert final_step in (9000,10500,12000)\n        assert json.loads((d/'effective_protocol_v3.json').read_text())['policy']==POLICY\n        by_step={int(r['step']):r for r in rows(d/'val_curve.csv')}\n        for endpoint in [s for s in (9000,10500,12000) if s<=final_step]:\n            computed=decision(by_step[endpoint-1500],by_step[endpoint])\n            saved=json.loads((d/'stopping_decisions'/f'step_{endpoint:05d}.json').read_text())\n            assert all(saved[k]==v for k,v in computed.items())\n            assert computed['stop']==(endpoint==final_step)\n        assert done['stop_reason']==computed['reason'] and done['practical_plateau']==computed['practical_plateau']\n        if final_step<12000:\n            stop_audit=json.loads((d/'policy_stop_audit.json').read_text())\n            assert stop_audit['status']=='POLICY_STOP_COMPLETE' and stop_audit['last_committed_step']==final_step\n        assert t.global_step.tolist()==list(range(1,final_step+1)),f'{model}: missing/duplicate training updates'")
repl('assert v.step.tolist()==STEPS and c.global_step.tolist()==STEPS and continuity.global_step.tolist()==STEPS','assert v.step.tolist()==expected_steps and c.global_step.tolist()==expected_steps and continuity.global_step.tolist()==expected_steps')
repl('for step in STEPS:','for step in expected_steps:')
repl("if continuity.optimizer_id.nunique()==1:","if not (d/'exact_resume_bridge.json').exists():")
repl("done['final_step']==12000","done['final_step']==final_step")
repl("validation_smoothing':'none; straight line segments only between actual prespecified endpoints'","validation_smoothing':'none; connect measured nodes only; stop at actual endpoint, never pad missing optional checkpoints'")
repl('nine measured checkpoints per model on the same247 visits/85 participants.','Seven mandatory checkpoints (0 through9000), with10500/12000 conditional on a shared validation stopping rule, on the same247 visits/85 participants. Different stopping endpoints are shown without extrapolation.')
main='''def main(out):
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
    (out/'CONVERGENCE_AUDIT.md').write_text('\\n'.join(audit_lines)+'\\n')
    (out/'CONVERGENCE_REPORT.md').write_text('\\n'.join(report)+'\\n')
    figures(train,val,out/'figures')
    (out/'REPORT_COMPLETE.json').write_text(json.dumps({'complete':True,'acceptance_pass':fullpass,'policy_version':POLICY['version'],'models':list(MODELS.values())},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args().out)
'''
new=prefix+main
compile(new,'report_v3.py','exec')
(R/'report_v3.py').write_text(new)
(R/'report_changes.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='validation_v2/report_v2.py',tofile='plateau_v3/report_v3.py')))
