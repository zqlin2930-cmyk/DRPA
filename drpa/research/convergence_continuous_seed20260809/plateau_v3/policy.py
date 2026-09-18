"""Prospective validation-only stopping rule, shared across all three models."""
import csv, hashlib, json, math, os, time
from decimal import Decimal
from pathlib import Path

POLICY = {
    'version': 'practical_plateau_v3',
    'minimum_updates': 9000, 'maximum_updates': 12000, 'interval': 1500,
    'required_endpoints': list(range(0, 9001, 1500)),
    'optional_endpoints': [10500, 12000],
    'gain_thresholds': {'dice_pp': 0.10, 'surface_dice_pp': 0.05, 'hd95_improvement_mm': 0.05},
    'regression_thresholds': {'dice_pp': 0.10, 'surface_dice_pp': 0.05, 'hd95_improvement_mm': 0.05},
    'comparison': 'adjacent endpoints; positive gains mean improvement; strict less-than plateau; regression at negative threshold inclusive',
    'at_9000': 'stop only if practical plateau; otherwise extend to10500',
    'at_10500': 'stop if plateau; extend only if at least one meaningful improvement and no material regression; stop deterioration/conflict separately',
    'undefined_metrics': 'never call plateau; extend to hard cap; retain undefined values and audit',
    'data': 'validation only; same247 visits and8 ROIs; no test access',
    'precision': 'use unrounded stored metrics; percentage points, not relative percentages',
    'claim_limit': 'operational practical plateau in one observed1500-update interval; no statistical equivalence or proof of permanent convergence',
}

def atomic_json(path, value):
    path=Path(path); temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); os.replace(temp,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def rows(path):
    path=Path(path)
    if not path.exists(): return []
    text=path.read_text()
    if not text.endswith('\n'): return []
    data=list(csv.DictReader(text.splitlines()))
    if any(None in row or None in row.values() for row in data): return []
    return data

def decision(previous, current):
    step=int(current['step']); before=int(previous['step'])
    assert step in (9000,10500,12000) and before==step-1500
    result={'policy_version':POLICY['version'],'step':step,'previous_step':before}
    fields=('mean_dice','surface_dice_2mm','hd95_mm')
    finite=all(math.isfinite(float(row[k])) for row in (previous,current) for k in fields)
    counts=all(int(float(row.get('hd95_undefined_roi_count',0)))==0 for row in (previous,current))
    if not finite or not counts:
        return {**result,'stop':step==12000,'reason':'unresolved_metrics_at_cap' if step==12000 else 'extend_unresolved_metrics',
                'practical_plateau':False,'gains':None,'material_regressions':[],'meaningful_improvements':[]}
    dec=lambda r,k:Decimal(str(r[k]))
    gains={'dice_pp':(dec(current,fields[0])-dec(previous,fields[0]))*100,
           'surface_dice_pp':(dec(current,fields[1])-dec(previous,fields[1]))*100,
           'hd95_improvement_mm':dec(previous,fields[2])-dec(current,fields[2])}
    limits={k:Decimal(str(v)) for k,v in POLICY['gain_thresholds'].items()}
    worse=[k for k,v in gains.items() if v<=-limits[k]]
    better=[k for k,v in gains.items() if v>=limits[k]]
    plateau=not worse and not better
    if plateau: stop,reason=True,'practical_plateau'
    elif step==12000: stop,reason=True,'budget_cap_without_plateau'
    elif step==9000: stop,reason=False,'extend_no_plateau'
    elif better and not worse: stop,reason=False,'extend_meaningful_improvement'
    else: stop,reason=True,'metric_conflict' if better and worse else 'material_deterioration'
    return {**result,'stop':stop,'reason':reason,'practical_plateau':plateau,
            'gains':{k:float(v) for k,v in gains.items()},'material_regressions':worse,'meaningful_improvements':better}

def record_decision(out, previous, current):
    result=decision(previous,current); path=Path(out)/'stopping_decisions'/f"step_{result['step']:05d}.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        saved=json.loads(path.read_text()); assert all(saved[k]==v for k,v in result.items()); return saved
    result.update(unix=time.time(),validation_csv_sha256_at_decision=sha(Path(out)/'val_curve.csv'),
                  previous_metrics={k:str(v) for k,v in previous.items()},current_metrics={k:str(v) for k,v in current.items()})
    atomic_json(path,result); return result
