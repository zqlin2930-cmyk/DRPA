"""Suspend only at a completed endpoint with no later committed optimizer update."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import json,os,signal,time
from pathlib import Path
ROOT=Path('__DRPA_WORKSPACE__/convergence_continuous_seed20260809')
DEST=ROOT/'validation_v2'
def js(p):return json.loads(p.read_text())
def write(name,obj):
 p=DEST/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));os.replace(t,p)
def latest_step(p):
 if not p.exists():return 0
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-4096));lines=f.read().decode().strip().splitlines()
 return int(lines[-1].split(',')[2])
def last_val(p):
 if not p.exists():return -1
 text=p.read_text()
 if not text.endswith('\n'):return -1
 lines=text.strip().splitlines()
 return int(lines[-1].split(',')[2]) if len(lines)>1 and len(lines[-1].split(','))==len(lines[0].split(',')) else -1
def main():
 q=js(ROOT/'queue_state.json');pid=q['child_pid'];sup=q['supervisor_pid'];model=q['stage'];d=ROOT/model
 assert model=='DRPA'
 assert b'experiment.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
 initial=js(d/'progress.json');target=((initial['step']+1499)//1500)*1500
 if initial['status']!='VALIDATING':target=(initial['step']//1500+1)*1500
 write('boundary_watch.json',{'status':'WAITING','pid':pid,'supervisor_pid':sup,'target_step':target,'initial':initial})
 start=time.time()
 while time.time()-start<7200:
  if not Path(f'/proc/{pid}').exists():raise RuntimeError('Original process exited before handoff')
  if last_val(d/'val_curve.csv')==target:
   os.kill(pid,signal.SIGSTOP)
   for _ in range(100):
    state=Path(f'/proc/{pid}/status').read_text()
    if '\nState:\tT' in state:break
    time.sleep(.01)
   else:raise RuntimeError('Cannot confirm process suspension')
   step=latest_step(d/'train_curve_raw.csv')
   if step==target:
    os.kill(sup,signal.SIGSTOP)
    write('TRANSITION_READY.json',{'status':'PAUSED_AT_COMPLETE_ENDPOINT','step':target,'pid':pid,'supervisor_pid':sup,
          'unix':time.time(),'last_committed_training_step':step,'completed_validation_step':last_val(d/'val_curve.csv'),
          'checkpoint':str(d/'checkpoints'/f'step_{target:05d}.pt'),'optimizer_rng':str(d/'latest_optimizer_rng.pt'),
          'policy':'No logged step rollback; possible in-flight uncommitted work is discarded upon exact continuation'})
    return
   os.kill(pid,signal.SIGCONT);target=(step//1500+1)*1500
   write('boundary_watch.json',{'status':'WAITING_NEXT_BOUNDARY','pid':pid,'target_step':target,'observed_step':step})
  time.sleep(.05)
 raise TimeoutError('Boundary watch expired; original job was left running')
if __name__=='__main__':main()
