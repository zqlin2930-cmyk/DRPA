"""Fresh-run supervisor using the shipped canonical training and tested stopping policy."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from .workspace import CONVERGENCE,verify_workspace,environment

def main(workspace):
    base=verify_workspace(workspace);root=base/CONVERGENCE
    sys.path.insert(0,str(root/'plateau_v3'))
    from policy import POLICY,atomic_json,sha
    from controller import monitor
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for model in ['DRPA','PDFT','FullFT']:
        out=root/model
        if (out/'train_curve_raw.csv').exists() or (out/'complete.json').exists() or list((out/'checkpoints').glob('*.pt')):
            raise RuntimeError('Fresh runs only; existing trajectory found for '+model)
    if (root/'queue_revision_v3.json').exists():raise RuntimeError('Queue already started; no automatic restart')
    env=environment(base);env['MPLCONFIGDIR']=str(root/'matplotlib_cache')
    # The original preflight validates cohort identity, disjoint PTIDs, prompt
    # embeddings, source hashes, native preprocessing, memory and disk capacity.
    with (root/'PREFLIGHT.log').open('x') as log:
        result=subprocess.run([sys.executable,'-u',str(root/'code/experiment.py'),'--preflight'],cwd=base,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Preflight failed; see PREFLIGHT.log. No model training was launched.')
    paths=[*list((root/'code').glob('*.py')),*list((root/'validation_v2').glob('*.py')),*list((root/'plateau_v3').glob('*.py'))]
    hashes={str(p):sha(p) for p in paths}
    atomic_json(root/'queue_revision_v3.json',{'pid':os.getpid(),'start_unix':time.time(),'code_sha256':hashes,
       'policy':POLICY,'order':['DRPA','PDFT','FullFT'],'automatic_retry':False,'fresh_distribution_run':True})
    try:
        for model in ['DRPA','PDFT','FullFT']:
            assert all(sha(p)==h for p,h in hashes.items()),'Source changed during queue'
            out=root/model;out.mkdir(exist_ok=True)
            atomic_json(out/'effective_protocol_v3.json',{'policy':POLICY,'original_training_config':'config.json',
              'overrides':['required_steps','scientific_scope: validation-based practical-plateau stopping'],
              'mandatory_endpoints':POLICY['required_endpoints'],'maximum_updates':12000,
              'validation_implementation':'raw_shared_edt_threads4_v2','optimizer_changes':False,'unix':time.time()})
            with (root/f'{model}.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(root/'validation_v2/experiment_v2.py'),'--model',model],
                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=base)
                atomic_json(root/'queue_state.json',{'status':'RUNNING','stage':model,'child_pid':proc.pid,'supervisor_pid':os.getpid(),
                   'start_unix':time.time(),'validation_revision':2,'stopping_revision':3})
                done=monitor(root,model,proc.pid,os.getpid(),proc)
            atomic_json(root/f'{model}.exit.json',{'status':'COMPLETE','runner_returncode':proc.returncode,
               'authorized_policy_stop':done['final_step']<12000,'final_step':done['final_step'],'stop_reason':done['stop_reason']})
        atomic_json(root/'queue_state.json',{'status':'RUNNING','stage':'REPORT','supervisor_pid':os.getpid()})
        with (root/'REPORT.log').open('x') as log:
            result=subprocess.run([sys.executable,str(root/'plateau_v3/report_v3.py'),'--out',str(root)],cwd=base,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Report generation failed; see REPORT.log')
        atomic_json(root/'queue_state.json',{'status':'COMPLETE','end_unix':time.time(),'stopping_revision':3})
    except BaseException as exc:
        atomic_json(root/'DISTRIBUTION_QUEUE_FAILURE.json',{'error':repr(exc),'traceback':traceback.format_exc(),'unix':time.time()})
        atomic_json(root/'queue_state.json',{'status':'FAILED','automatic_retry':False,'error':repr(exc)})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True);args=parser.parse_args();main(args.workspace)
