"""Safe entrypoints for inspecting and running the published research code."""
import argparse,json,subprocess,sys
from pathlib import Path
from .workspace import SOURCE,CONVERGENCE,prepare,verify_sources,verify_workspace,environment,missing_assets

def main(argv=None):
    p=argparse.ArgumentParser(description='DRPA: Decoder- and Rank-adaptive Parameter Adaptation')
    commands=p.add_subparsers(dest='command',required=True)
    commands.add_parser('inspect',help='Verify the public source inventory; no private inputs required')
    a=commands.add_parser('prepare',help='Create a separate configured workspace; never launch training')
    a.add_argument('--workspace',required=True);a.add_argument('--assets',help='Private asset mapping JSON; omit for a code-only workspace')
    a=commands.add_parser('doctor',help='Read-only workspace source and required-asset check')
    a.add_argument('--workspace',required=True)
    a=commands.add_parser('run',help='Execute one research Python/shell entrypoint in a prepared workspace')
    a.add_argument('--workspace',required=True);a.add_argument('entry');a.add_argument('arguments',nargs=argparse.REMAINDER)
    a=commands.add_parser('convergence',help='Start a fresh three-model FP32 queue with optimized validation and practical plateau stopping')
    a.add_argument('--workspace',required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='inspect':print(json.dumps(verify_sources(),indent=2));return 0
        if a.command=='prepare':print(json.dumps(prepare(a.workspace,a.assets),indent=2));return 0
        root=verify_workspace(a.workspace)
        if a.command=='doctor':
            missing=missing_assets(root);print(json.dumps({'source_integrity':'PASS','missing_convergence_assets':missing,'ready_for_preflight':not missing},indent=2));return 1 if missing else 0
        env=environment(root)
        if a.command=='run':
            entry=(root/a.entry).resolve()
            if not entry.is_relative_to(root) or entry.suffix not in {'.py','.sh'} or not entry.is_file():raise ValueError('Entry must be a Python/shell source inside the prepared workspace')
            args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
            command=[sys.executable if entry.suffix=='.py' else 'bash',str(entry),*args]
        else:
            if missing_assets(root):raise FileNotFoundError('Convergence assets missing; run doctor and configure the private mapping')
            if sys.platform!='linux':raise RuntimeError('Full training/controller requires Linux; inspection and preparation are platform independent')
            command=[sys.executable,'-m','drpa.convergence','--workspace',str(root)]
        return subprocess.run(command,cwd=str(root),env=env).returncode
    except (OSError,ValueError,RuntimeError) as exc:
        print('DRPA: '+str(exc),file=sys.stderr);return 2
