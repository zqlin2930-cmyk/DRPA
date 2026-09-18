"""Materialize isolated, relocatable copies without editing the research snapshot."""
import ast,hashlib,json,os,shutil,sys,time
from pathlib import Path,PurePosixPath

PACKAGE=Path(__file__).resolve().parent
SOURCE=PACKAGE/'research'
CONVERGENCE='convergence_continuous_seed20260809'
REQUIRED_ASSETS=[
 'VoxTell_weights/voxtell_v1.1/plans.json',
 'VoxTell_weights/voxtell_v1.1/fold_0/checkpoint_final.pth',
 'VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz',
 'quality_audit/voxtell_mtl_drpa8_full_data/full_data_train_cases.csv',
 'quality_audit/voxtell_mtl_drpa8_full_data/full_data_val_cases.csv',
 'quality_audit/drpa_data_capacity_scaling/manifests/train_100pct.csv',
 'quality_audit/drpa_data_capacity_scaling/manifests/val_100pct_frozen.csv',
]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def safe_literal_path(path):
    value=str(Path(path).expanduser().resolve())
    if any(c in value for c in "'\"\\`$\n\r"):
        raise ValueError('Workspace/root paths cannot contain quotes, backslashes, shell substitutions or newlines')
    return value

def inventory():
    return json.loads((PACKAGE/'source_inventory.json').read_text())

def verify_sources():
    records=inventory()['files'];failures=[]
    for row in records:
        p=SOURCE/row['source_relative_path']
        if not p.is_file() or sha(p)!=row['distributed_sha256']:failures.append(row['source_relative_path'])
    if failures:raise RuntimeError('Source inventory mismatch: '+', '.join(failures))
    return {'verified_source_files':len(records),'python_source_files':sum(x['path'].endswith('.py') for x in records)}

def prepare(destination,assets=None):
    verify_sources();dest=Path(safe_literal_path(destination))
    if dest.exists():raise FileExistsError('Use a new empty workspace path; existing directories are never overwritten')
    if dest.is_relative_to(PACKAGE.parent):raise ValueError('Runtime workspaces must be outside the distribution directory')
    conf=json.loads(Path(assets).read_text()) if assets else {}
    links=conf.get('links',{});roots=conf.get('roots',{});resolved={}
    targets=[PurePosixPath(p) for p in links]
    if any(a!=b and a in b.parents for a in targets for b in targets):raise ValueError('Asset destinations cannot overlap')
    for relative,source in links.items():
        rel=PurePosixPath(relative)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:raise ValueError('Asset destination must be a safe relative path')
        if (SOURCE/relative).exists():raise ValueError('Asset mapping would replace distributed source/config: '+relative)
        if rel.suffix in {'.py','.sh','.toml','.yaml','.yml'}:raise ValueError('Executable assets are not accepted')
        path=Path(source).expanduser().resolve()
        if not path.exists():raise FileNotFoundError('Missing private asset: '+relative)
        resolved[relative]=str(path)
    replacements={
      '__DRPA_WORKSPACE__':str(dest), '__DRPA_PYTHON__':safe_literal_path(sys.executable),
      '__DRPA_DATA_ROOT__':safe_literal_path(roots.get('data_root',dest/'private_data')),
      '__DRPA_OASIS_ROOT__':safe_literal_path(roots.get('oasis_root',dest/'private_data/oasis')),
    }
    shutil.copytree(SOURCE,dest)
    materialized={}
    for p in dest.rglob('*'):
        if not p.is_file():continue
        text=p.read_text()
        for token,value in replacements.items():text=text.replace(token,value)
        p.write_text(text)
        if p.suffix=='.py':ast.parse(text,filename=str(p))
        if p.suffix=='.sh':p.chmod(0o755)
        materialized[str(p.relative_to(dest))]=sha(p)
    for relative,source in resolved.items():
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() or target.is_symlink():raise FileExistsError('Overlapping asset mappings: '+relative)
        target.symlink_to(source,target_is_directory=Path(source).is_dir())
    receipt={'format':'drpa-prepared-workspace-v1','created_unix':time.time(),'workspace':str(dest),
      'distribution_version':'0.1.0','source_hashes':materialized,'private_asset_links':resolved,'replacements':replacements,
      'contains_private_paths':True,'publication':'Do not upload this runtime workspace; upload the clean DRPA distribution.'}
    (dest/'DRPA_WORKSPACE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return {'workspace':str(dest),'source_files':len(materialized),'linked_assets':len(resolved),'missing_convergence_assets':missing_assets(dest)}

def missing_assets(workspace):return [p for p in REQUIRED_ASSETS if not (Path(workspace)/p).is_file()]

def verify_workspace(workspace):
    workspace=Path(workspace).expanduser().resolve()
    receipt=json.loads((workspace/'DRPA_WORKSPACE.json').read_text())
    if receipt['format']!='drpa-prepared-workspace-v1' or receipt['workspace']!=str(workspace):raise ValueError('Workspace receipt does not match its location')
    failures=[]
    for relative,digest in receipt['source_hashes'].items():
        # Fresh preflight deliberately captures the configured dependency sources here.
        if '/source_snapshot/' in relative:continue
        # Historical example configurations are replaced by each fresh run's exact config.
        if relative.startswith(CONVERGENCE+'/') and relative.endswith('/config.json'):continue
        p=workspace/relative
        if not p.is_file() or sha(p)!=digest:failures.append(relative)
    if failures:raise RuntimeError('Prepared source changed: '+', '.join(failures))
    return workspace

def environment(workspace):
    workspace=verify_workspace(workspace)
    env=dict(os.environ,DRPA_PREPARED_WORKSPACE=str(workspace),MTL_MODEL_ROOT=str(workspace),PYTHONDONTWRITEBYTECODE='1')
    paths=[str(workspace),str(workspace/'VoxTell'),str(PACKAGE.parent)]
    if env.get('PYTHONPATH'):paths.append(env['PYTHONPATH'])
    env['PYTHONPATH']=os.pathsep.join(paths)
    for key,value in [('OMP_NUM_THREADS','4'),('MKL_NUM_THREADS','4'),('OPENBLAS_NUM_THREADS','1')]:env.setdefault(key,value)
    return env
