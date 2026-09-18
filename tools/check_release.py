"""Check release integrity and accidental private/artifact inclusion without printing secrets."""
import ast,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from drpa.workspace import verify_sources

def check():
    verify_sources();findings=[];files=[]
    forbidden={'.pt','.pth','.ckpt','.npz','.npy','.pkl','.pickle','.dcm','.log','.pid','.zip'}
    patterns={'patient_identifier':r'(?<![A-Za-z0-9])\d{3}S\d{4}(?!\d)|OAS\d[_-]?\d{4,}',
      'credential':r'gh[pousr]_[A-Za-z0-9]{25,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}',
      'private_host':r'connect\.[a-z]+\.seetacloud\.com',
      'private_key':r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'}
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=str(p.relative_to(ROOT))
        if any(part in {'__pycache__','.git','.pytest_cache','build','dist'} or part.endswith('.egg-info') for part in p.relative_to(ROOT).parts):continue
        files.append(p)
        if p.is_symlink():findings.append({'file':rel,'kind':'symlink'})
        if p.suffix in forbidden or p.name.endswith(('.nii','.nii.gz')):findings.append({'file':rel,'kind':'binary_or_runtime_artifact'})
        if p.name in {'.env','assets.local.json','DRPA_WORKSPACE.json'}:findings.append({'file':rel,'kind':'private_configuration'})
        text=p.read_text()
        if p.suffix=='.py':ast.parse(text,filename=rel)
        if p.suffix=='.json':json.loads(text)
        for name,pattern in patterns.items():
            if re.search(pattern,text):findings.append({'file':rel,'kind':name})
        if p.stat().st_size>=50*1024**2:findings.append({'file':rel,'kind':'unexpected_large_file'})
    result={'status':'PASS' if not findings else 'FAIL','files_checked':len(files),'findings':findings,
            'scope':'source integrity, Python/JSON syntax, patient-ID patterns, token/key patterns, private hosts, unexpected runtime/binary files; not a universal secret detector'}
    print(json.dumps(result,indent=2));return 0 if not findings else 1

if __name__=='__main__':raise SystemExit(check())
