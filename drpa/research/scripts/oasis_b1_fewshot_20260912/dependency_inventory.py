"""Readonly target dependency comparison; emit exact missing relative paths."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def expected_paths(receipt):
    keep = {}
    for p, h in receipt['guarded_hashes'].items():
        # No old checkpoints, STOP caches, dense features or results needed.
        if (p.endswith(('.py', '.nii.gz')) or '/embeddings/' in p or
                p.endswith('/crop_spec.json') or
                p.endswith('/preflight/external_subject_manifest_mapped.csv')):
            if '/scripts/oasis_fewshot_20260912/' not in p:
                keep[p] = h
    return keep


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    rows = []
    for p, expected in expected_paths(json.loads(args.receipt.read_text())).items():
        path = Path(p)
        observed = sha(path) if path.is_file() else None
        rows.append(dict(path=p, expected_sha256=expected, observed_sha256=observed,
                         status='MISSING' if observed is None else 'PASS' if observed == expected else 'CONFLICT'))
    print(json.dumps(rows, indent=2))
