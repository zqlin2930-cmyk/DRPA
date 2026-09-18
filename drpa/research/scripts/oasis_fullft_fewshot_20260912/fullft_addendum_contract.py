"""CPU-only contract preparation. Never imports torch or launches a process."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse
import csv
import hashlib
import json
from pathlib import Path

HASHES = {
    'support_manifest.csv': 'b2392c3b4051f47abd0e645d4c7d99853072134f091f13109806f90178f9b3d7',
    'query_manifest.csv': '29f972cb4f5e8d1d06807db8c01ec39e6ffcb0e041c3f4069db4b74646242ed7',
    'sample_order_500.csv': 'd2426d18338128813367f5b1cba1946a96aa64f80baabff1faaf37cb25cf4fc5',
    'FROZEN_CONFIG.json': '6f32ffff18290d6db1552b762fe06078b31ec6151fc1fa75d8498f0baf0f4041',
}
FULLFT_SHA256 = '7a4815ff9ecb68cf21fbb28bd355a7d84b6a52064061650ad7f8edc0d1710711'
SUPPORT = {f'oasis_trt20_{n:02d}' for n in [1, 6, 8, 16, 17]}


def validate_rows(support, query, order):
    s = [r['case_id'] for r in support]
    q = [r['case_id'] for r in query]
    if len(s) != 5 or set(s) != SUPPORT or len(q) != 15 or len(set(q)) != 15:
        raise ValueError('Participant count/identity mismatch')
    if set(s) & set(q):
        raise ValueError('Support/query overlap')
    if set(s + q) != {f'oasis_trt20_{n:02d}' for n in range(1, 21)}:
        raise ValueError('Not the original20 participants')
    if [int(r['step']) for r in order] != list(range(1, 501)):
        raise ValueError('Order must contain exactly500 consecutive updates')
    for index, row in enumerate(order):
        if row['case_id'] != s[int(row['support_index'])]:
            raise ValueError('Order case/index mismatch')
        if int(row['epoch']) != index // 5 + 1:
            raise ValueError('Order epoch mismatch')
    for i in range(0, 500, 5):
        if {r['case_id'] for r in order[i:i+5]} != set(s):
            raise ValueError('Epoch must visit each support participant once')


def prepare(source):
    source = Path(source)
    for name, expected in HASHES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source changed: {name}')
    def rows(name):
        with (source / name).open(newline='') as f:
            return list(csv.DictReader(f))
    support, query, order = [rows(n) for n in
                            ['support_manifest.csv', 'query_manifest.csv', 'sample_order_500.csv']]
    validate_rows(support, query, order)
    original = json.loads((source / 'FROZEN_CONFIG.json').read_text())
    if original['models'] != {'DRPA8': 10969696, 'B3Canonical': 81930624}:
        raise ValueError('Unexpected parent model inventory')
    matched = dict(original)
    matched['models'] = {'FullFT': 440029541}
    return {
        'status': 'CPU_CONTRACT_PASS__TARGET_PREFLIGHT_PENDING',
        'authorization': 'User approved matched FullFT addendum after DRPA/B3 completion',
        'target_gpu': 'RTX PRO 6000',
        'parent_file_sha256': HASHES,
        'fullft_adni100_step6000_sha256': FULLFT_SHA256,
        'protocol': matched,
        'support': [r['case_id'] for r in support],
        'query': [r['case_id'] for r in query],
        'do_not_rerun': ['DRPA8', 'B3Canonical', 'original_frozen_evaluations'],
        'new_run_count': 1,
        'retrospectively_preregistered': False,
        'no_claim_of_resource_or_model_runtime_pass': True,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'FULLFT_MATCHED_ADDENDUM.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'support': 5, 'query': 15, 'steps': 500,
                      'models_to_train': ['FullFT'], 'GPU_started': False}))
