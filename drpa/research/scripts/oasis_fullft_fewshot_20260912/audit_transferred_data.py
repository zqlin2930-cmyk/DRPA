"""Read-only NIfTI/manifest audit; emits only new transfer receipts, no model."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from dependency_inventory import expected_paths, sha
from fullft_addendum_contract import prepare

BASE = Path('__DRPA_WORKSPACE__')
OUT = BASE / 'quality_audit/oasis_fullft_fewshot_addendum_20260912/preparation'


def main():
    contract = prepare(OUT / 'parent_completed')
    expected = expected_paths(json.loads((OUT / 'parent_preflight.json').read_text()))
    inventory = []
    for p, digest in expected.items():
        assert Path(p).is_file(), p
        observed = sha(p)
        assert observed == digest, p
        inventory.append(dict(path=p, bytes=Path(p).stat().st_size,
                              sha256=observed, status='PASS'))
    rows = []
    for split in ('support', 'query'):
        with (OUT / 'parent_completed' / f'{split}_manifest.csv').open() as f:
            manifest = list(csv.DictReader(f))
        for r in manifest:
            paths = [r[k] for k in ('image_path', 'source_label_path', 'label_path')]
            hashes = [r[k] for k in ('image_sha256', 'source_label_sha256', 'derived_label_sha256')]
            objects = [nib.load(p) for p in paths]
            arrays = [np.asanyarray(im.dataobj) for im in objects]
            for p, h, im, a in zip(paths, hashes, objects, arrays):
                assert sha(p) == h, p
                assert np.isfinite(a).all(), p
                assert im.shape == objects[0].shape, p
                assert np.array_equal(im.affine, objects[0].affine), p
                assert np.array_equal(im.header.get_zooms(), objects[0].header.get_zooms()), p
            for a in arrays[1:]:
                assert np.equal(a, np.rint(a)).all()
            ids = sorted(int(x) for x in np.unique(arrays[2]))
            assert ids == list(range(9)), (r['case_id'], ids)
            rows.append(dict(case_id=r['case_id'], split=split,
                             image_path=paths[0], source_label_path=paths[1], label_path=paths[2],
                             shape='x'.join(map(str, objects[0].shape)),
                             spacing_mm=','.join(map(str, objects[0].header.get_zooms())),
                             orientation=''.join(nib.aff2axcodes(objects[0].affine)),
                             geometry_exact=True, finite=True, sha256_pass=True,
                             mapped_label_ids=','.join(map(str, ids))))
    assert len(rows) == 20 and len({r['case_id'] for r in rows}) == 20
    result = dict(status='TRANSFERRED_DATA_INTEGRITY_PASS', subjects=20, support=5, query=15,
                  MRI=20, source_labels=20, mapped_8ROI_labels=20,
                  nifti_files=60, verified_dependencies=len(inventory),
                  nifti_bytes=sum(r['bytes'] for r in inventory if r['path'].endswith('.nii.gz')),
                  all_manifest_paths_readable=True, source_target_file_hashes_match=True,
                  model_forward_started=False, training_started=False,
                  interpretation='Existing source NIfTI files copied byte-identically; original VoxTell reader retained. No new preprocessing or label mapping.',
                  frozen_manifest_hashes=contract['parent_file_sha256'], subjects_audit=rows)
    for name, data in [('DATA_TRANSFER_GEOMETRY_AUDIT.csv', rows),
                       ('DATA_TRANSFER_FILE_INVENTORY.csv', inventory)]:
        with (OUT / name).open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    with (OUT / 'DATA_TRANSFER_INTEGRITY.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'subjects_audit'}))


if __name__ == '__main__':
    main()
