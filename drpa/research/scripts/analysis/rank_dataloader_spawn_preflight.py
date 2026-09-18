#!/usr/bin/env python3
"""CPU-only equivalence and epoch-boundary test for the Rank DataLoader."""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler


BASE = Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__"))
AUDIT = BASE / "quality_audit/drpa_data_capacity_scaling"
PILOT = BASE / "quality_audit/voxtell_mtl_drpa8_pilot"
PREP = BASE / "quality_audit/voxtell_mtl_b1_bilateral_crop"
OUT = BASE / "quality_audit/rank_ablation_50/rank_spawn_preflight"
SEED = 20260809

sys.path[:0] = [str(PREP), str(BASE / "scripts/analysis")]
from b1_preprocessing import CaseRecord, load_crop_spec  # noqa: E402
from pro6000_pipeline_benchmark import (  # noqa: E402
    TimedReaderDataset,
    close_loader,
    collate,
    worker_init,
)


class MutableSampler(Sampler[int]):
    def __init__(self) -> None:
        self.indices: list[int] = []

    def set_indices(self, indices) -> None:
        self.indices = [int(index) for index in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def records(path: Path) -> list[CaseRecord]:
    frame = pd.read_csv(path, dtype=str)
    return [CaseRecord(str(row.case_id), str(row.image_path), str(row.label_path))
            for row in frame.itertuples()]


def digest_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def worker_state(loader: DataLoader) -> dict:
    iterator = getattr(loader, "_iterator", None)
    workers = list(getattr(iterator, "_workers", []) or [])
    return {
        "count": len(workers),
        "pids": [int(worker.pid) for worker in workers if worker.pid is not None],
        "alive": [bool(worker.is_alive()) for worker in workers],
    }


def make_loader(dataset, sampler) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate,
        worker_init_fn=worker_init,
        multiprocessing_context="spawn",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = AUDIT / "manifests/train_50pct.csv"
    train = records(manifest)
    if len(train) != 461:
        raise RuntimeError(f"expected 461 visits, got {len(train)}")
    dataset = TimedReaderDataset(train, load_crop_spec(PILOT / "crop_spec.json"), "disk")

    comparison_indices = np.random.default_rng(SEED + 1).permutation(len(dataset))[:32]
    expected = []
    for index in comparison_indices:
        item = dataset[int(index)]
        expected.append({
            "case_id": item["case_id"],
            "image": item["image"].clone(),
            "mask": item["mask"].clone(),
        })

    sampler = MutableSampler()
    sampler.set_indices(comparison_indices)
    loader = make_loader(dataset, sampler)
    rows = []
    try:
        for position, batch in enumerate(loader):
            reference = expected[position]
            image = batch["image"][0]
            mask = batch["mask"][0]
            rows.append({
                "position": position,
                "index": int(comparison_indices[position]),
                "expected_case": reference["case_id"],
                "actual_case": batch["case_id"][0],
                "order_equal": reference["case_id"] == batch["case_id"][0],
                "mri_exact_equal": bool(torch.equal(reference["image"], image)),
                "mri_max_abs_diff": float((reference["image"].float() - image.float()).abs().max()),
                "mask_exact_equal": bool(torch.equal(reference["mask"], mask)),
                "image_sha256_expected": digest_tensor(reference["image"]),
                "image_sha256_actual": digest_tensor(image),
                "mask_sha256_expected": digest_tensor(reference["mask"]),
                "mask_sha256_actual": digest_tensor(mask),
            })
    finally:
        close_loader(loader)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "spawn_equivalence_32_cases.csv", index=False)
    equivalence_pass = (
        len(comparison) == 32
        and bool(comparison[["order_equal", "mri_exact_equal", "mask_exact_equal"]].all().all())
        and float(comparison.mri_max_abs_diff.max()) == 0.0
    )

    lifecycle_sampler = MutableSampler()
    lifecycle_loader = make_loader(dataset, lifecycle_sampler)
    epoch_rows = []
    try:
        for epoch in range(1, 8):
            order = np.random.default_rng(SEED + epoch).permutation(len(dataset))
            lifecycle_sampler.set_indices(order)
            started = time.time()
            observed = []
            for batch in lifecycle_loader:
                observed.extend(batch["case_id"])
            state = worker_state(lifecycle_loader)
            expected_cases = [train[int(index)].case_id for index in order]
            epoch_rows.append({
                "epoch": epoch,
                "samples": len(observed),
                "first_case": observed[0] if observed else None,
                "last_case": observed[-1] if observed else None,
                "order_exact": observed == expected_cases,
                "worker_count": state["count"],
                "all_workers_alive": bool(state["alive"]) and all(state["alive"]),
                "elapsed_sec": time.time() - started,
            })
    finally:
        state_before_close = worker_state(lifecycle_loader)
        close_loader(lifecycle_loader)

    epochs = pd.DataFrame(epoch_rows)
    epochs.to_csv(OUT / "spawn_epoch_lifecycle.csv", index=False)
    lifecycle_pass = (
        len(epochs) == 7
        and bool((epochs.samples == 461).all())
        and bool(epochs.order_exact.all())
        and bool(epochs.all_workers_alive.all())
        and int(epochs.iloc[-1].epoch) == 7
    )
    final = {
        "status": "PASS" if equivalence_pass and lifecycle_pass else "FAIL",
        "pipeline_equivalence_pass": equivalence_pass,
        "epoch7_dataloader_lifecycle_pass": lifecycle_pass,
        "comparison_cases": len(comparison),
        "epochs_completed": len(epochs),
        "samples_per_epoch": epochs.samples.tolist(),
        "mri_max_abs_diff": float(comparison.mri_max_abs_diff.max()),
        "gt_mask_exact": bool(comparison.mask_exact_equal.all()),
        "sample_order_exact": bool(comparison.order_equal.all() and epochs.order_exact.all()),
        "multiprocessing_context": "spawn",
        "num_workers": 32,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "worker_state_before_close": state_before_close,
        "gpu_forward_backward_training_performed": False,
    }
    (OUT / "RANK_DATALOADER_SPAWN_PREFLIGHT.json").write_text(json.dumps(final, indent=2) + "\n")
    report = f"""# Rank DataLoader spawn preflight

Final status: `{'PIPELINE_EQUIVALENCE_PASS' if equivalence_pass else 'PIPELINE_EQUIVALENCE_FAIL'}`  
Lifecycle status: `{'EPOCH7_DATALOADER_LIFECYCLE_PASS' if lifecycle_pass else 'EPOCH7_DATALOADER_LIFECYCLE_FAIL'}`

- 32-case MRI maximum absolute difference: `{final['mri_max_abs_diff']}`
- GT/ROI masks exactly equal: `{final['gt_mask_exact']}`
- Sample order exactly equal: `{final['sample_order_exact']}`
- Epochs traversed: `{final['epochs_completed']}`
- Samples per epoch: `{final['samples_per_epoch']}`
- Worker configuration: spawn, 32 workers, prefetch=2, persistent workers
- GPU model forward/backward/training performed: `False`

This is an engineering equivalence/lifecycle gate only. It is not a Rank-4 training result and does not authorize an automatic restart.
"""
    (OUT / "RANK_DATALOADER_SPAWN_PREFLIGHT.md").write_text(report)
    if not equivalence_pass or not lifecycle_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
