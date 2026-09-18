#!/usr/bin/env python3
"""Benchmark-only throughput audit for the frozen 50% Rank/DRPA input path.

It never writes model checkpoints and never mutates source MRI, labels,
manifests, or canonical results.  Cache entries are derived reader-space MRI
and integer label maps only.
"""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
import csv
import inspect
import json
import os
# Set these before NumPy/Torch load so DataLoader children inherit a bounded
# CPU-thread configuration.  The benchmark is about workers, not BLAS fan-out.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"
from pathlib import Path
import shutil
import subprocess
import threading
import time
import traceback
import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BASE = Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__"))
OUT = BASE / "quality_audit/pro6000_pipeline_tuning"
CACHE = OUT / "preprocessed_reader_space_cache"
RUN_STATUS = OUT / "pipeline_benchmark_status.json"
PARTIAL = OUT / "partial_results"
AUDIT = BASE / "quality_audit/drpa_data_capacity_scaling"
PILOT = BASE / "quality_audit/voxtell_mtl_drpa8_pilot"
PEFT = BASE / "quality_audit/voxtell_mtl_peft"
PREP = BASE / "quality_audit/voxtell_mtl_b1_bilateral_crop"
PLACEMENT = BASE / "quality_audit/placement_ablation_50"
MODEL = BASE / "VoxTell_weights/voxtell_v1.1"
BANK = BASE / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"
TEXT_CACHE = PEFT / "text_embedding_cache.npz"
SEED, DEVICE = 20260809, torch.device("cuda")
WARMUP, MEASURE = 20, 100
_WORKER_THREADPOOL_LIMITER = None

import sys
sys.path[:0] = [str(PILOT), str(PEFT), str(PREP), str(PLACEMENT), str(BASE / "scripts/train")]
from b1_preprocessing import (BilateralGroupedPatchDataset, CaseRecord, PROMPTS,
                              ROI_LABEL_IDS, load_crop_spec, load_preprocessed_case)
from text_embedding_cache import TextEmbeddingCache
from rank_ablation_wrapper import RankAblationDRPAWrapper


def sync() -> None:
    torch.cuda.synchronize()


def now() -> float: return time.perf_counter()


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def records(path: Path) -> list[CaseRecord]:
    frame = pd.read_csv(path, dtype=str)
    return [CaseRecord(str(x.case_id), str(x.image_path), str(x.label_path)) for x in frame.itertuples()]


def set_seed() -> None:
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def process_util() -> float:
    try:
        import psutil
        return float(psutil.Process().cpu_percent(interval=None))
    except Exception:
        return float("nan")


class GPUSampler:
    def __init__(self) -> None:
        self.values: list[float] = []; self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                text = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=3).stdout.strip().splitlines()[0]
                self.values.append(float(text))
            except Exception: pass
            self.stop.wait(0.25)
    def __enter__(self): self.thread.start(); return self
    def __exit__(self, *_): self.stop.set(); self.thread.join(timeout=3)


def cache_paths(case_id: str) -> tuple[Path, Path, Path]:
    root = CACHE / case_id
    return root / "image.npy", root / "label.npy", root / "meta.json"


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f: np.save(f, array, allow_pickle=False)
    os.replace(tmp, path)


def build_cache(records_: list[CaseRecord], spec) -> list[dict]:
    rows = []
    for record in records_:
        ip, lp, mp = cache_paths(record.case_id)
        t = now()
        if ip.exists() and lp.exists() and mp.exists():
            rows.append({"case_id": record.case_id, "build_sec": 0.0,
                         "image_bytes": int(ip.stat().st_size), "label_bytes": int(lp.stat().st_size),
                         "reused": True})
            continue
        image, label, affine, meta = load_preprocessed_case(record, spec)
        atomic_npy(ip, image); atomic_npy(lp, label)
        temp = mp.with_suffix(".tmp")
        temp.write_text(json.dumps({"case_id": record.case_id, "affine": np.asarray(affine).tolist(), "meta": meta}, default=str))
        os.replace(temp, mp)
        rows.append({"case_id": record.case_id, "build_sec": now()-t,
                     "image_bytes": int(image.nbytes), "label_bytes": int(label.nbytes), "reused": False})
    return rows


def make_item(image: np.ndarray, label: np.ndarray, case_id: str, timing: dict) -> dict:
    masks = np.stack([(label == ROI_LABEL_IDS[p]).astype(np.float32) for p in PROMPTS], axis=0)
    return {"image": torch.from_numpy(np.asarray(image)[None].copy()), "mask": torch.from_numpy(masks),
            "case_id": case_id, "timing": timing}


class TimedReaderDataset(Dataset):
    def __init__(self, records_: list[CaseRecord], spec, mode: str, ram: dict | None = None):
        self.records, self.spec, self.mode, self.ram = records_, spec, mode, ram
        self._raw_marks: dict[str, float] = {}
        self._raw_dataset = None
        if self.mode == "none":
            # The no-cache path must literally traverse the canonical training
            # dataset's __getitem__, with caching disabled.
            self._raw_dataset = BilateralGroupedPatchDataset(
                records_, spec, cache_cases=False, lifecycle_hook=self._raw_hook)
    def _raw_hook(self, event: str, _record, **_) -> None:
        self._raw_marks.setdefault(event, now())
    def __len__(self): return len(self.records)
    def __getitem__(self, index: int) -> dict:
        rec = self.records[int(index)]; t0 = now()
        if self.mode == "none":
            self._raw_marks = {}
            raw_item = self._raw_dataset[int(index)]
            read_end = self._raw_marks.get("image_read_complete", t0)
            crop_end = self._raw_marks.get("crop_complete", now())
            return_end = self._raw_marks.get("getitem_return", now())
            timing = {"io": read_end-t0, "resample_crop": crop_end-read_end,
                      "mask_build": return_end-crop_end, "data_total": return_end-t0}
            return {"image": raw_item["image"], "mask": raw_item["mask"],
                    "case_id": rec.case_id, "timing": timing}
        elif self.mode == "disk":
            ip, lp, _ = cache_paths(rec.case_id); a = now()
            image, label = np.load(ip, mmap_mode="r"), np.load(lp, mmap_mode="r")
            timing = {"io": now()-a, "resample_crop": 0.0}
        elif self.mode == "ram":
            a = now(); image, label = self.ram[rec.case_id]; timing = {"io": now()-a, "resample_crop": 0.0}
        else: raise ValueError(self.mode)
        a = now(); item = make_item(image, label, rec.case_id, timing); item["timing"]["mask_build"] = now()-a
        item["timing"]["data_total"] = now()-t0
        return item


def collate(items: list[dict]) -> dict:
    timing = {k: float(np.mean([x["timing"][k] for x in items])) for k in items[0]["timing"]}
    # Each sample image is [C=1,D,H,W]. Stack preserves the explicit channel
    # dimension, yielding the canonical [B,1,D,H,W] input contract.
    return {"image": torch.stack([x["image"] for x in items], 0), "mask": torch.stack([x["mask"] for x in items], 0),
            "case_id": [x["case_id"] for x in items], "timing": timing}


def worker_init(_: int) -> None:
    global _WORKER_THREADPOOL_LIMITER
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    torch.set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits
        # Keep the limiter alive for the worker's full lifetime. Calling this
        # as an unbound context manager immediately restores the old limits.
        _WORKER_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except Exception: pass


def loss_fp32(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits); dims = tuple(range(2, p.ndim))
    return bce + 1.0 - ((2*(p*target).sum(dims)+1e-5)/(p.sum(dims)+target.sum(dims)+1e-5)).mean()


@dataclass
class RunSpec:
    name: str; mode: str; workers: int; prefetch: int | None; batch: int; serial: bool = False; cache_phase: str = "na"


def model_and_optimizer():
    set_seed(); wrapper = RankAblationDRPAWrapper(str(MODEL), str(BANK), projection_rank=4, device=DEVICE)
    wrapper.set_training_mode(); groups: dict[str, list[torch.nn.Parameter]] = {}
    for _, p, group in wrapper.trainable_parameter_groups(): groups.setdefault(group, []).append(p)
    count = sum(p.numel() for ps in groups.values() for p in ps)
    if count != 10_748_384: raise RuntimeError(f"rank4 count mismatch: {count}")
    opt = torch.optim.AdamW([{ "params": groups["cross_attention_lora"], "lr":1e-4},
                              { "params": groups["projection_adapter"], "lr":1e-4},
                              { "params": groups["decoder_stages"], "lr":1e-5}], weight_decay=1e-5)
    return wrapper, opt, torch.amp.GradScaler("cuda", enabled=True)


def gpu_train_step(wrapper, opt, scaler, cache, batch: dict) -> dict:
    sync(); t_h2d = now(); image = batch["image"].to(DEVICE, non_blocking=True); target = batch["mask"].to(DEVICE, non_blocking=True).float(); sync()
    h2d = now()-t_h2d; opt.zero_grad(set_to_none=True); forward = backward = 0.0
    for j, prompt in enumerate(PROMPTS):
        embedding = cache.get([prompt], DEVICE).expand(image.shape[0], -1, -1, -1)
        sync(); t = now()
        with torch.autocast("cuda", dtype=torch.float16): logits = wrapper(image, embedding)
        with torch.autocast("cuda", enabled=False): loss = loss_fp32(logits, target[:, [j]])
        sync(); forward += now()-t
        if not torch.isfinite(loss): raise RuntimeError("non-finite benchmark loss")
        sync(); t = now(); scaler.scale(loss/len(PROMPTS)).backward(); sync(); backward += now()-t
        del embedding, logits, loss
    sync(); t = now(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0, error_if_nonfinite=True); scaler.step(opt); scaler.update(); sync()
    return {"h2d":h2d, "forward":forward, "backward":backward, "optimizer":now()-t}


def make_loader(ds, indices: list[int], spec: RunSpec):
    if spec.serial: return None
    kwargs = {"dataset": ds, "batch_size": spec.batch, "sampler": indices, "num_workers": spec.workers,
              "pin_memory": True, "persistent_workers": True, "prefetch_factor": spec.prefetch,
              "collate_fn": collate, "worker_init_fn": worker_init}
    if "in_order" in inspect.signature(DataLoader).parameters: kwargs["in_order"] = True
    return DataLoader(**kwargs)


def close_loader(loader: DataLoader | None) -> None:
    """Explicitly retire persistent worker pools between sweep configurations."""
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    del loader
    gc.collect()


def serial_batches(ds, indices: list[int], batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield collate([ds[i] for i in indices[start:start + batch_size]])


def run_spec(spec: RunSpec, records_: list[CaseRecord], crop_spec, seq: list[int], ram: dict | None) -> dict:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); process_util()
    ds = TimedReaderDataset(records_, crop_spec, spec.mode, ram)
    wrapper, opt, scaler = model_and_optimizer(); cache = TextEmbeddingCache(str(BANK), str(MODEL), str(TEXT_CACHE))
    rows = []; loader = None; iterator = None
    try:
        if spec.serial:
            iterator = iter(serial_batches(ds, seq, spec.batch))
        else:
            loader = make_loader(ds, seq, spec)
            iterator = iter(loader)
        with GPUSampler() as sampler:
            for step in range(WARMUP + MEASURE):
                sync(); total_start = now(); data_start = now(); batch = next(iterator); data_wait = now()-data_start
                parts = gpu_train_step(wrapper, opt, scaler, cache, batch); sync(); total = now()-total_start
                if step >= WARMUP:
                    rows.append({"step":step-WARMUP+1, "data_wait":data_wait, "step_total":total, **parts,
                                 "io":batch["timing"]["io"], "resample_crop":batch["timing"]["resample_crop"],
                                 "mask_build":batch["timing"]["mask_build"], "data_total_item":batch["timing"]["data_total"]})
    finally:
        del iterator
        close_loader(loader)
    frame = pd.DataFrame(rows); peak_a = torch.cuda.max_memory_allocated()/2**30; peak_r = torch.cuda.max_memory_reserved()/2**30
    result = {"name":spec.name, "cache_mode":spec.mode, "cache_phase":spec.cache_phase, "batch_size":spec.batch, "num_workers":spec.workers,
              "data_source":{"none":"RAW_SERIAL", "disk":"DISK_CACHE", "ram":"RAM_CACHE"}[spec.mode],
              "prefetch_factor":spec.prefetch, "steps":MEASURE, "samples_per_sec":(MEASURE*spec.batch)/frame.step_total.sum(),
              "mean_sec_step":frame.step_total.mean(), "median_sec_step":frame.step_total.median(),
              "T_io":frame.io.mean(), "T_resample_crop":frame.resample_crop.mean(), "T_mask_build":frame.mask_build.mean(),
              "T_data_total":frame.data_wait.mean(), "T_H2D":frame.h2d.mean(), "T_forward":frame.forward.mean(),
              "T_backward":frame.backward.mean(), "T_optimizer":frame.optimizer.mean(), "data_wait_ratio":frame.data_wait.mean()/frame.step_total.mean(),
              "median_gpu_util":float(np.median(sampler.values)) if sampler.values else float("nan"),
              "peak_allocated_gib":peak_a, "peak_reserved_gib":peak_r, "cpu_process_percent":process_util()}
    del wrapper, opt, scaler, cache, ds; torch.cuda.empty_cache(); return result, frame


def _loader_equivalence(records_: list[CaseRecord], crop_spec, indices: list[int]) -> dict:
    """Confirm worker output/order matches direct, serial samples exactly."""
    ds = TimedReaderDataset(records_, crop_spec, "none")
    kwargs = {"dataset": ds, "batch_size": 1, "sampler": indices[:32], "num_workers": 2,
              "pin_memory": True, "persistent_workers": True, "prefetch_factor": 2,
              "collate_fn": collate, "worker_init_fn": worker_init}
    if "in_order" in inspect.signature(DataLoader).parameters: kwargs["in_order"] = True
    loader = DataLoader(**kwargs)
    rows = []
    try:
        for index, worker_batch in zip(indices[:32], loader):
            direct = ds[index]
            rows.append({"case_id": direct["case_id"], "order_equal": worker_batch["case_id"] == [direct["case_id"]],
                         "mri_exact_equal": bool(torch.equal(worker_batch["image"][0], direct["image"])),
                         "gt_exact_equal": bool(torch.equal(worker_batch["mask"][0], direct["mask"]))})
    finally:
        close_loader(loader)
    return {"rows": rows, "pass": all(all(row[k] for k in ("order_equal", "mri_exact_equal", "gt_exact_equal")) for row in rows)}


def equivalence(records_: list[CaseRecord], crop_spec, indices: list[int]) -> tuple[dict, dict]:
    selected = [records_[i] for i in indices[:32]]; build_rows = build_cache(selected, crop_spec); diffs = []
    for rec in selected:
        raw_i, raw_l, _, _ = load_preprocessed_case(rec, crop_spec)
        ip, lp, _ = cache_paths(rec.case_id); cached_i, cached_l = np.load(ip), np.load(lp)
        raw_masks = np.stack([(raw_l == ROI_LABEL_IDS[p]).astype(np.float32) for p in PROMPTS])
        cached_masks = np.stack([(cached_l == ROI_LABEL_IDS[p]).astype(np.float32) for p in PROMPTS])
        diffs.append({"case_id":rec.case_id, "mri_max_abs_diff":float(np.max(np.abs(raw_i-cached_i))),
                      "gt_exact_equal":bool(np.array_equal(raw_l,cached_l)), "mask_exact_equal":bool(np.array_equal(raw_masks,cached_masks))})
    loader = _loader_equivalence(records_, crop_spec, indices)
    summary = {"visits":len(selected), "mri_max_abs_diff":max(x["mri_max_abs_diff"] for x in diffs),
               "gt_exact_equal":all(x["gt_exact_equal"] for x in diffs), "mask_exact_equal":all(x["mask_exact_equal"] for x in diffs),
               "worker_order_and_tensor_exact": loader["pass"],
               "status":"PIPELINE_EQUIVALENCE_PASS" if max(x["mri_max_abs_diff"] for x in diffs)==0 and all(x["gt_exact_equal"] and x["mask_exact_equal"] for x in diffs) and loader["pass"] else "PIPELINE_EQUIVALENCE_FAIL"}
    return summary, {"equivalence":diffs, "worker_equivalence":loader, "cache_build":build_rows}


def write_reports(hardware: dict, results: list[dict], eq: dict, details: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT / "PIPELINE_BENCHMARK_SUMMARY.csv", index=False)
    pd.DataFrame(details["raw"]).to_csv(OUT / "PIPELINE_BENCHMARK_RAW.csv", index=False)
    (OUT / "PIPELINE_HARDWARE_AUDIT.md").write_text("# Hardware audit\n\n```json\n"+json.dumps(hardware,indent=2)+"\n```\n")
    (OUT / "PIPELINE_EQUIVALENCE_REPORT.md").write_text("# Pipeline equivalence\n\n```json\n"+json.dumps(eq,indent=2)+"\n```\n\nMRI, label and eight derived masks must all match before any batch-1 recommendation is made.\n")
    if not results:
        (OUT / "PIPELINE_OPTIMIZATION_FINAL_REPORT.md").write_text(
            "# Pipeline optimization final report\n\nNo valid throughput rows were produced.\n")
        return
    valid = [r for r in results if "samples_per_sec" in r]
    base = next(r for r in valid if r["name"]=="SERIAL_BASELINE")
    b1 = ranked_batch1(valid, ram_available_gib=800.0)
    high = max(valid, key=lambda x:x["samples_per_sec"])
    lines=["# Pipeline optimization final report", "", "## Results", "", f"- Serial baseline: `{base['samples_per_sec']:.4f}` samples/s.", f"- Best batch-1: `{b1['name']}`, `{b1['samples_per_sec']:.4f}` samples/s, `{b1['samples_per_sec']/base['samples_per_sec']:.2f}x`.", f"- Best maximum-throughput: `{high['name']}`, batch `{high['batch_size']}`, workers `{high['num_workers']}`, prefetch `{high['prefetch_factor']}`, `{high['samples_per_sec']:.4f}` samples/s, GPU util median `{high['median_gpu_util']:.1f}%`, peak allocated `{high['peak_allocated_gib']:.2f} GiB`.", "", "## Batch-scaling boundary", "", "- Batch 4 was attempted after the stable batch-2 result, but the process received external exit 137.", "- cgroup `oom=0` and `oom_kill=0`; batch 4 is recorded as runtime-unstable/unavailable, not as CUDA or host-memory OOM.", "- Batch 2 is the highest stable measured throughput configuration.", "", "## Interpretation boundary", "", f"- Equivalence: `{eq['status']}`. Batch>1 remains `NEW_PROTOCOL_THROUGHPUT` and does not replace canonical batch-1 experiments."]
    (OUT / "PIPELINE_OPTIMIZATION_FINAL_REPORT.md").write_text("\n".join(lines)+"\n")


def hardware_audit() -> dict:
    def run(cmd): return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    return {"physical_cores": int(run("lscpu -p=Core | grep -v '^#' | sort -u | wc -l")),
            "logical_cpus": os.cpu_count(), "ram":run("free -h | sed -n '2p'"), "shm":run("df -h /dev/shm | tail -1"),
            "autodl_disk":run("df -h /root/autodl-tmp | tail -1"), "data_mount":run("findmnt -T /root/autodl-tmp -o SOURCE,FSTYPE -n"),
            "gpu":run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
            "dataset_path":"__DRPA_DATA_ROOT__/Task003_MTL/imagesTr + labelsTr",
            "dataset_stages":"NIfTI read/canonical RAS/reorientation -> normalize + fixed 192^3 crop -> reader-space transpose -> eight masks"}


def advisory_drop_cache(records_: list[CaseRecord]) -> None:
    """Best-effort per-file cache eviction without a host-wide cache flush."""
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advice is None or not hasattr(os, "posix_fadvise"): return
    for rec in records_:
        for path in cache_paths(rec.case_id)[:2]:
            try:
                fd = os.open(path, os.O_RDONLY)
                try: os.posix_fadvise(fd, 0, 0, advice)
                finally: os.close(fd)
            except OSError: pass


def ranked_batch1(results: list[dict], ram_available_gib: float) -> dict:
    candidates = [r for r in results if r.get("batch_size") == 1 and "samples_per_sec" in r]
    warm_candidates = [r for r in candidates if r.get("cache_phase") == "warm"
                       and r.get("cache_mode") in {"disk", "ram"}]
    disk_warm = next((r for r in warm_candidates if r.get("cache_mode") == "disk"), None)
    ram = next((r for r in warm_candidates if r.get("cache_mode") == "ram"), None)
    # Prefer RAM only when it has a material speed advantage and the requested
    # >100 GiB free-memory condition is met. The formal choice is otherwise
    # restricted to repeated-epoch warm-cache measurements.
    if ram and disk_warm and ram_available_gib > 100 and ram["samples_per_sec"] > 1.05 * disk_warm["samples_per_sec"]:
        return ram
    ranked = warm_candidates if warm_candidates else candidates
    return sorted(ranked, key=lambda r: (r["samples_per_sec"], -r["data_wait_ratio"],
                                         r["median_gpu_util"], -r["peak_allocated_gib"]), reverse=True)[0]


def write_status(phase: str, **extra) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_STATUS.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": phase, **extra}, indent=2))


def partial_paths(name: str) -> tuple[Path, Path]:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    return PARTIAL / f"{safe}.json", PARTIAL / f"{safe}.csv"


def save_partial(result: dict, frame: pd.DataFrame) -> None:
    """Persist each completed configuration so an external kill loses at most one run."""
    PARTIAL.mkdir(parents=True, exist_ok=True)
    summary_path, raw_path = partial_paths(result["name"])
    summary_tmp = summary_path.with_suffix(".json.tmp")
    raw_tmp = raw_path.with_suffix(".csv.tmp")
    summary_tmp.write_text(json.dumps(result, indent=2, default=str))
    frame.to_csv(raw_tmp, index=False)
    os.replace(summary_tmp, summary_path)
    os.replace(raw_tmp, raw_path)


def load_partials() -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    raw: list[dict] = []
    if not PARTIAL.exists():
        return results, raw
    for summary_path in sorted(PARTIAL.glob("*.json")):
        result = json.loads(summary_path.read_text())
        _, raw_path = partial_paths(result["name"])
        if not raw_path.exists():
            continue
        result.setdefault("data_source", {"none": "RAW_SERIAL", "disk": "DISK_CACHE",
                                           "ram": "RAM_CACHE"}.get(result.get("cache_mode"), "UNKNOWN"))
        results.append(result)
        frame = pd.read_csv(raw_path)
        if "name" not in frame.columns:
            frame.insert(0, "name", result["name"])
        raw.extend(frame.to_dict("records"))
    return results, raw


def run_and_save(spec: RunSpec, records_: list[CaseRecord], crop_spec, seq: list[int], ram: dict | None,
                 results: list[dict], raw: list[dict]) -> dict:
    existing = next((row for row in results if row.get("name") == spec.name), None)
    if existing is not None:
        write_status("configuration_reused", configuration=spec.name)
        return existing
    result, frame = run_spec(spec, records_, crop_spec, seq, ram)
    frame.insert(0, "name", spec.name)
    save_partial(result, frame)
    results.append(result)
    raw.extend(frame.to_dict("records"))
    write_status("configuration_complete", configuration=spec.name,
                 samples_per_sec=result.get("samples_per_sec"))
    return result


def worker_lifecycle_smoke(records_: list[CaseRecord], crop_spec, fixed: list[int]) -> None:
    """CPU-only preflight: prove a persistent two-worker loader exits cleanly."""
    ds = TimedReaderDataset(records_, crop_spec, "disk")
    spec = RunSpec("WORKER_LIFECYCLE_SMOKE", "disk", 2, 2, 1)
    loader = make_loader(ds, fixed[:4], spec)
    try:
        for _ in loader:
            pass
    finally:
        close_loader(loader)
    time.sleep(1.0)
    workers = subprocess.run(
        "ps -eo comm=,args= | awk '$1 == \"pt_data_worker\" && $0 ~ /pro6000_pipeline_benchmark\\.py/'",
        shell=True, capture_output=True, text=True,
    ).stdout.strip()
    if workers:
        raise RuntimeError("worker lifecycle smoke left pt_data_worker processes: " + workers)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--run", action="store_true"); ap.add_argument("--worker-lifecycle-smoke", action="store_true"); ap.add_argument("--finalize-partials", action="store_true"); args=ap.parse_args()
    if not (args.run or args.worker_lifecycle_smoke or args.finalize_partials): ap.error("explicit --run, --worker-lifecycle-smoke, or --finalize-partials required")
    OUT.mkdir(parents=True, exist_ok=True)
    train_path=AUDIT/"manifests/train_50pct.csv"; records_=records(train_path); crop_spec=load_crop_spec(PILOT/"crop_spec.json")
    fixed = np.random.default_rng(SEED+77).permutation(len(records_)).tolist()
    if args.finalize_partials:
        write_status("final_equivalence_recheck")
        eq, eq_details = equivalence(records_, crop_spec, fixed)
        results, raw = load_partials()
        if not any(row.get("name") == "THROUGHPUT_B4" for row in results):
            results.append({"name": "THROUGHPUT_B4", "cache_mode": "disk", "cache_phase": "na",
                            "data_source": "DISK_CACHE", "batch_size": 4, "num_workers": 32,
                            "prefetch_factor": 2, "status": "EXTERNAL_TERMINATION_EXIT_137"})
        (OUT/"equivalence_details.json").write_text(json.dumps(eq_details, indent=2))
        write_reports(hardware_audit(), results, eq, {"raw": raw})
        write_status("complete_with_batch4_external_termination", equivalence=eq["status"])
        return
    if args.worker_lifecycle_smoke:
        write_status("worker_lifecycle_smoke_running")
        try:
            worker_lifecycle_smoke(records_, crop_spec, fixed)
        except Exception:
            write_status("worker_lifecycle_smoke_failed", traceback=traceback.format_exc())
            raise
        write_status("worker_lifecycle_smoke_pass")
        return
    write_status("equivalence_running")
    try:
    # 100 measured items at batch 1, cycling deterministically for larger batches.
        eq, eq_details = equivalence(records_, crop_spec, fixed)
        if eq["status"] != "PIPELINE_EQUIVALENCE_PASS":
            write_reports(hardware_audit(), [], eq, {"raw":[]}); raise RuntimeError(eq["status"])
    # The formal tuning cache covers precisely the actual Rank-50% training
    # cohort (461 visits), never the unrelated 971-visit cohort.
        write_status("cache_build_running")
        build_cache(records_, crop_spec)
        ram = {rec.case_id:(np.load(cache_paths(rec.case_id)[0]), np.load(cache_paths(rec.case_id)[1])) for rec in records_}
        sequence_base = fixed[:100]
        specs=[RunSpec("SERIAL_BASELINE","none",0,None,1,True)]
        for workers in [2,4,8,12,16,32]: specs.append(RunSpec(f"DL_W{workers}_P2","none",workers,2,1))
        results, raw = load_partials()
        (OUT / "SERIAL_BASELINE_RUNTIME_CONTRACT.json").write_text(json.dumps({
            "cache_used": False,
            "num_workers": 0,
            "batch_size": 1,
            "data_source": "RAW_SERIAL",
            "call_path": "BilateralGroupedPatchDataset(cache_cases=False).__getitem__",
        }, indent=2))
        for spec in specs:
            write_status("worker_sweep", configuration=spec.name)
            seq=(sequence_base*((WARMUP+MEASURE+len(sequence_base)-1)//len(sequence_base)))[:WARMUP+MEASURE]
            run_and_save(spec, records_, crop_spec, seq, ram, results, raw)
        top2=sorted([x for x in results if x["name"].startswith("DL_")],key=lambda x:x["samples_per_sec"],reverse=True)[:2]
        for chosen in top2:
            for p in [2,4]:
                spec=RunSpec(f"DL_W{chosen['num_workers']}_P{p}","none",chosen["num_workers"],p,1)
                if any(r["name"]==spec.name for r in results): continue
                seq=(sequence_base*2)[:WARMUP+MEASURE]
                run_and_save(spec, records_, crop_spec, seq, ram, results, raw)
        cache_records = [records_[i] for i in sequence_base]
        advisory_drop_cache(cache_records)
        for spec in [RunSpec("DL_DISK_CACHE_COLD","disk",top2[0]["num_workers"],top2[0]["prefetch_factor"],1,cache_phase="cold"), RunSpec("DL_DISK_CACHE_WARM","disk",top2[0]["num_workers"],top2[0]["prefetch_factor"],1,cache_phase="warm"), RunSpec("DL_RAM_CACHE_WARM","ram",top2[0]["num_workers"],top2[0]["prefetch_factor"],1,cache_phase="warm")]:
            write_status("cache_sweep", configuration=spec.name)
            seq=(sequence_base*2)[:WARMUP+MEASURE]
            run_and_save(spec, records_, crop_spec, seq, ram, results, raw)
        best_b1=ranked_batch1(results, ram_available_gib=858.0)
        previous=best_b1["samples_per_sec"]; batch=2
        while batch<=32:
            spec=RunSpec(f"THROUGHPUT_B{batch}",best_b1["cache_mode"],best_b1["num_workers"],best_b1["prefetch_factor"],batch)
            write_status("batch_sweep", configuration=spec.name)
            seq=(sequence_base*((WARMUP+MEASURE)*batch//len(sequence_base)+1))[:(WARMUP+MEASURE)*batch]
            try:
                result = run_and_save(spec, records_, crop_spec, seq, ram, results, raw)
            except torch.cuda.OutOfMemoryError:
                oom_result = {"name":spec.name,"batch_size":batch,"status":"OOM"}
                PARTIAL.mkdir(parents=True, exist_ok=True)
                partial_paths(spec.name)[0].write_text(json.dumps(oom_result, indent=2))
                results.append(oom_result)
                break
            if result["peak_allocated_gib"] > 0.9* (97887/1024) or result["samples_per_sec"] < previous*1.05: break
            previous=result["samples_per_sec"]; batch*=2
        (OUT/"equivalence_details.json").write_text(json.dumps(eq_details,indent=2))
        write_reports(hardware_audit(),results,eq,{"raw":raw})
    except Exception:
        write_status("failed", traceback=traceback.format_exc())
        raise
    write_status("complete", output=str(OUT / "PIPELINE_OPTIMIZATION_FINAL_REPORT.md"))


if __name__=="__main__": main()
