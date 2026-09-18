"""Isolated B1 preprocessing for a fixed bilateral MTL crop.

The official VoxTell source and the existing B0 pilot code are not modified.
B1 uses canonical RAS arrays on the native grid and a fixed 192^3 crop whose
center fraction is estimated from training labels only.
"""
from __future__ import annotations
import gc, json, sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
BASE = Path("__DRPA_WORKSPACE__"); PEFT = BASE / "quality_audit/voxtell_mtl_peft"; PILOT = BASE / "quality_audit/voxtell_mtl_peft_pilot"; sys.path.insert(0, str(PEFT))
sys.path.insert(0, str(PILOT))
from mtl_grounding_dataset import CaseRecord, PROMPTS, ROI_LABEL_IDS
from pilot_preprocessing import load_official_reader_case
from mtl_grounding_dataset import _make_centered_patch
@dataclass(frozen=True)
class CropSpec:
    mode: str = "bilateral_mtl_crop"
    patch_size: Tuple[int, int, int] = (192, 192, 192)
    center_fraction: Tuple[float, float, float] = (0.49819946, 0.48696001, 0.48282667)
    derived_from: str = "50 train PTID MALPEM labels; normalized canonical-RAS centroid median"
    target_spacing_mm: Tuple[float, float, float] | None = None
    image_interpolation: str = "continuous_order_1_when_resampled"
    label_interpolation: str = "nearest_neighbor_order_0_only"
    margin_policy: str = "fixed 192^3 model input; training-derived center; zero padding outside source"
def translation_affine(start: Sequence[int]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64); out[:3, 3] = np.asarray(start, dtype=np.float64); return out
def _label_array(nii: nib.Nifti1Image) -> np.ndarray:
    arr = np.asarray(nii.get_fdata())
    if arr.ndim == 4 and arr.shape[-1] == 1: arr = arr[..., 0]
    if arr.ndim != 3: raise ValueError(f"MALPEM label must be 3D, got {arr.shape}")
    return arr.astype(np.int16, copy=False)
def load_canonical_case(record: CaseRecord):
    raw_image_nii = nib.load(record.image_path); raw_label_nii = nib.load(record.label_path); raw_image = np.asarray(raw_image_nii.get_fdata(dtype=np.float32)); raw_label = _label_array(raw_label_nii)
    if raw_image.ndim != 3: raise ValueError(f"MRI must be 3D: {record.case_id}: {raw_image.shape}")
    if raw_image.shape != raw_label.shape: raise ValueError(f"raw shape mismatch for {record.case_id}: {raw_image.shape} vs {raw_label.shape}")
    if not np.allclose(raw_image_nii.affine, raw_label_nii.affine, rtol=0, atol=1e-4): raise ValueError(f"raw affine mismatch for {record.case_id}")
    image_nii = nib.as_closest_canonical(raw_image_nii); label_nii = nib.as_closest_canonical(raw_label_nii); image = np.asarray(image_nii.get_fdata(dtype=np.float32)); label = _label_array(label_nii)
    if image.shape != label.shape: raise ValueError(f"canonical shape mismatch for {record.case_id}: {image.shape} vs {label.shape}")
    if not np.allclose(image_nii.affine, label_nii.affine, rtol=0, atol=1e-4): raise ValueError(f"canonical affine mismatch for {record.case_id}")
    if nib.aff2axcodes(image_nii.affine) != ("R", "A", "S"): raise ValueError(f"MRI is not canonical RAS for {record.case_id}")
    if nib.aff2axcodes(label_nii.affine) != ("R", "A", "S"): raise ValueError(f"label is not canonical RAS for {record.case_id}")
    if not np.isfinite(image).all(): raise ValueError(f"non-finite MRI for {record.case_id}")
    missing = sorted(set(ROI_LABEL_IDS.values()) - set(np.unique(label).tolist()))
    if missing: raise ValueError(f"missing ROI IDs for {record.case_id}: {missing}")
    meta = {"case_id": record.case_id, "raw_shape": tuple(int(x) for x in raw_image.shape), "canonical_shape": tuple(int(x) for x in image.shape), "raw_affine": raw_image_nii.affine.astype(np.float64).tolist(), "canonical_affine": image_nii.affine.astype(np.float64).tolist(), "raw_orientation": tuple(str(x) for x in nib.aff2axcodes(raw_image_nii.affine)), "canonical_orientation": tuple(str(x) for x in nib.aff2axcodes(image_nii.affine)), "spacing_mm": tuple(float(x) for x in nib.affines.voxel_sizes(image_nii.affine)), "geometry_exact": True}
    return image, label, image_nii.affine.astype(np.float64), meta
def derive_training_crop_spec(cases: Sequence[CaseRecord], patch_size=(192, 192, 192)) -> CropSpec:
    centers = []
    for record in cases:
        image, label, _, _ = load_canonical_case(record); coords = np.argwhere(np.isin(label, list(ROI_LABEL_IDS.values())))
        if coords.size == 0: raise ValueError(f"no MTL ROI voxels in training case {record.case_id}")
        centers.append(coords.mean(axis=0) / np.asarray(image.shape, dtype=np.float64))
    return CropSpec(patch_size=tuple(int(x) for x in patch_size), center_fraction=tuple(float(x) for x in np.median(np.asarray(centers), axis=0)), derived_from=f"{len(cases)} train PTID MALPEM labels; normalized canonical-RAS centroid median")
def save_crop_spec(spec: CropSpec, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(asdict(spec), indent=2))
def load_crop_spec(path: str | Path) -> CropSpec:
    data = json.loads(Path(path).read_text()); data["patch_size"] = tuple(int(x) for x in data["patch_size"]); data["center_fraction"] = tuple(float(x) for x in data["center_fraction"])
    if data.get("target_spacing_mm") is not None: data["target_spacing_mm"] = tuple(float(x) for x in data["target_spacing_mm"])
    return CropSpec(**data)
def _crop_pad(arr: np.ndarray, start: np.ndarray, patch_size: np.ndarray) -> np.ndarray:
    stop = start + patch_size; src_lo = np.maximum(start, 0); src_hi = np.minimum(stop, np.asarray(arr.shape, dtype=int)); dst_lo = src_lo - start; out = np.zeros(tuple(int(x) for x in patch_size), dtype=arr.dtype)
    if np.all(src_hi > src_lo): out[tuple(slice(int(dst_lo[d]), int(dst_lo[d] + src_hi[d] - src_lo[d])) for d in range(3))] = arr[tuple(slice(int(src_lo[d]), int(src_hi[d])) for d in range(3))]
    return out
def _normalize_mri(image: np.ndarray) -> np.ndarray:
    nonzero = np.isfinite(image) & (np.abs(image) > 0)
    if not nonzero.any(): raise ValueError("MRI has no finite nonzero voxels")
    mean = float(image[nonzero].mean()); std = float(image[nonzero].std())
    if std < 1e-6 or not np.isfinite(std): std = 1.0
    return ((image - mean) / std).astype(np.float32)
def crop_case(image: np.ndarray, label: np.ndarray, affine: np.ndarray, spec: CropSpec):
    if spec.mode != "bilateral_mtl_crop": raise ValueError(f"crop_case only accepts bilateral_mtl_crop, got {spec.mode}")
    patch = np.asarray(spec.patch_size, dtype=int)
    if patch.shape != (3,) or np.any(patch <= 0): raise ValueError(f"invalid patch_size: {spec.patch_size}")
    start = np.floor(np.asarray(spec.center_fraction, dtype=np.float64) * np.asarray(image.shape, dtype=np.float64) - patch / 2.0).astype(int)
    cropped_image = _crop_pad(_normalize_mri(image), start, patch); cropped_label = _crop_pad(label.astype(np.int16, copy=False), start, patch); crop_affine = np.asarray(affine, dtype=np.float64) @ translation_affine(start)
    meta = {"mode": spec.mode, "source_shape": tuple(int(x) for x in image.shape), "crop_shape": tuple(int(x) for x in cropped_image.shape), "source_affine": np.asarray(affine, dtype=np.float64).tolist(), "crop_affine": crop_affine.tolist(), "crop_start_voxel_canonical": tuple(int(x) for x in start), "crop_stop_voxel_canonical": tuple(int(x) for x in start + patch), "padding_before": tuple(int(x) for x in np.maximum(-start, 0)), "padding_after": tuple(int(x) for x in np.maximum(start + patch - np.asarray(image.shape), 0)), "same_transform_for_image_and_label": True, "image_interpolation": spec.image_interpolation, "label_interpolation": spec.label_interpolation, "target_spacing_mm": spec.target_spacing_mm, "orientation": "RAS"}
    return cropped_image, cropped_label, crop_affine, meta
def restore_crop_to_canonical(crop: np.ndarray, source_shape: Sequence[int], meta: Dict) -> np.ndarray:
    source_shape = np.asarray(source_shape, dtype=int); start = np.asarray(meta["crop_start_voxel_canonical"], dtype=int); patch = np.asarray(crop);
    if meta.get("reader_space", False): patch = patch.transpose((2, 1, 0))
    out = np.zeros(tuple(int(x) for x in source_shape), dtype=patch.dtype); src_lo = np.maximum(-start, 0); src_hi = np.minimum(np.asarray(patch.shape), source_shape - start); dst_lo = np.maximum(start, 0)
    if np.all(src_hi > src_lo): out[tuple(slice(int(dst_lo[d]), int(dst_lo[d] + src_hi[d] - src_lo[d])) for d in range(3))] = patch[tuple(slice(int(src_lo[d]), int(src_hi[d])) for d in range(3))]
    return out
def restore_canonical_to_raw(canonical: np.ndarray, record: CaseRecord):
    raw = nib.load(record.image_path); can = nib.as_closest_canonical(raw); inverse = nib.orientations.ornt_transform(nib.orientations.io_orientation(can.affine), nib.orientations.io_orientation(raw.affine)); return nib.orientations.apply_orientation(np.asarray(canonical), inverse), raw.affine.astype(np.float64)
def _world_x(affine: np.ndarray, voxel: np.ndarray) -> float: return float((np.asarray(affine) @ np.r_[np.asarray(voxel, dtype=np.float64), 1.0])[0])
def validate_crop_integrity(source_label, crop_label, source_affine, crop_affine, case_id):
    rows = {"case_id": case_id, "all_roi_inside": True, "roi_voxel_loss": 0, "left_right_ok": True}
    for prompt, roi_id in ROI_LABEL_IDS.items():
        source_n = int(np.count_nonzero(source_label == roi_id)); crop_n = int(np.count_nonzero(crop_label == roi_id)); rows[f"{prompt.replace(' ', '_')}_source_voxels"] = source_n; rows[f"{prompt.replace(' ', '_')}_crop_voxels"] = crop_n
        if source_n != crop_n: rows["all_roi_inside"] = False; rows["roi_voxel_loss"] += source_n - crop_n
    for structure in ("hippocampus", "entorhinal cortex", "parahippocampal gyrus", "amygdala"):
        left = np.argwhere(crop_label == ROI_LABEL_IDS[f"left {structure}"]); right = np.argwhere(crop_label == ROI_LABEL_IDS[f"right {structure}"])
        if left.size and right.size:
            lx = _world_x(crop_affine, left.mean(0)); rx = _world_x(crop_affine, right.mean(0)); rows[f"{structure.replace(' ', '_')}_left_x_mm"] = lx; rows[f"{structure.replace(' ', '_')}_right_x_mm"] = rx; ok = lx < rx; rows[f"{structure.replace(' ', '_')}_left_right_ok"] = bool(ok); rows["left_right_ok"] = bool(rows["left_right_ok"] and ok)
    if not rows["all_roi_inside"]: raise AssertionError(f"ROI voxels were cropped for {case_id}: {rows['roi_voxel_loss']}")
    if not rows["left_right_ok"]: raise AssertionError(f"left/right semantic order changed for {case_id}")
    return rows
def load_preprocessed_case(record: CaseRecord, spec: CropSpec, lifecycle_hook=None):
    def event(name: str, **extra):
        if lifecycle_hook is not None:
            lifecycle_hook(name, record, **extra)
    if spec.mode == "full_volume":
        image, label, affine, meta = load_official_reader_case(record)
        event("image_read_complete", image_shape=list(image.shape))
        event("mask_read_complete", mask_shape=list(label.shape))
        image_patch, label_patch, patch_affine = _make_centered_patch(image, label, affine, None, spec.patch_size)
        event("crop_complete", crop_shape=list(image_patch.shape))
        meta = dict(meta); meta.update({"mode": "full_volume_compat_b0", "source_shape": tuple(int(x) for x in image.shape), "crop_shape": tuple(int(x) for x in image_patch.shape), "crop_affine": patch_affine.tolist(), "same_transform_for_image_and_label": True, "image_interpolation": "official_reader_native_grid", "label_interpolation": "official_reader_native_grid"}); validate_crop_integrity(label, label_patch, affine, patch_affine, record.case_id); return image_patch.astype(np.float32), label_patch.astype(np.int16), patch_affine, meta
    image, label, affine, source_meta = load_canonical_case(record)
    event("image_read_complete", image_shape=list(image.shape))
    event("mask_read_complete", mask_shape=list(label.shape))
    image_patch, label_patch, canonical_crop_affine, crop_meta = crop_case(image, label, affine, spec)
    event("crop_complete", crop_shape=list(image_patch.shape))
    validate_crop_integrity(label, label_patch, affine, canonical_crop_affine, record.case_id)
    perm = np.asarray([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]], dtype=float)
    reader_image = image_patch.transpose((2,1,0)).copy()
    reader_label = label_patch.transpose((2,1,0)).copy()
    reader_crop_affine = canonical_crop_affine @ perm
    crop_meta.update({"source_meta": source_meta, "reader_space": True,
                      "canonical_crop_affine": canonical_crop_affine.tolist(),
                      "crop_affine": reader_crop_affine.tolist(),
                      "reader_contract": "official NibabelIOWithReorient + transpose(2,1,0)",
                      "canonical_crop_shape": tuple(int(x) for x in label_patch.shape)})
    return reader_image, reader_label, reader_crop_affine, crop_meta
def _estimate_cache_bytes(cache: dict) -> int:
    total = 0
    for image, label, affine, meta in cache.values():
        total += int(getattr(image, 'nbytes', 0) + getattr(label, 'nbytes', 0) + getattr(affine, 'nbytes', 0))
    return total

class BilateralGroupedPatchDataset(Dataset):
    def __init__(self, cases: Sequence[CaseRecord], spec: CropSpec, cache_cases: bool = True, lifecycle_hook=None):
        self.cases = list(cases); self.spec = spec; self.cache_cases = bool(cache_cases); self._cache = {}; self.lifecycle_hook = lifecycle_hook
    def _event(self, event: str, record: CaseRecord, **extra):
        if self.lifecycle_hook is not None:
            self.lifecycle_hook(event, record, cache_item_count=len(self._cache), **extra)
    def __len__(self): return len(self.cases)
    def __getitem__(self, index: int):
        record = self.cases[int(index)]
        self._event('case_read_start', record)
        if self.cache_cases and record.case_id in self._cache:
            image, label, affine, meta = self._cache[record.case_id]
            self._event('cache_hit', record, estimated_cache_bytes=_estimate_cache_bytes(self._cache))
        else:
            self._event('preprocessing_start', record)
            image, label, affine, meta = load_preprocessed_case(record, self.spec, self._event)
            self._event('preprocessing_complete', record, image_shape=list(image.shape), mask_shape=list(label.shape))
            if self.cache_cases:
                self._cache[record.case_id] = (image, label, affine, meta)
                self._event('cache_insert_complete', record, estimated_cache_bytes=_estimate_cache_bytes(self._cache))
        masks = np.stack([(label == ROI_LABEL_IDS[p]).astype(np.float32) for p in PROMPTS], axis=0)
        if masks.shape[1:] != tuple(self.spec.patch_size): raise RuntimeError(f"batch image/mask shape mismatch: {masks.shape}")
        result = {"image": torch.from_numpy(image[None]), "mask": torch.from_numpy(masks), "case_id": record.case_id, "prompts": tuple(PROMPTS), "patch_affine": affine, "preprocess_meta": meta, "source_label": label, "foreground_fraction": float(np.count_nonzero(label) / label.size)}
        self._event('getitem_return', record, estimated_cache_bytes=_estimate_cache_bytes(self._cache), returned_tensor_bytes=int(result['image'].numpy().nbytes + result['mask'].numpy().nbytes))
        return result
    def prompt_item(self, index: int):
        case_index = int(index) // len(PROMPTS); prompt_index = int(index) % len(PROMPTS); item = dict(self[case_index]); prompt = PROMPTS[prompt_index]; item["prompt"] = prompt; item["mask"] = item["mask"][prompt_index:prompt_index + 1]; item["roi_label_id"] = ROI_LABEL_IDS[prompt]; return item
    def common_case_patch(self, case_id: str):
        record = next(x for x in self.cases if x.case_id == case_id); image, label, affine, meta = load_preprocessed_case(record, self.spec); return {"image": torch.from_numpy(image[None]), "label_patch": label, "patch_affine": affine, "preprocess_meta": meta, "case_id": case_id}
def audit_cases(cases: Sequence[CaseRecord], spec: CropSpec):
    rows = []
    for record in cases:
        image, label, affine, source_meta = load_canonical_case(record); cropped_image, cropped_label, crop_affine, crop_meta = crop_case(image, label, affine, spec); row = validate_crop_integrity(label, cropped_label, affine, crop_affine, record.case_id)
        row.update({"split_case_id": record.case_id, "shape": "x".join(map(str, image.shape)), "crop_shape": "x".join(map(str, cropped_image.shape)), "spacing_mm": "x".join(f"{x:.6f}" for x in nib.affines.voxel_sizes(crop_affine)), "canonical_orientation": "".join(nib.aff2axcodes(affine)), "foreground_fraction_source": float(np.count_nonzero(label) / label.size), "foreground_fraction_crop": float(np.count_nonzero(cropped_label) / cropped_label.size), "source_affine": json.dumps(source_meta["canonical_affine"]), "crop_affine": json.dumps(crop_affine.tolist()), "crop_start": json.dumps(crop_meta["crop_start_voxel_canonical"]), "crop_stop": json.dumps(crop_meta["crop_stop_voxel_canonical"])})
        rows.append(row)
        del image, label, affine, source_meta, cropped_image, cropped_label, crop_affine, crop_meta
        gc.collect()
    return rows
