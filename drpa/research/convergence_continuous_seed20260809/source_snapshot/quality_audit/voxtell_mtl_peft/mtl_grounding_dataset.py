"""Minimal canonical-RAS MRI + prompt + binary MALPEM mask interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


ROI_LABEL_IDS = {
    "right hippocampus": 1,
    "left hippocampus": 2,
    "right entorhinal cortex": 3,
    "left entorhinal cortex": 4,
    "right parahippocampal gyrus": 5,
    "left parahippocampal gyrus": 6,
    "right amygdala": 7,
    "left amygdala": 8,
}

PROMPTS = list(ROI_LABEL_IDS)


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    image_path: str
    label_path: str


def _load_canonical_case(record: CaseRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    image_nii = nib.as_closest_canonical(nib.load(record.image_path))
    label_nii = nib.as_closest_canonical(nib.load(record.label_path))
    image = np.asarray(image_nii.get_fdata(dtype=np.float32))
    label = np.asarray(label_nii.get_fdata())
    if image.ndim != 3:
        raise ValueError(f"MRI must be 3D: {record.case_id} has {image.shape}")
    if label.ndim == 4 and label.shape[-1] == 1:
        label = label[..., 0]
    if label.ndim != 3:
        raise ValueError(f"Label must be 3D: {record.case_id} has {label.shape}")
    if image.shape != label.shape:
        raise ValueError(f"MRI/label shape mismatch for {record.case_id}: {image.shape} vs {label.shape}")
    if not np.allclose(image_nii.affine, label_nii.affine, rtol=0, atol=1e-4):
        raise ValueError(f"MRI/label affine mismatch for {record.case_id}")
    if nib.aff2axcodes(image_nii.affine) != ("R", "A", "S"):
        raise ValueError(f"MRI is not canonical RAS for {record.case_id}")
    if nib.aff2axcodes(label_nii.affine) != ("R", "A", "S"):
        raise ValueError(f"Label is not canonical RAS for {record.case_id}")
    if not np.isfinite(image).all() or not np.isfinite(label).all():
        raise ValueError(f"Non-finite MRI/label values for {record.case_id}")
    required = set(ROI_LABEL_IDS.values())
    present = set(np.unique(label.astype(np.int16)).tolist())
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"Missing MALPEM label IDs for {record.case_id}: {missing}")
    meta = {
        "affine": image_nii.affine.astype(np.float64),
        "shape": tuple(int(x) for x in image.shape),
        "spacing_mm": tuple(float(x) for x in image_nii.header.get_zooms()[:3]),
        "orientation": "RAS",
    }
    return image, label.astype(np.int16), image_nii.affine.astype(np.float64), meta


def _translation_affine(start: Sequence[int]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.asarray(start, dtype=np.float64)
    return out


def _make_centered_patch(
    image: np.ndarray,
    label: np.ndarray,
    affine: np.ndarray,
    roi_label: Optional[int],
    patch_size: Sequence[int] = (192, 192, 192),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    patch_size = tuple(int(x) for x in patch_size)
    if len(patch_size) != 3:
        raise ValueError("patch_size must have three dimensions")

    nonzero = np.abs(image) > 0
    coords = np.argwhere(nonzero)
    if coords.size == 0:
        raise ValueError("MRI has no nonzero voxels")
    low = coords.min(axis=0)
    high = coords.max(axis=0) + 1
    image_crop = image[tuple(slice(int(low[d]), int(high[d])) for d in range(3))]
    label_crop = label[tuple(slice(int(low[d]), int(high[d])) for d in range(3))]
    crop_affine = affine @ _translation_affine(low)

    if roi_label is None:
        center_mask = np.isin(label_crop, list(ROI_LABEL_IDS.values()))
    else:
        center_mask = label_crop == int(roi_label)
    center_coords = np.argwhere(center_mask)
    if center_coords.size == 0:
        raise ValueError(f"Requested ROI label {roi_label} is empty")
    center = center_coords.mean(axis=0)
    start = np.floor(center - np.asarray(patch_size, dtype=np.float64) / 2.0).astype(int)
    stop = start + np.asarray(patch_size, dtype=int)
    clipped_start = np.maximum(start, 0)
    clipped_stop = np.minimum(stop, np.asarray(image_crop.shape, dtype=int))
    pad_before = np.maximum(-start, 0)
    pad_after = np.maximum(stop - np.asarray(image_crop.shape, dtype=int), 0)

    image_part = image_crop[
        tuple(slice(int(clipped_start[d]), int(clipped_stop[d])) for d in range(3))
    ]
    label_part = label_crop[
        tuple(slice(int(clipped_start[d]), int(clipped_stop[d])) for d in range(3))
    ]
    image_patch = np.pad(image_part, tuple((int(pad_before[d]), int(pad_after[d])) for d in range(3)))
    label_patch = np.pad(label_part, tuple((int(pad_before[d]), int(pad_after[d])) for d in range(3)))
    if image_patch.shape != patch_size or label_patch.shape != patch_size:
        raise RuntimeError(f"Patch shape error: {image_patch.shape}, {label_patch.shape}")

    finite_nonzero = image_crop[np.isfinite(image_crop) & (np.abs(image_crop) > 0)]
    mean = float(finite_nonzero.mean())
    std = float(finite_nonzero.std())
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    image_crop_norm = (image_crop - mean) / std
    image_part_norm = image_crop_norm[
        tuple(slice(int(clipped_start[d]), int(clipped_stop[d])) for d in range(3))
    ]
    image_patch = np.pad(image_part_norm, tuple((int(pad_before[d]), int(pad_after[d])) for d in range(3)))
    patch_affine = crop_affine @ _translation_affine(start)
    return image_patch.astype(np.float32), label_patch, patch_affine


class MTLGroundingDataset(Dataset):
    """One visit x one prompt -> one binary ROI mask, without source edits."""

    def __init__(self, cases: Sequence[CaseRecord], prompts: Sequence[str],
                 patch_size: Sequence[int] = (192, 192, 192), cache_cases: bool = True) -> None:
        self.cases = list(cases)
        self.prompts = [p.lower() for p in prompts]
        self.patch_size = tuple(int(x) for x in patch_size)
        self.cache_cases = bool(cache_cases)
        unknown = sorted(set(self.prompts) - set(ROI_LABEL_IDS))
        if unknown:
            raise ValueError(f"Unsupported prompts: {unknown}")
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]] = {}

    def __len__(self) -> int:
        return len(self.cases) * len(self.prompts)

    def _case(self, record: CaseRecord):
        if not self.cache_cases:
            return _load_canonical_case(record)
        if record.case_id not in self._cache:
            self._cache[record.case_id] = _load_canonical_case(record)
        return self._cache[record.case_id]

    def __getitem__(self, index: int) -> Dict:
        case_index = index // len(self.prompts)
        prompt_index = index % len(self.prompts)
        record = self.cases[case_index]
        prompt = self.prompts[prompt_index]
        image, label, affine, meta = self._case(record)
        roi_id = ROI_LABEL_IDS[prompt]
        image_patch, label_patch, patch_affine = _make_centered_patch(
            image, label, affine, roi_id, self.patch_size
        )
        binary = (label_patch == roi_id).astype(np.float32)
        return {
            "image": torch.from_numpy(image_patch[None]),
            "mask": torch.from_numpy(binary[None]),
            "case_id": record.case_id,
            "prompt": prompt,
            "ptid": record.case_id.split("_")[0],
            "patch_affine": patch_affine,
            "raw_orientation": meta["orientation"],
            "spacing_mm": meta["spacing_mm"],
        }

    def common_case_patch(self, case_id: str) -> Dict:
        """Return one common MTL-centered patch for prompt counterfactuals."""
        record = next(x for x in self.cases if x.case_id == case_id)
        image, label, affine, meta = self._case(record)
        image_patch, label_patch, patch_affine = _make_centered_patch(
            image, label, affine, None, self.patch_size
        )
        return {
            "image": torch.from_numpy(image_patch[None]),
            "label_patch": label_patch,
            "patch_affine": patch_affine,
            "case_id": case_id,
            "raw_orientation": meta["orientation"],
            "spacing_mm": meta["spacing_mm"],
        }


def audit_cases(cases: Sequence[CaseRecord]) -> List[Dict]:
    rows = []
    pairs = [
        ("hippocampus", 2, 1),
        ("entorhinal cortex", 4, 3),
        ("parahippocampal gyrus", 6, 5),
        ("amygdala", 8, 7),
    ]
    for record in cases:
        image, label, affine, meta = _load_canonical_case(record)
        row = {
            "case_id": record.case_id,
            "orientation": meta["orientation"],
            "shape": "x".join(map(str, image.shape)),
            "spacing_mm": "x".join(f"{x:.6f}" for x in meta["spacing_mm"]),
            "geometry_exact": True,
        }
        for name, left_id, right_id in pairs:
            left = np.argwhere(label == left_id).mean(axis=0)
            right = np.argwhere(label == right_id).mean(axis=0)
            left_x = float((affine @ np.r_[left, 1.0])[0])
            right_x = float((affine @ np.r_[right, 1.0])[0])
            row[f"{name}_left_x_mm"] = left_x
            row[f"{name}_right_x_mm"] = right_x
            row[f"{name}_left_is_left"] = bool(left_x < right_x)
        rows.append(row)
    return rows
