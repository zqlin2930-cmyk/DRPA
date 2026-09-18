from pathlib import Path
from typing import Sequence
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nibabel.orientations import io_orientation
from mtl_grounding_dataset import CaseRecord, PROMPTS, ROI_LABEL_IDS, _make_centered_patch

def load_official_reader_case(record: CaseRecord):
    raw_im=nib.load(record.image_path)
    raw_lb=nib.load(record.label_path)
    raw_image=np.asarray(raw_im.get_fdata(dtype=np.float32))
    raw_label=np.asarray(raw_lb.get_fdata())
    if raw_label.ndim==4 and raw_label.shape[-1]==1: raw_label=raw_label[...,0]
    if raw_image.shape!=raw_label.shape or not np.allclose(raw_im.affine,raw_lb.affine,rtol=0,atol=1e-4):
        raise ValueError(f'raw MRI/label geometry mismatch: {record.case_id}')
    if nib.aff2axcodes(nib.as_closest_canonical(raw_im).affine)!=('R','A','S'):
        raise ValueError(f'raw MRI not canonical-RAS-compatible: {record.case_id}')
    transform=io_orientation(raw_im.affine)
    reor_img=raw_im.as_reoriented(transform)
    reor_label=nib.Nifti1Image(raw_label,raw_lb.affine).as_reoriented(transform)
    perm=np.asarray([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]],dtype=float)
    reader_affine=reor_img.affine @ perm
    reader_image=np.asarray(reor_img.get_fdata(dtype=np.float32)).transpose((2,1,0))
    reader_label=np.asarray(reor_label.get_fdata()).transpose((2,1,0))
    if reader_image.shape!=reader_label.shape:
        raise ValueError(f'official reader shape mismatch: {record.case_id}')
    if not np.isfinite(reader_image).all():
        raise ValueError(f'nonfinite MRI: {record.case_id}')
    required=set(ROI_LABEL_IDS.values())
    if not required.issubset(set(np.unique(reader_label.astype(np.int16)).tolist())):
        raise ValueError(f'missing ROI IDs after official reorientation: {record.case_id}')
    return reader_image,reader_label.astype(np.int16),reader_affine.astype(np.float64),{
        'raw_shape':tuple(int(x) for x in raw_image.shape),
        'reader_shape':tuple(int(x) for x in reader_image.shape),
        'source_canonical_ras':True,
        'reader_contract':'NibabelIOWithReorient + transpose(2,1,0)',
        'spacing_mm':tuple(float(x) for x in nib.affines.voxel_sizes(reader_affine)),
    }

class OfficialGroupedPatchDataset(Dataset):
    def __init__(self,cases,patch_size=(192,192,192)):
        self.cases=list(cases); self.patch_size=tuple(int(x) for x in patch_size); self._cache={}
    def __len__(self): return len(self.cases)
    def __getitem__(self,index):
        rec=self.cases[int(index)]
        if rec.case_id not in self._cache:
            image,label,aff,meta=load_official_reader_case(rec)
            ip,lp,paff=_make_centered_patch(image,label,aff,None,self.patch_size)
            self._cache[rec.case_id]=(ip,lp,paff,meta)
        ip,lp,paff,meta=self._cache[rec.case_id]
        masks=np.stack([(lp==ROI_LABEL_IDS[p]).astype(np.float32) for p in PROMPTS],axis=0)
        return {'image':torch.from_numpy(ip[None]),'mask':torch.from_numpy(masks),'case_id':rec.case_id,'patch_affine':paff,'meta':meta}

