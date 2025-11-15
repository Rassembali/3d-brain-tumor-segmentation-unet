import os
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

from .bbox import get_brain_bbox
from .utils_io import load_nifti_sitk


def resample_image_sitk(image, new_spacing=(1,1,1), interpolator=sitk.sitkLinear):
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    new_spacing = np.array(new_spacing)
    new_size = (original_size * (original_spacing / new_spacing)).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(0)

    new_img = resampler.Execute(image)
    new_vol = sitk.GetArrayFromImage(new_img)

    return new_vol.astype(np.float32)


def normalize_brain(vol):
    mask = vol > 0
    if np.sum(mask) == 0:
        return vol

    mean = vol[mask].mean()
    std = vol[mask].std() + 1e-8

    out = (vol - mean) / std
    out[~mask] = 0
    return out.astype(np.float32)


def resize_volume(vol, target_shape=(128,128,128)):
    vol_t = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(vol_t, size=target_shape, mode="trilinear", align_corners=False)
    return resized.squeeze().numpy().astype(np.float32)


def resize_mask(mask, target_shape=(128,128,128)):
    m = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(m, size=target_shape, mode="nearest")
    return resized.squeeze().numpy().astype(np.uint8)


def preprocess_subject(sid, subject_path, target_shape=(128,128,128)):
    """
    Full preprocessing pipeline:
    - crop
    - resample
    - normalize
    - resize
    - remap labels (4 → 3)
    """

    modalities = ["flair", "t1", "t1ce", "t2"]
    vols = []
    sitk_mods = {}

    for mod in modalities:
        path = os.path.join(subject_path, f"{sid}_{mod}.nii.gz")
        img = sitk.ReadImage(path)
        sitk_mods[mod] = img

    flair_np = sitk.GetArrayFromImage(sitk_mods["flair"])
    z1, z2, y1, y2, x1, x2 = get_brain_bbox(flair_np)

    for mod in modalities:
        vol_np = sitk.GetArrayFromImage(sitk_mods[mod])
        vol_np = vol_np[z1:z2+1, y1:y2+1, x1:x2+1]

        cropped_sitk = sitk.GetImageFromArray(vol_np)
        cropped_sitk.SetSpacing(sitk_mods[mod].GetSpacing())

        resampled = resample_image_sitk(cropped_sitk, new_spacing=(1,1,1))
        resampled = normalize_brain(resampled)
        resized = resize_volume(resampled, target_shape)

        vols.append(resized)

    seg_path = os.path.join(subject_path, f"{sid}_seg.nii.gz")
    seg_np = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    seg_np = seg_np[z1:z2+1, y1:y2+1, x1:x2+1]

    seg_np[seg_np == 4] = 3
    seg_resized = resize_mask(seg_np, target_shape)

    X = np.stack(vols)  # (4,128,128,128)
    Y = seg_resized     # (128,128,128)

    return X.astype(np.float32), Y.astype(np.uint8)
