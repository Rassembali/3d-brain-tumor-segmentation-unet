import os
import SimpleITK as sitk
import numpy as np

def load_nifti_sitk(path):
    """
    Load NIfTI file using SimpleITK.
    Returns:
        volume (numpy array, shape: z,y,x),
        spacing (tuple: x,y,z)
    """
    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img)  # (z,y,x)
    spacing = img.GetSpacing()
    return vol.astype(np.float32), spacing


def save_numpy(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)
