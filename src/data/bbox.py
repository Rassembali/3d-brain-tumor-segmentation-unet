import numpy as np

def get_brain_bbox(volume):
    """
    Compute bounding box for non-zero region.
    volume shape: (Z, Y, X)
    """
    nonzero = np.where(volume != 0)
    if len(nonzero[0]) == 0:
        return (0, volume.shape[0], 0, volume.shape[1], 0, volume.shape[2])

    z1, z2 = nonzero[0].min(), nonzero[0].max()
    y1, y2 = nonzero[1].min(), nonzero[1].max()
    x1, x2 = nonzero[2].min(), nonzero[2].max()
    return (z1, z2, y1, y2, x1, x2)


def crop_volume(volume):
    """
    Crop MRI to brain region.
    """
    z1, z2, y1, y2, x1, x2 = get_brain_bbox(volume)
    return volume[z1:z2+1, y1:y2+1, x1:x2+1]
