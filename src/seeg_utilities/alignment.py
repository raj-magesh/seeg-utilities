from typing import TYPE_CHECKING

import mne.transforms
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def align_ct_to_t1w(
    *,
    ct: nib.spatialimages.SpatialImage,
    t1w: nib.spatialimages.SpatialImage,
) -> nib.spatialimages.SpatialImage:
    affine = mne.transforms.compute_volume_registration(ct, t1w, pipeline="rigids")[0]
    return mne.transforms.apply_volume_registration(ct, t1w, affine)


def plot_overlay(
    *,
    ct: nib.spatialimages.SpatialImage,
    t1w: nib.spatialimages.SpatialImage,
    threshold: float | None = None,
) -> Figure:
    t1w_data = nib.orientations.apply_orientation(
        np.asarray(t1w.dataobj),
        nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(t1w.affine)),
    )
    ct_data = nib.orientations.apply_orientation(
        np.asarray(ct.dataobj),
        nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(ct.affine)),
    )

    if threshold is not None:
        ct_data[ct_data < np.quantile(ct_data, threshold)] = np.nan

    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    for i_ax, ax in enumerate(axes.flat):
        ax.imshow(
            np.take(t1w_data, [t1w_data.shape[i_ax] // 2], axis=i_ax).squeeze().T,
            cmap="gray",
        )
        ax.imshow(
            np.take(ct_data, [ct_data.shape[i_ax] // 2], axis=i_ax).squeeze().T,
            cmap="plasma",
            alpha=0.75,
        )
        ax.invert_yaxis()
        ax.axis("off")

    fig.tight_layout()
    return fig
