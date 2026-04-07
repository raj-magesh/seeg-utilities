from typing import TYPE_CHECKING, Any

import k3d
import nibabel as nib
import numpy as np
from dipy.align import affine_registration
from k3d.colormaps import matplotlib_color_maps
from nilearn.plotting import view_img

from ._k3d_utilities import create_k3d_volume, set_k3d_camera

if TYPE_CHECKING:
    from nibabel.spatialimages import SpatialImage
    from nilearn.plotting.html_stat_map import StatMapView

CT_THRESHOLDS_IN_HU = (500, 5_000)


def align_ct_to_t1w(
    *,
    ct: SpatialImage,
    t1w: SpatialImage,
) -> SpatialImage:
    voxel_size = np.mean(ct.header.get_zooms())
    ct_aligned, _ = affine_registration(
        moving=ct,
        static=t1w,
        sigmas=[sigma / voxel_size for sigma in (3, 1, 0)],
        pipeline=["center_of_mass", "translation", "rigid"],
        optimizer_options={"gtol": 1e-4},
    )
    return nib.nifti1.Nifti1Image(ct_aligned, affine=t1w.affine, header=t1w.header)


def plot_2d_overlay(
    *,
    ct: SpatialImage,
    t1w: SpatialImage,
    **kwargs,
) -> StatMapView:
    kwargs_default = {
        "cut_coords": [0, 0, 0],
        "colorbar": False,
        "title": "CT overlaid on T1w",
        "threshold": None,
        "black_bg": True,
        "cmap": "plasma",
        "symmetric_cmap": False,
        "vmin": 300,
        "vmax": 1000,
        "width_view": 800,
        "opacity": 0.5,
        "dim": -1,
        "show_lr": False,
    }
    return view_img(
        ct,
        bg_img=t1w,
        **kwargs_default | kwargs,
    )


def plot_3d_overlay(
    *,
    ct: SpatialImage,
    t1w: SpatialImage,
    t1w_kwargs: dict[str, Any] | None = None,
    ct_kwargs: dict[str, Any] | None = None,
    camera_kwargs: dict[str, Any] | None = None,
) -> k3d.Plot:
    overlay_3d = k3d.plot(
        height=1024,
        grid_visible=False,
        camera_mode="orbit",
    )

    t1w_volume = create_k3d_volume(
        t1w,
        **(
            {
                "color_map": matplotlib_color_maps.Viridis_r,
                "name": "T1w",
            }
            | (t1w_kwargs or {})
        ),
    )
    overlay_3d += t1w_volume

    ct_volume = create_k3d_volume(
        ct,
        **(
            {
                "color_map": matplotlib_color_maps.Plasma,
                "name": "CT",
                "color_range": CT_THRESHOLDS_IN_HU,
            }
            | (ct_kwargs or {})
        ),
    )
    overlay_3d += ct_volume

    set_k3d_camera(overlay_3d, **(camera_kwargs or {}))

    return overlay_3d
