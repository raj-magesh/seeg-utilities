from typing import Literal

import k3d
import nibabel as nib
import numpy as np
from k3d.transform import get_bounds_fit_matrix, process_transform_arguments


def create_k3d_volume(img: nib.spatialimages.SpatialImage, /, **kwargs) -> k3d.Volume:
    # k3d plots an (x, y, z) array as (z, y, x), so I'm transposing it here
    # this doesn't seem to require a permutation of the affine matrix: not sure why
    # the outputs look perfect though
    volume = k3d.volume(
        img.get_fdata().transpose(2, 1, 0),
        **kwargs,
    )

    # k3d sets the default bounding box to (-0.5, 0.5)^3
    # this is fixed by applying the bounding box affine after the regular affine
    bounding_box_affine = get_bounds_fit_matrix(
        *np.stack([np.zeros(3), np.array(img.shape) - 1], axis=-1).ravel()
    )
    return process_transform_arguments(
        volume,
        model_matrix=img.affine @ bounding_box_affine,
    )


def set_k3d_camera(
    plot: k3d.Plot,
    *,
    view: Literal["coronal", "axial", "sagittal"] = "coronal",
    distance_factor: float = 2,
    axis_up: Literal[0, 1, 2] = 2,
) -> None:
    axes_up = np.zeros(3)
    axes_up[axis_up] = 1

    midpoints = tuple((plot.grid[dim + 3] + plot.grid[dim]) / 2 for dim in range(3))
    camera = [
        # initialize camera at center of volume
        *midpoints,
        # point camera towards center of volume
        *midpoints,
        # set one axis to be "up"
        *axes_up,
    ]

    # move camera away from center along some axis
    match view:
        case "coronal":
            dim = 1
        case "axial":
            dim = 2
        case "sagittal":
            dim = 0
        case _:
            raise ValueError

    camera[dim] = distance_factor * (plot.grid[dim + 3] - plot.grid[dim])
    plot.camera = camera
