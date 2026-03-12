import string
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import seaborn as sns
from hough_3d_line_detector import (
    Hough3DLineDetector,
    convert_parameters_to_coordinates,
    find_nearest_points_on_line,
)
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure

FloatArray = npt.NDArray[np.floating]


class ContactLocator:
    def __init__(
        self,
        *,
        t1w: nib.spatialimages.SpatialImage,
        ct: nib.spatialimages.SpatialImage,
        brain_mask: nib.spatialimages.SpatialImage,
        n_electrodes: int,
        n_contacts_per_electrode: int,
        contact_area_in_mm_2: float,
        contact_length_in_mm: float,
        contact_separation_in_mm: float,
        color_palette: sns.color_palette | None = None,
        threshold: float | None = None,
    ) -> None:
        self.n_electrodes = n_electrodes
        self.n_contacts_per_electrode = n_contacts_per_electrode
        self.contact_area_in_mm_2 = contact_area_in_mm_2
        self.contact_length_in_mm = contact_length_in_mm
        self.contact_separation_in_mm = contact_separation_in_mm

        self.t1w = t1w
        self.ct = ct
        self.brain_mask = brain_mask
        self.shape = self.t1w.shape

        self.voxels = np.stack(
            np.meshgrid(*[np.arange(dim) for dim in self.shape]),
            axis=-1,
        ).reshape(-1, 3)
        self.voxel_size_in_mm = nib.affines.voxel_sizes(self.t1w.affine)
        self.n_voxels_expected = int(
            self.n_electrodes
            * self.n_contacts_per_electrode
            * (self.contact_area_in_mm_2 * self.contact_length_in_mm)
            / np.prod(self.voxel_size_in_mm)
        )

        self.ct_data = self.ct.get_fdata().copy()
        self.ct_data[~brain_mask.get_fdata().astype(np.bool_)] = np.nan
        self.ct_data -= np.nanmin(self.ct_data)
        self.ct_data /= np.nanmax(self.ct_data)

        self.ct_flat = self.ct_data[*np.unstack(self.voxels, axis=-1)]
        self.threshold = (
            np.sort(self.ct_flat[~np.isnan(self.ct_flat)])[-self.n_voxels_expected]
            if threshold is None
            else threshold
        )

        self.colors = (
            sns.color_palette("colorblind", n_colors=self.n_electrodes)
            if color_palette is None
            else color_palette
        )

    def plot_voxels_above_threshold(
        self,
        *,
        threshold: float | None = None,
        bins: int = 100,
    ) -> Figure:
        threshold = self.threshold if threshold is None else threshold

        fig = plt.figure(figsize=(10, 4))

        ax = fig.add_subplot(121)
        ax.hist(self.ct_flat, bins=bins, facecolor="gray")
        ax.axvline(threshold, c="k", ls="--")
        ax.set_yscale("log")
        ax.set_title(f"threshold={threshold:.3f}")
        ax.set_ylabel("number of voxels")
        ax.set_xlabel("intensity (normalized)")

        ax_3d = fig.add_subplot(122, projection="3d")
        ax_3d.scatter(
            *np.nonzero(self.ct_data > threshold),
            color="k",
            s=5,
            linewidths=0,
        )
        ax_3d.set_aspect("equal")

        return fig

    def fit(self, *, n_grid: int = 64, **kwargs):
        estimator = Hough3DLineDetector(max_lines=self.n_electrodes, n_grid=n_grid)
        voxels_above_threshold = np.stack(
            np.nonzero(self.ct_data > self.threshold), axis=-1
        )

        self.outputs = [
            [
                line,
                voxels,
                *self._run_on_single_electrode(line=line, voxels=voxels, **kwargs),
            ]
            for line, voxels in estimator(voxels_above_threshold)
        ]

        self.contact_positions = pd.concat(
            [
                pd.DataFrame(contact_positions, columns=["x", "y", "z"]).assign(
                    shaft=string.ascii_uppercase[i_line],
                    electrode=np.arange(len(contact_positions)),
                    label=lambda x: x.apply(
                        lambda y: f"{y['shaft']}-{y['electrode']:02}", axis=1
                    ),
                )
                for i_line, (_, _, contact_positions, _, _, _, _, _) in enumerate(
                    self.outputs
                )
            ],
            axis=0,
        ).set_index(["shaft", "electrode"])

    def _run_on_single_electrode(
        self,
        *,
        line: tuple[FloatArray, FloatArray],
        voxels: FloatArray,
        fraction_extend_parameter_range: float = 0.025,
        min_contact_separation_factor: float = 0.75,
        kde_bandwidth: float = 0.035,
        n_parameter_samples: int = 1_000,
        max_distance_to_electrode_in_mm: float = 4,
    ):
        # slightly extend the parameter range describing the electrode shaft
        t = find_nearest_points_on_line(voxels, line=line)
        t_min, t_max = t.min(), t.max()
        t_range = t_max - t_min
        t_min -= fraction_extend_parameter_range * t_range
        t_max += fraction_extend_parameter_range * t_range

        # find all voxels that are within a distance of the electrode shaft
        t, d = find_nearest_points_on_line(
            self.voxels, line=line, return_distances=True
        )
        filter_ = (
            (t >= t_min)
            & (t <= t_max)
            & (d <= max_distance_to_electrode_in_mm)
            & (~np.isnan(self.ct_flat))
        )

        # compute a KDE of the parameter distribution
        # weighted by the CT brightness
        ts = np.linspace(t_min, t_max, n_parameter_samples)
        weights = self.ct_flat[filter_]
        weights = scipy.special.softmax(weights)
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        kde = scipy.stats.gaussian_kde(
            t[filter_],
            bw_method=kde_bandwidth,
            weights=weights,
        )(ts)

        # identify peaks of KDE, corresponding to contact locations
        peak_indices, _ = scipy.signal.find_peaks(
            kde,
            distance=(
                min_contact_separation_factor
                * self.contact_separation_in_mm
                * n_parameter_samples
                / ((t_max - t_min) * np.linalg.norm(self.voxel_size_in_mm))
            ),
        )
        contact_positions = convert_parameters_to_coordinates(
            ts[peak_indices],
            line=line,
        )

        return contact_positions, ts, kde, peak_indices, filter_, weights

    def plot_contact_positions(self) -> Figure:
        fig = plt.figure(figsize=(12, 8))
        subfigures = fig.subfigures(nrows=2)

        axes = subfigures[0].subplots(ncols=self.n_electrodes)

        for i_line, (_, _, _, ts, kde, peak_indices, _, _) in enumerate(self.outputs):
            ax = axes[i_line]
            ax.plot(ts, kde, c=self.colors[i_line])
            ax.scatter(ts[peak_indices], kde[peak_indices], marker="x", c="k")
            separation = np.diff(ts[peak_indices])
            ax.set_title(
                f"{string.ascii_uppercase[i_line]} | {len(peak_indices)}\n"
                f"{separation.mean():.2f} +/- {separation.std():.2f} mm"
            )
            ax.axis("off")

        axes = subfigures[1].subplots(
            ncols=2,
            subplot_kw={"projection": "3d"},
        )

        for i_line, (
            _,
            voxels,
            contact_coordinates,
            _,
            _,
            _,
            filter_,
            weights,
        ) in enumerate(self.outputs):
            axes[0].scatter(
                *np.unstack(voxels, axis=-1),
                color=self.colors[i_line],
                s=5,
                alpha=0.25,
                linewidths=0,
            )

            axes[1].scatter(
                *np.unstack(self.voxels[filter_], axis=-1),
                color=self.colors[i_line],
                alpha=0.5 * weights,
                s=5,
                linewidths=0,
            )

            axes[1].scatter(
                *np.unstack(contact_coordinates, axis=-1),
                c=sns.dark_palette(
                    self.colors[i_line],
                    reverse=True,
                    n_colors=len(contact_coordinates),
                ),
                s=25,
                linewidths=0,
            )

            for i_contact, coordinates in enumerate(contact_coordinates):
                axes[1].text(
                    *coordinates,
                    s=f"{string.ascii_uppercase[i_line]}-{i_contact + 1:02}",
                    ha="center",
                    va="baseline",
                    c="gray",
                    alpha=0.5,
                    fontsize="xx-small",
                )

        for ax in axes.flat:
            ax.set_aspect("equal")
        return fig
