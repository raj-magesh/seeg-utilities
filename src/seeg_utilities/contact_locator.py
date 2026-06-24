from __future__ import annotations

import string
from typing import TYPE_CHECKING

import ipywidgets
import k3d
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import seaborn as sns
from hough_3d_line_detector import (
    Hough3DLineDetector,
    compute_distances_to_line,
    convert_parameters_to_coordinates,
    find_nearest_points_on_line,
)
from matplotlib import pyplot as plt
from nibabel.affines import apply_affine

from ._k3d_utilities import create_k3d_volume, set_k3d_camera

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from nibabel.spatialimages import SpatialImage

FloatArray = npt.NDArray[np.floating]


class ContactLocator:
    def __init__(
        self,
        *,
        ct: SpatialImage,
        brain_mask: SpatialImage,
        n_electrodes: int,
        contact_separation_in_mm: float,
        color_palette: sns.color_palette | None = None,
    ) -> None:
        self.ct = ct

        self.n_electrodes = n_electrodes
        self.contact_separation_in_mm = contact_separation_in_mm

        if color_palette is None:
            color_palette = sns.color_palette("colorblind", n_colors=n_electrodes)
        self.colors = color_palette.as_hex()

        self.ct_data = self.ct.get_fdata().ravel()
        self.brain_mask = brain_mask.get_fdata().astype(np.bool_).ravel()
        self.voxel_positions = apply_affine(
            ct.affine,
            np.indices(ct.shape, sparse=False).reshape(3, -1).T,
        )

        self._contact_positions = n_electrodes * [None]

    def fit(
        self,
        *,
        thresholds: tuple[float, float] = (2_000, 5_000),
        hough_threshold: float | None = None,
        max_distance_to_electrode_in_mm: float = 1.25,
        min_contact_separation_in_mm: float | None = None,
        kde_bandwidth_in_mm: float | None = None,
        kde_height_threshold: float = 0.05,
        kde_n_samples: int = 1_000,
    ) -> Figure:
        self._initialize_qc_figure()
        self._initialize_ct_figure(thresholds=thresholds)

        kwargs = {
            "disabled": False,
            "continuous_update": False,
            "style": {"description_width": "initial"},
            "layout": ipywidgets.Layout(display="flex", width="75%"),
        }
        ipywidgets.interact(
            self._fit,
            thresholds=ipywidgets.FloatRangeSlider(
                value=thresholds,
                min=-1_000,
                max=20_000,
                step=100,
                description="CT intensity thresholds (Hounsfield Units)",
                **kwargs,
            ),
            hough_threshold=ipywidgets.FloatSlider(
                value=hough_threshold or self.contact_separation_in_mm,
                min=0.1,
                max=10,
                step=0.01,
                description="threshold for Hough assignment of point to line (mm)",
                readout_format=".2f",
                **kwargs,
            ),
            max_distance_to_electrode_in_mm=ipywidgets.FloatSlider(
                value=max_distance_to_electrode_in_mm,
                min=0,
                max=10,
                step=0.01,
                description="maximum distance to electrode (mm)",
                readout_format=".2f",
                **kwargs,
            ),
            min_contact_separation_in_mm=ipywidgets.FloatSlider(
                value=min_contact_separation_in_mm
                or 0.75 * self.contact_separation_in_mm,
                min=0,
                max=2 * self.contact_separation_in_mm,
                step=0.1,
                description="minimum allowed contact separation (mm)",
                readout_format=".1f",
                **kwargs,
            ),
            kde_bandwidth_in_mm=ipywidgets.FloatSlider(
                # default to making 6 * sigma = contact_separation (i.e. 3 SD)
                value=kde_bandwidth_in_mm or self.contact_separation_in_mm / 6,
                min=0.1,
                max=1,
                step=0.01,
                description="bandwidth of Gaussian kernel used for KDE (mm)",
                readout_format=".2f",
                **kwargs,
            ),
            kde_height_threshold=ipywidgets.FloatSlider(
                value=kde_height_threshold,
                min=0,
                max=1,
                step=0.01,
                description="minimum KDE height for peak detection",
                readout_format=".2f",
                **kwargs,
            ),
            kde_n_samples=ipywidgets.IntSlider(
                value=kde_n_samples,
                min=1_000,
                max=5_000,
                step=100,
                description="number of samples for KDE evaluation",
                **kwargs,
            ),
        )

    def _fit(
        self,
        thresholds: tuple[float, float],
        hough_threshold: float,
        min_contact_separation_in_mm: float,
        kde_bandwidth_in_mm: float,
        kde_height_threshold: float,
        kde_n_samples: int,
        max_distance_to_electrode_in_mm: float,
    ) -> None:
        points_within_thresholds = self.voxel_positions[
            (self.ct_data > thresholds[0])
            & (self.ct_data < thresholds[1])
            & self.brain_mask
        ]
        hough_3d_line_detector = Hough3DLineDetector(
            max_lines=self.n_electrodes,
            threshold=hough_threshold,
        )
        electrodes = hough_3d_line_detector(points_within_thresholds)
        points_line_membership = np.argmin(
            np.stack(
                [
                    compute_distances_to_line(points_within_thresholds, line=electrode)
                    for electrode in electrodes
                ],
                axis=0,
            ),
            axis=0,
        )

        contact_positions = []
        for i_electrode in range(self.n_electrodes):
            points = points_within_thresholds[points_line_membership == i_electrode]

            contact_positions_, ts, kde, peak_indices, endpoints = (
                self._run_on_single_electrode(
                    points=points,
                    min_contact_separation_in_mm=min_contact_separation_in_mm,
                    kde_bandwidth_in_mm=kde_bandwidth_in_mm,
                    kde_height_threshold=kde_height_threshold,
                    kde_n_samples=kde_n_samples,
                    max_distance_to_electrode_in_mm=max_distance_to_electrode_in_mm,
                )
            )
            contact_positions.append(contact_positions_)

            self._update_qc_figure(
                i_electrode=i_electrode,
                ts=ts,
                kde=kde,
                kde_height_threshold=kde_height_threshold,
                peak_indices=peak_indices,
            )
            self._update_ct_figure(
                i_electrode=i_electrode,
                contact_positions=contact_positions_,
                thresholds=thresholds,
                endpoints=endpoints,
                max_distance_to_electrode_in_mm=max_distance_to_electrode_in_mm,
                points=points,
            )

        self.positions = pd.concat(
            [
                pd.DataFrame(positions, columns=["x", "y", "z"]).assign(
                    electrode=string.ascii_uppercase[i_electrode],
                    contact=np.arange(len(positions)),
                )
                for i_electrode, positions in enumerate(contact_positions)
            ],
            axis=0,
        ).set_index(["electrode", "contact"])

    def _run_on_single_electrode(
        self,
        *,
        points: FloatArray,
        min_contact_separation_in_mm: float,
        kde_bandwidth_in_mm: float,
        kde_height_threshold: float,
        kde_n_samples: int,
        max_distance_to_electrode_in_mm: float,
    ):
        # find the best-fit line (a, b) for the points assigned to the electrode
        a = points.mean(axis=0)
        _, _, v_t = np.linalg.svd(points - a)
        b = v_t[0, :]

        # slightly extend the parameter range describing the electrode shaft
        t = find_nearest_points_on_line(points, line=(a, b))
        t_min = t.min() - min_contact_separation_in_mm
        t_max = t.max() + min_contact_separation_in_mm

        endpoints = convert_parameters_to_coordinates(
            np.array([t_min, t_max]),
            line=(a, b),
        )

        # check if b points inward (i.e. t_max is closer to center of volume
        # than t_min) if so, flip contacts
        if np.diff(np.linalg.norm(endpoints, axis=-1)) < 0:
            b = -b
            t_min, t_max = -t_max, -t_min

        # find all points that are within a distance of the electrode shaft
        t = find_nearest_points_on_line(self.voxel_positions, line=(a, b))
        d = compute_distances_to_line(self.voxel_positions, line=(a, b))

        filter_ = (t >= t_min) & (t <= t_max) & (d <= max_distance_to_electrode_in_mm)
        t = t[filter_]

        weights = self.ct_data[filter_]
        weights -= weights.min()
        weights /= weights.sum()

        # compute a KDE of the parameter distribution
        # weighted by the CT brightness
        ts = np.linspace(t_min, t_max, kde_n_samples)
        kde_estimator = scipy.stats.gaussian_kde(
            t,
            bw_method=kde_bandwidth_in_mm / t.std(),
            weights=weights,
        )
        kde = kde_estimator(ts)
        kde = (kde - kde.min()) / (kde.max() - kde.min())

        # identify peaks of KDE, corresponding to contact locations
        peak_indices, _ = scipy.signal.find_peaks(
            kde,
            distance=(min_contact_separation_in_mm / (t_max - t_min) * kde_n_samples),
            height=kde_height_threshold,
        )
        contact_positions = convert_parameters_to_coordinates(
            ts[peak_indices],
            line=(a, b),
        )

        return contact_positions, ts, kde, peak_indices, endpoints

    def _initialize_qc_figure(self) -> None:
        self._qc_fig, self._axes_qc = plt.subplots(
            nrows=2,
            ncols=self.n_electrodes,
            figsize=(self.n_electrodes * 3, 6),
            sharex="col",
            sharey="row",
        )
        axes = self._axes_qc

        self._axes_qc_elements = {
            "kde": [],
            "threshold": [],
            "peaks": [],
            "separation": [],
        }
        for i_electrode in range(self.n_electrodes):
            ax = axes[0, i_electrode]

            kde = ax.plot([0, 1], [0, 0], c=self.colors[i_electrode])[0]
            self._axes_qc_elements["kde"].append(kde)

            threshold = ax.axhline(0, c="k", ls="--")
            self._axes_qc_elements["threshold"].append(threshold)

            indices = np.linspace(0, 1, 10)
            peaks = ax.scatter(
                indices,
                np.ones(len(indices)),
                marker="x",
                c="k",
            )
            self._axes_qc_elements["peaks"].append(peaks)

            separation = np.diff(indices)
            ax.set_title(
                f"{string.ascii_uppercase[i_electrode]} | {len(indices)} contacts\n"
                f"{separation.mean():.2f} +/- {separation.std():.2f} mm",
            )
            ax.spines[["top", "right"]].set_visible(False)

            ax = axes[1, i_electrode]
            x = indices[:-1] + separation / 2
            separation = ax.plot(
                x,
                self.contact_separation_in_mm * np.ones(len(x)),
                c=self.colors[i_electrode],
                marker="o",
            )[0]
            self._axes_qc_elements["separation"].append(separation)

            ax.axhline(self.contact_separation_in_mm, ls="--", c="k")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylim(
                bottom=self.contact_separation_in_mm - 0.75,
                top=self.contact_separation_in_mm + 0.75,
            )

        axes[0, 0].set_ylabel("CT intensity")
        axes[1, 0].set_ylabel("contact separation (mm)")

        self._qc_fig.supxlabel("position along electrode (mm)")

    def _update_qc_figure(
        self,
        i_electrode: int,
        ts: FloatArray,
        kde: FloatArray,
        kde_height_threshold: float,
        peak_indices: FloatArray,
    ) -> None:
        self._axes_qc_elements["kde"][i_electrode].set_data(ts, kde)
        self._axes_qc_elements["threshold"][i_electrode].set_ydata([
            kde_height_threshold,
            kde_height_threshold,
        ])
        self._axes_qc_elements["peaks"][i_electrode].set_offsets(
            np.stack([ts[peak_indices], kde[peak_indices]], axis=-1),
        )
        self._axes_qc[0, i_electrode].set_xlim(left=ts.min() - 3, right=ts.max() + 3)
        self._axes_qc[0, i_electrode].set_ylim(bottom=-0.05, top=1.05)

        separation = np.diff(ts[peak_indices])
        x = ts[peak_indices[:-1]] + separation / 2

        self._axes_qc_elements["separation"][i_electrode].set_data(x, separation)
        self._axes_qc[0, i_electrode].set_title(
            f"{string.ascii_uppercase[i_electrode]} | {len(peak_indices)} contacts\n"
            f"{separation.mean():.2f} +/- {separation.std():.2f} mm",
        )

    def _initialize_ct_figure(self, thresholds: tuple[float, float]) -> None:
        self._ct_fig = k3d.plot(
            height=1000,
            grid_visible=False,
            camera_mode="orbit",
        )
        self._contacts_3d = self.n_electrodes * [None]
        self._electrodes_3d = self.n_electrodes * [None]
        self._points_3d = self.n_electrodes * [None]

        self._ct_volume = create_k3d_volume(
            self.ct,
            color_map=k3d.colormaps.matplotlib_color_maps.Gray_r,
            name="CT",
            color_range=list(thresholds),
            alpha_coef=10,
        )
        self._ct_fig += self._ct_volume

        set_k3d_camera(self._ct_fig)

        self._ct_fig.display()

    def _update_ct_figure(
        self,
        *,
        i_electrode: int,
        contact_positions: FloatArray,
        thresholds: tuple[float, float],
        endpoints: FloatArray,
        max_distance_to_electrode_in_mm: float,
        points: FloatArray,
    ) -> None:
        self._ct_volume.color_range = list(thresholds)
        self._ct_volume.push_data("color_range")

        color = int(self.colors[i_electrode][1:], base=16)

        if self._contacts_3d[i_electrode] is not None:
            self._ct_fig -= self._contacts_3d[i_electrode]

        contacts = k3d.points(
            positions=contact_positions,
            name=f"{string.ascii_uppercase[i_electrode]} (contacts)",
            point_size=2,
            shader="flat",
            opacity=0.5,
            color=color,
        )
        self._ct_fig += contacts
        self._contacts_3d[i_electrode] = contacts

        if self._electrodes_3d[i_electrode] is not None:
            self._ct_fig -= self._electrodes_3d[i_electrode]

        electrode = k3d.line(
            vertices=endpoints,
            color=color,
            opacity=0.1,
            name=f"{string.ascii_uppercase[i_electrode]} (electrode)",
            width=2 * max_distance_to_electrode_in_mm,
            shader="mesh",
        )
        self._ct_fig += electrode
        self._electrodes_3d[i_electrode] = electrode

        if self._points_3d[i_electrode] is not None:
            self._ct_fig -= self._points_3d[i_electrode]

        points_ = k3d.points(
            positions=points,
            name=f"{string.ascii_uppercase[i_electrode]} (points)",
            point_size=0.5,
            shader="flat",
            opacity=0.5,
            color=color,
        )
        self._ct_fig += points_
        self._points_3d[i_electrode] = points_
