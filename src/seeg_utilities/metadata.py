"""Instructions for digitizing sEEG electrode metadata."""

import itertools
import runpy
from typing import TYPE_CHECKING

import mne
import numpy as np
import pandas as pd
import scipy

if TYPE_CHECKING:
    from pathlib import Path


def create_dig_montage(
    *,
    subject: str,
    eeg_channels: list[str],
    eeg_montage: str = "standard_1020",
    dataset_home: Path,
) -> mne.channels.DigMontage:
    # read channel locations in ACPC space
    channel_positions = pd.read_csv(
        dataset_home
        / "sourcedata"
        / f"sub-{subject}"
        / "ieeg"
        / f"sub-{subject}_space-ACPC_electrodes.tsv",
        delimiter="\t",
    )

    # compute fiducials in subject's MRI space
    fiducials = dict(
        zip(
            ("LPA", "nasion", "RPA"),
            mne.coreg.get_mni_fiducials(
                subject=f"sub-{subject}",
                subjects_dir=dataset_home / "derivatives" / "freesurfer",
            ),
            strict=True,
        ),
    )
    head_size = 0.6 * np.linalg.norm(fiducials["LPA"]["r"] - fiducials["RPA"]["r"])

    eeg_positions = mne.channels.make_standard_montage(
        eeg_montage,
        head_size=head_size,
    ).get_positions()["ch_pos"]

    return mne.channels.make_dig_montage(
        ch_pos=dict(
            zip(
                channel_positions["name"],
                channel_positions[["x", "y", "z"]].to_numpy() / 1e3,
                strict=True,
            ),
        )
        | {eeg_channel: eeg_positions[eeg_channel] for eeg_channel in eeg_channels},
        coord_frame="mri",
        **{label.lower(): fiducial["r"] for label, fiducial in fiducials.items()},
    )


def load_electrode_metadata(
    metadata: Path,
    jackbox: Path,
    *,
    exclude_ground: bool = True,
    exclude_reference: bool = False,
) -> pd.DataFrame:
    electrode_metadata = pd.read_csv(
        jackbox,
        sep=",",
        names=["electrode", "contact", "channel_type"],
    )
    extra = runpy.run_path(str(metadata))

    electrode_metadata = pd.concat(
        [
            electrode_metadata,
            pd.DataFrame(
                [
                    extra["GROUND_CONTACT"],
                    extra["REFERENCE_CONTACT"],
                ],
                columns=electrode_metadata.columns,
            ),
        ],
        axis=0,
        ignore_index=True,
    )

    electrode_metadata = (
        electrode_metadata
        .astype({"contact": np.uint8})
        .assign(
            ch_name=extra["CHANNEL_NAMES"] + ["ground", "reference"],
            label=lambda x: (
                x["electrode"] + "-" + (1 + x["contact"]).astype(str).str.zfill(2)
            ),
            ground=pd.col("ch_name") == "ground",
            reference=pd.col("ch_name") == "reference",
        )
        .set_index(["electrode", "contact"])
        .sort_index(level=["electrode", "contact"])
    )

    # add bad contacts
    electrode_metadata.loc[extra["BAD_CONTACTS"].keys(), "bad"] = True
    electrode_metadata.loc[extra["BAD_CONTACTS"].keys(), "details"] = list(
        extra["BAD_CONTACTS"].values(),
    )

    if exclude_ground:
        electrode_metadata = electrode_metadata.loc[~electrode_metadata["ground"]]
    if exclude_reference:
        electrode_metadata = electrode_metadata.loc[~electrode_metadata["reference"]]

    return electrode_metadata


def compute_adjacency(ch_names: list[str]) -> scipy.sparse.csr_array:
    adjacency = np.zeros((len(ch_names), len(ch_names)), dtype=bool)
    for (i_channel_1, channel_1), (i_channel_2, channel_2) in itertools.product(
        enumerate(ch_names),
        repeat=2,
    ):
        electrode_1, contact_1 = channel_1.split("-")
        electrode_2, contact_2 = channel_2.split("-")

        contact_1, contact_2 = int(contact_1), int(contact_2)

        if (electrode_1 == electrode_2) and (contact_2 - contact_1 in {-1, 0, 1}):
            adjacency[i_channel_1, i_channel_2] = True

    return scipy.sparse.csr_array(adjacency)


def apply_bipolar_referencing(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    kwargs = {key: [] for key in ("anode", "cathode", "ch_name")}
    for electrode_1, electrode_2 in itertools.pairwise(sorted(raw.ch_names)):
        shaft_1, contact_1 = electrode_1.split("-")
        shaft_2, _ = electrode_2.split("-")
        if shaft_1 == shaft_2:
            if shaft_1 == "None":
                continue
            kwargs["anode"].append(electrode_1)
            kwargs["cathode"].append(electrode_2)
            kwargs["ch_name"].append(f"d{shaft_1}-{contact_1}")
    return mne.set_bipolar_reference(raw.load_data(), **kwargs)
