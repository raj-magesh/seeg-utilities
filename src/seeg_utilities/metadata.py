"""Instructions for digitizing sEEG electrode metadata."""

import runpy
from typing import TYPE_CHECKING

import mne
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


def create_dig_montage(
    *,
    subject: str,
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
                subjects_dir=dataset_home / "derivatives" / "fastsurfer",
            ),
            strict=True,
        ),
    )
    return mne.channels.make_dig_montage(
        ch_pos=dict(
            zip(
                channel_positions["name"],
                channel_positions[["x", "y", "z"]].to_numpy() / 1e3,
                strict=True,
            ),
        ),
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
