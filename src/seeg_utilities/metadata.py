"""Instructions for digitizing sEEG electrode metadata."""

import runpy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


def load_electrode_metadata(
    filepath: Path,
    *,
    exclude_ground: bool = True,
    exclude_reference: bool = False,
) -> pd.DataFrame:
    metadata = _collate_electrode_metadata(**{
        k.lower(): v
        for k, v in runpy.run_path(filepath).items()
        if k
        in {
            "CHANNEL_NAMES",
            "CONTACT_LOCATIONS",
            "GROUND_CONTACT",
            "REFERENCE_CONTACT",
            "BAD_CONTACTS",
        }
    })
    if exclude_ground:
        metadata = metadata.loc[~metadata["ground"]]
    if exclude_reference:
        metadata = metadata.loc[~metadata["reference"]]
    return metadata


def _collate_electrode_metadata(
    *,
    channel_names: list[str],
    contact_locations: dict[tuple(str, int), str],
    ground_contact: tuple[str, int],
    reference_contact: tuple[str, int],
    bad_contacts: dict[str, tuple[list[int], str]],
) -> pd.DataFrame:
    metadata = {
        "electrode": [],
        "contact": [],
        "location": [],
        "label": [],
        "bad": [],
        "details": [],
        "jackbox_index": [],
        "ch_name": [],
    }

    jackbox_index = 0
    for (electrode, contact_index), location in contact_locations.items():
        metadata["electrode"].append(electrode)
        metadata["contact"].append(contact_index)
        metadata["location"].append(location)
        metadata["label"].append(f"{electrode}-{1 + contact_index:02}")
        metadata["bad"].append((electrode, contact_index) in bad_contacts)
        metadata["details"].append(
            bad_contacts[electrode, contact_index] if metadata["bad"][-1] else ""
        )
        if (electrode, contact_index) in {
            ground_contact,
            reference_contact,
        }:
            metadata["ch_name"].append(
                "ground"
                if (electrode, contact_index) == ground_contact
                else "reference"
            )
            metadata["jackbox_index"].append(-1)
        else:
            metadata["ch_name"].append(channel_names[jackbox_index])
            metadata["jackbox_index"].append(jackbox_index)
            jackbox_index += 1

    return (
        pd
        .DataFrame(metadata)
        .astype({"contact": np.uint8})
        .set_index(["electrode", "contact"])
        .sort_index(level=["electrode", "contact"])
        .assign(
            ground=lambda x: x.index == ground_contact,
            reference=lambda x: x.index == reference_contact,
        )
    )
