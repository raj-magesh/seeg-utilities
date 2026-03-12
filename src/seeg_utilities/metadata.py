import runpy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


def load_electrode_metadata(
    filepath: Path,
    *,
    only_recording_electrodes: bool = True,
) -> pd.DataFrame:
    metadata = _collate_electrode_metadata(**{
        k.lower(): v
        for k, v in runpy.run_path(filepath).items()
        if k
        in {
            "ELECTRODE_LOCATIONS",
            "GROUND_ELECTRODE",
            "REFERENCE_ELECTRODE",
            "BAD_ELECTRODES",
            "JACKBOX_INDICES",
        }
    })
    if only_recording_electrodes:
        metadata = metadata.loc[metadata["jackbox_index"] >= 0]
    return metadata


def _collate_electrode_metadata(
    *,
    electrode_locations: dict[str, dict[frozenset[int], str]],
    ground_electrode: tuple[str, int],
    reference_electrode: tuple[str, int],
    bad_electrodes: dict[str, tuple[frozenset[int], str]],
    jackbox_indices: dict[str, frozenset[int]],
) -> pd.DataFrame:
    jackbox_electrodes = {
        shaft: range(sum(len(key) for key in locations))
        for shaft, locations in electrode_locations.items()
    }
    for shaft, electrode in (ground_electrode, reference_electrode):
        jackbox_electrodes[shaft] = sorted(
            set(jackbox_electrodes[shaft]) - {electrode},
        )

    jackbox_mapping = {
        shaft: dict(
            zip(jackbox_electrodes[shaft], jackbox_indices_, strict=False),
        )
        for shaft, jackbox_indices_ in jackbox_indices.items()
    }

    metadata = {
        "label": [],
        "shaft": [],
        "electrode": [],
        "location": [],
        "bad": [],
        "details": [],
        "jackbox_index": [],
    }

    for shaft, values in electrode_locations.items():
        for electrodes_, location in values.items():
            for electrode in electrodes_:
                jackbox_index_valid = (shaft, electrode) not in {
                    ground_electrode,
                    reference_electrode,
                }

                metadata["label"].append(f"{shaft}-{1 + electrode:02}")
                metadata["shaft"].append(shaft)
                metadata["electrode"].append(electrode)
                metadata["location"].append(location)
                metadata["jackbox_index"].append(
                    jackbox_mapping[shaft][electrode] if jackbox_index_valid else -1,
                )

                if not jackbox_index_valid:
                    metadata["bad"].append(True)
                    metadata["details"].append("")
                elif shaft in bad_electrodes:
                    bad_electrodes_, details = bad_electrodes[shaft]
                    metadata["bad"].append(electrode in bad_electrodes_)
                    metadata["details"].append(
                        details if electrode in bad_electrodes_ else "",
                    )
                else:
                    metadata["bad"].append(False)
                    metadata["details"].append("")

    return (
        pd
        .DataFrame(metadata)
        .astype({
            "shaft": "string",
            "location": "string",
            "electrode": np.uint8,
            "label": "string",
            "details": "string",
        })
        .set_index(["shaft", "electrode"])
        .sort_values("jackbox_index")
        .assign(
            ground=lambda x: x.index == ground_electrode,
            reference=lambda x: x.index == reference_electrode,
        )
    )
