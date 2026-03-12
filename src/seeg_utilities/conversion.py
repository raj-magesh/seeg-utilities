import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def convert_ct_dicoms_to_nifti(
    filename: str,
    *,
    input_directory: Path,
    output_directory: Path,
) -> None:
    subprocess.run(
        [
            "/usr/bin/env",
            "dcm2niix",
            # create BIDS sidecar
            "-b",
            "y",
            # anonymize BIDS sidecar
            "-ba",
            "y",
            # output filename
            "-f",
            filename,
            # merge 2D slices
            "-m",
            "y",
            # output directory
            "-o",
            str(output_directory),
            # disable single-file mode
            "-s",
            "n",
            # don't crop unnecessary regions
            "-x",
            "n",
            # output compression
            "-z",
            "y",
            # input directory
            str(input_directory),
        ],
        check=False,
    )


def run_heudiconv(
    *,
    dicoms: list[Path],
    output_directory: Path,
    heuristic: Path,
    subject: str,
) -> None:
    _ = subprocess.run(
        [
            "/usr/bin/env",
            "heudiconv",
            "--files",
            *[str(filepath) for filepath in dicoms],
            "--outdir",
            str(output_directory),
            "--heuristic",
            str(heuristic),
            "--subjects",
            subject,
            "--converter",
            "dcm2niix",
            "--grouping",
            "all",
            "--bids",
            "notop",
            "--minmeta",
            "--overwrite",
            "--random-seed",
            "0",
        ],
        check=False,
    )
