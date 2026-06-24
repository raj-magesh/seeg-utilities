from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import xipppy as xp
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

PARALLEL_PORT_INDEX = 4

SamplingRate = Literal[1_000, 2_000, 7_500, 30_000]


def send_trigger(trigger: int = 0) -> None:
    with xp.xipppy_open(use_tcp=True):
        try:
            xp.digout(
                outputs=[PARALLEL_PORT_INDEX],
                values=[trigger],
            )
            logger.info(
                "Sent trigger {trigger} to parallel port",
                trigger=trigger,
            )
        except Exception:
            logger.warning(
                "Failed to send trigger {trigger} to parallel port",
                trigger=trigger,
            )


def start_recording(*, operator_id: int = 129, filepath_base: Path, **kwargs) -> None:
    with xp.xipppy_open(use_tcp=True):
        xp.add_operator(oper_addr=operator_id)
        logger.info("Added operator {operator_id}", operator_id=operator_id)
        filepath_base.parent.mkdir(exist_ok=True, parents=True)
        status, *_ = xp.trial(
            oper=operator_id,
            file_name_base=str(filepath_base),
            **kwargs,
        )

        if status == "recording":
            logger.info(
                "sEEG recording started automatically with output saved at {filepath_base}",
                filepath_base=filepath_base,
            )
        else:
            logger.error(
                "sEEG recording was not started automatically. Start the recording manually in the Trellis GUI now, using the filepath: {filepath_base}",
                filepath_base=filepath_base,
            )


def stop_recording(*, operator_id: int = 129) -> None:
    with xp.xipppy_open(use_tcp=True):
        xp.trial(oper=operator_id, status="paused")
        status, *_ = xp.trial(oper=operator_id, status="stopped")
        if status == "stopped":
            logger.info("sEEG recording stopped.")
        else:
            logger.warning(
                "sEEG recording may not have been stopped automatically. Stop the recording manually in the Trellis GUI now.",
            )


def read_analog_inputs(n_points: int, sampling_rate: SamplingRate, **kwargs):
    with xp.xipppy_open(use_tcp=True):
        analog_inputs = xp.list_elec(fe_type="analog")

        match sampling_rate:
            case 30_000:
                func = xp.cont_raw
            case 7_500:
                func = xp.cont_hifreq
            case 2_000:
                func = xp.cont_hires
            case 1_000:
                func = xp.cont_lfp
            case _:
                error = f"Unsupported sampling rate {sampling_rate}"
                raise ValueError(error)

        data, t = func(npoints=n_points, elecs=analog_inputs, **kwargs)

        return analog_inputs, data, t


with xp.xipppy_open(use_tcp=True):
    logger.info(
        "Connected to RippleNeuroMed Explorer Summit | {version_info}",
        version_info=xp.get_version(),
    )
