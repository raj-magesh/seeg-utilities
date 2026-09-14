import itertools

import mne
import numpy as np
import numpy.typing as npt
import scipy
from loguru import logger
from scipy.ndimage import median_filter

POWER_LINE_FREQUENCY_IN_HZ = 60


def find_binary_state_transitions(
    signal: npt.NDArray[np.floating],
    *,
    low: float,
    high: float,
    median_size: int = 1,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Return ON and OFF transition indices for a noisy two-state signal.

    low:  switch OFF when signal falls below this value
    high: switch ON when signal rises above this value
    """
    x = np.asarray(signal)

    if median_size > 1:
        x = median_filter(x, size=median_size)

    state = np.zeros(len(x), dtype=bool)
    on = False

    for idx, value in enumerate(x):
        if not on and value >= high:
            on = True
        elif on and value <= low:
            on = False

        state[idx] = on

    transitions = np.flatnonzero(np.diff(state.astype(int)) != 0) + 1

    on_transitions = transitions[state[transitions]]
    off_transitions = transitions[~state[transitions]]

    return on_transitions, off_transitions


def filter_raw_data(
    raw: mne.io.BaseRaw,
    *,
    l_freq: float | None = None,
    h_freq: float | None = None,
    notch_freq: float | None = None,
    **kwargs,
) -> mne.io.BaseRaw:
    if notch_freq is not None:
        raw = raw.load_data().notch_filter(
            freqs=notch_freq * np.arange(1, (raw.info["sfreq"] / 2) // notch_freq),
            **kwargs,
        )
    if (l_freq is not None) or (h_freq is not None):
        raw = raw.load_data().filter(l_freq=l_freq, h_freq=h_freq, **kwargs)
    return raw


def compute_adjacency_for_seeg_channels(ch_names: list[str]) -> scipy.sparse.csr_array:
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


def apply_bipolar_referencing_to_seeg_channels[Data: mne.io.BaseRaw | mne.BaseEpochs](
    inst: Data,
) -> Data:
    inst = inst.load_data()

    kwargs = {key: [] for key in ("anode", "cathode", "ch_name", "ch_info")}
    positions = inst.get_montage().get_positions()["ch_pos"]

    for electrode_1, electrode_2 in itertools.pairwise(
        sorted(inst.copy().pick("seeg", exclude="bads").ch_names),
    ):
        shaft_1, contact_1 = electrode_1.split("-")
        shaft_2, contact_2 = electrode_2.split("-")
        if shaft_1 == shaft_2:
            if shaft_1 == "None":
                continue
            kwargs["anode"].append(electrode_1)
            kwargs["cathode"].append(electrode_2)
            kwargs["ch_name"].append(f"d{shaft_1}-{contact_1}-{contact_2}")
            kwargs["ch_info"].append(
                {
                    "loc": np.concatenate(
                        [
                            np.mean(
                                np.stack(
                                    [positions[electrode_1], positions[electrode_2]],
                                    axis=0,
                                ),
                                axis=0,
                            ),
                            np.zeros(3),
                            np.full((6,), np.nan),
                        ],
                    ),
                },
            )
            if int(contact_2) - int(contact_1) != 1:
                logger.warning(
                    "Using non-neighboring contacts in bipolar referencing: {electrode_1}/{electrode_2}",
                    electrode_1=electrode_1,
                    electrode_2=electrode_2,
                )
    return mne.set_bipolar_reference(inst, **kwargs)
