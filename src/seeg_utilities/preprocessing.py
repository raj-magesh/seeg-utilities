from typing import Literal

import mne
import numpy as np
import numpy.typing as npt
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


def _z_score(x):
    return (
        (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)
        if ~np.any(x.std(axis=-1, keepdims=True) == 0)
        else x
    )


def _center(x):
    return x - x.mean(axis=-1, keepdims=True)


def normalize_raw_data(
    raw: mne.io.BaseRaw,
    *,
    mode: Literal["zscore", "mean", "robust"] | None = "zscore",
) -> mne.io.BaseRaw:
    match mode:
        case "zscore":
            return raw.load_data().apply_function(
                _z_score,
                channel_wise=True,
            )
        case "mean":
            return raw.load_data().apply_function(
                _center,
                channel_wise=True,
            )
        case None:
            return raw
        case _:
            raise ValueError


def filter_raw_data(
    raw: mne.io.Raw,
    *,
    l_freq: float | None = None,
    h_freq: float | None = None,
    notch_freq: float = POWER_LINE_FREQUENCY_IN_HZ,
) -> mne.io.Raw:
    return (
        raw
        .load_data()
        .notch_filter(
            freqs=notch_freq * np.arange(1, (raw.info["sfreq"] / 2) // notch_freq),
        )
        .filter(l_freq=l_freq, h_freq=h_freq)
    )
