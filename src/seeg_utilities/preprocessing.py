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
