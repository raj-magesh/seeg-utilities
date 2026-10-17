from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_temporal_generalization_matrix(
    ax: Axes,
    scores: npt.NDArray[np.floating],
    *,
    t: npt.NDArray[np.floating],
    t_spacing: float,
    t_start: float,
    t_end: float,
):
    ticks = np.linspace(
        t_start,
        t_end,
        1 + np.ceil((t_end - t_start) / t_spacing).astype(int),
    )
    tick_indices = np.searchsorted(t, ticks)

    sns.heatmap(
        ax=ax,
        data=pd.DataFrame(
            scores,
            columns=t,
            index=t,
        ),
        xticklabels=False,
        yticklabels=False,
        square=True,
        center=0.5,
        cbar_kws={"label": "ROC-AUC"},
        cmap="RdBu_r",
    )
    _ = ax.set(
        xlabel="time",
        ylabel="time",
        xticks=tick_indices,
        xticklabels=[f"{tick:.1f}" for tick in ticks],
        yticks=tick_indices,
        yticklabels=[f"{tick:.1f}" for tick in ticks],
    )

    t_0 = np.searchsorted(t, 0).astype(float)
    ax.invert_yaxis()
    ax.axvline(t_0, ls="-", lw=0.5, c="gray")
    ax.axhline(t_0, ls="-", lw=0.5, c="gray")
    ax.axline((t_start, t_start), (t_end, t_end), ls="--", lw=0.5, c="gray")
