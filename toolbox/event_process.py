"""Shared event-history terms for binned Poisson occurrence models.

These small helpers support event-process analyses; the archived Barker workflow
has its own frozen copy.  Keeping
them here makes the clean workspace independent of the old Rousseau/Cheng
main-analysis script from which they were originally factored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


HISTORY_TERM = "same_type_history_count"
CLIMATE_TERMS = ("lr04_scaled", "co2_scaled")
PHASE_TERMS = ("pre_phase_sin", "pre_phase_cos")


def add_same_type_history(
    binned: pd.DataFrame, history_window_ka: float
) -> pd.DataFrame:
    """Count earlier events of the analyzed type in ``(t, t + window]``.

    Ages increase into the past, so older bins have larger ages.  The current
    bin is excluded to prevent an event from predicting itself.
    """

    frames = []
    for _, group in binned.groupby("dataset_id", sort=False):
        group = group.sort_values("bin_center_ka").copy()
        centers = group["bin_center_ka"].to_numpy(dtype=float)
        counts = group["event_count"].to_numpy(dtype=float)

        cumulative = np.concatenate([[0.0], np.cumsum(counts)])
        left = np.searchsorted(centers, centers + 1e-9, side="right")
        right = np.searchsorted(centers, centers + history_window_ka, side="right")
        history = cumulative[right] - cumulative[left]

        coverage = np.minimum(centers[-1], centers + history_window_ka) - centers
        bin_width_ka = float(np.nanmedian(group["dt_ka"].to_numpy(dtype=float)))
        group[HISTORY_TERM] = history
        group["same_type_history_complete"] = coverage >= (
            history_window_ka - 0.5 * bin_width_ka
        )
        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def model_frame(binned: pd.DataFrame) -> pd.DataFrame:
    """Keep bins with a fully observed older event-history window."""

    return binned[binned["same_type_history_complete"].astype(bool)].copy()
