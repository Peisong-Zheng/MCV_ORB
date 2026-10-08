#!/usr/bin/env python3
"""Prepare the two Barker et al. (2011) event catalogues from SOM Sheet1.

Retain the source SpeleoAge values on the project's working kyr BP coordinate.
The source epoch has not been verified; no 50-year conversion is applied.
Event IDs preserve the original Excel row so saved chronologies remain aligned.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE_XLS = ROOT / "data/raw/Barker et al-2011-SOM.xls"
OUTPUT_DIR = ROOT / "data/processed"


def main():

    raw = pd.read_excel(SOURCE_XLS, sheet_name="Sheet1", header=8)
    definitions = {
        "variable_threshold": ("DO pick variable threshold", 70),
        "fixed_threshold": ("DO pick", 59),
    }
    catalogues = {}
    for definition, (pick_column, expected_count) in definitions.items():
        events = raw[["SpeloAge (kyr).1", pick_column]].apply(pd.to_numeric, errors="coerce")
        events.columns = ["event_age_kyr_bp", "pick_value"]
        # header=8 puts the first data row on Excel row 10; retain it before filtering.
        events["event_id"] = [f"Barker_S3_{row + 10:03d}" for row in raw.index]
        selected = events.pick_value.eq(1) & events.event_age_kyr_bp.between(0, 400)
        events = events.loc[selected, ["event_id", "event_age_kyr_bp"]]
        events = events.sort_values("event_age_kyr_bp").reset_index(drop=True)

        ages = events.event_age_kyr_bp.to_numpy(float)
        if len(events) != expected_count or not events.event_id.is_unique:
            raise ValueError(f"Expected {expected_count} unique {definition} events")
        if not np.isfinite(ages).all() or np.any(np.diff(ages) <= 0):
            raise ValueError(f"{definition} ages must be finite and strictly increasing")
        catalogues[definition] = events

    variable = catalogues["variable_threshold"].set_index("event_id")
    fixed = catalogues["fixed_threshold"].set_index("event_id")
    if not fixed.index.isin(variable.index).all():
        raise ValueError("Fixed-threshold event IDs must be a subset of variable-threshold IDs")
    if not np.array_equal(fixed.event_age_kyr_bp, variable.loc[fixed.index, "event_age_kyr_bp"]):
        raise ValueError("The two definitions must retain the same age for each shared event ID")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for definition, events in catalogues.items():
        output = OUTPUT_DIR / f"barker_events_{definition}.csv"
        events.to_csv(output, index=False, float_format="%.17g")
        ages = events.event_age_kyr_bp
        print(f"{definition}: {len(events)} events, {ages.min():.6f}–{ages.max():.6f} kyr BP; {output}")


if __name__ == "__main__":
    main()
