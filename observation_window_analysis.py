#!/usr/bin/env python3
"""
How long must mobility be observed to separate emergency from normal behavior?

For each observation window, users are classified as k-returners or
k-explorers for k in {2, 3, 4, 5, 6, 8, 10}, and the percentage of
returners is reported. Three analyses are run:

1. Matched windows: the first N days (N = 1..14) of the normal period
   (from day 43) and of the emergency period (from day 60). For every N
   the script reports, for each k, whether both periods agree on the
   majority group. The shortest N at which they disagree is the minimum
   observation length reported in the paper.

2. Robustness segments: three other 14-day normal segments
   (days 1-14, 15-28, 29-42), each observed for 7 and 14 days.

3. Extended normal windows: 2, 4, 6 and 8 weeks ending on day 56.

Radius of gyration is invariant to merging consecutive records at the
same cell, so this analysis works directly on the record table.

Default input
-------------
data/processed/records.parquet

Default outputs (results/observation_window/)
---------------------------------------------
matched_window_shares.csv     Returner share per N, k and period
matched_window_agreement.csv  Majority group agreement per N and k
segment_shares.csv            Returner share for the robustness segments
extended_window_shares.csv    Returner share for the extended windows
crossover_summary.csv         Crossover k for every window analysed

Run
---
python observation_window_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import mobility_metrics as mm


MAX_MATCHED_DAYS = 14

ROBUSTNESS_SEGMENTS = {"days_1-14": 1, "days_15-28": 15, "days_29-42": 29}

EXTENDED_WINDOWS = {
    "2_weeks": (43, 56),
    "4_weeks": (29, 56),
    "6_weeks": (15, 56),
    "8_weeks": (1, 56),
}


def window_shares(records: pd.DataFrame, window, **labels) -> pd.DataFrame:
    """Returner/explorer shares for each k within one window."""
    shares = mm.returner_shares_by_k(mm.filter_window(records, window))
    for key, value in labels.items():
        shares[key] = value
    shares["first_day"], shares["last_day"] = window
    return shares


def crossover_rows(table: pd.DataFrame, keys: list[str]) -> list[dict]:
    """Crossover k for every window in a shares table."""
    return [
        {**dict(zip(keys, key if isinstance(key, tuple) else (key,))),
         "crossover_k": mm.crossover_k(group)}
        for key, group in table.groupby(keys, sort=False)
    ]


def matched_windows(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First N days of the normal and emergency periods, N = 1..14."""
    rows = []
    for n_days in range(1, MAX_MATCHED_DAYS + 1):
        for period, start in [("normal", mm.NORMAL_WINDOW[0]),
                              ("emergency", mm.EMERGENCY_WINDOW[0])]:
            window = (start, start + n_days - 1)
            rows.append(window_shares(records, window, period=period, n_days=n_days))
        mm.log(f"  first {n_days} day(s) done")
    shares = pd.concat(rows, ignore_index=True)

    wide = shares.pivot_table(index=["n_days", "k"], columns="period",
                              values="returner_pct").reset_index()
    wide["normal_majority"] = np.where(wide["normal"] > 50, mm.RETURNER, mm.EXPLORER)
    wide["emergency_majority"] = np.where(wide["emergency"] > 50, mm.RETURNER, mm.EXPLORER)
    wide["agree"] = wide["normal_majority"] == wide["emergency_majority"]

    first = wide.loc[~wide["agree"], "n_days"]
    if len(first):
        mm.log(f"  periods first disagree on the majority group at N = {first.min()} days")
    else:
        mm.log(f"  periods agree at every k for all N <= {MAX_MATCHED_DAYS}")
    return shares, wide


def robustness_segments(records: pd.DataFrame) -> pd.DataFrame:
    """Three other 14-day normal segments, observed for 7 and 14 days."""
    rows = []
    for label, start in ROBUSTNESS_SEGMENTS.items():
        for n_days in (7, 14):
            rows.append(window_shares(records, (start, start + n_days - 1),
                                      segment=label, n_days=n_days))
    return pd.concat(rows, ignore_index=True)


def extended_windows(records: pd.DataFrame) -> pd.DataFrame:
    """2-, 4-, 6- and 8-week normal windows."""
    rows = [window_shares(records, window, window_label=label)
            for label, window in EXTENDED_WINDOWS.items()]
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    out = mm.ensure_dir(Path(args.out_dir) / "observation_window")
    records = pd.read_parquet(
        Path(args.data_dir) / "records.parquet",
        columns=["uid", "d", "t", "x", "y", "time_spent"],
    )

    mm.log("1. Matched windows (first 1-14 days)")
    matched, agreement = matched_windows(records)
    matched.to_csv(out / "matched_window_shares.csv", index=False)
    agreement.to_csv(out / "matched_window_agreement.csv", index=False)

    mm.log("2. Robustness segments")
    segments = robustness_segments(records)
    segments.to_csv(out / "segment_shares.csv", index=False)

    mm.log("3. Extended normal windows")
    extended = extended_windows(records)
    extended.to_csv(out / "extended_window_shares.csv", index=False)

    summary = (
        [{"analysis": "matched", **r} for r in crossover_rows(matched, ["period", "n_days"])]
        + [{"analysis": "segment", **r} for r in crossover_rows(segments, ["segment", "n_days"])]
        + [{"analysis": "extended", **r} for r in crossover_rows(extended, ["window_label"])]
    )
    summary = pd.DataFrame(summary)
    summary["n_days"] = summary["n_days"].astype("Int64")
    summary.to_csv(out / "crossover_summary.csv", index=False)
    mm.log(f"Done. Results saved under {out}/")


if __name__ == "__main__":
    main()
