#!/usr/bin/env python3
"""
Prepare the YJMob100K Dataset 2 records used in
"Mobility Behavior Evolution During an Emergency".

Default inputs
--------------
data/raw/yjmob100k-dataset2.csv.gz
    YJMob100K Dataset 2 (25,000 users, days 0-74), columns uid, d, t, x, y.
    A .csv or .parquet copy with the same columns also works.

data/raw/cell_POIcat.csv.gz
    Number of POIs per grid cell and category (x, y, POIcategory, POI_count).
    A pre-aggregated table with columns x, y, poi_sum is also accepted.

Default outputs (data/processed/)
---------------------------------
records.parquet          All records with time spent, trip distance,
                         calendar flags, home cell and neighborhood flag.
stops_normal.parquet     Stops for the normal window (days 43-57).
stops_emergency.parquet  Stops for the emergency window (days 60-74).
poi_cells.csv            Total number of POIs per grid cell.

Run
---
python preprocess_yjmob100k.py
python preprocess_yjmob100k.py --raw path/to/Data2.parquet --poi path/to/poi.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import mobility_metrics as mm


# ============================================================
# 1. Load
# ============================================================

def load_records(path: Path) -> pd.DataFrame:
    """Load raw records and drop the excluded days."""
    df = mm.read_table(path, columns=["uid", "d", "t", "x", "y"])
    missing = {"uid", "d", "t", "x", "y"}.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    n_before = len(df)
    df = df[~df["d"].isin(mm.EXCLUDED_DAYS)].copy()
    mm.log(f"Loaded {n_before:,} records; {len(df):,} kept after excluding "
           f"days {sorted(mm.EXCLUDED_DAYS)}")
    mm.log(f"  {df['uid'].nunique():,} users, days {df['d'].min()}-{df['d'].max()}")
    return df


# ============================================================
# 2. Home location
# ============================================================

def infer_home(df: pd.DataFrame) -> pd.DataFrame:
    """
    Home = grid cell with the most records between 8:00 PM and 8:00 AM
    over the normal days (d <= 59), following the dataset manual.
    Ties are broken by (x, y) so the result is deterministic.
    """
    night = df[
        (df["d"] <= mm.HOME_INFERENCE_LAST_DAY)
        & ((df["t"] >= 40) | (df["t"] < 16))
    ]
    home = (
        night.groupby(["uid", "x", "y"]).size().rename("n").reset_index()
        .sort_values(["uid", "n", "x", "y"],
                     ascending=[True, False, True, True], kind="mergesort")
        .drop_duplicates("uid")
        .rename(columns={"x": "x_h", "y": "y_h"})
        [["uid", "x_h", "y_h"]]
    )
    mm.log(f"Home location inferred for {len(home):,} users")
    return home


# ============================================================
# 3. Time spent and trip distance
# ============================================================

def add_time_and_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    time_spent: hours until the user's next record, capped at
                MAX_GAP_HOURS (0 for a user's last record).
    dist:       straight-line distance (km) from the previous record.
    """
    df = df.sort_values(["uid", "d", "t"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("uid", sort=False)

    time_index = df["d"] * mm.SLOTS_PER_DAY + df["t"]
    next_index = time_index.groupby(df["uid"], sort=False).shift(-1)
    df["time_spent"] = (
        ((next_index - time_index) * mm.SLOT_HOURS)
        .clip(lower=0, upper=mm.MAX_GAP_HOURS)
        .fillna(0.0)
    )

    dx = df["x"] - g["x"].shift(1)
    dy = df["y"] - g["y"].shift(1)
    df["dist"] = (np.sqrt(dx ** 2 + dy ** 2) * mm.GRID_CELL_KM).fillna(0.0)
    return df


# ============================================================
# 4. POI table
# ============================================================

def load_poi_cells(path: Path) -> pd.DataFrame:
    """Return one row per grid cell with its total number of POIs (poi_sum)."""
    poi = mm.read_table(path)
    if "poi_sum" in poi.columns:
        cells = poi.groupby(["x", "y"], as_index=False)["poi_sum"].first()
    elif "POI_count" in poi.columns:
        cells = poi.groupby(["x", "y"], as_index=False)["POI_count"].sum()
        cells = cells.rename(columns={"POI_count": "poi_sum"})
    else:
        raise ValueError(f"{path} needs a 'poi_sum' or 'POI_count' column")
    mm.log(f"POI table: {len(cells):,} grid cells with POIs")
    return cells


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/raw/yjmob100k-dataset2.csv.gz")
    parser.add_argument("--poi", default="data/raw/cell_POIcat.csv.gz")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    out_dir = mm.ensure_dir(args.out_dir)

    df = load_records(Path(args.raw))
    df = mm.add_calendar(df)
    df = df.merge(infer_home(df), on="uid", how="left")
    df = add_time_and_distance(df)
    df["walk_d_home"] = mm.in_home_neighborhood(df)

    df.to_parquet(out_dir / "records.parquet", index=False)
    mm.log(f"Saved records.parquet ({len(df):,} rows)")

    for name, window in [("normal", mm.NORMAL_WINDOW),
                         ("emergency", mm.EMERGENCY_WINDOW)]:
        stops = mm.collapse_stops(mm.filter_window(df, window))
        stops.to_parquet(out_dir / f"stops_{name}.parquet", index=False)
        mm.log(f"Saved stops_{name}.parquet: days {window[0]}-{window[1]}, "
               f"{len(stops):,} stops, {stops['uid'].nunique():,} users")

    poi_path = Path(args.poi)
    if poi_path.exists():
        load_poi_cells(poi_path).to_csv(out_dir / "poi_cells.csv", index=False)
        mm.log("Saved poi_cells.csv")
    else:
        mm.log(f"POI file not found at {poi_path}; skipping poi_cells.csv")


if __name__ == "__main__":
    main()
