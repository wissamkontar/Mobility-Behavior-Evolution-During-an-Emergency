#!/usr/bin/env python3
"""
Single-period analysis of returners and explorers.

The same pipeline is applied to the normal period (days 43-57) and the
emergency period (days 60-74); choose the period on the command line.

Default inputs (data/processed/, produced by preprocess_yjmob100k.py)
---------------------------------------------------------------------
stops_normal.parquet or stops_emergency.parquet

Default outputs (results/<period>/)
-----------------------------------
rg_fit_full_period.csv          Lognormal parameters and R statistics for r_g
rg_fit_by_day.csv               Same fit for each of the 15 days
rk_fit.csv                      Lognormal parameters for r_g^(k), k = 2..5, and r_g
s_k_by_user.csv                 r_g, r_g^(k) and S_k for every user and k = 2..5
returner_shares_by_k.csv        Percentage of returners and explorers per k
user_classification.csv         Returner/explorer label per user (k = 4)
user_metrics.csv                Max distance from home, non-home dwelling time,
                                and real entropy per user
group_tests.csv                 Mann-Whitney U and KS tests with bootstrap CIs
max_distance_shares.csv         Binned max distance from home by group
non_home_dwelling_shares.csv    Binned non-home dwelling time by group
daily_max_distance_shares.csv   Same, for each day of the period
daily_non_home_dwelling_shares.csv
spatial_stops_per_person.csv    Stops per person in each grid cell, by group
spatial_stay_minutes_per_person.csv
onn_time_shares.csv             Group composition by daily time outside the
                                nearby neighborhood (weekdays / holidays)
onn_distance_shares.csv         Same, by daily distance outside the neighborhood
home_returner_share.csv         Share of returners among residents of each cell

Run
---
python period_analysis.py --period normal
python period_analysis.py --period emergency
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import mobility_metrics as mm


WINDOWS = {"normal": mm.NORMAL_WINDOW, "emergency": mm.EMERGENCY_WINDOW}


# ============================================================
# 1. Distribution of the radius of gyration
# ============================================================

def fit_radius_of_gyration(stops: pd.DataFrame, window, out: Path) -> pd.Series:
    """Fit r_g over the full period and for each day separately."""
    rg = mm.radius_of_gyration(stops)

    full = mm.fit_distribution(rg.to_numpy() / 1000.0)
    pd.DataFrame([full]).to_csv(out / "rg_fit_full_period.csv", index=False)
    mm.log(f"  r_g lognormal: mu={full['lognormal_mu']:.2f}, "
           f"sigma={full['lognormal_sigma']:.2f}")

    rows = []
    for day_idx, day in enumerate(range(window[0], window[1] + 1), start=1):
        rg_day = mm.radius_of_gyration(stops[stops["d"] == day])
        if len(rg_day) < 10:  # skip days with too few users
            continue
        rows.append({"day": day_idx, "d": day,
                     **mm.fit_distribution(rg_day.to_numpy() / 1000.0)})
    pd.DataFrame(rows).to_csv(out / "rg_fit_by_day.csv", index=False)
    return rg


def fit_k_radius_of_gyration(stops: pd.DataFrame, rg: pd.Series, out: Path) -> None:
    """Lognormal fits of r_g^(k) for k = 2..5, plus S_k for every user."""
    fits, sk_tables = [], []
    for k in mm.K_FIT_VALUES:
        rk = mm.k_radius_of_gyration(stops, k)
        fits.append({"metric": f"r_{k}", **mm.fit_distribution(rk.to_numpy() / 1000.0)})

        c = mm.classify(stops, k=k, rg=rg)
        sk_tables.append(c.assign(k=k).reset_index())
    fits.append({"metric": "r_g", **mm.fit_distribution(rg.to_numpy() / 1000.0)})

    pd.DataFrame(fits).to_csv(out / "rk_fit.csv", index=False)
    pd.concat(sk_tables, ignore_index=True).to_csv(out / "s_k_by_user.csv", index=False)


# ============================================================
# 2. Returner / explorer classification
# ============================================================

def classify_users(stops: pd.DataFrame, rg: pd.Series, out: Path) -> pd.Series:
    """Shares by k, and the k = 4 classification used in the rest of the paper."""
    shares = mm.returner_shares_by_k(stops)
    shares.to_csv(out / "returner_shares_by_k.csv", index=False)
    mm.log(f"  crossover k (returners > explorers): {mm.crossover_k(shares):.0f}")

    c = mm.classify(stops, k=mm.K_CLASSIFY, rg=rg)
    c.reset_index().to_csv(out / "user_classification.csv", index=False)
    n_ret = (c["cls"] == mm.RETURNER).sum()
    mm.log(f"  k={mm.K_CLASSIFY}: {n_ret:,} returners, {len(c) - n_ret:,} explorers")
    return c["cls"]


# ============================================================
# 3. Max distance from home, non-home dwelling time, entropy
# ============================================================

def compare_groups(stops: pd.DataFrame, cls: pd.Series, out: Path) -> pd.DataFrame:
    """Per-user metrics and returner-versus-explorer tests."""
    metrics = pd.concat(
        [mm.max_distance_from_home(stops),
         mm.non_home_dwelling_time(stops),
         mm.real_entropy(stops),
         cls.rename("cls")],
        axis=1,
    ).dropna(subset=["cls"])
    metrics.reset_index().to_csv(out / "user_metrics.csv", index=False)

    tests = []
    for col, n_boot in [("max_dist_km", mm.N_BOOT),
                        ("non_home_hours", mm.N_BOOT),
                        ("real_entropy", mm.N_BOOT_ENTROPY)]:
        mm.log(f"  tests for {col} ({n_boot} bootstrap replicates)")
        res = mm.two_group_tests(
            metrics.loc[metrics["cls"] == mm.RETURNER, col],
            metrics.loc[metrics["cls"] == mm.EXPLORER, col],
            n_boot=n_boot, seed=mm.RANDOM_SEED,
        )
        tests.append({"metric": col, **res})
    pd.DataFrame(tests).to_csv(out / "group_tests.csv", index=False)
    return metrics


def binned_group_shares(stops: pd.DataFrame, cls: pd.Series, metrics: pd.DataFrame,
                        window, out: Path) -> None:
    """Binned distributions over the full period and for each day."""
    dist_bins = np.arange(0, 130, 5)
    mm.binned_shares(metrics["max_dist_km"], metrics["cls"], dist_bins).to_csv(
        out / "max_distance_shares.csv", index=False)

    width = metrics["non_home_hours"].max() / 30
    dwell_bins = np.arange(0, metrics["non_home_hours"].max() + width, width)
    mm.binned_shares(metrics["non_home_hours"], metrics["cls"], dwell_bins).to_csv(
        out / "non_home_dwelling_shares.csv", index=False)

    # Daily distributions use the full-period classification.
    daily_dist, daily_dwell = [], []
    for day_idx, day in enumerate(range(window[0], window[1] + 1), start=1):
        s = stops[stops["d"] == day]
        if s.empty:
            continue
        daily_dist.append(mm.binned_shares(
            mm.max_distance_from_home(s), cls, dist_bins).assign(day=day_idx, d=day))
        daily_dwell.append(mm.binned_shares(
            mm.non_home_dwelling_time(s), cls, np.arange(0, 25, 1)).assign(day=day_idx, d=day))
    pd.concat(daily_dist).to_csv(out / "daily_max_distance_shares.csv", index=False)
    pd.concat(daily_dwell).to_csv(out / "daily_non_home_dwelling_shares.csv", index=False)


# ============================================================
# 4. Spatial distribution of stops and stay time
# ============================================================

def spatial_patterns(stops: pd.DataFrame, cls: pd.Series, out: Path) -> None:
    """Stops per person and stay time per person in each grid cell, by group."""
    s = stops.join(cls.rename("cls"), on="uid").dropna(subset=["cls"])
    n_users = s.groupby("cls")["uid"].nunique()

    stops_pp = s.groupby(["cls", "x", "y"]).size().rename("stops").reset_index()
    stops_pp["stops_per_person"] = stops_pp["stops"] / stops_pp["cls"].map(n_users)
    stops_pp.to_csv(out / "spatial_stops_per_person.csv", index=False)

    stay = s.groupby(["cls", "x", "y"])["time_spent"].sum().rename("hours").reset_index()
    stay["stay_minutes_per_person"] = stay["hours"] * 60 / stay["cls"].map(n_users)
    stay.to_csv(out / "spatial_stay_minutes_per_person.csv", index=False)


# ============================================================
# 5. Mobility outside nearby neighborhoods (ONN)
# ============================================================

ONN_TIME_BINS = [0, 0.5, 1, 2, 4, 8, 24]
ONN_TIME_LABELS = ["<0.5", "0.5-1", "1-2", "2-4", "4-8", ">8"]
ONN_DIST_BINS = [0, 0.5, 2, 5, 10, 20, 100]
ONN_DIST_LABELS = ["<0.5", "0.5-2", "2-5", "5-10", "10-20", ">20"]


def outside_neighborhood(stops: pd.DataFrame, out: Path) -> pd.Series:
    """
    Average daily time (hours) and distance (km) outside the nearby
    neighborhood of home, separately for weekdays and weekends/holidays,
    and the returner/explorer composition of each bin.
    """
    cls_onn = mm.classify(stops, k=mm.K_ONN)["cls"]

    time_rows, dist_rows = [], []
    for holiday, day_type in [(0, "weekday"), (1, "holiday")]:
        days = stops[stops["holiday"] == holiday]
        n_days = days.groupby("uid")["d"].nunique()
        outside = days[days["walk_d_home"] == 0]

        hours = (outside.groupby("uid")["time_spent"].sum() / n_days).dropna()
        km = (outside.groupby("uid")["dist"].sum() / n_days).dropna()

        time_rows.append(mm.row_shares(hours, cls_onn, ONN_TIME_BINS, ONN_TIME_LABELS)
                         .assign(day_type=day_type))
        dist_rows.append(mm.row_shares(km, cls_onn, ONN_DIST_BINS, ONN_DIST_LABELS)
                         .assign(day_type=day_type))

    pd.concat(time_rows).to_csv(out / "onn_time_shares.csv", index=False)
    pd.concat(dist_rows).to_csv(out / "onn_distance_shares.csv", index=False)
    return cls_onn


def home_location_shares(stops: pd.DataFrame, cls_onn: pd.Series, out: Path) -> None:
    """Percentage of returners among the residents of each home cell."""
    home = stops.groupby("uid")[["x_h", "y_h"]].first().join(cls_onn.rename("cls"))
    home = home.dropna(subset=["cls"])
    grid = home.groupby(["x_h", "y_h"])["cls"].agg(
        n_users="size",
        n_returners=lambda c: int((c == mm.RETURNER).sum()),
    ).reset_index()
    grid["returner_pct"] = 100 * grid["n_returners"] / grid["n_users"]
    grid.to_csv(out / "home_returner_share.csv", index=False)


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--period", choices=sorted(WINDOWS), required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    window = WINDOWS[args.period]
    out = mm.ensure_dir(Path(args.out_dir) / args.period)
    stops = pd.read_parquet(Path(args.data_dir) / f"stops_{args.period}.parquet")
    mm.log(f"{args.period.capitalize()} period, days {window[0]}-{window[1]}: "
           f"{len(stops):,} stops, {stops['uid'].nunique():,} users")

    mm.log("1. Fitting r_g and r_g^(k) distributions")
    rg = fit_radius_of_gyration(stops, window, out)
    fit_k_radius_of_gyration(stops, rg, out)

    mm.log("2. Classifying returners and explorers")
    cls = classify_users(stops, rg, out)

    mm.log("3. Comparing groups")
    metrics = compare_groups(stops, cls, out)
    binned_group_shares(stops, cls, metrics, window, out)

    mm.log("4. Spatial patterns")
    spatial_patterns(stops, cls, out)

    mm.log("5. Mobility outside nearby neighborhoods")
    cls_onn = outside_neighborhood(stops, out)
    home_location_shares(stops, cls_onn, out)

    mm.log(f"Done. Results saved under {out}/")


if __name__ == "__main__":
    main()
