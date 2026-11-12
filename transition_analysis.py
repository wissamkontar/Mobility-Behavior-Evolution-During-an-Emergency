#!/usr/bin/env python3
"""
Behavioral transitions between the normal and emergency periods.

Every user is classified as a returner or explorer (k = 4) separately in
each window, and the analyses below compare those labels:

1. Transition matrix from the normal period (days 43-57) to the emergency
   period (days 60-74), with row-normalized percentages.
2. Placebo baseline: the same comparison from the normal period to three
   other normal windows (days 1-15, 15-29, 29-43).
3. Threshold sensitivity: transitions recomputed with the S_k cutoff set
   to r_g/1.8, r_g/2 and r_g/2.2.
4. Four transition groups (R-R, R-E, E-E, E-R): daily maximum distance
   from home and non-home dwelling time during the emergency period.
5. POI comparison across the four groups: average POI count of visited
   cells outside the nearby neighborhood (weekdays and holidays, both
   periods) and of the home neighborhood, with Kruskal-Wallis tests,
   Holm-corrected pairwise Mann-Whitney tests and a balanced-subsampling
   check.

Default inputs (data/processed/)
--------------------------------
records.parquet, stops_normal.parquet, stops_emergency.parquet, poi_cells.csv

Default outputs (results/transitions/)
--------------------------------------
transition_matrices.csv         Counts and row percentages for every comparison
threshold_sensitivity.csv       Transition percentages at three S_k cutoffs
user_transition_groups.csv      R-R / R-E / E-E / E-R label per user
group_daily_max_distance.csv    Binned daily max distance by group (emergency)
group_daily_non_home_dwelling.csv
poi_group_summary.csv           Mean POI counts per group, period and day type
poi_kruskal_wallis.csv          Kruskal-Wallis H, p and epsilon-squared
poi_pairwise.csv                Pairwise Mann-Whitney with rank-biserial r and Holm p
poi_balanced_subsampling.csv    Epsilon-squared and group means after equalizing group sizes

Run
---
python transition_analysis.py
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

import mobility_metrics as mm


LABEL_ORDER = [mm.RETURNER, mm.EXPLORER]


# ============================================================
# 1-2. Transition matrices
# ============================================================

def classify_window(records: pd.DataFrame, window) -> pd.DataFrame:
    """k = 4 classification (with S_k) for one window."""
    return mm.classify(mm.filter_window(records, window), k=mm.K_CLASSIFY)


def transition_matrix(origin: pd.Series, destination: pd.Series,
                      comparison: str) -> pd.DataFrame:
    """Counts and row percentages (each origin group sums to 100%)."""
    j = pd.concat([origin.rename("from"), destination.rename("to")], axis=1).dropna()
    counts = pd.crosstab(j["from"], j["to"]).reindex(
        index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100

    rows = []
    for a in LABEL_ORDER:
        for b in LABEL_ORDER:
            rows.append({"comparison": comparison, "n_matched_users": len(j),
                         "from": a, "to": b,
                         "n_users": int(counts.loc[a, b]),
                         "row_pct": pct.loc[a, b]})
    out = pd.DataFrame(rows)
    re_ = pct.loc[mm.RETURNER, mm.EXPLORER]
    er_ = pct.loc[mm.EXPLORER, mm.RETURNER]
    mm.log(f"  {comparison:<28} n={len(j):>6,}   R->E {re_:6.2f}%   E->R {er_:6.2f}%")
    return out


def threshold_sensitivity(origin: pd.DataFrame, destination: pd.DataFrame) -> pd.DataFrame:
    """Transitions recomputed with alternative S_k cutoffs."""
    j = pd.concat([origin["s_k"].rename("s_from"),
                   destination["s_k"].rename("s_to")], axis=1).dropna()
    rows = []
    for name, cutoff in mm.THRESHOLD_SENSITIVITY.items():
        ret_from = j["s_from"] > cutoff
        ret_to = j["s_to"] > cutoff
        rows.append({
            "threshold": name,
            "s_k_cutoff": cutoff,
            "n_returners": int(ret_from.sum()),
            "n_explorers": int((~ret_from).sum()),
            "R_to_R_pct": 100 * ret_to[ret_from].mean(),
            "R_to_E_pct": 100 * (~ret_to[ret_from]).mean(),
            "E_to_R_pct": 100 * ret_to[~ret_from].mean(),
            "E_to_E_pct": 100 * (~ret_to[~ret_from]).mean(),
        })
    out = pd.DataFrame(rows)
    out["E_to_R_minus_R_to_E"] = out["E_to_R_pct"] - out["R_to_E_pct"]
    return out


# ============================================================
# 4. Four transition groups
# ============================================================

def transition_groups(normal_cls: pd.Series, emergency_cls: pd.Series) -> pd.Series:
    """R-R, R-E, E-E or E-R label for users classified in both periods."""
    j = pd.concat([normal_cls.rename("n"), emergency_cls.rename("e")], axis=1).dropna()
    groups = pd.Series(
        [mm.TRANSITION_GROUPS[(a, b)] for a, b in zip(j["n"], j["e"])],
        index=j.index, name="group",
    )
    mm.log("  group sizes: " + ", ".join(
        f"{g}={(groups == g).sum():,}" for g in mm.TRANSITION_ORDER))
    return groups


def group_daily_distributions(stops: pd.DataFrame, groups: pd.Series,
                              window, out: Path) -> None:
    """Binned daily max distance and non-home dwelling time for each group."""
    dist_rows, dwell_rows = [], []
    for day_idx, day in enumerate(range(window[0], window[1] + 1), start=1):
        s = stops[stops["d"] == day]
        if s.empty:
            continue
        dist_rows.append(mm.binned_shares(
            mm.max_distance_from_home(s), groups, np.arange(0, 130, 5)
        ).assign(day=day_idx, d=day))
        dwell_rows.append(mm.binned_shares(
            mm.non_home_dwelling_time(s), groups, np.arange(0, 25, 1)
        ).assign(day=day_idx, d=day))
    pd.concat(dist_rows).to_csv(out / "group_daily_max_distance.csv", index=False)
    pd.concat(dwell_rows).to_csv(out / "group_daily_non_home_dwelling.csv", index=False)


# ============================================================
# 5. Points of interest by transition group
# ============================================================

def user_poi(stops: pd.DataFrame, poi: pd.DataFrame, groups: pd.Series,
             mask: pd.Series, label: str) -> pd.DataFrame:
    """Mean POI count over each user's stops that satisfy mask."""
    s = stops[mask].merge(poi, on=["x", "y"], how="left", validate="many_to_one")
    s["poi_sum"] = s["poi_sum"].fillna(0)
    per_user = s.groupby("uid")["poi_sum"].mean().rename("avg_poi")
    return (per_user.to_frame().join(groups, how="inner")
            .assign(comparison=label).reset_index())


def kruskal_wallis(data: pd.DataFrame) -> dict:
    arrays = [data.loc[data["group"] == g, "avg_poi"].to_numpy()
              for g in mm.TRANSITION_ORDER]
    h_stat, p = kruskal(*arrays)
    n = sum(len(a) for a in arrays)
    return {"H": h_stat, "p": p, "N": n,
            "epsilon_squared": mm.epsilon_squared(h_stat, n, len(arrays))}


def pairwise_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g1, g2 in combinations(mm.TRANSITION_ORDER, 2):
        x = data.loc[data["group"] == g1, "avg_poi"].to_numpy()
        y = data.loc[data["group"] == g2, "avg_poi"].to_numpy()
        u_stat, p = mannwhitneyu(x, y, alternative="two-sided")
        rows.append({"pair": f"{g1} vs {g2}", "n1": len(x), "n2": len(y),
                     "mean1": x.mean(), "mean2": y.mean(),
                     "U": u_stat, "p_raw": p,
                     "rank_biserial": mm.rank_biserial(u_stat, len(x), len(y))})
    out = pd.DataFrame(rows)
    out["p_holm"] = mm.holm_adjust(out["p_raw"])
    return out


def balanced_subsampling(data: pd.DataFrame) -> dict:
    """Repeat Kruskal-Wallis after downsampling every group to the smallest size."""
    rng = np.random.default_rng(mm.RANDOM_SEED)
    values = {g: data.loc[data["group"] == g, "avg_poi"].to_numpy()
              for g in mm.TRANSITION_ORDER}
    n_min = min(len(v) for v in values.values())

    eps, means = [], {g: [] for g in mm.TRANSITION_ORDER}
    for _ in range(mm.N_BALANCED_ITER):
        samples = {g: rng.choice(v, size=n_min, replace=False) for g, v in values.items()}
        for g, s in samples.items():
            means[g].append(s.mean())
        h_stat, _ = kruskal(*samples.values())
        eps.append(mm.epsilon_squared(h_stat, n_min * len(samples), len(samples)))

    row = {"n_per_group": n_min,
           "epsilon_squared_median": np.median(eps),
           "epsilon_squared_2.5": np.percentile(eps, 2.5),
           "epsilon_squared_97.5": np.percentile(eps, 97.5)}
    for g in mm.TRANSITION_ORDER:
        row[f"mean_{g}"] = np.mean(means[g])
    return row


def poi_analysis(stops_normal: pd.DataFrame, stops_emergency: pd.DataFrame,
                 poi: pd.DataFrame, groups: pd.Series, out: Path) -> None:
    datasets = []
    for period, stops in [("normal", stops_normal), ("emergency", stops_emergency)]:
        for holiday, day_type in [(1, "holiday"), (0, "weekday")]:
            mask = (stops["walk_d_home"] == 0) & (stops["holiday"] == holiday)
            datasets.append(user_poi(stops, poi, groups, mask,
                                     f"{period} | visited ONN | {day_type}"))
    datasets.append(user_poi(stops_normal, poi, groups,
                             stops_normal["walk_d_home"] == 1,
                             "normal | home neighborhood"))

    summary, kw, pairwise, balanced = [], [], [], []
    for data in datasets:
        label = data["comparison"].iloc[0]
        summary.append(
            data.groupby("group")["avg_poi"].agg(["count", "mean", "median", "std"])
            .reindex(mm.TRANSITION_ORDER).reset_index().assign(comparison=label))
        kw.append({"comparison": label, **kruskal_wallis(data)})
        pairwise.append(pairwise_tests(data).assign(comparison=label))
        balanced.append({"comparison": label, **balanced_subsampling(data)})
        mm.log(f"  {label}: epsilon^2 = {kw[-1]['epsilon_squared']:.4f}")

    pd.concat(summary).to_csv(out / "poi_group_summary.csv", index=False)
    pd.DataFrame(kw).to_csv(out / "poi_kruskal_wallis.csv", index=False)
    pd.concat(pairwise).to_csv(out / "poi_pairwise.csv", index=False)
    pd.DataFrame(balanced).to_csv(out / "poi_balanced_subsampling.csv", index=False)


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = mm.ensure_dir(Path(args.out_dir) / "transitions")
    records = pd.read_parquet(data_dir / "records.parquet",
                              columns=["uid", "d", "t", "x", "y", "time_spent"])

    mm.log("1-2. Transition matrices")
    normal = classify_window(records, mm.NORMAL_WINDOW)
    emergency = classify_window(records, mm.EMERGENCY_WINDOW)

    matrices = [transition_matrix(normal["cls"], emergency["cls"],
                                  "normal -> emergency")]
    for name, window in mm.PLACEBO_WINDOWS.items():
        placebo = classify_window(records, window)
        matrices.append(transition_matrix(
            normal["cls"], placebo["cls"],
            f"normal -> {name} ({window[0]}-{window[1]})"))
    pd.concat(matrices).to_csv(out / "transition_matrices.csv", index=False)

    mm.log("3. Threshold sensitivity")
    threshold_sensitivity(normal, emergency).to_csv(
        out / "threshold_sensitivity.csv", index=False)

    mm.log("4. Four transition groups")
    groups = transition_groups(normal["cls"], emergency["cls"])
    groups.reset_index().to_csv(out / "user_transition_groups.csv", index=False)

    stops_normal = pd.read_parquet(data_dir / "stops_normal.parquet")
    stops_emergency = pd.read_parquet(data_dir / "stops_emergency.parquet")
    group_daily_distributions(stops_emergency, groups, mm.EMERGENCY_WINDOW, out)

    poi_path = data_dir / "poi_cells.csv"
    if poi_path.exists():
        mm.log("5. POI comparison across transition groups")
        poi_analysis(stops_normal, stops_emergency, pd.read_csv(poi_path), groups, out)
    else:
        mm.log(f"5. Skipped: {poi_path} not found")

    mm.log(f"Done. Results saved under {out}/")


if __name__ == "__main__":
    main()
