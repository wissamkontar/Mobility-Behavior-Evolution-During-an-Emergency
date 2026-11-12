#!/usr/bin/env python3
"""
Shared study configuration and mobility metrics for
"Mobility Behavior Evolution During an Emergency".

This module is imported by every analysis script in the repository.
It contains:

1. the study configuration (analysis windows, calendar, k values,
   classification threshold, bootstrap settings, random seed);
2. data helpers (file reading, calendar flags, stop construction);
3. mobility indicators (radius of gyration, k-radius of gyration,
   returner/explorer classification, maximum distance from home,
   non-home dwelling time, real entropy);
4. statistical helpers (distribution fitting, Mann-Whitney U and
   Kolmogorov-Smirnov tests with bootstrap confidence intervals,
   Kruskal-Wallis epsilon-squared, Holm correction, binned shares).

All spatial quantities are computed on the 500 m x 500 m YJMob100K
grid. Radii of gyration are returned in meters; distances from home
are returned in kilometers; durations are returned in hours.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu


# ============================================================
# 1. Study configuration
# ============================================================

# --- Spatial and temporal resolution -------------------------
GRID_CELL_M = 500.0          # side of one grid cell in meters
GRID_CELL_KM = 0.5           # side of one grid cell in kilometers
SLOTS_PER_DAY = 48           # 30-minute slots per day
SLOT_HOURS = 0.5             # duration of one slot in hours
MAX_GAP_HOURS = 5.0          # cap on the time attributed to one record

# --- Analysis windows (dataset day indices, inclusive) -------
NORMAL_WINDOW = (43, 57)     # representative 15-day normal period
EMERGENCY_WINDOW = (60, 74)  # emergency period fixed by the dataset creators
HOME_INFERENCE_LAST_DAY = 59 # home locations use the 60 normal days only

# Additional normal windows used as a placebo baseline for transitions
PLACEBO_WINDOWS = {
    "normal_1": (1, 15),
    "normal_2": (15, 29),
    "normal_3": (29, 43),
}

# Days excluded from all analyses (advised by the dataset creators)
EXCLUDED_DAYS = {27}

# --- Calendar ------------------------------------------------
# Normal days (d < 60): day d % 7 == 0 is a Sunday and 6 is a Saturday.
# Non-weekend public holidays were identified from the daily ping counts.
NORMAL_HOLIDAYS = {
    0, 1, 6, 7, 8, 13, 14, 20, 21, 27, 28, 29,
    34, 35, 37, 41, 42, 48, 49, 50, 55, 56,
}

# The emergency block (days 60-74) is spliced separately in the dataset.
# It is re-indexed so that day 60 is a Monday, which gives it the same
# weekly and holiday structure as the normal window (weekend on days 6-7
# and a holiday on day 8 of each 15-day period).
EMERGENCY_FIRST_DAY_DOW = 1  # 0 = Sunday, 1 = Monday, ..., 6 = Saturday
EMERGENCY_HOLIDAYS = {67}

# --- Returner / explorer classification ----------------------
K_CLASSIFY = 4               # k used for classification (normal-period crossover)
K_ONN = 6                    # k used in the analysis notebooks for the
                             # outside-nearby-neighborhood and home-location maps
S_K_THRESHOLD = 0.5          # returner if r_g^(k) / r_g > 0.5
K_VALUES = [2, 3, 4, 5, 6, 8, 10]
K_FIT_VALUES = [2, 3, 4, 5]
THRESHOLD_SENSITIVITY = {"rg/1.8": 1 / 1.8, "rg/2.0": 1 / 2.0, "rg/2.2": 1 / 2.2}

# --- Nearby neighborhood -------------------------------------
NEIGHBORHOOD_RADIUS_CELLS = 2  # |dx| <= 2 and |dy| <= 2 cells around home

# --- Distribution fitting ------------------------------------
FIT_XMIN_KM = 1.0

# --- Statistics and reproducibility --------------------------
RANDOM_SEED = 12345
N_BOOT = 2000                # bootstrap replicates for distance and dwell tests
N_BOOT_ENTROPY = 200         # bootstrap replicates for entropy tests
N_BALANCED_ITER = 500        # iterations for the balanced-subsampling check

RETURNER = "returner"
EXPLORER = "explorer"

TRANSITION_GROUPS = {
    (RETURNER, RETURNER): "R-R",
    (RETURNER, EXPLORER): "R-E",
    (EXPLORER, EXPLORER): "E-E",
    (EXPLORER, RETURNER): "E-R",
}
TRANSITION_ORDER = ["R-R", "R-E", "E-E", "E-R"]


# ============================================================
# 2. Data helpers
# ============================================================

def log(msg: str) -> None:
    """Print a progress message immediately."""
    print(msg, flush=True)


def read_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a .parquet, .csv, or .csv.gz table."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def window_days(window: tuple[int, int]) -> list[int]:
    """Days in an inclusive window, excluding EXCLUDED_DAYS."""
    lo, hi = window
    return [d for d in range(lo, hi + 1) if d not in EXCLUDED_DAYS]


def filter_window(df: pd.DataFrame, window: tuple[int, int]) -> pd.DataFrame:
    """Rows whose day index falls inside an inclusive window."""
    lo, hi = window
    return df[(df["d"] >= lo) & (df["d"] <= hi)]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add day_of_week (0 = Sunday ... 6 = Saturday), weekend and holiday flags.

    Normal days use d % 7. Emergency days are re-indexed so that day 60
    falls on EMERGENCY_FIRST_DAY_DOW.
    """
    df = df.copy()
    d = df["d"]
    emergency = d >= EMERGENCY_WINDOW[0]

    dow = d % 7
    dow = dow.where(
        ~emergency,
        (d - EMERGENCY_WINDOW[0] + EMERGENCY_FIRST_DAY_DOW) % 7,
    )
    df["day_of_week"] = dow.astype("int8")
    df["weekend"] = dow.isin([0, 6]).astype("int8")

    is_holiday = np.where(
        emergency, d.isin(EMERGENCY_HOLIDAYS), d.isin(NORMAL_HOLIDAYS)
    )
    df["holiday"] = (df["weekend"].astype(bool) | is_holiday).astype("int8")
    return df


def collapse_stops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge consecutive records of a user at the same grid cell into a stop.

    The stop keeps all attributes of its first record (day, slot, trip
    distance into the stop, calendar flags, ...) and the summed
    time_spent of the merged records.
    """
    df = df.sort_values(["uid", "d", "t"], kind="mergesort")
    new_stop = (df["uid"] != df["uid"].shift()) | (
        (df["x"] != df["x"].shift()) | (df["y"] != df["y"].shift())
    )
    stop_id = new_stop.cumsum().to_numpy()

    first = df[new_stop.to_numpy()].copy()
    first["time_spent"] = (
        df.groupby(stop_id, sort=False)["time_spent"].sum().to_numpy()
    )
    return first.reset_index(drop=True)


# ============================================================
# 3. Mobility indicators
# ============================================================

def radius_of_gyration(df: pd.DataFrame) -> pd.Series:
    """
    Dwell-time-weighted radius of gyration per user (meters).

        r_g = sqrt( (1/T) * sum_i t_i * |r_i - r_cm|^2 ),

    with r_cm the dwell-time-weighted center of mass. Users with zero
    total time are dropped.
    """
    d = df[["uid", "x", "y", "time_spent"]].copy()
    total = d.groupby("uid")["time_spent"].transform("sum")
    d = d[total > 0]
    d["w"] = d["time_spent"] / total[total > 0]

    center = (
        d.assign(wx=d["w"] * d["x"], wy=d["w"] * d["y"])
        .groupby("uid")[["wx", "wy"]]
        .sum()
    )
    d = d.join(center, on="uid")
    d["s"] = d["w"] * ((d["x"] - d["wx"]) ** 2 + (d["y"] - d["wy"]) ** 2)
    rg = np.sqrt(d.groupby("uid")["s"].sum()) * GRID_CELL_M
    return rg.rename("rg")


def top_k_locations(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    The k grid cells where each user spent the most time.

    Ties in time are broken by (x, y) so the selection is deterministic.
    """
    loc = df.groupby(["uid", "x", "y"], as_index=False)["time_spent"].sum()
    loc = loc.sort_values(
        ["uid", "time_spent", "x", "y"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return loc.groupby("uid", sort=False).head(k)


def k_radius_of_gyration(df: pd.DataFrame, k: int) -> pd.Series:
    """Radius of gyration over each user's top-k locations by time (meters)."""
    return radius_of_gyration(top_k_locations(df, k)).rename(f"rg_k{k}")


def classify(
    df: pd.DataFrame,
    k: int = K_CLASSIFY,
    threshold: float = S_K_THRESHOLD,
    rg: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Classify users as k-returners or k-explorers.

    S_k = r_g^(k) / r_g. A user is a returner if S_k > threshold and an
    explorer otherwise. Users with r_g = 0 cannot be classified and are
    dropped.

    Returns a DataFrame indexed by uid with columns rg, rk, s_k, cls.
    """
    if rg is None:
        rg = radius_of_gyration(df)
    rk = k_radius_of_gyration(df, k).rename("rk")
    out = pd.concat([rg.rename("rg"), rk], axis=1).dropna()
    out = out[out["rg"] > 0]
    out["s_k"] = out["rk"] / out["rg"]
    out["cls"] = np.where(out["s_k"] > threshold, RETURNER, EXPLORER)
    return out


def returner_shares_by_k(
    df: pd.DataFrame, k_values: list[int] = K_VALUES
) -> pd.DataFrame:
    """Percentage of returners and explorers for each k."""
    rg = radius_of_gyration(df)
    rows = []
    for k in k_values:
        c = classify(df, k=k, rg=rg)
        n = len(c)
        n_ret = int((c["cls"] == RETURNER).sum())
        rows.append({
            "k": k,
            "n_users": n,
            "returner_pct": 100 * n_ret / n if n else np.nan,
            "explorer_pct": 100 * (n - n_ret) / n if n else np.nan,
        })
    return pd.DataFrame(rows)


def crossover_k(shares: pd.DataFrame) -> float:
    """Smallest k at which returners outnumber explorers (NaN if none)."""
    hit = shares.loc[shares["returner_pct"] > shares["explorer_pct"], "k"]
    return float(hit.min()) if len(hit) else np.nan


def max_distance_from_home(df: pd.DataFrame) -> pd.Series:
    """Maximum Euclidean distance between any visited cell and home (km)."""
    dist = np.sqrt((df["x"] - df["x_h"]) ** 2 + (df["y"] - df["y_h"]) ** 2)
    return (dist * GRID_CELL_KM).groupby(df["uid"]).max().rename("max_dist_km")


def non_home_dwelling_time(df: pd.DataFrame) -> pd.Series:
    """Total time spent in any cell other than the home cell (hours)."""
    away = (df["x"] != df["x_h"]) | (df["y"] != df["y_h"])
    hours = df["time_spent"].where(away, 0.0)
    return hours.groupby(df["uid"]).sum().rename("non_home_hours")


def real_entropy(stops: pd.DataFrame, window_size: int = 2) -> pd.Series:
    """
    Real entropy of each user's time-ordered stop sequence.

        E(u) = - sum P(T'_u) log2 P(T'_u),

    where T'_u are the consecutive subsequences of length window_size
    (default 2, i.e. ordered pairs of consecutive stops). Users with
    fewer than window_size stops have entropy 0.
    """
    s = stops.sort_values(["uid", "d", "t"], kind="mergesort")
    loc = (s["x"].astype("int64") * 1000 + s["y"].astype("int64")).to_numpy()
    uid = s["uid"].to_numpy()

    keys = [loc]
    valid = np.ones(len(s), dtype=bool)
    for lag in range(1, window_size):
        shifted_uid = np.roll(uid, -lag)
        shifted_loc = np.roll(loc, -lag)
        ok = shifted_uid == uid
        ok[-lag:] = False
        valid &= ok
        keys.append(shifted_loc)

    seq = pd.DataFrame({"uid": uid[valid]})
    for i, key in enumerate(keys):
        seq[f"l{i}"] = key[valid]

    counts = seq.groupby(list(seq.columns)).size().rename("n").reset_index()
    counts["p"] = counts["n"] / counts.groupby("uid")["n"].transform("sum")
    entropy = (-counts["p"] * np.log2(counts["p"])).groupby(counts["uid"]).sum()

    return entropy.reindex(pd.Index(np.unique(uid), name="uid"), fill_value=0.0
                           ).rename("real_entropy")


def in_home_neighborhood(df: pd.DataFrame) -> pd.Series:
    """1 if the record lies within the nearby neighborhood of home, else 0."""
    r = NEIGHBORHOOD_RADIUS_CELLS
    inside = ((df["x"] - df["x_h"]).abs() <= r) & ((df["y"] - df["y_h"]).abs() <= r)
    return inside.astype("int8")


# ============================================================
# 4. Statistical helpers
# ============================================================

def fit_distribution(values_km: np.ndarray, xmin: float = FIT_XMIN_KM) -> dict:
    """
    Fit exponential, lognormal and truncated power-law distributions
    by maximum likelihood (powerlaw package) and compare lognormal with
    the two alternatives through log-likelihood ratios R.
    Positive R favors the lognormal distribution.
    """
    import powerlaw  # imported lazily: only needed for fitting

    values_km = np.asarray(values_km, dtype=float)
    values_km = values_km[np.isfinite(values_km)]
    fit = powerlaw.Fit(values_km, xmin=xmin, discrete=False, verbose=False)
    r_tpl, p_tpl = fit.distribution_compare("lognormal", "truncated_power_law")
    r_exp, p_exp = fit.distribution_compare("lognormal", "exponential")
    return {
        "n": int(len(values_km)),
        "n_above_xmin": int((values_km >= xmin).sum()),
        "xmin_km": fit.lognormal.xmin,
        "lognormal_mu": fit.lognormal.mu,
        "lognormal_sigma": fit.lognormal.sigma,
        "R_lognormal_vs_truncated_power_law": r_tpl,
        "p_lognormal_vs_truncated_power_law": p_tpl,
        "R_lognormal_vs_exponential": r_exp,
        "p_lognormal_vs_exponential": p_exp,
    }


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """
    Rank-biserial correlation from a Mann-Whitney U statistic.
    Positive values mean the first group tends to have larger values.
    """
    return 2 * u_stat / (n1 * n2) - 1


def two_group_tests(
    returners: np.ndarray,
    explorers: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Compare returners and explorers with Mann-Whitney U and two-sample
    Kolmogorov-Smirnov tests. Percentile 95% confidence intervals for the
    rank-biserial correlation and the KS statistic D are obtained by
    resampling users independently within each group.
    """
    r = np.asarray(returners, dtype=float)
    e = np.asarray(explorers, dtype=float)
    r = r[np.isfinite(r)]
    e = e[np.isfinite(e)]
    n_r, n_e = len(r), len(e)

    u_stat, p_mw = mannwhitneyu(r, e, alternative="two-sided")
    rbc = rank_biserial(u_stat, n_r, n_e)
    ks = ks_2samp(r, e, alternative="two-sided")

    rng = np.random.default_rng(seed)
    rbc_boot = np.empty(n_boot)
    ks_boot = np.empty(n_boot)
    for i in range(n_boot):
        rb = rng.choice(r, size=n_r, replace=True)
        eb = rng.choice(e, size=n_e, replace=True)
        u_b, _ = mannwhitneyu(rb, eb, alternative="two-sided")
        rbc_boot[i] = rank_biserial(u_b, n_r, n_e)
        ks_boot[i] = ks_2samp(rb, eb, alternative="two-sided").statistic

    rbc_lo, rbc_hi = np.percentile(rbc_boot, [2.5, 97.5])
    ks_lo, ks_hi = np.percentile(ks_boot, [2.5, 97.5])
    return {
        "n_returners": n_r,
        "n_explorers": n_e,
        "mean_returners": r.mean(),
        "mean_explorers": e.mean(),
        "median_returners": np.median(r),
        "median_explorers": np.median(e),
        "mw_U": u_stat,
        "mw_p": p_mw,
        "rank_biserial": rbc,
        "rank_biserial_ci_low": rbc_lo,
        "rank_biserial_ci_high": rbc_hi,
        "ks_D": ks.statistic,
        "ks_p": ks.pvalue,
        "ks_D_ci_low": ks_lo,
        "ks_D_ci_high": ks_hi,
    }


def epsilon_squared(h_stat: float, n_total: int, n_groups: int) -> float:
    """Kruskal-Wallis effect size, epsilon^2 = (H - k + 1) / (N - k), floored at 0."""
    return max(0.0, (h_stat - n_groups + 1) / (n_total - n_groups))


def holm_adjust(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running_max
    return adjusted


def binned_shares(
    values: pd.Series,
    labels: pd.Series,
    bins: np.ndarray,
) -> pd.DataFrame:
    """
    Share of all users (in percent) falling in each bin, split by label.

    values and labels are Series indexed by uid. Bins are left-closed.
    The percentages of all labels and bins sum to 100 (users outside the
    bin range are counted in the denominator).
    """
    data = pd.concat([values.rename("value"), labels.rename("label")], axis=1).dropna()
    total = len(data)
    centers = bins[:-1] + np.diff(bins) / 2
    data["bin"] = pd.cut(data["value"], bins=bins, labels=centers, right=False)
    counts = data.groupby(["bin", "label"], observed=False).size().rename("n_users")
    out = counts.reset_index()
    out["pct_of_users"] = 100 * out["n_users"] / total if total else np.nan
    return out.rename(columns={"bin": "bin_center"})


def row_shares(values: pd.Series, labels: pd.Series, bins, bin_labels) -> pd.DataFrame:
    """
    Composition of each bin by label (percentages sum to 100 within a bin).
    Used for the outside-nearby-neighborhood analysis.
    """
    data = pd.concat([values.rename("value"), labels.rename("label")], axis=1).dropna()
    data["bin"] = pd.cut(data["value"], bins=bins, labels=bin_labels, right=False)
    counts = data.groupby(["bin", "label"], observed=False).size().unstack(fill_value=0)
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    out = counts.add_prefix("n_").join(pct.add_suffix("_pct"))
    return out.reset_index()
