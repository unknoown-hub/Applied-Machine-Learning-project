"""
Baseline models — replicating all benchmarks from Z&T Table 1.

All baselines follow the same interface:
  run_<model>(panel, splits) → predictions DataFrame
  columns: week_num, district, y_true, y_pred

Baselines implemented:
  1. Random Walk          — ŷ_t = y_{t-1}
  2. AR(1)                — per-district OLS: y_t = α + β·y_{t-1}, refit each step
  3. Rolling Average (w)  — ŷ_t = mean(y_{t-1}, ..., y_{t-w})
  4. Long Run Mean        — ŷ_t = mean of all y in training window for that district
  5. Long Run Median      — ŷ_t = median of all y in training window for that district
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.walkforward import rolling_week_splits


def _base_records(panel: pd.DataFrame, splits: list) -> list:
    """Pre-extract arrays once to avoid repeated pandas indexing inside loops."""
    return panel.set_index(['district', 'week_num'])


def run_random_walk(panel: pd.DataFrame, splits: list) -> pd.DataFrame:
    """ŷ_t = y_{t-1} for each district — uses y_lag1 column directly."""
    records = []
    for train_weeks, test_week in splits:
        test = panel[panel['week_num'] == test_week]
        for _, row in test.iterrows():
            records.append({
                'week_num':  test_week,
                'district':  row['district'],
                'y_true':    row['y'],
                'y_pred':    row['y_lag1'],   # y at t-1, already in panel_tabular
            })
    return pd.DataFrame(records)


def run_ar1(panel: pd.DataFrame, splits: list) -> pd.DataFrame:
    """
    Per-district AR(1): y_t = α_i + β_i · y_{t-1}.
    OLS refitted at every walk-forward step using only the training window.
    """
    districts = sorted(panel['district'].unique())
    records   = []

    for train_weeks, test_week in splits:
        train = panel[panel['week_num'].isin(train_weeks)]
        test  = panel[panel['week_num'] == test_week]

        # Fit OLS per district on training window
        coefs = {}
        for d in districts:
            sub = train[train['district'] == d][['y_lag1', 'y']].dropna()
            if len(sub) < 3:
                coefs[d] = (0.0, 1.0)
                continue
            A = np.column_stack([np.ones(len(sub)), sub['y_lag1'].values])
            c, *_ = np.linalg.lstsq(A, sub['y'].values, rcond=None)
            coefs[d] = (float(c[0]), float(c[1]))

        for _, row in test.iterrows():
            alpha, beta = coefs.get(row['district'], (0.0, 1.0))
            y_pred = max(0.0, alpha + beta * row['y_lag1'])
            records.append({
                'week_num': test_week,
                'district': row['district'],
                'y_true':   row['y'],
                'y_pred':   y_pred,
            })
    return pd.DataFrame(records)


def run_rolling_average(panel: pd.DataFrame, splits: list, window: int) -> pd.DataFrame:
    """ŷ_t = mean of the last `window` values of y before t, per district."""
    records = []
    for train_weeks, test_week in splits:
        # Use only the most recent `window` training weeks per district
        recent_weeks = sorted(train_weeks)[-window:]
        recent = panel[panel['week_num'].isin(recent_weeks)]
        means  = recent.groupby('district')['y'].mean()

        test = panel[panel['week_num'] == test_week]
        for _, row in test.iterrows():
            records.append({
                'week_num': test_week,
                'district': row['district'],
                'y_true':   row['y'],
                'y_pred':   float(means.get(row['district'], 0.0)),
            })
    return pd.DataFrame(records)


def run_longrun_mean(panel: pd.DataFrame, splits: list) -> pd.DataFrame:
    """ŷ_t = mean of all y in the training window for that district."""
    records = []
    for train_weeks, test_week in splits:
        train = panel[panel['week_num'].isin(train_weeks)]
        means = train.groupby('district')['y'].mean()
        test  = panel[panel['week_num'] == test_week]
        for _, row in test.iterrows():
            records.append({
                'week_num': test_week,
                'district': row['district'],
                'y_true':   row['y'],
                'y_pred':   float(means.get(row['district'], 0.0)),
            })
    return pd.DataFrame(records)


def run_longrun_median(panel: pd.DataFrame, splits: list) -> pd.DataFrame:
    """ŷ_t = median of all y in the training window for that district."""
    records = []
    for train_weeks, test_week in splits:
        train   = panel[panel['week_num'].isin(train_weeks)]
        medians = train.groupby('district')['y'].median()
        test    = panel[panel['week_num'] == test_week]
        for _, row in test.iterrows():
            records.append({
                'week_num': test_week,
                'district': row['district'],
                'y_true':   row['y'],
                'y_pred':   float(medians.get(row['district'], 0.0)),
            })
    return pd.DataFrame(records)


def run_all_baselines(panel: pd.DataFrame) -> dict:
    """
    Run all baselines and return a dict of predictions DataFrames.
    panel must be panel_tabular (contains y_lag1 column).
    """
    splits = rolling_week_splits()
    print("Running baselines …")

    results = {}

    print("  Random Walk …")
    results['Random Walk'] = run_random_walk(panel, splits)

    print("  AR(1) …")
    results['AR(1)'] = run_ar1(panel, splits)

    print("  Rolling Average (4 wks) …")
    results['Rolling Avg (4)'] = run_rolling_average(panel, splits, window=4)

    print("  Rolling Average (8 wks) …")
    results['Rolling Avg (8)'] = run_rolling_average(panel, splits, window=8)

    print("  Rolling Average (12 wks) …")
    results['Rolling Avg (12)'] = run_rolling_average(panel, splits, window=12)

    print("  Long Run Mean …")
    results['Long Run Mean'] = run_longrun_mean(panel, splits)

    print("  Long Run Median …")
    results['Long Run Median'] = run_longrun_median(panel, splits)

    print("  Done.")
    return results
