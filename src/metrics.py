"""
Evaluation metrics — matching Z&T's reporting exactly.

From the paper:
  "we report RMSE, MAE, MAPE and correlation of point predictions and true
   values.  All criteria are averaged across the 52 hold-out periods.
   RMSE, MAE, and MAPE are reported relative to the random walk benchmark."

So the procedure is:
  1. For each of the 52 holdout weeks, compute RMSE / MAE / MAPE / Corr
     across all 74 districts in that week.
  2. Average those 52 per-week values.
  3. Divide RMSE / MAE / MAPE by the random walk's corresponding values.
     Correlation is reported as-is (not relative).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # eps=1.0 avoids division by zero on zero-displacement weeks
    denom = np.where(np.abs(y_true) < 1.0, 1.0, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Symmetric MAPE: treats over- and under-forecasting equally
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.std() < 1e-9 or y_pred.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def per_week_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RMSE, MAE, MAPE, Corr for each holdout week separately.
    predictions_df must have columns: week_num, y_true, y_pred.
    Returns a DataFrame with one row per holdout week.
    """
    rows = []
    for week, grp in predictions_df.groupby('week_num'):
        yt = grp['y_true'].values
        yp = grp['y_pred'].values
        rows.append({
            'week_num': week,
            'rmse':  _rmse(yt, yp),
            'mae':   _mae(yt, yp),
            'mape':  _mape(yt, yp),
            'smape': _smape(yt, yp),
            'corr':  _corr(yt, yp),
        })
    return pd.DataFrame(rows)


def evaluate(predictions_df: pd.DataFrame) -> dict:
    """
    Average per-week metrics across all 52 holdout weeks.
    Returns absolute (non-relative) metrics.
    """
    pw = per_week_metrics(predictions_df)
    return {
        'rmse':  float(pw['rmse'].mean()),
        'mae':   float(pw['mae'].mean()),
        'mape':  float(pw['mape'].mean()),
        'smape': float(pw['smape'].mean()),
        'corr':  float(pw['corr'].mean()),
    }


def build_results_table(model_preds: dict) -> pd.DataFrame:
    """
    Build the final results table with metrics relative to random walk.

    model_preds — dict mapping model name → predictions DataFrame
                  (must include 'Random Walk' as the denominator)

    Returns a DataFrame matching Z&T Table 1 format.
    """
    assert 'Random Walk' in model_preds, "Random Walk must be included to compute relative metrics"

    raw = {name: evaluate(df) for name, df in model_preds.items()}
    rw  = raw['Random Walk']

    rows = []
    for name, m in raw.items():
        rows.append({
            'Model':  name,
            'RMSE':   round(m['rmse']  / rw['rmse'],  3),
            'MAE':    round(m['mae']   / rw['mae'],   3),
            'MAPE':   round(m['mape']  / rw['mape'],  3),
            'sMAPE':  round(m['smape'] / rw['smape'], 3),
            'Corr':   round(m['corr'], 3),
        })
    return pd.DataFrame(rows)
