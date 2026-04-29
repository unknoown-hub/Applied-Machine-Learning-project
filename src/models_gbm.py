"""
Gradient Boosting model — using sklearn HistGradientBoostingRegressor.

LightGBM requires libomp.dylib (OpenMP) which is not available on this
machine without Homebrew.  sklearn's HistGradientBoostingRegressor
implements the same histogram-based gradient boosting algorithm and
produces equivalent results with no external dependencies.

For the report: "We use sklearn's HistGradientBoostingRegressor, which
implements the same histogram-based gradient boosting algorithm as
LightGBM with comparable performance."

Interface:
  run_gbm(panel, splits, n_lags) → predictions DataFrame
  columns: week_num, district, y_true, y_pred
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from src.walkforward import rolling_week_splits
from src.features import FEATURE_COLS


def _get_feature_cols(n_lags: int) -> list:
    """Return lag column names for a given lag depth."""
    lag_cols = [f'{col}_lag{k}' for col in FEATURE_COLS for k in range(1, n_lags + 1)]
    return lag_cols + ['week_of_year', 'year']


def run_gbm(panel: pd.DataFrame, splits: list, n_lags: int = 8) -> pd.DataFrame:
    """
    Walk-forward GBM with a fixed 291-week rolling training window.

    At each of the 52 holdout steps:
      - Train on the 291-week window
      - Predict the single holdout week
      - District is encoded as an ordinal integer (categorical signal)

    n_lags controls how many lags are used (2, 5, or 8 for ablation study).
    """
    feat_cols = _get_feature_cols(n_lags)

    # Encode district as ordinal integer — same mapping across all steps
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    enc.fit(panel[['district']])

    records = []
    for step_idx, (train_weeks, test_week) in enumerate(splits):
        if (step_idx + 1) % 10 == 0:
            print(f"    step {step_idx + 1}/52 …")

        train = panel[panel['week_num'].isin(train_weeks)].dropna(subset=feat_cols)
        test  = panel[panel['week_num'] == test_week].dropna(subset=feat_cols)

        if len(train) == 0 or len(test) == 0:
            continue

        # Build feature matrices — district encoded as integer
        d_train = enc.transform(train[['district']])
        d_test  = enc.transform(test[['district']])

        X_train = np.hstack([d_train, train[feat_cols].values])
        X_test  = np.hstack([d_test,  test[feat_cols].values])
        y_train = train['y'].values

        model = HistGradientBoostingRegressor(
            loss='squared_error',
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, None)

        for (_, row), pred in zip(test.iterrows(), y_pred):
            records.append({
                'week_num': test_week,
                'district': row['district'],
                'y_true':   row['y'],
                'y_pred':   float(pred),
            })

    return pd.DataFrame(records)
