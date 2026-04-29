"""
Stage 2 — Feature engineering.

Produces two representations from data/processed/panel.parquet:

  1. Tabular features  → data/processed/panel_tabular.parquet
     For LightGBM. Each row is a flat feature vector:
     40 lag columns (5 variables × 8 lags) + calendar + district (categorical).

  2. Sequence arrays   → data/processed/sequences.npz
     For LSTM and Transformer. Each sample is a sliding-window of raw values.
     Window length is a parameter of build_sequences() — not fixed here.

Both are built from the same clean panel.  Both use week_num to align with the
52-step walk-forward splits in walkforward.py — no split logic lives here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROC = ROOT / 'data' / 'processed'

# Variables to lag / use in sequences — target first, then conflict types
FEATURE_COLS = ['y', 'battles', 'explosions', 'strategic_dev', 'viol_civ']
N_LAGS       = 8   # matches Z&T lag window of t−1 … t−8


# ── 1. Tabular features for LightGBM ──────────────────────────────────────────

def build_tabular(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 40 lag columns to the panel (5 variables × 8 lags, computed within
    each district so no cross-district leakage).

    Also adds:
      week_of_year  — integer 1–53, captures seasonal pattern
      year          — integer 2017–2023, captures long-run trend

    District is kept as a string column so LightGBM can use it as a native
    categorical without one-hot encoding.

    The first 8 rows per district have NaN lags and are dropped.
    Final row count: 25,382 − (74 × 8) = 24,790.
    """
    df = panel.copy().sort_values(['district', 'week_start']).reset_index(drop=True)

    # Compute lags within each district — shift(k) on the grouped series
    for col in FEATURE_COLS:
        grp = df.groupby('district')[col]
        for k in range(1, N_LAGS + 1):
            df[f'{col}_lag{k}'] = grp.shift(k)

    # Calendar features
    df['week_of_year'] = df['week_start'].dt.isocalendar().week.astype(int)
    df['year']         = df['week_start'].dt.year

    # Drop rows where any lag is NaN (the first 8 weeks per district)
    lag_cols = [f'{col}_lag{k}' for col in FEATURE_COLS for k in range(1, N_LAGS + 1)]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    return df


# ── 2. Sequence arrays for LSTM / Transformer ─────────────────────────────────

def build_sequences(panel: pd.DataFrame, seq_len: int = 8) -> dict:
    """
    Builds sliding-window sequence arrays for every (district, week) pair
    where a full seq_len-step history exists.

    seq_len is intentionally a parameter — you may want to try different
    window lengths (e.g. 4, 8, 12) during model training without rebuilding
    the tabular features.  Pass the chosen value when calling this function.

    For each sample at time t, the input window covers [t−seq_len … t−1].
    The target is y at time t.

    Returns a dict with:
      X_seq          — float32 array, shape (n_samples, seq_len, 5)
                       5 features per timestep: [y, battles, explosions,
                       strategic_dev, viol_civ]  (same order as FEATURE_COLS)
      X_dist         — int32 array, shape (n_samples,)
                       District index 0–73 for the embedding layer
      y              — float32 array, shape (n_samples,)
                       Log(1+outflows) at time t
      week_num       — int32 array, shape (n_samples,)
                       Sequential week index 1–343 at time t; used by the
                       walk-forward evaluator to split train / test
      district_names — list of 74 district names in index order
                       (district_names[X_dist[i]] gives the district string)
      seq_len        — the seq_len value used, stored for reference
    """
    districts      = sorted(panel['district'].unique())
    district_index = {d: i for i, d in enumerate(districts)}

    seqs, dist_ids, targets, week_nums = [], [], [], []

    for district in districts:
        d_id  = district_index[district]
        sub   = (panel[panel['district'] == district]
                 .sort_values('week_start')
                 .reset_index(drop=True))
        vals  = sub[FEATURE_COLS].values   # (343, 5) — all weeks for this district
        wnums = sub['week_num'].values      # (343,)

        # Slide a seq_len-step window: window covers [t-seq_len : t], target is t
        for t in range(seq_len, len(sub)):
            seqs.append(vals[t - seq_len : t])    # (seq_len, 5)
            dist_ids.append(d_id)
            targets.append(float(vals[t, 0]))      # y is first column in FEATURE_COLS
            week_nums.append(int(wnums[t]))

    return {
        'X_seq'         : np.array(seqs,      dtype=np.float32),
        'X_dist'        : np.array(dist_ids,  dtype=np.int32),
        'y'             : np.array(targets,   dtype=np.float32),
        'week_num'      : np.array(week_nums, dtype=np.int32),
        'district_names': districts,
        'seq_len'       : seq_len,
    }


# ── 3. Entry point ─────────────────────────────────────────────────────────────

def build_and_save() -> None:
    print("Loading panel …")
    panel = pd.read_parquet(PROC / 'panel.parquet')
    print(f"  Panel loaded: {panel.shape[0]:,} rows × {panel.shape[1]} cols")

    # --- Tabular ---
    print("Building tabular features for LightGBM …")
    tabular = build_tabular(panel)
    lag_cols = [f'{col}_lag{k}' for col in FEATURE_COLS for k in range(1, N_LAGS + 1)]
    print(f"  Rows after dropping NaN lags : {len(tabular):,}  (expected 24,790)")
    print(f"  Lag columns added            : {len(lag_cols)}  (expected 40)")
    print(f"  All columns                  : {tabular.columns.tolist()}")
    tabular.to_parquet(PROC / 'panel_tabular.parquet', index=False)
    print(f"  Saved data/processed/panel_tabular.parquet")

    # --- Sequences (default seq_len=8, matching N_LAGS) ---
    seq_len = 8
    print(f"Building sequence arrays for LSTM / Transformer (seq_len={seq_len}) …")
    seqs = build_sequences(panel, seq_len=seq_len)
    print(f"  X_seq shape    : {seqs['X_seq'].shape}   (n_samples, {seq_len} steps, 5 features)")
    print(f"  X_dist shape   : {seqs['X_dist'].shape}")
    print(f"  y shape        : {seqs['y'].shape}")
    print(f"  week_num range : {seqs['week_num'].min()} – {seqs['week_num'].max()}")
    print(f"  Districts      : {len(seqs['district_names'])}")

    np.savez_compressed(
        PROC / 'sequences.npz',
        X_seq          = seqs['X_seq'],
        X_dist         = seqs['X_dist'],
        y              = seqs['y'],
        week_num       = seqs['week_num'],
        district_names = np.array(seqs['district_names']),
    )
    print(f"  Saved data/processed/sequences.npz")

    expected = 74 * (343 - seq_len)
    assert len(seqs['y']) == expected, \
        f"Expected {expected} sequence samples, got {len(seqs['y'])}"
    print(f"  Sequence count check: {len(seqs['y']):,} == {expected:,}  ✓")


if __name__ == '__main__':
    build_and_save()
