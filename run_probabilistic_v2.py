"""
Probabilistic forecasting — v2.
Key change over v1: replaces Gaussian CRPS with empirical CRPS via the
pinball-loss integral (Gneiting & Raftery, 2007; Laio & Tamea, 2007).
Quantile GBM is retrained at 19 quantile levels so the CRPS approximation
is essentially exact. Conformal models are scored at their 3 stored quantiles
(q05, median, q95) — coarser but assumption-free.
"""
import sys
sys.path.insert(0, '/Users/noha/Desktop/Hilary_term /Applied ML/Summative/project')

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT = Path('/Users/noha/Desktop/Hilary_term /Applied ML/Summative/project')

from src.walkforward import rolling_week_splits
from src.features import FEATURE_COLS, build_tabular

MODEL_COLORS = {
    'Random Walk'    : '#999999',
    'AR(1)'          : '#56B4E9',
    'Rolling Avg (4)': '#009E73',
    'Rolling Avg (8)': '#00836a',
    'Rolling Avg(12)': '#006452',
    'GBM (lags=5)'   : '#D55E00',
    'Quantile GBM'   : '#E69F00',
    'LSTM (seq=5)'   : '#9B3F80',
    'Hurdle (lags=5)': '#0072B2',
}

# Quantile levels
TAUS_3  = np.array([0.05, 0.50, 0.95])
TAUS_19 = np.round(np.arange(0.05, 1.00, 0.05), 2)  # [0.05 … 0.95], 19 levels

print('=== Section 1: Load data and splits ===')
panel   = pd.read_parquet(ROOT / 'data' / 'processed' / 'panel.parquet')
tabular = build_tabular(panel)
splits  = rolling_week_splits()
print(f'Panel   : {panel.shape}')
print(f'Tabular : {tabular.shape}')
print(f'Splits  : {len(splits)} walk-forward steps')


# ── Metrics ───────────────────────────────────────────────────────────────────

print('\n=== Section 2: Metric functions ===')

def empirical_crps(y_true, q_preds, taus):
    """
    CRPS via pinball-loss integral (Gneiting & Raftery 2007, eq. 21).

    CRPS(y) = 2 * mean_k [ rho_{tau_k}(y, q_k) ]
    No parametric distribution assumed.

    y_true  : (N,)
    q_preds : (N, K)  predicted quantile levels
    taus    : (K,)    quantile levels in (0,1)
    Returns : (N,)    per-observation CRPS
    """
    y = np.asarray(y_true, dtype=float)[:, None]   # (N,1)
    q = np.asarray(q_preds, dtype=float)            # (N,K)
    t = np.asarray(taus,    dtype=float)[None, :]   # (1,K)
    diff    = y - q
    pinball = np.where(diff >= 0, t * diff, (t - 1) * diff)
    return 2.0 * pinball.mean(axis=1)               # (N,)


def pinball_loss(y_true, y_pred_q, q):
    e = y_true - y_pred_q
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def coverage(y_true, lower, upper):
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def _q_col(tau):
    return f'q{int(round(tau * 100)):02d}'


def evaluate_probabilistic(preds_df, taus=TAUS_3):
    """Score a predictions DataFrame.

    For conformal models   → pass taus=TAUS_3  (3 quantiles: q05, q50=y_pred, q95)
    For Quantile GBM       → pass taus=TAUS_19 (19 columns q05…q95 in the df)
    """
    yt  = preds_df['y_true'].values
    mu  = preds_df['y_pred'].values
    q05 = preds_df['q05'].values
    q95 = preds_df['q95'].values

    if len(taus) == 3:
        # conformal: 3 stored quantiles
        q_mat = np.column_stack([q05, mu, q95])
    else:
        # Quantile GBM: 19 columns
        cols  = [_q_col(t) for t in taus]
        q_mat = preds_df[cols].values

    crps_vals = empirical_crps(yt, q_mat, taus)
    return {
        'CRPS (mean)'     : round(float(crps_vals.mean()), 4),
        'Coverage 90%'    : round(coverage(yt, q05, q95), 3),
        'Interval Width'  : round(float((q95 - q05).mean()), 4),
        'Pinball (q=0.10)': round(pinball_loss(yt, q05, 0.10), 4),
        'Pinball (q=0.50)': round(pinball_loss(yt, mu,  0.50), 4),
        'Pinball (q=0.90)': round(pinball_loss(yt, q95, 0.90), 4),
        'RMSE (point)'    : round(float(np.sqrt(np.mean((yt - mu)**2))), 4),
    }


# R² helper — computed globally across all 3848 test observations
def compute_r2(preds_df, panel):
    from src.walkforward import FIRST_TEST
    y_test_all = panel[panel['week_num'] >= FIRST_TEST]['y'].values
    ss_tot = float(np.sum((y_test_all - y_test_all.mean()) ** 2))
    yt = preds_df['y_true'].values
    yp = preds_df['y_pred'].values
    ss_res = float(np.sum((yt - yp) ** 2))
    return round(1.0 - ss_res / ss_tot, 3)


print('Metric functions defined.')


# ── Conformal wrapper ──────────────────────────────────────────────────────────

print('\n=== Section 3: Conformal wrapper ===')

def conformal_intervals(cal_residuals, point_pred, alpha=0.10):
    n     = len(cal_residuals)
    level = min((1 - alpha) * (1 + 1 / n), 1.0)
    q_hat = float(np.quantile(cal_residuals, level))
    return (max(0.0, point_pred - q_hat), point_pred + q_hat)

print('conformal_intervals() defined.')


# ── Probabilistic baselines ────────────────────────────────────────────────────

print('\n=== Section 4: Probabilistic baselines ===')

def run_rw_probabilistic(panel, splits, alpha=0.10):
    records = []
    for train_weeks, test_week in splits:
        cal_weeks = sorted(train_weeks)[-26:]
        cal_data  = panel[panel['week_num'].isin(cal_weeks)].copy()
        cal_resid = np.abs(cal_data['y'].values - cal_data['y_lag1'].fillna(0).values)
        test = panel[panel['week_num'] == test_week]
        for _, row in test.iterrows():
            pt = max(0.0, float(row['y_lag1']))
            q05, q95 = conformal_intervals(cal_resid, pt, alpha)
            records.append({'week_num': test_week, 'district': row['district'],
                            'y_true': row['y'], 'y_pred': pt,
                            'q05': q05, 'q95': q95})
    return pd.DataFrame(records)


def run_ar1_probabilistic(panel, splits, alpha=0.10):
    districts = sorted(panel['district'].unique())
    records   = []
    for train_weeks, test_week in splits:
        train     = panel[panel['week_num'].isin(train_weeks)]
        cal_weeks = sorted(train_weeks)[-26:]
        coefs = {}
        for d in districts:
            sub = train[train['district'] == d][['y_lag1','y']].dropna()
            if len(sub) < 3:
                coefs[d] = (0.0, 1.0); continue
            A = np.column_stack([np.ones(len(sub)), sub['y_lag1'].values])
            c, *_ = np.linalg.lstsq(A, sub['y'].values, rcond=None)
            coefs[d] = (float(c[0]), float(c[1]))
        cal_data  = panel[panel['week_num'].isin(cal_weeks)]
        cal_resid = []
        for _, row in cal_data.iterrows():
            a, b = coefs.get(row['district'], (0.0, 1.0))
            pt = max(0.0, a + b * float(row['y_lag1']))
            cal_resid.append(abs(row['y'] - pt))
        cal_resid = np.array(cal_resid)
        test = panel[panel['week_num'] == test_week]
        for _, row in test.iterrows():
            a, b = coefs.get(row['district'], (0.0, 1.0))
            pt   = max(0.0, a + b * float(row['y_lag1']))
            q05, q95 = conformal_intervals(cal_resid, pt, alpha)
            records.append({'week_num': test_week, 'district': row['district'],
                            'y_true': row['y'], 'y_pred': pt,
                            'q05': q05, 'q95': q95})
    return pd.DataFrame(records)


def run_rolling_avg_probabilistic(panel, splits, window=4, alpha=0.10):
    records = []
    for train_weeks, test_week in splits:
        cal_weeks    = sorted(train_weeks)[-26:]
        recent_weeks = sorted(train_weeks)[-window:]
        means    = panel[panel['week_num'].isin(recent_weeks)].groupby('district')['y'].mean()
        cal_data = panel[panel['week_num'].isin(cal_weeks)]
        cal_resid = []
        for _, row in cal_data.iterrows():
            pt = float(means.get(row['district'], 0.0))
            cal_resid.append(abs(row['y'] - pt))
        cal_resid = np.array(cal_resid)
        test = panel[panel['week_num'] == test_week]
        for _, row in test.iterrows():
            pt = float(means.get(row['district'], 0.0))
            q05, q95 = conformal_intervals(cal_resid, pt, alpha)
            records.append({'week_num': test_week, 'district': row['district'],
                            'y_true': row['y'], 'y_pred': pt,
                            'q05': q05, 'q95': q95})
    return pd.DataFrame(records)


print('Running Random Walk ...')
rw_prob  = run_rw_probabilistic(tabular, splits)
print(f'  {len(rw_prob)} predictions')

print('Running AR(1) ...')
ar1_prob = run_ar1_probabilistic(tabular, splits)
print(f'  {len(ar1_prob)} predictions')

print('Running Rolling Avg (4) ...')
ra4_prob = run_rolling_avg_probabilistic(tabular, splits, window=4)
print(f'  {len(ra4_prob)} predictions')

print('Running Rolling Avg (8) ...')
ra8_prob = run_rolling_avg_probabilistic(tabular, splits, window=8)
print(f'  {len(ra8_prob)} predictions')

print('Running Rolling Avg (12) ...')
ra12_prob = run_rolling_avg_probabilistic(tabular, splits, window=12)
print(f'  {len(ra12_prob)} predictions')


# ── GBM (conformal) ───────────────────────────────────────────────────────────

print('\n=== Section 5: GBM (conformal) ===')

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


def _feat_cols(n_lags):
    return [f'{c}_lag{k}' for c in FEATURE_COLS for k in range(1, n_lags+1)] \
         + ['week_of_year', 'year']


def run_gbm_probabilistic(panel, splits, n_lags=5, alpha=0.10):
    feat_cols = _feat_cols(n_lags)
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    enc.fit(panel[['district']])
    records = []
    for step_idx, (train_weeks, test_week) in enumerate(splits):
        if (step_idx + 1) % 10 == 0:
            print(f'    step {step_idx+1}/52 ...')
        cal_weeks  = sorted(train_weeks)[-26:]
        ptr_weeks  = sorted(train_weeks)[:-26]
        ptr  = panel[panel['week_num'].isin(ptr_weeks)].dropna(subset=feat_cols)
        cal  = panel[panel['week_num'].isin(cal_weeks)].dropna(subset=feat_cols)
        test = panel[panel['week_num'] == test_week].dropna(subset=feat_cols)
        if len(ptr) == 0 or len(test) == 0:
            continue
        X_ptr  = np.hstack([enc.transform(ptr[['district']]),  ptr[feat_cols].values])
        X_cal  = np.hstack([enc.transform(cal[['district']]),  cal[feat_cols].values])
        X_test = np.hstack([enc.transform(test[['district']]), test[feat_cols].values])
        model = HistGradientBoostingRegressor(
            loss='squared_error', max_iter=500, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
        model.fit(X_ptr, ptr['y'].values)
        cal_preds  = np.clip(model.predict(X_cal), 0, None)
        cal_resid  = np.abs(cal['y'].values - cal_preds)
        test_preds = np.clip(model.predict(X_test), 0, None)
        for (_, row), pt in zip(test.iterrows(), test_preds):
            q05, q95 = conformal_intervals(cal_resid, float(pt), alpha)
            records.append({'week_num': test_week, 'district': row['district'],
                            'y_true': row['y'], 'y_pred': float(pt),
                            'q05': q05, 'q95': q95})
    return pd.DataFrame(records)


print('Running GBM (conformal, lags=5) ...')
gbm_prob = run_gbm_probabilistic(tabular, splits, n_lags=5)
print(f'Done. {len(gbm_prob)} predictions.')


# ── Quantile GBM — 19 quantiles ───────────────────────────────────────────────

print('\n=== Section 6: Quantile GBM (19 quantiles) ===')
print(f'Tau levels: {list(TAUS_19)}')


def run_quantile_gbm_19(panel, splits, n_lags=5, taus=TAUS_19):
    """
    Trains 19 separate quantile GBMs per walk-forward step.
    Stores each quantile as column q05, q10, ..., q95.
    Point prediction = q50 (median).
    """
    feat_cols = _feat_cols(n_lags)
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    enc.fit(panel[['district']])
    records = []
    BASE = dict(max_iter=400, learning_rate=0.05,
                max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
    for step_idx, (train_weeks, test_week) in enumerate(splits):
        if (step_idx + 1) % 10 == 0:
            print(f'    step {step_idx+1}/52 ({len(taus)} quantiles) ...')
        train = panel[panel['week_num'].isin(train_weeks)].dropna(subset=feat_cols)
        test  = panel[panel['week_num'] == test_week].dropna(subset=feat_cols)
        if len(train) == 0 or len(test) == 0:
            continue
        X_tr = np.hstack([enc.transform(train[['district']]), train[feat_cols].values])
        X_te = np.hstack([enc.transform(test[['district']]),  test[feat_cols].values])
        y_tr = train['y'].values

        # Train one model per quantile level
        q_preds = {}
        for tau in taus:
            m = HistGradientBoostingRegressor(loss='quantile', quantile=tau, **BASE)
            m.fit(X_tr, y_tr)
            q_preds[tau] = np.clip(m.predict(X_te), 0, None)

        q50 = q_preds[0.50]
        q05 = q_preds[0.05]
        q95 = q_preds[0.95]

        for i, (_, row) in enumerate(test.iterrows()):
            record = {
                'week_num': test_week,
                'district': row['district'],
                'y_true'  : row['y'],
                'y_pred'  : float(q50[i]),
                'q05'     : float(q05[i]),
                'q95'     : float(q95[i]),
            }
            for tau in taus:
                record[_q_col(tau)] = float(q_preds[tau][i])
            records.append(record)
    return pd.DataFrame(records)


print('Running Quantile GBM (19 quantiles × 52 steps = 988 model fits) ...')
print('This takes ~30-50 minutes. Progress printed every 10 steps.')
qgbm_prob = run_quantile_gbm_19(tabular, splits, n_lags=5)
print(f'Done. {len(qgbm_prob)} predictions, {len(TAUS_19)} quantile columns.')


# ── LSTM (conformal) ──────────────────────────────────────────────────────────

print('\n=== Section 7: LSTM (conformal) ===')

import torch
import torch.nn as nn
from src.features import build_sequences
from src.models_lstm import LSTMForecaster, _standardise, _train_one_step


def run_lstm_probabilistic(panel, splits, seq_len=5, seed=42, alpha=0.10):
    seqs           = build_sequences(panel, seq_len=seq_len)
    X_seq_all      = seqs['X_seq']
    X_dist_all     = seqs['X_dist']
    y_all          = seqs['y']
    week_all       = seqs['week_num']
    district_names = seqs['district_names']
    records = []
    for step_idx, (train_weeks, test_week) in enumerate(splits):
        if (step_idx + 1) % 10 == 0:
            print(f'    step {step_idx+1}/52 (seed={seed}) ...')
        tr_mask   = np.isin(week_all, list(train_weeks))
        te_mask   = week_all == test_week
        val_weeks = sorted(train_weeks)[-26:]
        val_mask  = np.isin(week_all, val_weeks)
        ptr_mask  = tr_mask & ~val_mask
        if ptr_mask.sum() == 0 or te_mask.sum() == 0:
            continue
        X_seq_tr  = X_seq_all[ptr_mask]
        X_seq_val = X_seq_all[val_mask]
        X_seq_te  = X_seq_all[te_mask]
        X_seq_tr_n,  X_seq_val_n = _standardise(X_seq_tr,  X_seq_val)
        _,           X_seq_te_n  = _standardise(X_seq_tr,  X_seq_te)
        model = _train_one_step(
            X_seq_tr_n,  X_dist_all[ptr_mask], y_all[ptr_mask],
            X_seq_val_n, X_dist_all[val_mask],  y_all[val_mask],
            seq_len=seq_len, seed=seed,
        )
        model.eval()
        with torch.no_grad():
            val_preds = model(
                torch.tensor(X_seq_val_n, dtype=torch.float32),
                torch.tensor(X_dist_all[val_mask], dtype=torch.long),
            ).numpy()
            te_preds = model(
                torch.tensor(X_seq_te_n, dtype=torch.float32),
                torch.tensor(X_dist_all[te_mask], dtype=torch.long),
            ).numpy()
        val_preds = np.clip(val_preds, 0, None)
        te_preds  = np.clip(te_preds,  0, None)
        cal_resid = np.abs(y_all[val_mask] - val_preds)
        dist_names_te = [district_names[i] for i in X_dist_all[te_mask]]
        for district, y_true, pt in zip(dist_names_te, y_all[te_mask], te_preds):
            q05, q95 = conformal_intervals(cal_resid, float(pt), alpha)
            records.append({'week_num': test_week, 'district': district,
                            'y_true': float(y_true), 'y_pred': float(pt),
                            'q05': q05, 'q95': q95})
    return pd.DataFrame(records)


print('Running LSTM (conformal, seq=5) ...')
lstm_prob = run_lstm_probabilistic(panel, splits, seq_len=5, seed=42)
print(f'Done. {len(lstm_prob)} predictions.')


# ── Hurdle (conformal) ────────────────────────────────────────────────────────

print('\n=== Section 8: Hurdle (conformal) ===')

from sklearn.ensemble import HistGradientBoostingClassifier


def run_hurdle_probabilistic(panel, splits, n_lags=5, alpha=0.10):
    feat_cols = _feat_cols(n_lags)
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    enc.fit(panel[['district']])
    records = []
    for step_idx, (train_weeks, test_week) in enumerate(splits):
        if (step_idx + 1) % 10 == 0:
            print(f'    step {step_idx+1}/52 ...')
        cal_weeks = sorted(train_weeks)[-26:]
        ptr_weeks = sorted(train_weeks)[:-26]
        ptr  = panel[panel['week_num'].isin(ptr_weeks)].dropna(subset=feat_cols)
        cal  = panel[panel['week_num'].isin(cal_weeks)].dropna(subset=feat_cols)
        test = panel[panel['week_num'] == test_week].dropna(subset=feat_cols)
        if len(ptr) == 0 or len(test) == 0:
            continue
        X_ptr = np.hstack([enc.transform(ptr[['district']]),  ptr[feat_cols].values])
        X_cal = np.hstack([enc.transform(cal[['district']]),  cal[feat_cols].values])
        X_te  = np.hstack([enc.transform(test[['district']]), test[feat_cols].values])
        z_ptr = (ptr['y'].values > 0).astype(int)
        BASE  = dict(max_iter=300, learning_rate=0.05,
                     max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
        clf = HistGradientBoostingClassifier(**BASE)
        clf.fit(X_ptr, z_ptr)
        nz_mask = ptr['y'].values > 0
        reg = HistGradientBoostingRegressor(loss='squared_error', max_iter=500,
                                            learning_rate=0.05, max_leaf_nodes=31,
                                            min_samples_leaf=10, random_state=42)
        if nz_mask.sum() >= 10:
            reg.fit(X_ptr[nz_mask], ptr['y'].values[nz_mask])
            cal_count = np.clip(reg.predict(X_cal), 0, None)
            te_count  = np.clip(reg.predict(X_te),  0, None)
        else:
            cal_count = np.zeros(len(X_cal))
            te_count  = np.zeros(len(X_te))
        cal_prob = clf.predict_proba(X_cal)[:, 1]
        te_prob  = clf.predict_proba(X_te)[:, 1]
        cal_preds = np.clip(cal_prob * cal_count, 0, None)
        te_preds  = np.clip(te_prob  * te_count,  0, None)
        cal_resid = np.abs(cal['y'].values - cal_preds)
        for (_, row), pt in zip(test.iterrows(), te_preds):
            q05, q95 = conformal_intervals(cal_resid, float(pt), alpha)
            records.append({'week_num': test_week, 'district': row['district'],
                            'y_true': row['y'], 'y_pred': float(pt),
                            'q05': q05, 'q95': q95})
    return pd.DataFrame(records)


print('Running Hurdle (conformal, lags=5) ...')
hurdle_prob = run_hurdle_probabilistic(tabular, splits, n_lags=5)
print(f'Done. {len(hurdle_prob)} predictions.')


# ── Results table ─────────────────────────────────────────────────────────────

print('\n=== Section 9: Results table (empirical CRPS) ===')

prob_preds = {
    'Random Walk'    : (rw_prob,     TAUS_3),
    'AR(1)'          : (ar1_prob,    TAUS_3),
    'Rolling Avg (4)': (ra4_prob,    TAUS_3),
    'Rolling Avg (8)': (ra8_prob,    TAUS_3),
    'Rolling Avg(12)': (ra12_prob,   TAUS_3),
    'GBM (lags=5)'   : (gbm_prob,    TAUS_3),
    'Quantile GBM'   : (qgbm_prob,   TAUS_19),
    'LSTM (seq=5)'   : (lstm_prob,   TAUS_3),
    'Hurdle (lags=5)': (hurdle_prob, TAUS_3),
}

rows = []
rw_metrics = evaluate_probabilistic(rw_prob, taus=TAUS_3)

for name, (df, taus) in prob_preds.items():
    m = evaluate_probabilistic(df, taus=taus)
    r2 = compute_r2(df, panel)
    rows.append({
        'Model'           : name,
        'CRPS'            : m['CRPS (mean)'],
        'Rel. CRPS'       : round(m['CRPS (mean)'] / rw_metrics['CRPS (mean)'], 3),
        'Coverage 90%'    : m['Coverage 90%'],
        'Interval Width'  : m['Interval Width'],
        'Pinball (q=0.10)': m['Pinball (q=0.10)'],
        'Pinball (q=0.90)': m['Pinball (q=0.90)'],
        'RMSE (point)'    : m['RMSE (point)'],
        'R²'              : r2,
    })

prob_table = pd.DataFrame(rows)

print('\n=== Probabilistic Forecasting Results (empirical CRPS) ===')
print('CRPS method: pinball-loss integral (Gneiting & Raftery, 2007)')
print('  Conformal models scored at 3 quantiles {0.05, 0.50, 0.95}')
print('  Quantile GBM scored at 19 quantiles {0.05, 0.10, ..., 0.95}')
print('(Rel. CRPS < 1.0 = better than Random Walk)')
print()
print(prob_table.to_string(index=False))

out_dir = ROOT / 'results' / 'tables'
prob_table.to_csv(out_dir / 'probabilistic_results_v2.csv', index=False)
print('\nSaved probabilistic_results_v2.csv')


# ── Figures ───────────────────────────────────────────────────────────────────

print('\n=== Section 10: Figures ===')

DISPLAY_ORDER = [
    'Random Walk', 'AR(1)', 'Rolling Avg (4)',
    'GBM (lags=5)', 'Quantile GBM', 'LSTM (seq=5)', 'Hurdle (lags=5)',
]
plot_table = prob_table[prob_table['Model'].isin(DISPLAY_ORDER)].set_index('Model').loc[DISPLAY_ORDER].reset_index()

# Figure 1: CRPS bar chart
fig, ax = plt.subplots(figsize=(11, 5))
models   = plot_table['Model'].tolist()
rel_crps = plot_table['Rel. CRPS'].tolist()
colors   = [MODEL_COLORS.get(m, '#aaa') for m in models]
bars = ax.bar(range(len(models)), rel_crps, color=colors, width=0.6, alpha=0.88)
ax.axhline(1.0, color='black', linewidth=1.0, linestyle='--', label='Random Walk = 1.0')
for bar, val in zip(bars, rel_crps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('CRPS relative to Random Walk', fontsize=10)
ax.set_title('Probabilistic Forecast Quality — Empirical CRPS\n'
             '(pinball-loss integral, no distributional assumption; lower = better)',
             fontsize=10)
ax.set_ylim(0.70, 1.10)
plt.tight_layout()
plt.savefig(out_dir / 'fig_crps_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig_crps_v2.png')

# Figure 2: Coverage vs Width
fig, ax = plt.subplots(figsize=(8, 6))
for _, row in plot_table.iterrows():
    color = MODEL_COLORS.get(row['Model'], '#aaa')
    ax.scatter(row['Interval Width'], row['Coverage 90%'],
               color=color, s=130, zorder=3, edgecolors='white', linewidths=1)
    ax.annotate(row['Model'], (row['Interval Width'], row['Coverage 90%']),
                xytext=(5, 3), textcoords='offset points', fontsize=8)
ax.axhline(0.90, color='red', linestyle='--', linewidth=1.0, label='Target 90%')
ax.set_xlabel('Mean 90% interval width (log-displacement units)', fontsize=10)
ax.set_ylabel('Empirical 90% coverage rate', fontsize=10)
ax.set_title('Calibration: Coverage Rate vs Interval Width', fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0.50, 1.05)
plt.tight_layout()
plt.savefig(out_dir / 'fig_coverage_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig_coverage_v2.png')

# Figure 3: Prediction intervals for Belet Weyne
DISTRICT = 'Belet Weyne'
SHOW     = ['Random Walk', 'GBM (lags=5)', 'Hurdle (lags=5)']
preds_dict = {n: df for n, (df, _) in prob_preds.items()}

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for ax, name in zip(axes, SHOW):
    df = preds_dict[name]
    d  = df[df['district'] == DISTRICT].sort_values('week_num')
    if len(d) == 0:
        ax.set_title(f'{name} — {DISTRICT} not found'); continue
    color = MODEL_COLORS.get(name, '#888')
    ax.fill_between(d['week_num'], d['q05'], d['q95'],
                    alpha=0.25, color=color, label='90% interval')
    ax.plot(d['week_num'], d['y_pred'], color=color,   lw=1.5, label='Predicted (median)')
    ax.plot(d['week_num'], d['y_true'], color='black', lw=1.0, alpha=0.8, label='Actual')
    ax.set_ylabel('log(1 + outflows)', fontsize=9)
    ax.set_title(f'{name}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=3)
    ax.tick_params(labelsize=8)
axes[-1].set_xlabel('Week number (holdout period: weeks 292–343)', fontsize=9)
fig.suptitle(f'90% Prediction Intervals — {DISTRICT}', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / 'fig_intervals_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig_intervals_v2.png')

print('\n=== ALL DONE ===')
print('\nKey files written:')
print('  results/tables/probabilistic_results_v2.csv')
print('  results/tables/fig_crps_v2.png')
print('  results/tables/fig_coverage_v2.png')
print('  results/tables/fig_intervals_v2.png')
