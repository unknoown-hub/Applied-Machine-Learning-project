"""
Generates notebooks/04_final_models.ipynb programmatically.
Run once: python3 create_notebook_04.py
"""
import nbformat
from pathlib import Path

ROOT = Path('/Users/noha/Desktop/Hilary_term /Applied ML/Summative/project')

def md(source):
    return nbformat.v4.new_markdown_cell(source)

def code(source):
    return nbformat.v4.new_code_cell(source.strip('\n'))

# ── CELLS ──────────────────────────────────────────────────────────────────────

cells = []

# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(md("""# Notebook 04 — Final Models

All model specifications match Section 3.4 of the paper exactly.

| Section | Model |
|---|---|
| 1 | Random Walk |
| 2 | AR(1) |
| 3 | Rolling Average (4 weeks main / 8 and 12 appendix) |
| 4 | GBM Point Regressor — HistGradientBoostingRegressor (lag ablation) |
| 5 | Quantile GBM — 5th / 50th / 95th Percentile |
| 6 | Two-Stage Hurdle Model (lag ablation) |
| 7 | Long Short-Term Memory — single layer, 64 hidden units (sequence-length ablation) |
| 8 | Point Prediction Results Table |
| 9 | Probabilistic Results Table |
| 10 | Figures |

**Runtime:** Sections 1–6 ≈ 25 min. Section 7 (LSTM × 3 seq-lengths) ≈ 60 min.
"""))

# ── Setup ──────────────────────────────────────────────────────────────────────
cells.append(md("---\n## Setup — Imports, Palette, Metric Functions, Conformal Wrapper"))

cells.append(code("""
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from sklearn.ensemble import (HistGradientBoostingRegressor,
                               HistGradientBoostingClassifier)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import permutation_importance as sk_perm_imp

from pathlib import Path
ROOT = Path('..')

from src.walkforward import rolling_week_splits
from src.features    import FEATURE_COLS, build_tabular, build_sequences
from src.models_lstm import LSTMForecaster, _standardise, _train_one_step

SAVE = ROOT / 'results' / 'tables'

# ── Consistent colour palette (Okabe-Ito / Wong 2011, colorblind-safe) ────────
MODEL_COLORS = {
    'Random Walk'         : '#999999',
    'AR(1)'               : '#56B4E9',
    'Rolling Avg (4)'     : '#009E73',
    'Rolling Avg (8)'     : '#00836a',
    'Rolling Avg (12)'    : '#006452',
    'Long Run Mean'       : '#CCBB44',
    'Long Run Median'     : '#AA3377',
    'GBM (lags=1)'        : '#FFAA77',
    'GBM (lags=2)'        : '#FF8833',
    'GBM (lags=5)'        : '#D55E00',
    'GBM (lags=8)'        : '#A34400',
    'Quantile GBM'        : '#E69F00',
    'LSTM (seq=2)'        : '#CC79A7',
    'LSTM (seq=5)'        : '#9B3F80',
    'LSTM (seq=8)'        : '#6B1F55',
    'Hurdle (lags=2)'     : '#88CCEE',
    'Hurdle (lags=5)'     : '#0072B2',
    'Hurdle (lags=8)'     : '#004E80',
    'Z&T Bayesian DL-DLM' : '#000000',
}

# ── Point prediction metrics (per-week average, matching Z&T convention) ──────
def _rmse(yt, yp):  return float(np.sqrt(np.mean((yt - yp) ** 2)))
def _mae(yt, yp):   return float(np.mean(np.abs(yt - yp)))
def _smape(yt, yp):
    d = (np.abs(yt) + np.abs(yp)) / 2
    return float(np.mean(np.abs(yt - yp) / np.where(d < 1e-8, 1e-8, d)))
def _corr(yt, yp):
    return 0.0 if (np.std(yt) < 1e-9 or np.std(yp) < 1e-9) else float(np.corrcoef(yt, yp)[0,1])

def evaluate_point(df):
    rows = []
    for _, grp in df.groupby('week_num'):
        yt, yp = grp['y_true'].values, grp['y_pred'].values
        rows.append({'rmse': _rmse(yt,yp), 'mae': _mae(yt,yp),
                     'smape': _smape(yt,yp), 'corr': _corr(yt,yp)})
    pw = pd.DataFrame(rows)
    return {c: pw[c].mean() for c in pw.columns}

def global_rmse(df):
    return float(np.sqrt(np.mean((df['y_true'].values - df['y_pred'].values)**2)))

# ── Probabilistic metrics ──────────────────────────────────────────────────────
def gaussian_crps(mu, sigma, y):
    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma
    return sigma * (z*(2*sp_stats.norm.cdf(z)-1) + 2*sp_stats.norm.pdf(z) - 1/np.sqrt(np.pi))

def evaluate_probabilistic(df):
    yt, mu, sg = df['y_true'].values, df['y_pred'].values, df['sigma'].values
    q05, q95   = df['q05'].values, df['q95'].values
    crps = gaussian_crps(mu, sg, yt)
    return {
        'crps'    : float(crps.mean()),
        'coverage': float(np.mean((yt >= q05) & (yt <= q95))),
        'width'   : float((q95 - q05).mean()),
    }

# ── Split-conformal prediction interval ───────────────────────────────────────
def conformal_interval(cal_resid, pt, alpha=0.10):
    n     = len(cal_resid)
    level = min((1 - alpha) * (1 + 1/n), 1.0)
    q     = float(np.quantile(cal_resid, level))
    sigma = float(cal_resid.std()) + 1e-8
    return max(0.0, pt - q), pt + q, sigma

# ── Feature column names for a given lag depth ────────────────────────────────
def feat_cols(n_lags):
    return [f'{c}_lag{k}' for c in FEATURE_COLS for k in range(1, n_lags+1)] \
         + ['week_of_year', 'year']

print('Setup complete.')
"""))

# ── Load data ─────────────────────────────────────────────────────────────────
cells.append(md("---\n## Load Data and Walk-Forward Splits"))

cells.append(code("""
panel   = pd.read_parquet(ROOT / 'data' / 'processed' / 'panel.parquet')
tabular = build_tabular(panel)
splits  = rolling_week_splits()

# Ordinal encoder fitted once; reused across all GBM / Hurdle runs
ENCODER = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ENCODER.fit(tabular[['district']])

# Prediction stores
point_preds = {}   # model_name → DataFrame (week_num, district, y_true, y_pred)
prob_preds  = {}   # model_name → DataFrame (+ q05, q95, sigma)

# Test-set SS_total for R²  (weeks 292–343)
TEST_WEEKS = set(range(292, 344))
y_test_all = tabular[tabular['week_num'].isin(TEST_WEEKS)]['y'].values
SS_tot     = float(np.sum((y_test_all - y_test_all.mean()) ** 2))
n_test     = len(y_test_all)

print(f'Panel   : {panel.shape}')
print(f'Tabular : {tabular.shape}')
print(f'Splits  : {len(splits)} walk-forward steps')
print(f'y_test  : n={n_test}, mean={y_test_all.mean():.3f}, std={y_test_all.std():.3f}')
"""))

# ── 1. Random Walk ────────────────────────────────────────────────────────────
cells.append(md("---\n## 1. Random Walk\n\n$\\hat{y}_{i,t+1} = y_{i,t}$\n\nPoint prediction uses the previous week's observed value directly. Conformal intervals are derived from calibration residuals on the final 26 training weeks."))

cells.append(code("""
records = []
for train_weeks, test_week in splits:
    cal_weeks = sorted(train_weeks)[-26:]
    cal       = tabular[tabular['week_num'].isin(cal_weeks)].copy()
    cal_resid = np.abs(cal['y'].values - cal['y_lag1'].fillna(0).values)

    test = tabular[tabular['week_num'] == test_week]
    for _, row in test.iterrows():
        pt = max(0.0, float(row['y_lag1']))
        q05, q95, sig = conformal_interval(cal_resid, pt)
        records.append({'week_num': test_week, 'district': row['district'],
                        'y_true': row['y'], 'y_pred': pt,
                        'q05': q05, 'q95': q95, 'sigma': sig})

rw_preds = pd.DataFrame(records)
point_preds['Random Walk'] = rw_preds
prob_preds['Random Walk']  = rw_preds
print(f'Random Walk: {len(rw_preds)} predictions')
"""))

# ── 2. AR(1) ──────────────────────────────────────────────────────────────────
cells.append(md("---\n## 2. AR(1)\n\nPer-district OLS: $y_t = \\alpha_i + \\beta_i \\cdot y_{t-1}$, refitted at every walk-forward step using only weeks in the current training window."))

cells.append(code("""
districts = sorted(tabular['district'].unique())
records   = []

for train_weeks, test_week in splits:
    train     = tabular[tabular['week_num'].isin(train_weeks)]
    cal_weeks = sorted(train_weeks)[-26:]

    coefs = {}
    for d in districts:
        sub = train[train['district'] == d][['y_lag1', 'y']].dropna()
        if len(sub) < 3:
            coefs[d] = (0.0, 1.0); continue
        A = np.column_stack([np.ones(len(sub)), sub['y_lag1'].values])
        c, *_ = np.linalg.lstsq(A, sub['y'].values, rcond=None)
        coefs[d] = (float(c[0]), float(c[1]))

    cal      = tabular[tabular['week_num'].isin(cal_weeks)]
    cal_resid = np.array([abs(row['y'] - max(0., coefs.get(row['district'],(0.,1.))[0]
                                              + coefs.get(row['district'],(0.,1.))[1]
                                              * float(row['y_lag1'])))
                           for _, row in cal.iterrows()])

    test = tabular[tabular['week_num'] == test_week]
    for _, row in test.iterrows():
        a, b = coefs.get(row['district'], (0.0, 1.0))
        pt   = max(0.0, a + b * float(row['y_lag1']))
        q05, q95, sig = conformal_interval(cal_resid, pt)
        records.append({'week_num': test_week, 'district': row['district'],
                        'y_true': row['y'], 'y_pred': pt,
                        'q05': q05, 'q95': q95, 'sigma': sig})

ar1_preds = pd.DataFrame(records)
point_preds['AR(1)'] = ar1_preds
prob_preds['AR(1)']  = ar1_preds
print(f'AR(1): {len(ar1_preds)} predictions')
"""))

# ── 3. Rolling Average ─────────────────────────────────────────────────────────
cells.append(md("---\n## 3. Rolling Average — 4 Weeks (main) / 8 and 12 Weeks (appendix)\n\n$\\hat{y}_{i,t} = \\frac{1}{w}\\sum_{k=1}^{w} y_{i,t-k}$\n\nThe 4-week window is retained in the main results table. Windows of 8 and 12 weeks produced uniformly weaker forecasts and are reported in the appendix."))

cells.append(code("""
def run_rolling(tabular, splits, window, with_conformal=True):
    records = []
    for train_weeks, test_week in splits:
        cal_weeks    = sorted(train_weeks)[-26:]
        recent_weeks = sorted(train_weeks)[-window:]
        means    = (tabular[tabular['week_num'].isin(recent_weeks)]
                    .groupby('district')['y'].mean())
        if with_conformal:
            cal       = tabular[tabular['week_num'].isin(cal_weeks)]
            cal_resid = np.array([abs(row['y'] - float(means.get(row['district'], 0.)))
                                   for _, row in cal.iterrows()])
        test = tabular[tabular['week_num'] == test_week]
        for _, row in test.iterrows():
            pt  = float(means.get(row['district'], 0.0))
            rec = {'week_num': test_week, 'district': row['district'],
                   'y_true': row['y'], 'y_pred': pt}
            if with_conformal:
                q05, q95, sig = conformal_interval(cal_resid, pt)
                rec.update({'q05': q05, 'q95': q95, 'sigma': sig})
            records.append(rec)
    return pd.DataFrame(records)

print('Running Rolling Avg (4) ...')
ra4 = run_rolling(tabular, splits, window=4, with_conformal=True)
point_preds['Rolling Avg (4)'] = ra4
prob_preds['Rolling Avg (4)']  = ra4

print('Running Rolling Avg (8) ...')
point_preds['Rolling Avg (8)'] = run_rolling(tabular, splits, window=8,  with_conformal=False)

print('Running Rolling Avg (12) ...')
point_preds['Rolling Avg (12)'] = run_rolling(tabular, splits, window=12, with_conformal=False)

print(f'Done. {len(ra4)} predictions each.')
"""))

# ── Long Run Mean / Median ─────────────────────────────────────────────────────
cells.append(md("#### Long Run Mean and Long Run Median (appendix)"))

cells.append(code("""
def run_longrun(tabular, splits, agg='mean'):
    records = []
    for train_weeks, test_week in splits:
        train = tabular[tabular['week_num'].isin(train_weeks)]
        stat  = (train.groupby('district')['y'].mean()
                 if agg == 'mean'
                 else train.groupby('district')['y'].median())
        test = tabular[tabular['week_num'] == test_week]
        for _, row in test.iterrows():
            records.append({'week_num': test_week, 'district': row['district'],
                            'y_true': row['y'],
                            'y_pred': float(stat.get(row['district'], 0.0))})
    return pd.DataFrame(records)

print('Running Long Run Mean ...')
point_preds['Long Run Mean']   = run_longrun(tabular, splits, 'mean')
print('Running Long Run Median ...')
point_preds['Long Run Median'] = run_longrun(tabular, splits, 'median')
print('Done.')
"""))

# ── 4. GBM ─────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 4. GBM Point Regressor — HistGradientBoostingRegressor (Lag Ablation)

Scikit-learn's `HistGradientBoostingRegressor` (Pedregosa et al., 2011).
Fixed hyperparameters: 500 boosting rounds, learning rate 0.05, max 31 leaves per tree, min 20 samples per leaf.
District encoded as an ordinal integer.
Lag ablation: $L \\in \\{1, 2, 5, 8\\}$.  Representative for probabilistic results: lags = 5.
"""))

cells.append(code("""
def run_gbm(tabular, splits, n_lags, with_conformal=False):
    fc = feat_cols(n_lags)
    records = []
    for step_idx, (train_weeks, test_week) in enumerate(splits):
        if (step_idx + 1) % 10 == 0:
            print(f'    step {step_idx+1}/52 ...')

        ptr_weeks = sorted(train_weeks)[:-26] if with_conformal else list(train_weeks)
        ptr  = tabular[tabular['week_num'].isin(ptr_weeks)].dropna(subset=fc)
        test = tabular[tabular['week_num'] == test_week].dropna(subset=fc)
        if len(ptr) == 0 or len(test) == 0:
            continue

        X_ptr  = np.hstack([ENCODER.transform(ptr[['district']]),  ptr[fc].values])
        X_test = np.hstack([ENCODER.transform(test[['district']]), test[fc].values])

        model = HistGradientBoostingRegressor(
            loss='squared_error', max_iter=500, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
        model.fit(X_ptr, ptr['y'].values)

        if with_conformal:
            cal_weeks = sorted(train_weeks)[-26:]
            cal  = tabular[tabular['week_num'].isin(cal_weeks)].dropna(subset=fc)
            X_cal = np.hstack([ENCODER.transform(cal[['district']]), cal[fc].values])
            cal_resid = np.abs(cal['y'].values - np.clip(model.predict(X_cal), 0, None))

        preds = np.clip(model.predict(X_test), 0, None)
        for (_, row), pt in zip(test.iterrows(), preds):
            rec = {'week_num': test_week, 'district': row['district'],
                   'y_true': row['y'], 'y_pred': float(pt)}
            if with_conformal:
                q05, q95, sig = conformal_interval(cal_resid, float(pt))
                rec.update({'q05': q05, 'q95': q95, 'sigma': sig})
            records.append(rec)
    return pd.DataFrame(records)

# Ablation models (point only)
for n_lags, label in [(1,'GBM (lags=1)'), (2,'GBM (lags=2)'), (8,'GBM (lags=8)')]:
    print(f'Running {label} ...')
    point_preds[label] = run_gbm(tabular, splits, n_lags, with_conformal=False)

# Representative model (point + conformal)
print('Running GBM (lags=5) with conformal intervals ...')
gbm5 = run_gbm(tabular, splits, 5, with_conformal=True)
point_preds['GBM (lags=5)'] = gbm5
prob_preds['GBM (lags=5)']  = gbm5
print('Done.')
"""))

# GBM feature importance
cells.append(md("#### GBM Feature Importance (fitted on final training window)"))

cells.append(code("""
# Refit GBM on the LAST training window for feature importance
fc5  = feat_cols(5)
last_train_weeks, last_test_week = splits[-1]
last_train = tabular[tabular['week_num'].isin(last_train_weeks)].dropna(subset=fc5)
last_test  = tabular[tabular['week_num'] == last_test_week].dropna(subset=fc5)

X_last = np.hstack([ENCODER.transform(last_train[['district']]), last_train[fc5].values])
y_last = last_train['y'].values

gbm_final = HistGradientBoostingRegressor(
    loss='squared_error', max_iter=500, learning_rate=0.05,
    max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
gbm_final.fit(X_last, y_last)

# Permutation importance on test week
X_te_last = np.hstack([ENCODER.transform(last_test[['district']]), last_test[fc5].values])
all_feat_names = ['district'] + fc5
perm = sk_perm_imp(gbm_final, X_te_last, last_test['y'].values,
                    n_repeats=10, random_state=42, scoring='neg_mean_squared_error')

imp_df = pd.DataFrame({
    'feature'   : all_feat_names,
    'importance': perm.importances_mean,
    'std'       : perm.importances_std,
}).sort_values('importance', ascending=False).head(20)

print(imp_df.to_string(index=False))
"""))

# ── 5. Quantile GBM ────────────────────────────────────────────────────────────
cells.append(md("""---
## 5. Quantile GBM — 5th / 50th / 95th Percentile Regressor

Three `HistGradientBoostingRegressor` models trained with pinball (quantile) loss at $\\tau \\in \\{0.05, 0.50, 0.95\\}$.
The 90% prediction interval is $[\\hat{q}_{0.05},\\, \\hat{q}_{0.95}]$; the point prediction is the median $\\hat{q}_{0.50}$.
Native probabilistic model — no conformal calibration required.
"""))

cells.append(code("""
fc5     = feat_cols(5)
records = []

for step_idx, (train_weeks, test_week) in enumerate(splits):
    if (step_idx + 1) % 10 == 0:
        print(f'    step {step_idx+1}/52 ...')

    train = tabular[tabular['week_num'].isin(train_weeks)].dropna(subset=fc5)
    test  = tabular[tabular['week_num'] == test_week].dropna(subset=fc5)
    if len(train) == 0 or len(test) == 0:
        continue

    X_tr = np.hstack([ENCODER.transform(train[['district']]), train[fc5].values])
    X_te = np.hstack([ENCODER.transform(test[['district']]),  test[fc5].values])
    y_tr = train['y'].values

    BASE = dict(max_iter=500, learning_rate=0.05,
                max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
    m05 = HistGradientBoostingRegressor(loss='quantile', quantile=0.05, **BASE)
    m50 = HistGradientBoostingRegressor(loss='quantile', quantile=0.50, **BASE)
    m95 = HistGradientBoostingRegressor(loss='quantile', quantile=0.95, **BASE)
    m05.fit(X_tr, y_tr)
    m50.fit(X_tr, y_tr)
    m95.fit(X_tr, y_tr)

    p05 = np.clip(m05.predict(X_te), 0, None)
    p50 = np.clip(m50.predict(X_te), 0, None)
    p95 = np.clip(m95.predict(X_te), 0, None)

    for (_, row), q05, q50, q95 in zip(test.iterrows(), p05, p50, p95):
        sigma = max((q95 - q05) / (2 * 1.6449), 1e-8)
        records.append({'week_num': test_week, 'district': row['district'],
                        'y_true': row['y'], 'y_pred': float(q50),
                        'q05': float(q05), 'q95': float(q95), 'sigma': sigma})

qgbm = pd.DataFrame(records)
point_preds['Quantile GBM'] = qgbm
prob_preds['Quantile GBM']  = qgbm
print(f'Quantile GBM: {len(qgbm)} predictions')
"""))

# ── 6. Hurdle ──────────────────────────────────────────────────────────────────
cells.append(md("""---
## 6. Two-Stage Hurdle Model (Lag Ablation)

**Stage 1** — `HistGradientBoostingClassifier` estimates $P(y_{i,t+1} > 0 \\mid \\mathbf{x}_{i,t})$.
**Stage 2** — `HistGradientBoostingRegressor` estimates $\\mathbb{E}[y_{i,t+1} \\mid y_{i,t+1} > 0,\\, \\mathbf{x}_{i,t}]$ trained on non-zero rows only.
**Prediction** — product of the two stages (Cragg, 1971; Mullahy, 1986).
Motivated by the 32% zero-inflation in the panel.
Lag ablation: $L \\in \\{2, 5, 8\\}$.  Representative for probabilistic results: lags = 5.
"""))

cells.append(code("""
def run_hurdle(tabular, splits, n_lags, with_conformal=False):
    fc = feat_cols(n_lags)
    records = []
    for step_idx, (train_weeks, test_week) in enumerate(splits):
        if (step_idx + 1) % 10 == 0:
            print(f'    step {step_idx+1}/52 ...')

        ptr_weeks = sorted(train_weeks)[:-26] if with_conformal else list(train_weeks)
        ptr  = tabular[tabular['week_num'].isin(ptr_weeks)].dropna(subset=fc)
        test = tabular[tabular['week_num'] == test_week].dropna(subset=fc)
        if len(ptr) == 0 or len(test) == 0:
            continue

        X_ptr = np.hstack([ENCODER.transform(ptr[['district']]),  ptr[fc].values])
        X_te  = np.hstack([ENCODER.transform(test[['district']]), test[fc].values])
        z_ptr = (ptr['y'].values > 0).astype(int)

        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
        clf.fit(X_ptr, z_ptr)

        nz  = ptr['y'].values > 0
        reg = HistGradientBoostingRegressor(
            loss='squared_error', max_iter=500, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=10, random_state=42)
        if nz.sum() >= 10:
            reg.fit(X_ptr[nz], ptr['y'].values[nz])
            te_count = np.clip(reg.predict(X_te), 0, None)
        else:
            te_count = np.zeros(len(X_te))

        te_preds = np.clip(clf.predict_proba(X_te)[:,1] * te_count, 0, None)

        if with_conformal:
            cal_weeks = sorted(train_weeks)[-26:]
            cal  = tabular[tabular['week_num'].isin(cal_weeks)].dropna(subset=fc)
            X_cal = np.hstack([ENCODER.transform(cal[['district']]), cal[fc].values])
            cal_count = np.clip(reg.predict(X_cal), 0, None) if nz.sum() >= 10 else np.zeros(len(X_cal))
            cal_preds = np.clip(clf.predict_proba(X_cal)[:,1] * cal_count, 0, None)
            cal_resid = np.abs(cal['y'].values - cal_preds)

        for (_, row), pt in zip(test.iterrows(), te_preds):
            rec = {'week_num': test_week, 'district': row['district'],
                   'y_true': row['y'], 'y_pred': float(pt)}
            if with_conformal:
                q05, q95, sig = conformal_interval(cal_resid, float(pt))
                rec.update({'q05': q05, 'q95': q95, 'sigma': sig})
            records.append(rec)
    return pd.DataFrame(records)

for n_lags, label in [(2, 'Hurdle (lags=2)'), (8, 'Hurdle (lags=8)')]:
    print(f'Running {label} ...')
    point_preds[label] = run_hurdle(tabular, splits, n_lags, with_conformal=False)

print('Running Hurdle (lags=5) with conformal intervals ...')
h5 = run_hurdle(tabular, splits, 5, with_conformal=True)
point_preds['Hurdle (lags=5)'] = h5
prob_preds['Hurdle (lags=5)']  = h5
print('Done.')
"""))

# ── 7. LSTM ────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 7. Long Short-Term Memory — Single Layer, 64 Hidden Units (Sequence-Length Ablation)

**Architecture:** `Embedding(74, 8)` → `LSTM(input=13, hidden=64, num_layers=1)` → `Linear(64→32)` → `ReLU` → `Linear(32→1)`
**Training:** MSE loss, Adam ($\\text{lr} = 10^{-3}$), batch size 256, early stopping patience 10 on the final 26 training weeks.
Sequence-length ablation: $\\text{seq\\_len} \\in \\{2, 5, 8\\}$.  Representative for probabilistic results: seq\\_len = 5.

⚠ **Runtime:** ≈ 60 minutes for all three sequence lengths.
"""))

cells.append(code("""
def run_lstm(panel, splits, seq_len, seed=42, with_conformal=False, alpha=0.10):
    seqs           = build_sequences(panel, seq_len=seq_len)
    X_seq_all      = seqs['X_seq']
    X_dist_all     = seqs['X_dist']
    y_all          = seqs['y']
    week_all       = seqs['week_num']
    district_names = seqs['district_names']
    records = []

    for step_idx, (train_weeks, test_week) in enumerate(splits):
        if (step_idx + 1) % 10 == 0:
            print(f'    step {step_idx+1}/52  seq={seq_len} ...')

        tr_mask   = np.isin(week_all, list(train_weeks))
        te_mask   = week_all == test_week
        val_weeks = sorted(train_weeks)[-26:]
        val_mask  = np.isin(week_all, val_weeks)
        ptr_mask  = tr_mask & ~val_mask

        if ptr_mask.sum() == 0 or te_mask.sum() == 0:
            continue

        X_seq_tr_n, X_seq_val_n = _standardise(X_seq_all[ptr_mask], X_seq_all[val_mask])
        _,          X_seq_te_n  = _standardise(X_seq_all[ptr_mask], X_seq_all[te_mask])

        model = _train_one_step(
            X_seq_tr_n,  X_dist_all[ptr_mask], y_all[ptr_mask],
            X_seq_val_n, X_dist_all[val_mask],  y_all[val_mask],
            seq_len=seq_len, seed=seed,
        )
        model.eval()
        with torch.no_grad():
            te_preds = np.clip(model(
                torch.tensor(X_seq_te_n,          dtype=torch.float32),
                torch.tensor(X_dist_all[te_mask], dtype=torch.long),
            ).numpy(), 0, None)

        if with_conformal:
            with torch.no_grad():
                val_preds = np.clip(model(
                    torch.tensor(X_seq_val_n,           dtype=torch.float32),
                    torch.tensor(X_dist_all[val_mask],  dtype=torch.long),
                ).numpy(), 0, None)
            cal_resid = np.abs(y_all[val_mask] - val_preds)

        for dist, y_t, pt in zip([district_names[i] for i in X_dist_all[te_mask]],
                                   y_all[te_mask], te_preds):
            rec = {'week_num': test_week, 'district': dist,
                   'y_true': float(y_t), 'y_pred': float(pt)}
            if with_conformal:
                q05, q95, sig = conformal_interval(cal_resid, float(pt), alpha)
                rec.update({'q05': q05, 'q95': q95, 'sigma': sig})
            records.append(rec)
    return pd.DataFrame(records)

for seq_len, label in [(2, 'LSTM (seq=2)'), (8, 'LSTM (seq=8)')]:
    print(f'Running {label} ...')
    point_preds[label] = run_lstm(panel, splits, seq_len, with_conformal=False)

print('Running LSTM (seq=5) with conformal intervals ...')
lstm5 = run_lstm(panel, splits, 5, with_conformal=True)
point_preds['LSTM (seq=5)'] = lstm5
prob_preds['LSTM (seq=5)']  = lstm5
print('Done.')
"""))

# ── 8. Point Results ──────────────────────────────────────────────────────────
cells.append(md("""---
## 8. Point Prediction Results

RMSE, MAE, and sMAPE are reported relative to the Random Walk benchmark (ratio < 1 = better).
Correlation and R² are reported as absolute values.

Z&T benchmark added for comparison (from their Table 1).
"""))

cells.append(code("""
POINT_ORDER = [
    'Random Walk', 'AR(1)', 'Rolling Avg (4)', 'Rolling Avg (8)', 'Rolling Avg (12)',
    'Long Run Mean', 'Long Run Median',
    'GBM (lags=1)', 'GBM (lags=2)', 'GBM (lags=5)', 'GBM (lags=8)',
    'Quantile GBM',
    'LSTM (seq=2)', 'LSTM (seq=5)', 'LSTM (seq=8)',
    'Hurdle (lags=2)', 'Hurdle (lags=5)', 'Hurdle (lags=8)',
]

raw_pt = {n: evaluate_point(df) for n, df in point_preds.items()}
rw_pt  = raw_pt['Random Walk']

rows = []
for name in POINT_ORDER:
    if name not in raw_pt:
        continue
    m  = raw_pt[name]
    df = point_preds[name]
    r2 = round(1 - n_test * global_rmse(df)**2 / SS_tot, 3)
    rows.append({
        'Model' : name,
        'RMSE'  : round(m['rmse']  / rw_pt['rmse'],  3),
        'MAE'   : round(m['mae']   / rw_pt['mae'],   3),
        'sMAPE' : round(m['smape'] / rw_pt['smape'], 3),
        'Corr'  : round(m['corr'], 3),
        'R²'    : r2,
    })

# Add Z&T benchmark row
rows.append({'Model': 'Z&T Bayesian DL-DLM',
             'RMSE': 0.737, 'MAE': 0.907, 'sMAPE': 0.804, 'Corr': 0.639, 'R²': None})

point_table = pd.DataFrame(rows)
print('=== Point Prediction Results (relative to Random Walk) ===')
print(point_table.to_string(index=False))
point_table.to_csv(SAVE / 'point_results_final.csv', index=False)
print('\\nSaved point_results_final.csv')
"""))

# ── 9. Probabilistic Results ──────────────────────────────────────────────────
cells.append(md("""---
## 9. Probabilistic Results

CRPS relative to Random Walk benchmark (ratio < 1 = better).
Coverage target: 0.900. Interval width in log(1+outflows) units.
"""))

cells.append(code("""
PROB_ORDER = [
    'Random Walk', 'AR(1)', 'Rolling Avg (4)',
    'GBM (lags=5)', 'Quantile GBM',
    'LSTM (seq=5)', 'Hurdle (lags=5)',
]

raw_prob = {n: evaluate_probabilistic(df) for n, df in prob_preds.items()}
rw_crps  = raw_prob['Random Walk']['crps']

rows = []
for name in PROB_ORDER:
    if name not in raw_prob:
        continue
    m  = raw_prob[name]
    df = prob_preds[name]
    r2 = round(1 - n_test * global_rmse(df)**2 / SS_tot, 3)
    rows.append({
        'Model'       : name,
        'CRPS'        : round(m['crps'], 4),
        'Rel. CRPS'   : round(m['crps'] / rw_crps, 3),
        'Coverage 90%': round(m['coverage'], 3),
        'Width'       : round(m['width'], 4),
        'R²'          : r2,
    })

prob_table = pd.DataFrame(rows)
print('=== Probabilistic Results ===')
print(prob_table.to_string(index=False))
prob_table.to_csv(SAVE / 'prob_results_final.csv', index=False)
print('\\nSaved prob_results_final.csv')
"""))

# ── 10. Figures ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 10. Figures"))

# Fig 1: RMSE bar chart
cells.append(md("### Figure 1 — Point Prediction Accuracy (Relative RMSE)"))
cells.append(code("""
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,
                     'axes.spines.right':False,'axes.grid':True,
                     'grid.alpha':0.3,'grid.linestyle':'--'})

# Main-paper models only (no appendix ablations)
MAIN_MODELS = ['Random Walk','AR(1)','Rolling Avg (4)','GBM (lags=5)',
               'Quantile GBM','LSTM (seq=5)','Hurdle (lags=5)','Z&T Bayesian DL-DLM']
pt_main = point_table[point_table['Model'].isin(MAIN_MODELS)].copy()
pt_main['color'] = pt_main['Model'].map(MODEL_COLORS)

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(pt_main)), pt_main['RMSE'],
              color=pt_main['color'], width=0.6, alpha=0.90,
              edgecolor='white', linewidth=0.5)
ax.axhline(1.0, color='#333333', lw=1.2, ls='--', zorder=5)

for bar, val in zip(bars, pt_main['RMSE']):
    if pd.notna(val):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(range(len(pt_main)))
ax.set_xticklabels(pt_main['Model'], rotation=30, ha='right', fontsize=9)
ax.set_ylabel('RMSE relative to Random Walk', fontsize=10)
ax.set_ylim(0.65, 1.08)
ax.set_title('Point Prediction Accuracy — Relative RMSE\\n'
             'Lower = better  |  Dashed line = Random Walk benchmark', fontsize=11)
ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(SAVE / 'fig_rmse_final.png', dpi=160, bbox_inches='tight')
plt.show()
print('Saved fig_rmse_final.png')
"""))

# Fig 2: CRPS bar chart
cells.append(md("### Figure 2 — Probabilistic Forecast Quality (Relative CRPS)"))
cells.append(code("""
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(range(len(prob_table)), prob_table['Rel. CRPS'],
              color=[MODEL_COLORS.get(m,'#aaa') for m in prob_table['Model']],
              width=0.6, alpha=0.90, edgecolor='white', linewidth=0.5)
ax.axhline(1.0, color='#333333', lw=1.2, ls='--', zorder=5)
for bar, val in zip(bars, prob_table['Rel. CRPS']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_xticks(range(len(prob_table)))
ax.set_xticklabels(prob_table['Model'], rotation=30, ha='right', fontsize=9)
ax.set_ylabel('CRPS relative to Random Walk', fontsize=10)
ax.set_ylim(0.75, 1.10)
ax.set_title('Probabilistic Forecast Quality — Relative CRPS\\n'
             'Lower = better calibrated and sharper prediction intervals', fontsize=11)
ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(SAVE / 'fig_crps_final.png', dpi=160, bbox_inches='tight')
plt.show()
print('Saved fig_crps_final.png')
"""))

# Fig 3: Coverage vs Width
cells.append(md("### Figure 3 — Calibration: Coverage Rate vs Interval Width"))
cells.append(code("""
fig, ax = plt.subplots(figsize=(8, 6))
for _, row in prob_table.iterrows():
    color = MODEL_COLORS.get(row['Model'], '#aaa')
    ax.scatter(row['Width'], row['Coverage 90%'],
               color=color, s=140, zorder=4, edgecolors='white', linewidths=1.2)
    ax.annotate(row['Model'], (row['Width'], row['Coverage 90%']),
                xytext=(6, 3), textcoords='offset points',
                fontsize=7.5, color=color, fontweight='bold')
ax.axhline(0.90, color='#cc0000', ls='--', lw=1.2, label='Target 90% coverage')
ax.set_xlabel('Mean 90% interval width  (log-displacement units)', fontsize=10)
ax.set_ylabel('Empirical 90% coverage rate', fontsize=10)
ax.set_title('Calibration: Coverage vs Interval Width\\n'
             'Ideal: on the red line, as far left as possible', fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0.84, 0.97)
ax.yaxis.grid(True, ls='--', alpha=0.4); ax.xaxis.grid(True, ls='--', alpha=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(SAVE / 'fig_coverage_final.png', dpi=160, bbox_inches='tight')
plt.show()
print('Saved fig_coverage_final.png')
"""))

# Fig 4: Prediction intervals
cells.append(md("### Figure 4 — 90% Prediction Intervals (Belet Weyne)"))
cells.append(code("""
DISTRICT = 'Belet Weyne'
SHOW = [
    ('Random Walk',     prob_preds['Random Walk']),
    ('GBM (lags=5)',    prob_preds['GBM (lags=5)']),
    ('Hurdle (lags=5)', prob_preds['Hurdle (lags=5)']),
]
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for ax, (name, df) in zip(axes, SHOW):
    d = df[df['district'] == DISTRICT].sort_values('week_num')
    color = MODEL_COLORS[name]
    upper = np.minimum(d['q95'].values, d['y_true'].max() + 2)
    ax.fill_between(d['week_num'], d['q05'].values, upper,
                    alpha=0.18, color=color, label='90% interval')
    ax.plot(d['week_num'], d['y_pred'], color=color, lw=2.0, label='Predicted (median)', zorder=3)
    ax.plot(d['week_num'], d['y_true'], color='#222222', lw=1.2, alpha=0.85, label='Actual', zorder=4)
    inside = ((d['y_true'] >= d['q05']) & (d['y_true'] <= d['q95'])).mean()
    ax.set_title(f'{name}    (district coverage: {inside:.0%})',
                 fontsize=10, fontweight='bold', color=color, loc='left')
    ax.set_ylabel('log(1 + outflows)', fontsize=9)
    ax.legend(fontsize=8, loc='upper right', ncol=3, framealpha=0.8)
    ax.yaxis.grid(True, ls='--', alpha=0.3)
axes[-1].set_xlabel('Week number  (holdout: weeks 292–343)', fontsize=9)
fig.suptitle(f'90% Prediction Intervals — {DISTRICT}\\n'
             'Shaded band = conformal 90% interval  |  Black = actual displacement',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(SAVE / 'fig_intervals_final.png', dpi=160, bbox_inches='tight')
plt.show()
print('Saved fig_intervals_final.png')
"""))

# Fig 5: Feature importance
cells.append(md("### Figure 5 — GBM Permutation Feature Importance"))
cells.append(code("""
top20 = imp_df.head(20)
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(range(len(top20)), top20['importance'],
        xerr=top20['std'], color='#D55E00', alpha=0.85,
        edgecolor='white', capsize=3)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['feature'], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel('Mean decrease in MSE (permutation importance)', fontsize=10)
ax.set_title('GBM (lags=5) — Top 20 Features by Permutation Importance\\n'
             'Error bars = ±1 std over 10 repetitions', fontsize=10)
ax.xaxis.grid(True, ls='--', alpha=0.4); ax.set_axisbelow(True)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(SAVE / 'fig_importance_final.png', dpi=160, bbox_inches='tight')
plt.show()
print('Saved fig_importance_final.png')
"""))

# Fig 6: Residuals
cells.append(md("### Figure 6 — Residual Distributions (Key Models)"))
cells.append(code("""
KEY_MODELS = ['Random Walk', 'GBM (lags=5)', 'LSTM (seq=5)', 'Hurdle (lags=5)']
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()
for ax, name in zip(axes, KEY_MODELS):
    df    = point_preds[name]
    resid = df['y_true'].values - df['y_pred'].values
    color = MODEL_COLORS.get(name, '#888')
    ax.hist(resid, bins=60, color=color, alpha=0.75, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', lw=1.0, ls='--')
    ax.axvline(resid.mean(), color='red', lw=1.0, ls='-',
               label=f'mean={resid.mean():.3f}')
    ax.set_title(name, fontsize=10, fontweight='bold', color=color)
    ax.legend(fontsize=8)
    ax.set_xlabel('Residual  (actual − predicted)', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Residual Distributions — Key Models\\n'
             'Dashed = zero  |  Red = mean residual', fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(SAVE / 'fig_residuals_final.png', dpi=160, bbox_inches='tight')
plt.show()
print('Saved fig_residuals_final.png')
"""))

# ── Assemble and write ─────────────────────────────────────────────────────────
nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    },
    'language_info': {
        'name': 'python',
        'version': '3.9.6',
    },
}

out = ROOT / 'notebooks' / '04_final_models.ipynb'
nbformat.write(nb, out)
print(f'Written: {out}')
print(f'Cells  : {len(nb.cells)}')
