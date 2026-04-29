"""
LSTM forecasting model — PyTorch implementation.

Architecture (from the project spec):
  - District embedding: nn.Embedding(74, embed_dim=8)
  - LSTM: input (5 features + 8 embedding) → hidden_size=64, 1 layer
  - Head: Linear(64, 32) → ReLU → Linear(32, 1)

Training setup (matching Z&T's walk-forward):
  - Fixed 291-week rolling window refit at every holdout step
  - Loss: MSE on log target (y = log(1+outflows))
  - Optimiser: Adam, lr=1e-3
  - Max 50 epochs with early stopping (patience=10) on last 26 weeks of training window
  - Batch size: 256
  - Inputs standardised per feature using training-window statistics only
  - Seed set for reproducibility; caller can loop over seeds

Interface:
  run_lstm(panel, splits, seq_len, seed) → predictions DataFrame
  columns: week_num, district, y_true, y_pred
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.walkforward import rolling_week_splits
from src.features import build_sequences, FEATURE_COLS


# ── Architecture ──────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int = 5, n_districts: int = 74,
                 embed_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.dist_embed = nn.Embedding(n_districts, embed_dim)
        self.lstm = nn.LSTM(
            input_size=n_features + embed_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_dist: torch.Tensor) -> torch.Tensor:
        # x_seq : (B, seq_len, n_features)
        # x_dist: (B,)
        emb = self.dist_embed(x_dist)                      # (B, embed_dim)
        emb = emb.unsqueeze(1).expand(-1, x_seq.size(1), -1)  # (B, seq_len, embed_dim)
        x   = torch.cat([x_seq, emb], dim=-1)              # (B, seq_len, n_features+embed_dim)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)         # (B,)


# ── Training helpers ───────────────────────────────────────────────────────────

def _standardise(X_train: np.ndarray, X_test: np.ndarray):
    """
    Z-score normalise per feature using training statistics only.
    X shape: (n, seq_len, n_features).
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)   # (1, 1, n_features)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def _train_one_step(
    X_seq_tr, X_dist_tr, y_tr,
    X_seq_val, X_dist_val, y_val,
    seq_len: int, seed: int,
    max_epochs: int = 50, patience: int = 10, batch_size: int = 256,
) -> LSTMForecaster:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    model  = LSTMForecaster(n_features=len(FEATURE_COLS)).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Build DataLoaders
    tr_ds  = TensorDataset(
        torch.tensor(X_seq_tr,  dtype=torch.float32),
        torch.tensor(X_dist_tr, dtype=torch.long),
        torch.tensor(y_tr,      dtype=torch.float32),
    )
    tr_dl  = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    val_seq  = torch.tensor(X_seq_val,  dtype=torch.float32)
    val_dist = torch.tensor(X_dist_val, dtype=torch.long)
    val_y    = torch.tensor(y_val,      dtype=torch.float32)

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, db, yb in tr_dl:
            opt.zero_grad()
            loss_fn(model(xb, db), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_seq, val_dist), val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


# ── Main walk-forward runner ───────────────────────────────────────────────────

def run_lstm(panel: pd.DataFrame, splits: list, seq_len: int = 8,
             seed: int = 42) -> pd.DataFrame:
    """
    Walk-forward LSTM with a fixed 291-week rolling training window.

    For each of the 52 holdout steps:
      1. Build sequences from the training window (seq_len-step windows)
      2. Use last 26 weeks of training window as validation for early stopping
      3. Train until convergence (max 50 epochs)
      4. Predict the holdout week

    seq_len is passed through to build_sequences — change it to try different
    window lengths (4, 8, 12) without modifying this file.
    """
    # Pre-build sequences for the full panel once (fast — pure numpy)
    seqs         = build_sequences(panel, seq_len=seq_len)
    X_seq_all    = seqs['X_seq']      # (n_samples, seq_len, 5)
    X_dist_all   = seqs['X_dist']     # (n_samples,)
    y_all        = seqs['y']          # (n_samples,)
    week_all     = seqs['week_num']   # (n_samples,)
    district_names = seqs['district_names']

    records = []

    for step_idx, (train_weeks, test_week) in enumerate(splits):
        if (step_idx + 1) % 10 == 0:
            print(f"    step {step_idx + 1}/52 (seed={seed}) …")

        # Masks for this walk-forward step
        tr_mask   = np.isin(week_all, list(train_weeks))
        te_mask   = week_all == test_week
        # Validation: last 26 weeks of training window
        val_weeks = sorted(train_weeks)[-26:]
        val_mask  = np.isin(week_all, val_weeks)
        # Pure training: training window minus validation weeks
        ptr_mask  = tr_mask & ~val_mask

        if ptr_mask.sum() == 0 or te_mask.sum() == 0:
            continue

        X_seq_tr  = X_seq_all[ptr_mask]
        X_dist_tr = X_dist_all[ptr_mask]
        y_tr      = y_all[ptr_mask]

        X_seq_val  = X_seq_all[val_mask]
        X_dist_val = X_dist_all[val_mask]
        y_val      = y_all[val_mask]

        X_seq_te  = X_seq_all[te_mask]
        X_dist_te = X_dist_all[te_mask]
        y_te      = y_all[te_mask]

        # Standardise inputs using training statistics only
        X_seq_tr_n, X_seq_val_n = _standardise(X_seq_tr, X_seq_val)
        _,          X_seq_te_n  = _standardise(X_seq_tr, X_seq_te)

        model = _train_one_step(
            X_seq_tr_n, X_dist_tr, y_tr,
            X_seq_val_n, X_dist_val, y_val,
            seq_len=seq_len, seed=seed,
        )

        model.eval()
        with torch.no_grad():
            preds = model(
                torch.tensor(X_seq_te_n, dtype=torch.float32),
                torch.tensor(X_dist_te,  dtype=torch.long),
            ).numpy()

        preds = np.clip(preds, 0, None)

        # Recover district names for each test sample
        dist_names_te = [district_names[i] for i in X_dist_all[te_mask]]

        for district, y_true, y_pred in zip(dist_names_te, y_te, preds):
            records.append({
                'week_num': test_week,
                'district': district,
                'y_true':   float(y_true),
                'y_pred':   float(y_pred),
            })

    return pd.DataFrame(records)
