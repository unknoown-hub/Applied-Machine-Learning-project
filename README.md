# Predicting Conflict-Driven Internal Displacement in Somalia

A comparative study of short-term district-level forecasting models, evaluating whether machine learning and deep learning methods can improve one-week-ahead forecasts of conflict-driven internal displacement across 74 Somali districts.

## Overview

This project benchmarks multiple forecasting approaches against a Bayesian dynamic linear model baseline, using weekly district-level displacement data from UNHCR's Protection and Return Monitoring Network (PRMN) and conflict event data from ACLED (2017–2023).

**Key finding:** Gradient boosting with 5 lags achieves the best point-forecast performance (relative RMSE = 0.824, R² = 0.511), while the Hurdle model delivers the strongest probabilistic forecasting (relative CRPS = 0.920, 90% coverage = 0.903). However, all ML models remain within a narrow margin of simple autoregressive baselines, suggesting a low intrinsic predictability ceiling for short-term sub-national displacement forecasting.

## Repository Structure

```
├── data/                  # Raw and processed datasets (PRMN displacement + ACLED conflict)
├── notebooks/             # Jupyter notebooks for EDA, modelling, and results
├── results/tables/        # Output tables and evaluation metrics
├── src/                   # Full model implementation and utilities
│   ├── preprocessing/     # Data cleaning and feature engineering scripts
│   ├── models/            # GBM, Hurdle, Quantile GBM, and LSTM implementations
│   └── evaluation/       # Rolling-origin walk-forward evaluation framework
├── requirements.txt       # Python dependencies
└── README.md
```

## Models Evaluated

| Model | Rel. RMSE | R² |
|---|---|---|
| Random Walk (baseline) | 1.000 | 0.282 |
| AR(1) | 0.883 | 0.443 |
| Rolling Average (4-week) | 0.853 | 0.476 |
| GBM (5 lags) | **0.824** | **0.511** |
| LSTM (seq=5) | 0.833 | 0.503 |
| Hurdle (5 lags) | 0.829 | 0.506 |
| Z&T Bayesian DL-DLM | 0.737 | — |

## Data Sources

- **PRMN** — UNHCR Protection and Return Monitoring Network: weekly district-level displacement outflows across Somalia
- **ACLED** — Armed Conflict Location & Event Data Project: battles, explosions/remote violence, violence against civilians, and strategic developments

## Reproducing the Results

1. Clone the repository:
   ```bash
   git clone https://github.com/unknoown-hub/Applied-Machine-Learning-Summative.git
   cd Applied-Machine-Learning-Summative
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks in order (see `notebooks/`) or execute the scripts in `src/` directly. Full preprocessing, modelling, and evaluation pipelines are included.

## Reference

This project builds directly on:
> Zens, G. & Thalheimer, L. (2025). The short-term dynamics of conflict-driven displacement: Bayesian modeling of disaggregated data from Somalia. *The Annals of Applied Statistics*, 19(1), 286–301.
