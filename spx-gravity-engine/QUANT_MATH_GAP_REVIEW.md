# Market Gravity — Quant Math Delta Review (2026-09-07)

This review treats generic quant-finance formula sheets as an **audit checklist**, not as a feature backlog.

## Review rule

For every proposed concept:

1. Search the current engine/research stack.
2. Identify the exact problem the concept would solve.
3. Reject duplicates and generic methods without a current data/model gap.
4. Admit only components that preserve causality and can be falsified OOS.
5. Add tests and explicit non-probability / non-signal guards where appropriate.

## Reviewed themes and decisions

| Theme | Existing coverage | Delta decision |
|---|---|---|
| Probability theory | calibrated direction/boundary research metrics; empirical tail risk; Brownian baseline | no generic probability layer added |
| Statistics / regression / hypothesis testing | purged walk-forward, empirical-prior benchmark, DSR, regime diagnostics | no generic OLS/significance layer added |
| Machine learning | constrained QuantAI hypothesis layer with engine-owned model/evaluator | no RF/XGBoost/LSTM/RL added by default |
| Optimization / efficient frontier | not part of the current per-asset decision problem | defer to a future portfolio/allocation layer |
| Linear algebra / PCA | multivariate structure required by canonical F6 | **adopt robust shrinkage-covariance structural mass**; no generic PCA feature added |
| Random processes / GBM | Brownian diagnostic exists; market tails/dependence make GBM too strong as a production assumption | **adopt empirical moving-block path simulation**; do not promote GBM Monte Carlo |
| Numerical methods / binomial trees | European Black-Scholes gamma diagnostics exist | **adopt CRR tree for American ETF/stock option valuation diagnostics** |
| Stochastic calculus / Black-Scholes PDE | foundational pricing theory already represented by BSM Greeks/scenario gamma | no PDE solver added; no value for current direction engine |
| Put-call parity | European integrity audit exists | **repair exercise-style scope: American options use bounds, not European equality** |
| Time-series indicators / GARCH / ARIMA | causal session/volatility features and model tournament framework exist | keep as candidate models only; no canonical model added without OOS gain |
| Portfolio return/risk formulas | costs, DSR, drawdown, Sortino/Calmar, empirical VaR/ES already present | no duplicate risk module |
| Generic stock-return predictor | duplicates QuantAI and risks feature-mining | rejected |

## Newly admitted components

### 1. Robust structural mass

`structural_mass.py`

- robust median/MAD normalization;
- Ledoit-Wolf shrinkage covariance;
- Mahalanobis structural mass;
- eigenvalue/condition diagnostics;
- distance/reachability explicitly excluded;
- RMS fallback only for already-normalized layers and explicitly non-canonical.

The output is **not a probability**.

### 2. Empirical boundary/reachability simulation

`empirical_path_simulation.py`

- moving-block bootstrap of historical log returns;
- finite-horizon UpperFirst / LowerFirst / Neither frequencies;
- short-range dependence preserved within sampled blocks;
- no GBM/normal-return assumption;
- caller must provide a causally eligible history and pre-chosen block size.

The output remains **uncalibrated** until chronological OOS validation.

### 3. American-style option valuation

`american_option_pricing.py`

- Cox-Ross-Rubinstein tree;
- European or American exercise;
- continuous dividend yield;
- early-exercise-aware valuation;
- tree Delta/Gamma diagnostics.

Market bid/ask remains execution truth. This module must not create directional signals.

### 4. Exercise-style-aware option integrity

`option_integrity.py`

- European options retain exact put-call parity diagnostics;
- American ETF/stock options use admissible parity **bounds** instead of a false equality;
- quote pairing remains upstream and point-in-time.

## Explicitly not added

- generic Monte Carlo GBM as a production reachability model;
- Black-Scholes PDE/finite-difference solver;
- generic PCA feature generator;
- mean-variance portfolio optimizer;
- automatic XGBoost/LSTM/RL strategy selection;
- arbitrary hypothesis-test p-value gates;
- any new trading probability without calibration.

These may be reconsidered only when a concrete unresolved problem and OOS admission test exist.
