# Market Gravity Engine

## Scope

The project is no longer SPX-only. `SPX Gravity` becomes the first calibrated profile inside a broader **Market Gravity Engine** that supports three target asset classes:

- **INDEX** — e.g. SPX, NDX, RUT
- **ETF** — e.g. SPY, QQQ, IWM, sector ETFs
- **STOCK** — e.g. TSLA, AMD, MU, SNDK, WDC, BE, SMCI

Futures are **not target assets** in the current project scope. A future implementation may optionally use futures as contextual market-state inputs, but no futures-specific trading path belongs to the canonical target universe unless explicitly enabled later.

## Core principle

The engine is shared, but the evidence is asset-specific. We do **not** force every instrument through an SPX-shaped feature set.

Common state machine:

1. **Data Integrity / Causality Gate**
2. **Instrument & Session State**
3. **Market / Macro / Regime State**
4. **Derivatives State** — GEX / DEX / VEX / CHEX / OI / IV / skew when available
5. **Asset Microstructure State**
6. **Structural Mass / Geometry**
7. **Direction Model**
8. **Reachability**
9. **Boundary Competition** — reversal vs continuation
10. **Trigger State**
11. **Underlying / Contract EV**
12. **Execution & Risk**
13. **AI Hypothesis Research Layer**
14. **Walk-forward Validation / Learning / Alpha Decay**

The output states remain explicit:

- `WAIT`
- `LONG_ACTIVE`
- `SHORT_ACTIVE`
- `NO_TRADE`
- `INSUFFICIENT_DATA`

Probability is never an entry signal by itself.

## Asset-specific profiles

### INDEX profile

Primary evidence:

- cash-index price truth
- index options chain
- GEX / DEX / VEX / CHEX / OI
- gamma flip / walls / pockets / strike gradients
- VIX / skew / rates / macro-event regime
- index-level market breadth or constituent confirmation when causally available

Index-specific constraints:

- cash indices generally do not have a normal premarket/after-hours tape like listed ETFs/stocks
- any proxy series must be explicitly tagged as a proxy and must never overwrite the cash-index identity
- SPX/SPXW 0DTE remains its own calibrated profile

### ETF profile

Primary evidence:

- regular + extended-hours ETF price path where the data source supports it
- ETF options GEX / DEX / VEX / CHEX / OI / IV / skew
- ETF liquidity and spread state
- premium/discount or NAV state when relevant and available
- reference-index relationship
- macro / rates / volatility regime

ETF-specific examples:

- SPY ↔ SPX
- QQQ ↔ NDX / Nasdaq-100 state
- IWM ↔ Russell 2000 state

### STOCK profile

Primary evidence:

- regular + extended-hours stock path
- stock options GEX / DEX / VEX / CHEX / OI / IV / skew
- dark-pool / off-exchange structure when available
- FINRA short-volume data using the correct publication lag
- short interest / borrow fee / fails-to-deliver when available
- earnings / guidance / corporate-event state
- sector and index-relative state
- liquidity / spread / volume / order-flow state

Stock-specific constraints:

- company events can dominate macro/dealer structure
- corporate-event windows require their own regime state
- short-volume, FTD and short-interest fields must obey their publication/as-of timestamps

## Shared structural mathematics

### Structural mass

For a strike or structural node, use a robust vector such as

`z = [z_GEX, z_DEX, z_VEX, z_CHEX, z_OI]^T`

and a covariance-aware mass such as

`M_struct = sqrt(z^T Sigma^-1 z)`

when the historical sample is sufficient to estimate the covariance matrix without leakage.

A missing Greek is not silently imputed into the production score. The data gate must either lower the module quality or block the dependent fitted model.

### Gamma flip

A local current-net-GEX sign transition is only a proxy. A true scenario gamma flip requires scenario repricing. The engine must preserve that distinction.

### Negative-gamma corridor

The path statistic excludes the destination strike itself:

`NGC_path(P,K) = sum(max(-G_j,0)) / sum(|G_j|)` over strictly intermediate strikes.

### Reachability

Canonical reachability uses a remaining-move scale appropriate to the active horizon:

`R_K(t) = exp(-|K - P_t| / EM_remaining(t))`

ATR-based versions are diagnostics unless explicitly validated as a separate model.

### Boundary competition

Touching a structural level opens a new episode. A break is not automatically continuation.

Each boundary episode asks which event occurs first:

- reversal / reclaim
- continuation / acceptance
- censor / unresolved

This is the core state machine for index, ETF and stock profiles.

## Direction is upstream of monetization

The engine must keep four questions separate:

1. **Direction:** upper-first, lower-first, neither
2. **Destination / geometry:** which structural level is reachable
3. **Boundary outcome:** reversal vs continuation after touch
4. **Monetization:** whether the stock position or option contract has positive expected value

A correct underlying direction does not imply a profitable option contract.

## AI research layer

The AI layer is a hypothesis generator only. It may propose interactions among a frozen feature whitelist but cannot:

- execute generated Python
- change the target
- change train/calibration/test partitions
- see test results during search
- modify structural levels
- overwrite asset-profile invariants

The deterministic evaluator owns calibration, OOS testing and admission.

## Deployment model

The intended Railway deployment should expose the generic engine as a service, with asset profiles loaded by symbol / asset class. Storage is intentionally separated from the runtime service:

- **GitHub**: source code, schemas, tests, research rules, configuration
- **Google Drive / data store**: raw and processed datasets, snapshots, reports, model artifacts
- **Railway**: running API / scheduler / workers

This conversation must not mutate Google Drive storage while another workflow is managing the Drive data hierarchy. GitHub code changes are independent of that storage workflow.
