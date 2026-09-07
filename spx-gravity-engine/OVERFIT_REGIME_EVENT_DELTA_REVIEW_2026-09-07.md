# Market Gravity — Overfit / Regime / Event Delta Review (2026-09-07)

This review starts from the current `main` implementation and treats the four supplied images as proposals to audit, not as requirements to copy.

## Method

1. Search the existing implementation.
2. Decompose the proposal into the actual problem it claims to solve.
3. Compare it with current controls.
4. Identify only the unresolved delta.
5. Implement only the delta that adds measurable value.
6. Add tests and preserve causal / OOS boundaries.

## 1. Backtest overfitting / Deflated Sharpe Ratio

### Existing coverage

`strategy_evaluator.py` already implements a Bailey/López-de-Prado-style Deflated Sharpe Ratio diagnostic with:

- the actually tested Sharpe distribution rather than an arbitrary `N`;
- skew and kurtosis of realized returns;
- a deflated benchmark Sharpe;
- a probability-style DSR diagnostic;
- a hard separation between classifier metrics and realized strategy returns.

`ai_research_loop.py` also prevents the LLM from seeing untouched test results and uses TRAIN-only purged inner walk-forward feedback before freezing candidates and selecting on calibration.

### Decision

**No second DSR implementation added.** Duplicating the formula would increase inconsistency risk without adding information.

The image's broad warning is accepted; its simplified code/formula is not copied.

## 2. AI strategy search lab / regime mapping

### Existing coverage

The current research loop already:

- constrains the LLM to a frozen causal feature whitelist;
- forbids LLM-generated executable strategy code;
- rejects duplicate feature/interaction signatures;
- performs purged inner walk-forward evaluation on TRAIN;
- freezes the candidate set before calibration;
- evaluates TEST once;
- reports strategy metrics by an upstream regime label when supplied.

### Real gap

Regime-conditioned metrics did not explicitly audit whether each regime slice had enough total and active observations to support interpretation. A tiny regime slice can otherwise display unstable Sharpe/return metrics and look like a genuine conditional edge.

### Implemented delta

`regime_support.py`

- counts total observations by regime;
- counts active strategy observations by regime;
- reports sample fractions;
- accepts optional explicit minimum-support requirements;
- invents no universal threshold;
- never changes a trading signal.

`t-SNE` or other visual clustering is **not** used as a strategy-selection criterion.

## 3. Prediction-market arbitrage

### Comparison to project scope

This is a different trading venue, contract-definition problem, settlement regime, and execution stack. It does not improve the current INDEX / ETF / STOCK Gravity decision engine.

A prediction-market price could someday be researched as an external event-expectation feature, but that would require:

- a verified event-identity mapping;
- publication / availability timestamps;
- resolution-rule equivalence checks;
- independent OOS incremental-value testing.

### Decision

**Rejected from the current engine.** No Polymarket/Kalshi arbitrage module, Kelly sizing module, order-book parser, or venue execution code was added.

## 4. News-to-price diffusion / Hawkes process

### Evidence review

The cited Rambaldi–Pennesi–Lillo work on macroeconomic news and FX market activity is a real Hawkes-process research direction. Its useful idea is the separation of a scheduled exogenous event from self-excited market activity and the estimation of how the response decays.

### Existing gap

Market Gravity already has macro/event regime handling, but it did not have a dedicated, causal measure of *post-event absorption speed*.

### Data-feasibility constraint

The current historical stack includes coarse/intraday bar data suitable for event studies, but a genuine Hawkes point-process model should use event-time observations such as trades, quotes, or price-change events at materially finer resolution. Treating 5-minute bars as point-process events would create false precision.

### Implemented delta

`event_response.py`

- `causal_event_absorption_state`: uses only observations at or before `as_of` and reports current activity relative to a frozen pre-event baseline;
- `completed_event_decay_diagnostic`: fits an offline exponential decay to observed excess activity and reports `beta`, half-life, fit quality, and `available_at`;
- explicitly labels the result as **not a Hawkes fit** and **not a trading probability**;
- forbids automatic NO_TRADE thresholds;
- preserves the possibility of a later true Hawkes tournament when event-time data is available.

## Net additions from the four images

Only two implementation deltas survived review:

1. **Regime support audit** — prevents tiny conditional samples from being mistaken for regime edge.
2. **Causal event absorption / decay diagnostics** — adds a useful macro/news response state without pretending coarse bars are a Hawkes process.

Everything else was either already present, outside scope, or methodologically premature.
