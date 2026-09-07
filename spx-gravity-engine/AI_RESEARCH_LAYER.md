# Market Gravity Engine — Quant Research Layer

## Role

`QuantAI` is a constrained senior-quant hypothesis generator inside Market Gravity Engine. It is not a trading guru, not the execution engine, and not the authority that scores its own ideas.

The useful parts of the Quant Trading Strategist prompt are preserved:

- hypothesis first;
- economic / microstructure intuition before model complexity;
- explicit model/market assumptions;
- transaction costs and slippage;
- regime testing;
- drawdown, turnover and trade-frequency reporting;
- walk-forward / multiple-testing control;
- explicit attempt to falsify every strategy.

The unsafe parts are deliberately changed:

- no LLM-generated Python is executed;
- no `exec()` / `eval()`;
- no repeated optimization on the untouched test set;
- no arbitrary `N=100` for DSR;
- no direct optimization of Sharpe inside the hypothesis loop;
- yfinance is smoke-test / research convenience only, not the 23h production data source.

## System placement

1. Causal data gate
2. **Option-chain integrity gate when options are used** (`option_integrity.py`)
3. Session reconstruction / asset profile
4. Macro + volatility regime
5. Options / dealer state where applicable
6. **IV-surface state features where available** (`iv_surface_dynamics.py`)
7. Structural geometry
8. Reachability / NGC
9. Boundary competition
10. **QuantAI hypothesis layer**
11. Engine-owned calibration / OOS test
12. Trigger / position generation
13. **Cost-aware strategy evaluator**
14. Tail / liquidity-capacity / **execution-latency** / stability audit
15. Underlying EV
16. Option-contract EV, if applicable

The same research contract supports `index`, `etf`, and `stock` profiles. SPX remains the first specialized profile.

## QuantAI output contract

QuantAI may return only a structured hypothesis:

- `name`
- `mechanism`
- `assumptions`
- `features`
- `interactions`
- `expected_regimes`
- `failure_modes`
- `rationale`

It cannot change the target, split, labels, evaluator, structural levels, costs, or risk methodology. At least one explicit assumption is required; assumption count is not assigned an arbitrary score or weight.

## Validation protocol

Candidate-generation feedback is not in-sample.

```text
TRAIN
  └─ purged inner walk-forward feedback to QuantAI
        ↓
freeze candidate set
        ↓
CALIBRATION selects one candidate
        ↓
refit TRAIN + CALIBRATION
        ↓
TEST exactly once
```

The untouched TEST result is never returned to QuantAI for another iteration.

## Prediction metrics

For direction / destination / boundary models:

- log loss
- multiclass Brier score
- calibration
- first-hit accuracy
- time-to-hit
- reversal / continuation quality

Sharpe is not a valid replacement for probability calibration.

## Strategy / execution metrics

After a model and trigger are frozen, `strategy_evaluator.py` applies:

- transaction costs;
- slippage;
- turnover;
- annualized Sharpe;
- explicit-MAR Sortino;
- maximum drawdown;
- compounded net return, reported separately from arithmetic return;
- diagnostic annualized compounded growth and Calmar ratio;
- recovery gain required after maximum drawdown;
- profit factor;
- active-period win rate;
- exposure;
- entries per year;
- regime breakdown;
- empirical tail VaR / Expected Shortfall without a normality assumption;
- optional spread / dollar-volume / participation feasibility limits;
- optional measured decision→order→fill latency diagnostics;
- optional recent-vs-prior stability diagnostics.

Annualized compounded growth and Calmar are diagnostic only. `periods_per_year` must match the actual decision frequency; the engine does not infer a misleading annualization from timestamps.

Liquidity limits are active only when explicit data columns and limits are configured. Liquidity diagnostics do not silently add a second spread/slippage charge on top of the execution-cost model.

Latency is not represented by an invented constant. It is evaluated only when actual `decision_time`, `order_time`, and `fill_time` fields exist. Any latency SLO must be explicitly configured from measured infrastructure or execution policy.

Stability diagnostics do not trigger automatic retraining or model switching. Adaptation requires a separately specified governance rule and OOS evidence.

### Deflated Sharpe Ratio

DSR is used only on realized strategy return series.

The number of trials must reflect the strategies actually tested (or a justified effective-independent count). The engine does **not** accept an arbitrary prompt instruction such as `N=100` unless 100 trials were genuinely part of the search.

## Option-chain integrity: put-call parity

Put-call parity is admitted as a **data-quality / synchronization invariant**, not as a directional signal.

When a forward is available:

```text
C - P = exp(-rT) * (F - K)
```

Otherwise, with spot and dividend yield:

```text
C - P = S*exp(-qT) - K*exp(-rT)
```

`option_integrity.py` supports a threshold-free executable bid/ask interval check when synchronized call and put quotes exist. It deliberately does not pair calls and puts itself: contract matching and as-of synchronization must happen upstream to avoid look-ahead or root/expiry mismatches.

## IV-surface dynamics

The engine adopts the **problem** of forward IV-surface dynamics, not a preselected deep-learning method.

`iv_surface_dynamics.py` creates auditable decision-time surface state features such as:

- nearest-forward ATM IV;
- descriptive skew slope;
- descriptive curvature;
- ATM term-structure slope.

A later synchronized snapshot can be converted to `FUTURE_LABEL_ONLY` surface-change outcomes. Those future changes may never be merged back into the same decision row as features.

LSTM, Transformer, GARCH, HAR, PCA/state-space or simpler baselines are model candidates only. They enter the canonical stack only after chronological OOS comparison demonstrates incremental value for option-contract EV or another defined target.

## Regime testing

Regimes are engine-built causal states, not post-hoc labels chosen to make a strategy look good. Examples may include:

- calm / normal / volatile VIX state;
- macro-event state;
- positive / negative gamma state;
- liquidity regime;
- Asia / London / NY path state;
- stock-specific earnings / borrow / short / dark-pool state.

## Data hierarchy

Production research uses the canonical Market Gravity dataset assembled from Railway / QuantData / market-data sources and the 23h session architecture where relevant.

`yfinance` remains optional for smoke tests and independent checks. It must not silently replace the canonical historical source.

## Falsification rule

Every hypothesis must answer:

> What market inefficiency or conditional mechanism should exist?

> Which explicit assumptions must remain true for the mechanism to make sense?

and

> What observable condition should make the strategy fail?

A strategy with no explicit assumptions or no falsifier is not admitted to the research tournament.

## Formula/project-image audit: what is deliberately NOT added

The “42 Laws of Quant Finance” and “100 Quant Formulas / Quant & ML Projects” images are treated as audit checklists, not as canonical specifications. Most items restate principles or methods already enforced by the engine.

No new module is added merely because an image mentions calculus, linear algebra, stochastic calculus, Monte Carlo simulation, C++, reinforcement learning, correlation, SMA/EMA, ARIMA, GARCH, XGBoost, LSTM, Transformers, or generic stock-return prediction. Such methods are admitted only when they solve a defined model/data/execution problem and demonstrate incremental OOS value.

The latest delta review admitted only the gaps with direct system value:

1. put-call parity as an option-chain integrity check;
2. compounded return plus explicit-MAR Sortino and diagnostic Calmar reporting;
3. measured decision→order→fill execution latency;
4. causal IV-surface state / future-label separation, without selecting a neural model in advance.

Benchmark-relative alpha/beta/information-ratio belongs to ETF/stock or future portfolio comparison when a benchmark question exists. Earnings-call NLP belongs to a stock-event companion layer and must pass timestamp causality and ablation. RL hedging is not admitted without a validated simulator, dense option history, realistic costs and independent OOS evidence.

Correlation-breakdown stress belongs to the future multi-asset / portfolio layer rather than the current single-instrument strategy evaluator. Capital sizing/risk budgeting likewise remains a separate downstream layer because index, ETF, stock and option contracts require different capital and liquidity mechanics.

This policy prevents feature/method accumulation from being mistaken for research quality.
