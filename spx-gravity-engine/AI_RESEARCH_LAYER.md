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
2. Session reconstruction / asset profile
3. Macro + volatility regime
4. Options / dealer state where applicable
5. Structural geometry
6. Reachability / NGC
7. Boundary competition
8. **QuantAI hypothesis layer**
9. Engine-owned calibration / OOS test
10. Trigger / position generation
11. **Cost-aware strategy evaluator**
12. Tail / liquidity-capacity / stability audit
13. Underlying EV
14. Option-contract EV, if applicable

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
- maximum drawdown;
- recovery gain required after maximum drawdown;
- profit factor;
- active-period win rate;
- exposure;
- entries per year;
- regime breakdown;
- empirical tail VaR / Expected Shortfall without a normality assumption;
- optional spread / dollar-volume / participation feasibility limits;
- optional recent-vs-prior stability diagnostics.

Liquidity limits are active only when explicit data columns and limits are configured. Liquidity diagnostics do not silently add a second spread/slippage charge on top of the execution-cost model.

Stability diagnostics do not trigger automatic retraining or model switching. Adaptation requires a separately specified governance rule and OOS evidence.

### Deflated Sharpe Ratio

DSR is used only on realized strategy return series.

The number of trials must reflect the strategies actually tested (or a justified effective-independent count). The engine does **not** accept an arbitrary prompt instruction such as `N=100` unless 100 trials were genuinely part of the search.

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

## 42-laws audit: what is deliberately NOT added

The “42 Laws of Quant Finance” images are treated as an audit checklist, not as a canonical specification. Most items restate principles already enforced by the engine.

No new module is added merely because an image mentions calculus, linear algebra, stochastic calculus, Monte Carlo simulation, C++, reinforcement learning, correlation, or automation. Such methods are admitted only when they solve a defined model/data/execution problem and demonstrate incremental OOS value.

Correlation-breakdown stress belongs to the future multi-asset / portfolio layer rather than the current single-instrument strategy evaluator. Capital sizing/risk budgeting likewise remains a separate downstream layer because index, ETF, stock and option contracts require different capital and liquidity mechanics.

This policy prevents feature/method accumulation from being mistaken for research quality.
