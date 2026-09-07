# Market Gravity Engine — Quant Research Layer

## Role

`QuantAI` is a constrained senior-quant hypothesis generator inside Market Gravity Engine. It is not a trading guru, not the execution engine, and not the authority that scores its own ideas.

The useful parts of the Quant Trading Strategist prompt are preserved:

- hypothesis first;
- economic / microstructure intuition before model complexity;
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
12. Underlying EV
13. Option-contract EV, if applicable

The same research contract supports `index`, `etf`, and `stock` profiles. SPX remains the first specialized profile.

## QuantAI output contract

QuantAI may return only a structured hypothesis:

- `name`
- `mechanism`
- `features`
- `interactions`
- `expected_regimes`
- `failure_modes`
- `rationale`

It cannot change the target, split, labels, evaluator, structural levels, costs, or risk methodology.

## Validation protocol

Candidate-generation feedback is no longer in-sample.

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
- profit factor;
- active-period win rate;
- exposure;
- entries per year;
- regime breakdown.

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

and

> What observable condition should make the strategy fail?

A strategy that has no falsifier is not admitted to the research tournament.
