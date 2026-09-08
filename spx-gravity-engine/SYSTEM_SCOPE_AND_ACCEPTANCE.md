# Market Gravity System: scope and acceptance

## Scope

The system models the next movement and boundary behavior of optionable
underlyings.  Supported profiles are `index`, `etf`, and `stock`.  SPX is the
first specialized profile; TSLA and other equities are not disposable test
artifacts.

For stocks, admissible point-in-time evidence includes dark-pool prints and
levels, short interest/short volume/borrow state, lit flow, options flow and open
interest, GEX/DEX/VEX/CHEX, IV term structure/skew, price/session structure,
events and benchmark/sector context.  Missing fields remain explicit.

## Mandatory order

1. Freeze provider, symbol, market date, retrieval time, availability time and raw hash.
2. Reject stale, future, duplicated or unsynchronized panels.
3. Build the underlying state and direction before considering an option contract.
4. Generate structural boundaries and reversal/continuation triggers.
5. Run deterministic walk-forward and untouched holdout evaluation.
6. Apply the auditable research loss and full trade-cost function, including 1.5x and 2x stress.
7. Run the Devil falsification layer on the frozen candidate.
8. Promote only when baselines, calibration, DSR/multiple-testing and liquidity checks pass.
9. Select expiration/strike/contract last, in a separate option-economics test.

## Agents

Agents may propose bounded hypotheses, identify contradictions and run a Devil
critique.  They may not alter canonical data, labels, splits, costs, probability,
structural levels or acceptance thresholds.  No agent inference is required in
the deterministic backtest loop.

## Current admission state

Infrastructure health is not strategy validity.  Until a frozen point-in-time
dataset, executable signal/trigger pipeline, walk-forward results and untouched
holdout artifact exist, the strategy status is `RESEARCH_ONLY` and live trading
eligibility is `false`.
