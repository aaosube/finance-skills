# SPX Gravity Engine — AI Research Layer

## Position in the system

The AI layer is **not** a replacement for the Gravity Engine and is not allowed to mutate canonical structural stages.

Pipeline:

1. Causal data gate
2. 23h session reconstruction (ES/NQ + SPX cash identity)
3. Macro/regime state
4. GEX/DEX/VEX/CHEX + OI reconstruction
5. Structural mass / walls / local flip / pockets
6. Reachability / NGC / barrier geometry
7. Boundary competition labels
8. **AI hypothesis layer** — proposes interactions over the frozen feature matrix
9. Calibration / OOS test owned by the evaluator
10. Trigger / execution
11. Contract EV and monetization

The AI sees only feature names and TRAIN-only results during candidate generation.

## Hard invariants

- No `exec()` or `eval()` of model-generated text.
- API keys are read from environment variables only.
- The target column, split dates/fractions, metrics and evaluator are engine-owned.
- Time series are split chronologically; no random train/test split.
- Candidate generation stops before calibration/test evaluation.
- Calibration selects the winner from the frozen candidate set.
- The untouched test set is evaluated once and is never fed back to the LLM.
- Structural Gravity levels cannot be changed by the AI layer.
- Contract profitability remains downstream from underlying/boundary probability.

## Why the original snippet was changed

The original prototype had four research-integrity problems:

1. It reused the same dataset for AI generation and scoring, creating data-snooping / multiple-testing bias.
2. It selected the maximum Sharpe from repeated attempts on the same sample.
3. It executed free-form LLM-generated Python using `exec()`.
4. It was RTH-centric (`^GSPC`/`^VIX`) and therefore could not represent the 23h ES/NQ Asia → London → premarket path used by Gravity.

The new module preserves the useful idea — AI-assisted hypothesis generation — while moving all execution and evaluation authority back into the deterministic research engine.

## Suggested feature pool for the full Gravity matrix

Examples (only include fields that are causal at the checkpoint):

- `asia_return`, `asia_range_z`, `asia_sweep_state`
- `london_return`, `london_range_z`, `london_reclaim_state`
- `overnight_range_z`, `premarket_position`
- `es_nq_relative_strength`, `vix_change`, `rv_state`, `skew_state`
- `gex_net`, `gex_gradient`, `gex_pocket_width`, `gex_pocket_velocity`
- `dex_local`, `vex_local`, `chex_local`
- `gamma_flip_distance`, `gamma_flip_stability`
- `ngc_upper`, `ngc_lower`
- `reach_upper`, `reach_lower`
- `macro_regime`, `event_state`
- `boundary_touch_depth`, `reclaim_speed`, `acceptance_state`

## Model selection objective

Do not optimize Sharpe directly in the hypothesis loop. The core research objective is calibrated state prediction:

- log loss
- multiclass Brier score
- calibration
- first-hit accuracy / time-to-hit
- boundary reversal vs continuation quality

Sharpe, profit factor, drawdown and contract EV belong to the downstream execution/monetization evaluation after probability and boundary models are frozen.
