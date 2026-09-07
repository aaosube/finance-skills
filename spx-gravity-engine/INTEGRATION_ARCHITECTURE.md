# Curated Quant + Infrastructure Integration

## Objective

Add only capabilities that improve data integrity, research validity, reproducibility, or engineering throughput. Verify existing infrastructure before adding replacements. No third-party or internal AI agent is allowed to become the trading authority.

## Resulting topology

```text
Barchart / ChartExchange / QuantWheel / approved broker exports
                    |
                    v
          data_ingestion.py
      provenance + schema audit
                    |
                    v
      frozen causal feature matrix
                    |
       +------------+----------------------+------------------+
       |                                   |                  |
       v                                   v                  v
ai_research_loop.py                 shadow_research.py   UnoRouter Agent
hypothesis generation               counterfactual       evidence review
TRAIN-only feedback                  replay               + adversarial review
       |                                   |                  |
       +-------------------+---------------+------------------+
                           v
                    validation.py
          purged walk-forward + prior baseline
                           |
                           v
                 SPX Gravity Stage 1/2/3
```

The existing UnoRouter analytical agent lives in `aaosube/market-research-gatewa` and is already mounted by the Market Research Gateway. Its canonical roles are `primary_reasoner`, `independent_critic`, `finance_reviewer`, and `fast_monitor`. It consumes a versioned deterministic `MarketSnapshot` behind an integrity gate. Model disagreement is preserved and no model vote is interpreted as an empirical probability.

The UnoRouter layer is **not** a replacement for `ai_research_loop.py`. The latter is a constrained hypothesis-generation loop whose candidates are fitted and selected by deterministic code using TRAIN/CALIBRATION/TEST separation. The UnoRouter agent is an evidence-interpretation/review layer. They must remain semantically separate unless a future, explicitly tested research-agent contract is added.

`gamma_scenario.js` is a Stage-1 diagnostic input only. It recomputes Black-Scholes gamma across an explicit spot grid when the full required fields are available. It does **not** reuse current gamma as though it were a true flip.

## Adopted capabilities

- **Existing UnoRouter agent:** canonical AI evidence/review layer in the Market Research Gateway; verified live on Railway with the dedicated `/ai/unorouter/health` route, a direct UnoRouter JSON probe, and an end-to-end `/analyze` smoke through `fast_monitor`.
- **Vibe-Trading patterns:** local file ingestion, structured shadow/counterfactual research, and OOS discipline. Autonomous execution and generated executable strategy code are rejected. Vibe does not replace the existing UnoRouter agent.
- **Scenario-repriced gamma:** requires IV and positive time-to-expiry for every option row; missing data fails closed.
- **Context-mode:** development-only MCP infrastructure; not imported by the engine.
- **Camofox:** optional authorized/public browser-ingestion service; browser results remain non-canonical until validated.
- **Fincept / GammaGrid:** reference-only; no source copied because their licensing creates obligations/restrictions unsuitable for casual vendoring.
- **ClawRouter:** rejected as redundant with the deployed UnoRouter orchestration layer; it adds no market-data or validation capability required by the engine.

## Non-negotiable invariants

1. Stage 1 structural levels are immutable to AI and external agents.
2. Stage 2 confirmation cannot rewrite Stage 1.
3. Stage 3 monetization remains downstream.
4. No `exec`/`eval` of generated strategy text.
5. No external broker execution from research modules.
6. Test/OOS information is not fed back into candidate generation.
7. Vendor/browser data requires provenance and timestamp validation.
8. A missing required market field is reported; it is never fabricated.
9. External packages are pinned when invoked from infrastructure scripts.
10. Secrets stay in environment variables and are excluded from Git.
11. UnoRouter model output is advisory evidence interpretation; numeric probabilities must originate from deterministic/backtested methods.
12. New routers/agents are rejected when they duplicate an already verified capability without measurable incremental value.
