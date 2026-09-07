# Curated Quant + Infrastructure Integration

## Objective

Add only capabilities that improve data integrity, research validity, reproducibility, or engineering throughput. No third-party agent is allowed to become the trading authority.

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
       +------------+-------------+
       |                          |
       v                          v
ai_research_loop.py        shadow_research.py
hypothesis generation      structured counterfactual replay
       |                          |
       +------------+-------------+
                    v
             validation.py
     purged walk-forward + prior baseline
                    |
                    v
          SPX Gravity Stage 1/2/3
```

`gamma_scenario.js` is a Stage-1 diagnostic input only. It recomputes Black-Scholes gamma across an explicit spot grid when the full required fields are available. It does **not** reuse current gamma as though it were a true flip.

## Adopted capabilities

- **Vibe-Trading patterns:** local file ingestion, structured shadow/counterfactual research, and OOS discipline. Autonomous execution and generated executable strategy code are rejected.
- **Scenario-repriced gamma:** requires IV and positive time-to-expiry for every option row; missing data fails closed.
- **Context-mode:** development-only MCP infrastructure; not imported by the engine.
- **Camofox:** optional authorized/public browser-ingestion service; browser results remain non-canonical until validated.
- **Fincept / GammaGrid:** reference-only; no source copied because their licensing creates obligations/restrictions unsuitable for casual vendoring.

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
