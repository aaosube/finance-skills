# Camofox (optional authorized web-ingestion service)

Pinned package: `@askjo/camofox-browser@1.14.0` (Node >=22).

This service is intentionally outside the SPX engine. Use it only when an authorized/public webpage must be ingested and a normal API/CSV route is unavailable.

## Guardrails

- Do not use it to bypass authentication or access controls.
- Do not commit cookies, API keys, or session state.
- Browser-extracted prices/market fields are non-canonical until timestamp, symbol, and source are independently validated.
- Prefer CSV/API/vendor exports when available.

Run:

```bash
export CAMOFOX_API_KEY="..."
./infra/camofox/run.sh
```

The upstream service listens on port 9377 by default. Configure remote exposure only behind authentication and network controls.
