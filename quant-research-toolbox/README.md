# Quant Research Toolbox

This directory activates the GitHub repositories that passed the Quant Research Intake & Validation filter **without turning them into global production dependencies**.

## Operating rule

A repository can be present in the system in one of four ways:

1. **Owned/retained repo** — already present under `aaosube/*` and available for governed use.
2. **Sandbox clone on demand** — upstream is approved for experimentation/toolbox use but is not automatically installed.
3. **Conceptual/reference only** — knowledge or patterns may be used, but the code is not a runtime dependency.
4. **Inactive** — PARK/REJECT items remain recorded to prevent accidental reintroduction.

The source of truth for repository status remains the governed Quant Research Registry in Google Drive. `repos.json` is the GitHub execution projection of that registry.

## Active approved set

| Registry | Repo | Decision | Activation |
|---|---|---|---|
| R001 | PaperGym | ADOPT | Conceptual/isolated; owned fork exists |
| R017 | QuantResearch | KEEP / REFERENCE | Owned reference fork |
| R020 | Kronos | EXPERIMENT | Sandbox clone on demand |
| R021 | skfolio | ADOPT FOR RESEARCH | Sandbox install on demand |
| R070 | vectorbt | KEEP / TOOLBOX | Sandbox clone on demand |
| R071 | hftbacktest | KEEP / TOOLBOX | Owned sandbox repo |
| R072 | Qlib | KEEP OPTIONAL | Owned; project-specific |
| R073 | LEAN | KEEP / TOOLBOX | Sandbox clone on demand |
| R074 | OpenBB | KEEP / TOOLBOX | Owned; use only for source gaps |
| R075 | TradingAgents | KEEP / TOOLBOX | Owned; isolated only |
| R077 | Vibe-Trading | RETAIN EXISTING | Patterns only; not the system platform |
| R078 | market-research-gateway | EXISTING / RETAIN | System infrastructure boundary |
| R093 | quantdataapi_sdk | EXISTING / RETAIN | Project-specific integration tool |
| R094 | FinGPT | KEEP / TOOLBOX | Owned; isolated |
| R095 | FinRL | KEEP / EXPERIMENT TOOLBOX | Owned sandbox |
| R096 | finrobot | KEEP / TOOLBOX | Owned; isolated |
| R097 | rd-agent | KEEP / TOOLBOX | Owned; governed automation only |
| R098 | playwright-mcp | KEEP / INFRASTRUCTURE TOOL | Owned; on demand |
| R099 | no-ai-slop | KEEP / REFERENCE | Research-writing quality only |
| R100 | Jesse | EXPERIMENT TOOLBOX | Sandbox clone on demand |

## Explicitly inactive

- `gstack` — PARK.
- `NautilusTrader` — PARK / evaluate only if an execution/backtest bottleneck appears.
- `Vibe-Trading` as a platform — REJECT; only existing retained patterns remain.

## Activation

Use the bootstrap utility to clone a selected approved upstream into a local `.toolbox/` sandbox. It does not install anything globally and it refuses inactive entries.

```bash
python quant-research-toolbox/bootstrap.py list
python quant-research-toolbox/bootstrap.py clone Kronos
python quant-research-toolbox/bootstrap.py clone vectorbt
python quant-research-toolbox/bootstrap.py clone LEAN
python quant-research-toolbox/bootstrap.py clone Jesse
```

For owned repositories, the manifest points to the owned GitHub copy and the bootstrap utility clones that copy by default.

## Promotion boundary

Repository presence is **not** production approval. Any model, strategy, feature, execution logic, or data adapter derived from these repos must still pass the project firewall, experiment ledger, point-in-time controls, and Validation Firewall before production use.
