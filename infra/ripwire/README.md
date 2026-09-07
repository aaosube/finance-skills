# Ripwire — optional deterministic code-impact tooling

Pinned upstream: `redhat-et/ripwire` **v0.4.0** (Apache-2.0).

## Why it is useful here

Ripwire is not a trading, market-data, ML, or execution component. It is a local development aid that builds a deterministic code/call graph and ranks symbols relevant to a requested change. That gives Market Gravity development a pre-change map of:

- likely symbols/files to inspect;
- callers/callees;
- blast radius / change amplification;
- complexity and git churn;
- tests that reach changed symbols;
- PR/diff context.

This is complementary to `context-mode`: context-mode manages coding-agent context budget, while Ripwire maps repository structure and change impact. It does not replace tests, code review, or the engine's causal/data validation.

## Hard boundaries

- `runtime_dependency = false`.
- Never deploy Ripwire inside the Railway market runtime.
- Never use Ripwire output as market evidence, a trading feature, or a probability.
- Never allow it to mutate canonical Stage 1/2/3 output.
- No broker execution.
- No bundled/advisory hooks are activated by this repository.
- Do not vendor the upstream binary or source here.
- Pin the release before use; do not install an unbounded `latest` in controlled environments.

## Upstream release verification

The reviewed upstream release is `v0.4.0`. Upstream publishes per-asset SHA-256 files and its installer refuses missing checksums, checks archive paths before extraction, and verifies the installed binary version.

For a local development machine, prefer a reviewed pinned installation and disable automatic agent-skill activation if you only want the CLI:

```bash
export RIPWIRE_REPO=redhat-et/ripwire
export RIPWIRE_VERSION=v0.4.0
export RIPWIRE_NO_ACTIVATE=1
# Review the upstream scripts/install.sh before running it in a new environment.
```

The repository helper below intentionally **does not install anything**. It only uses an already-installed matching binary.

## Use before risky changes

```bash
./infra/ripwire/review-change.sh "change option-chain parity integrity and execution latency diagnostics"
```

Use the result as an orientation/impact report, then inspect the actual files and run the real test suite. A high Ripwire ranking/confidence is not evidence that a proposed implementation is correct.
