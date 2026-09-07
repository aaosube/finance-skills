# Context-mode (optional development infrastructure)

Pinned package: `context-mode@1.0.169`.

Use this only in coding/research-agent environments where large tool outputs are consuming context. It is not a market-data or trading dependency.

Before enabling it in a new environment, review the Elastic-2.0 license and run:

```bash
npx -y context-mode@1.0.169 doctor
```

The example configs deliberately pin the package version rather than executing an unbounded `latest` release.
