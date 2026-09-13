#!/usr/bin/env python3
"""Governed bootstrap for approved Quant Research Toolbox repositories.

This utility clones only entries that are already approved/retained in repos.json.
It never installs packages globally and never activates PARK/REJECT entries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "repos.json"
DEFAULT_DEST = ROOT.parent / ".toolbox"


def load_manifest() -> dict:
    with MANIFEST.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def repo_map(manifest: dict) -> dict[str, dict]:
    return {item["name"].lower(): item for item in manifest["repos"]}


def choose_url(item: dict) -> str | None:
    return item.get("owned_repo") or item.get("upstream")


def list_entries(manifest: dict) -> int:
    for item in manifest["repos"]:
        url = choose_url(item) or "(no clone target; conceptual/reference entry)"
        print(f"{item['name']:<24} {item['decision']:<42} {item['activation']:<30} {url}")
    return 0


def clone_entry(manifest: dict, name: str, destination: Path) -> int:
    mapping = repo_map(manifest)
    item = mapping.get(name.lower())
    if item is None:
        print(f"ERROR: {name!r} is not in the approved toolbox manifest.", file=sys.stderr)
        return 2

    activation = item["activation"]
    if activation in {"conceptual_isolated", "reference_only", "owned_repo_patterns_only", "owned_repo_reference"}:
        print(f"REFUSED: {item['name']} is governed as {activation}; it is not an automatic runtime dependency.")
        return 3

    url = choose_url(item)
    if not url:
        print(f"ERROR: {item['name']} has no governed clone target.", file=sys.stderr)
        return 4

    destination.mkdir(parents=True, exist_ok=True)
    target = destination / item["name"]
    if target.exists():
        print(f"EXISTS: {target}")
        return 0

    print(f"Cloning {url} -> {target}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
    print("Clone complete. Production promotion is NOT implied; project validation rules still apply.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Governed Quant Research Toolbox bootstrap")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List approved/retained toolbox entries")

    clone = sub.add_parser("clone", help="Clone one approved entry into the local sandbox")
    clone.add_argument("name", help="Manifest repo name, e.g. Kronos, vectorbt, LEAN, Jesse")
    clone.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Sandbox directory")

    args = parser.parse_args()
    manifest = load_manifest()

    if args.command == "list":
        return list_entries(manifest)
    if args.command == "clone":
        return clone_entry(manifest, args.name, args.dest)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
