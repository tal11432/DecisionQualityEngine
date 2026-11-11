"""Explicit, opt-in download of Metrica sample data (pinned to a commit)."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from src.config import ROOT, load_config


def plan(game: int, cfg: dict | None = None) -> list[tuple[str, Path]]:
    cfg = cfg or load_config()
    m = cfg["metrica"]
    g = m["games"].get(str(game))
    if g is None:
        raise ValueError(f"Unknown game {game}; configured: {sorted(m['games'])}")
    out_dir = ROOT / m["local_dir"] / g["name"]
    return [(m["base_url"].format(commit=m["commit"], game=g["name"], file=f), out_dir / f) for f in g["files"]]


def remote_size(url: str) -> int | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            return int(r.headers.get("Content-Length", "")) or None
    except Exception:
        return None


def download(game: int, confirm: bool = False, cfg: dict | None = None) -> list[Path]:
    """Download one game. Without ``confirm`` only reports what would be fetched (no network writes)."""
    items = plan(game, cfg)
    sizes = [remote_size(u) for u, _ in items]
    for (u, dest), s in zip(items, sizes):
        print(f"  {u}\n    -> {dest}  ({'?' if s is None else f'{s / 1e6:.1f} MB'})")
    if not confirm:
        print(f"Total ~{sum(s or 0 for s in sizes) / 1e6:.0f} MB. Nothing downloaded; re-run with --yes to proceed.")
        return []
    manifest = {}
    for (url, dest), _ in zip(items, sizes):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"exists, skipping: {dest.name}")
        else:
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
        manifest[dest.name] = {"url": url, "bytes": dest.stat().st_size,
                               "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    (items[0][1].parent / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return [d for _, d in items]
