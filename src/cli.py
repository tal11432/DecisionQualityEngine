"""Command line: python -m src.cli {download,validate,plot} ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

from src.config import ROOT, load_config


def _paths(cfg: dict, game: int) -> tuple[Path, Path, Path]:
    g = cfg["metrica"]["games"][str(game)]
    d = ROOT / cfg["metrica"]["local_dir"] / g["name"]
    return tuple(d / f for f in g["files"])  # home, away, events


def _load(cfg: dict, game: int):
    from src.ingestion.metrica import read_events, read_tracking
    from src.schema import Pitch
    home, away, ev = _paths(cfg, game)
    pitch = Pitch.from_config(cfg)
    td = read_tracking(home, away, pitch)
    return td, (read_events(ev, pitch) if ev.exists() else None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dq", description=__doc__)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download a Metrica sample game (needs --yes)")
    d.add_argument("--game", type=int, default=1)
    d.add_argument("--yes", action="store_true")
    v = sub.add_parser("validate", help="load, validate and sync events; print report")
    v.add_argument("--game", type=int, default=1)
    p = sub.add_parser("plot", help="save a PNG of a frame, time or event")
    p.add_argument("--game", type=int, default=1)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--frame", type=int, help="provider frame number")
    g.add_argument("--time", type=float, help="seconds (nearest frame, within tolerance)")
    g.add_argument("--event", type=int, help="row index in events table")
    p.add_argument("--orient", choices=["Home", "Away"], help="rotate so this team attacks right")
    p.add_argument("--label", choices=["jersey", "id", "none"], default="jersey")
    p.add_argument("--out", default="outputs/frame.png")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)

    try:
        if a.cmd == "download":
            from src.ingestion.download import download
            download(a.game, a.yes, cfg)
            return 0

        td, ev = _load(cfg, a.game)
        tol = cfg["sync"]["tolerance_s"]
        if a.cmd == "validate":
            from src.processing.sync import synchronize_events
            from src.processing.validation import validate_tracking
            print(f"Loaded {len(td.frames)} frames, {len(td.players)} players; attack dirs: {td.attack_dir}")
            rep = validate_tracking(td, cfg)
            print(rep.to_text())
            if ev is not None:
                print(synchronize_events(ev, td.frames, tol).summary())
            return 0 if rep.ok else 1

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.processing.sync import synchronize_events
        from src.visualization.pitch import plot_event, plot_frame
        if a.event is not None:
            if ev is None:
                raise FileNotFoundError("events file not found")
            res = synchronize_events(ev.loc[[a.event]], td.frames, tol)
            if res.matched.empty:
                print(f"Event {a.event} unmatched: {res.unmatched['reason'].iloc[0]}")
                return 1
            row = int(res.matched["frame_row"].iloc[0])
            plot_event(td, ev.loc[a.event].to_dict(), row, a.orient, label=a.label)
        else:
            if a.frame is not None:
                rows = td.frames.index[td.frames["frame"] == a.frame]
                if len(rows) != 1:
                    raise ValueError(f"frame {a.frame}: {len(rows)} matching rows")
                row = int(rows[0])
            else:
                row = int((td.frames["time_s"] - a.time).abs().to_numpy().argmin())
                if abs(td.frames["time_s"].iloc[row] - a.time) > tol:
                    raise ValueError(f"no frame within {tol}s of t={a.time}")
            plot_frame(td, row, a.orient, label=a.label)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved {out}")
        return 0
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
