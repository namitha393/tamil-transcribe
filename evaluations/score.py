"""Score every results/<dataset>/<model>.jsonl and print a comparison table.

  python score.py               # all datasets
  python score.py fleurs_ta     # one dataset
  python score.py --examples 5  # also show worst clips per model

Only clips that EVERY model on that dataset has transcribed are scored, so
the numbers stay comparable while some runs are still in progress.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer

from normalize import normalize, score

ROOT = Path(__file__).resolve().parent


def load(p: Path) -> dict:
    return {r["id"]: r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}


def table(ds_dir: Path, examples: int) -> str:
    runs = {p.stem: load(p) for p in sorted(ds_dir.glob("*.jsonl"))}
    if not runs:
        return ""
    common = set.intersection(*(set(r) for r in runs.values()))
    if not any(next(iter(r.values()))["ref"] for r in runs.values()):
        return ""  # no reference text (e.g. customer audio before correction)
    lines = [f"\n## {ds_dir.name}  ({len(common)} clips common to all {len(runs)} models)\n",
             "| model | WER % | CER % | errors | empty | RTF (CPU) | audio h |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    rows = []
    for name, r in runs.items():
        recs = [r[i] for i in sorted(common)]
        s = score([x["ref"] for x in recs], [x["hyp"] for x in recs])
        audio = sum(x["duration"] for x in recs)
        rtf = sum(x["seconds"] for x in recs) / audio if audio else float("nan")
        errs = sum(1 for x in recs if x.get("error"))
        rtf_s = "API" if recs and recs[0].get("api") else f"{rtf:.2f}"
        rows.append((s["wer"], f"| {name} | {100*s['wer']:.1f} | {100*s['cer']:.1f} | {errs} | "
                               f"{s['empty_outputs']} | {rtf_s} | {audio/3600:.2f} |", name, recs))
    rows.sort(key=lambda x: x[0])
    lines += [r[1] for r in rows]
    if examples:
        for _, _, name, recs in rows:
            worst = sorted(recs, key=lambda x: -jiwer.wer(normalize(x["ref"]) or "-", normalize(x["hyp"])))
            lines.append(f"\n### {name}: worst {examples}")
            for x in worst[:examples]:
                lines.append(f"- `{x['id']}`\n  - REF: {x['ref']}\n  - HYP: {x['hyp']}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--examples", type=int, default=0)
    a = ap.parse_args()
    dirs = [ROOT / "results" / a.dataset] if a.dataset else sorted((ROOT / "results").glob("*/"))
    out = "\n".join(table(d, a.examples) for d in dirs if d.is_dir())
    print(out)
    (ROOT / "results" / "SUMMARY.md").write_text(out + "\n", encoding="utf-8")
