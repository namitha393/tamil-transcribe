"""Un-blind the judges' scores and summarise per system (and per input transcript).
Writes results.md next to this file."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

H = Path(__file__).parent
key = json.loads((H / "key.json").read_text())
scores = {}
for f in sorted(H.glob("scores_*.json")):
    scores.update(json.loads(f.read_text()))
missing = [k for k in key if k not in scores]
rows = defaultdict(list)
for k, s in scores.items():
    if k in key:
        rows[key[k]].append((k.split(":")[0], s))

def summary(items):
    m = [s["meaning"] for _, s in items]; f = [s["fluency"] for _, s in items]
    t = [s["terms"] for _, s in items if s.get("terms") is not None]
    inv = sum(bool(s.get("invented")) for _, s in items)
    return mean(m), mean(f), (mean(t) if t else float("nan")), inv, len(items)

NICE = {"claude-opus-5.5": "LLM, whole paragraph (Claude)", "sarvam-speech-translate": "Sarvam speech → English (cloud)",
        "sarvam-translate": "Sarvam-Translate text (cloud)", "indictrans2-200M-sentences": "IndicTrans2-200M, per sentence (offline)",
        "indictrans2-200M-paragraph": "IndicTrans2-200M, whole paragraph (offline)",
        "indictrans2-1B-sentences": "IndicTrans2-1B half-precision, per sentence (offline)",
        "indictrans2-1B-paragraph": "IndicTrans2-1B half-precision, whole paragraph (offline)"}
INPUT = {"ours": "our transcript", "gemini": "Gemini's clean transcript", "audio": "audio"}
lines = ["# Translation scores (blind LLM judges, 1–5, higher is better)", "",
         f"Candidates scored: {sum(len(v) for v in rows.values())}; missing: {len(missing)}", "",
         "| System | Input | Meaning | Fluency | Terms | Invented | n |", "|---|---|--:|--:|--:|--:|--:|"]
order = sorted(rows, key=lambda k: -summary(rows[k])[0])
for k in order:
    sysname, inp = k.split("__")
    m, f, t, inv, n = summary(rows[k])
    lines.append(f"| {NICE.get(sysname, sysname)} | {INPUT[inp]} | **{m:.2f}** | {f:.2f} | {t:.2f} | {inv}/{n} | {n} |")
# fair comparison on the 5 units where IndicTrans2-1B finished
units_1b = sorted({u for u, _ in rows.get("indictrans2-1B-sentences__ours", [])})
if units_1b:
    lines += ["", f"## Same {len(units_1b)} units only ({', '.join(units_1b)}) — includes IndicTrans2-1B", "",
              "| System | Input | Meaning | Fluency | Terms |", "|---|---|--:|--:|--:|"]
    for k in order:
        sub = [(u, s) for u, s in rows[k] if u in units_1b]
        if sub:
            sysname, inp = k.split("__"); m, f, t, inv, n = summary(sub)
            lines.append(f"| {NICE.get(sysname, sysname)} | {INPUT[inp]} | {m:.2f} | {f:.2f} | {t:.2f} |")
lines += ["", "## Judges' notes (lowest meaning first, 2 per system)", ""]
for k in order:
    worst = sorted(rows[k], key=lambda x: x[1]["meaning"])[:2]
    lines.append(f"- **{k}**: " + "; ".join(f"{u}: {s.get('note','')}" for u, s in worst))
(H / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
