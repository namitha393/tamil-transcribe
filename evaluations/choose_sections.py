"""Pick test sections from a long recording using several models' outputs.

Difficulty per piece = average pairwise character disagreement among the good
models (no correct transcript exists yet, so disagreement is the stand-in).
Then slide a 5-minute window and pick: A typical, B astrology-heavy,
C interaction/English-heavy - non-overlapping.
"""
import itertools, json, sys
import numpy as np, jiwer
from normalize import normalize
from pipeline.tanglish import LOANS, ASTRO

name = "astro_class"
MODELS = ["indic-transcribe-core", "indic-transcribe-flex", "sravaani-tamil", "hima-asr"]
man = [json.loads(l) for l in open(f"manifests/{name}.jsonl")]
H = {m: {json.loads(l)["id"]: normalize(json.loads(l)["hyp"]) for l in open(f"results/{name}/{m}.jsonl")} for m in MODELS}
EN = [k for k in LOANS if k not in ASTRO]
rows = []
for p in man:
    i = p["id"]
    d = np.mean([jiwer.cer(H[a][i].replace(" ", "") or "-", H[b][i].replace(" ", "") or "-")
                 for a, b in itertools.combinations(MODELS, 2)])
    w = H["indic-transcribe-core"][i].split()
    rows.append(dict(id=i, t=p["start"], dur=p["duration"], dis=min(d, 1.5), n=len(w),
                     en=sum(any(x.startswith(k) for k in EN) for x in w) + sum(any(c in "ஜஷஸஹ" for c in x) for x in w),
                     astro=sum(any(x.startswith(k) for k in ASTRO) for x in w)))
json.dump(rows, open(f"results/{name}/profile_v2.json", "w"))
print(f"pieces {len(rows)}; mean pairwise disagreement {100*np.mean([r['dis'] for r in rows]):.1f}% (was 44.9% with the 2 buggy runs)")
for a, b in itertools.combinations(MODELS, 2):
    print(f"  {a} vs {b}: {100*np.mean([jiwer.cer(H[a][r['id']].replace(' ','') or '-', H[b][r['id']].replace(' ','') or '-') for r in rows]):.1f}%")
W = 300
def win(t0):
    g = [r for r in rows if t0 <= r["t"] < t0 + W]
    n = sum(r["n"] for r in g) or 1
    return dict(t0=t0, dis=np.mean([r["dis"] for r in g]), en=sum(r["en"] for r in g)/n, astro=sum(r["astro"] for r in g)/n, n=n)
wins = [win(t) for t in range(60, int(rows[-1]["t"]) - W, 30)]
med = {k: np.median([w[k] for w in wins]) for k in ("dis", "en", "astro")}
picked = []
def ok(w): return all(abs(w["t0"] - p["t0"]) >= W for _, p in picked)
B = max(wins, key=lambda w: w["astro"]); picked.append(("B astrology-heavy", B))
C = max([w for w in wins if ok(w)], key=lambda w: w["en"] + 0.5 * w["dis"]); picked.append(("C interaction/English", C))
A = min([w for w in wins if ok(w)], key=lambda w: sum(abs(w[k] - med[k]) / (med[k] + 1e-9) for k in med)); picked.append(("A typical", A))
ts = lambda s: f"{int(s//60)}:{int(s%60):02d}"
print(f"\nmedians: disagreement {100*med['dis']:.1f}%, English {100*med['en']:.1f}%, astrology {100*med['astro']:.1f}%")
for lab, w in sorted(picked):
    print(f"{lab:24s} {ts(w['t0'])}-{ts(w['t0']+W)}  disagreement {100*w['dis']:.1f}%  English {100*w['en']:.1f}%  astrology {100*w['astro']:.1f}%  words {w['n']}")
json.dump([{"label": l, "start": w["t0"], "end": w["t0"] + W} for l, w in sorted(picked)], open(f"results/{name}/sections.json", "w"))
