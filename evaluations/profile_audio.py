"""Profile a long recording minute by minute to choose test sections.

For each piece: how much the two best models disagree (a stand-in for
difficulty, since there is no correct transcript), English-word density,
astrology-term density, speaking rate and loudness. Aggregated per minute.
"""
import json, sys
import numpy as np, soundfile as sf, jiwer
from normalize import normalize

name = sys.argv[1] if len(sys.argv) > 1 else "astro_class"
A, B = "sravaani", "indicconformer-rnnt"
man = [json.loads(l) for l in open(f"manifests/{name}.jsonl")]
ha = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in open(f"results/{name}/{A}.jsonl")}
hb = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in open(f"results/{name}/{B}.jsonl")}

ASTRO = ["சனி","ராகு","கேது","லக்","நட்சத்","ராசி","குரு","சுக்கிர","செவ்வா","புதன","சந்திர","சூரிய","ஜாதக","தசை","புத்தி",
         "பாவ","கிரக","அம்ச","உச்ச","நீச்ச","ஆட்சி","மேஷ","ரிஷப","மிதுன","கடக","சிம்ம","கன்னி","துலா","விருச்சி","தனுசு","மகர",
         "கும்ப","மீன","அஸ்வினி","பரணி","கார்த்திகை","ரோகிணி","திருவாதிரை","பூசம்","மகம்","அனுஷ","கேட்டை","மூலம்","திசை","கோச்சார",
         "பெயர்ச்சி","வக்ர","அஸ்தம","பார்வை","யோக","தோஷ"]
GRANTHA = set("ஜஷஸஹ")
LOAN = ["ஓகே","சார்","மேடம்","போட்டோ","வீடியோ","ஆடியோ","கரெக்ட","மைண்ட்","ஸ்பீட்","ஸ்லோ","தேங்க்","கிளாஸ்","டிவி","போன்","பாயிண்ட்",
        "நோட்","புக்","சிம்பிள்","டைம்","லைன்","சார்ட்","ஹவுஸ்","பிராக்டிகல்","எக்ஸாம்பிள்","கேஸ்","கிளியர்","டவுட்","லேடி","ஜென்ட்ஸ்","மேட்டர்"]

def is_loan(w): return any(w.startswith(l) for l in LOAN) or (any(c in GRANTHA for c in w) and not any(w.startswith(t) for t in ASTRO))
def is_astro(w): return any(w.startswith(t) for t in ASTRO)

rows = []
for m in man:
    a, b = normalize(ha[m["id"]]), normalize(hb[m["id"]])
    y, _ = sf.read(m["audio"], dtype="float32")
    words = a.split()
    dis = jiwer.cer(a.replace(" ", "") or "-", b.replace(" ", "") or "-")
    rows.append(dict(id=m["id"], start=m["start"], dur=m["duration"], dis=dis, n=len(words),
                     loan=sum(map(is_loan, words)), astro=sum(map(is_astro, words)),
                     db=float(20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-9))))
json.dump(rows, open(f"results/{name}/profile.json", "w"), ensure_ascii=False)
print("min | disagree% | words/s | english% | astro% | loudness dB")
for mn in range(int(max(r["start"] for r in rows) // 60) + 1):
    g = [r for r in rows if mn * 60 <= r["start"] < (mn + 1) * 60]
    if not g: continue
    n = sum(r["n"] for r in g) or 1
    print(f"{mn:3d} | {100*np.mean([r['dis'] for r in g]):5.1f} | {n/sum(r['dur'] for r in g):4.2f} | "
          f"{100*sum(r['loan'] for r in g)/n:5.1f} | {100*sum(r['astro'] for r in g)/n:5.1f} | {np.mean([r['db'] for r in g]):6.1f}")
d = [r["dis"] for r in rows]
print(f"\noverall disagreement CER {100*np.mean(d):.1f}%, median piece {100*np.median(d):.1f}%, "
      f"pieces >30%: {sum(x>0.3 for x in d)}")
