"""Where does each system go wrong on words that Gemini AND Sarvam v3 agree on?
'Confirmed' words = aligned and identical in both cloud transcripts. For each
system we classify what happens to every confirmed word."""
import json, re, collections, jiwer
from normalize import normalize
from pipeline.tanglish import LOANS, ASTRO
exec(open("compare_sections.py").read().split("cer = lambda")[0])
for t in "BC": out[t]["Gemini 3.1 Pro"] = normalize(open(f"../sravaani_eval/references/gemini_section_{t}_tamil.txt").read())
EN = [k for k in LOANS if k not in ASTRO]
ASTRO_EXTRA = ["நட்சத்திர", "லக்ன", "திசை", "தசை", "உச்ச", "நீச்ச", "ஆட்சி", "ஜாதக", "ஜோசிய", "கிரக", "ராசி", "பாவ",
               "சனி", "குரு", "ராகு", "கேது", "செவ்வா", "புதன", "சுக்கிர", "சூரிய", "சந்திர", "பாத", "ரஜ்ஜு", "சந்திராஷ்டம", "கற்ப"]
NUM = ["ஒன்", "ரெண்டு", "இரண்டு", "மூணு", "மூன்று", "நாலு", "நான்கு", "அஞ்சு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது",
       "பத்து", "பதினொ", "பன்னிர", "பதின", "இருபத", "முப்பத", "நாற்பத"]
def cat(w):
    if any(w.startswith(a) for a in ASTRO_EXTRA) or any(w.startswith(a) for a in ASTRO): return "astrology term"
    if any(w.startswith(k) for k in EN) or re.search("[ஜஷஸஹ]", w): return "English word"
    if any(w.startswith(n) for n in NUM): return "number"
    if len(w) <= 4: return "short word (≤4 letters)"
    return "other Tamil"
def pairs(ref, hyp):
    """map ref word index -> ('equal'|'sub'|'del', hyp word)"""
    r, h = ref.split(), hyp.split(); m = {}
    for a in jiwer.process_words(ref, hyp).alignments[0]:
        for k in range(a.ref_end_idx - a.ref_start_idx):
            i = a.ref_start_idx + k
            if a.type == "equal": m[i] = ("ok", h[a.hyp_start_idx + k])
            elif a.type == "substitute": m[i] = ("wrong", h[a.hyp_start_idx + k])
            elif a.type == "delete": m[i] = ("missing", "")
    return r, m
SYS = ["SraVaani pipeline", "Indic-Transcribe Core", "hima-asr", "Sarvam v4"]
stats = {s: collections.Counter() for s in SYS}; tot = collections.Counter(); examples = collections.defaultdict(list)
missing_runs = collections.Counter()
for t in "ABC":
    g = out[t]["Gemini 3.1 Pro"]; r, ms = pairs(g, out[t]["Sarvam v3"])
    confirmed = [i for i in range(len(r)) if ms.get(i, ("",))[0] == "ok"]
    for i in confirmed: tot[cat(r[i])] += 1
    for s in SYS:
        _, m = pairs(g, out[t][s]); run = 0
        for i in range(len(r)):
            st = m.get(i, ("missing", ""))[0]
            if i in set(confirmed):
                stats[s][(cat(r[i]), st)] += 1
                if s == SYS[0] and st == "wrong" and len(examples[cat(r[i])]) < 8: examples[cat(r[i])].append(f"{r[i]}→{m[i][1]}")
            if s == SYS[0]:
                if st == "missing": run += 1
                else:
                    if run: missing_runs["1-2 words" if run <= 2 else "3-5 words" if run <= 5 else "6+ words"] += run
                    run = 0
print(f"confirmed words: {sum(tot.values())} of {sum(len(out[t]['Gemini 3.1 Pro'].split()) for t in 'ABC')}\n")
print(f"{'category':26s} {'n':>4s} | " + " | ".join(f"{s[:18]:>18s}" for s in SYS))
print(f"{'':26s} {'':>4s} | " + " | ".join(f"{'wrong%  missing%':>18s}" for s in SYS))
for c in ["astrology term", "English word", "number", "short word (≤4 letters)", "other Tamil"]:
    n = tot[c]; cells = [f"{100*stats[s][(c,'wrong')]/n:6.1f}  {100*stats[s][(c,'missing')]/n:7.1f}" for s in SYS]
    print(f"{c:26s} {n:4d} | " + " | ".join(f"{x:>18s}" for x in cells))
n = sum(tot.values())
print(f"{'ALL':26s} {n:4d} | " + " | ".join(f"{100*sum(v for (c,st),v in stats[s].items() if st=='wrong')/n:6.1f}  {100*sum(v for (c,st),v in stats[s].items() if st=='missing')/n:7.1f}".rjust(18) for s in SYS))
print("\nSraVaani missing words (all words, not only confirmed), by length of the missing run:", dict(missing_runs))
print("\nSraVaani wrong-word examples (confirmed word → SraVaani):")
for c, ex in examples.items(): print(f"  {c}: " + ", ".join(ex))
