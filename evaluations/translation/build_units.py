"""Test units for translation: whole teaching points (>= 45 s), built by joining the
pipeline's paragraphs. Each unit has two Tamil versions of the same stretch:
  ours   = our pipeline's transcript (SraVaani + punctuation)
  gemini = Gemini 3.1 Pro's transcript of the same stretch (cleaner; found by word alignment)
Writes units.json."""
import json, re, sys
from pathlib import Path
import jiwer, soundfile as sf
sys.path.insert(0, "..")
from normalize import normalize
from pipeline.punctuate import pauses, punctuate, paragraphs
SECS = {"A": "section_A_38m46s", "B": "section_B_59m34s", "C": "section_C_29m11s"}
AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../../tamil_asr_eval/for_customer/sections")
MIN_S = 45.0
units = []
for tag, f in SECS.items():
    words = json.load(open(f"../../sravaani_eval/class_transcripts/{f}/words.json"))
    y, _ = sf.read(AUDIO / f"{f}.wav", dtype="float32")
    paras = paragraphs(punctuate(words, pauses(words, y)))
    groups, cur = [], []
    for p in paras:
        cur += p
        if cur[-1]["end"] - cur[0]["start"] >= MIN_S:
            groups.append(cur); cur = []
    if cur:
        if groups and cur[-1]["end"] - cur[0]["start"] < MIN_S / 2: groups[-1] += cur
        else: groups.append(cur)
    # map our word positions -> Gemini word positions
    graw = Path(f"../../sravaani_eval/references/gemini_section_{tag}_tamil.txt").read_text().split()
    gtok = [(i, normalize(t)) for i, t in enumerate(graw) if normalize(t)]
    g_norm = " ".join(t for _, t in gtok); flat = [w for s in sum(groups, []) for w in s["words"]]
    h_norm = " ".join(normalize(w["w"]) or "-" for w in flat)
    amap = {}
    for a in jiwer.process_words(g_norm, h_norm).alignments[0]:
        if a.type in ("equal", "substitute"):
            for k in range(a.ref_end_idx - a.ref_start_idx): amap[a.hyp_start_idx + k] = a.ref_start_idx + k
    pos = 0; prev_g = -1
    for k, g in enumerate(groups):
        n = sum(len(s["words"]) for s in g)
        idx = [amap[i] for i in range(pos, pos + n) if i in amap]
        g_end = max(idx) if idx else prev_g
        if k == len(groups) - 1: g_end = len(gtok) - 1
        gem = " ".join(graw[gtok[i][0]] for i in range(prev_g + 1, g_end + 1)) if g_end > prev_g else ""
        units.append({"id": f"{tag}{k+1}", "section": tag, "start": round(g[0]["start"], 1), "end": round(g[-1]["end"], 1),
                      "ours": " ".join(s["text"] for s in g), "gemini": gem})
        prev_g = g_end; pos += n
json.dump(units, open("units.json", "w"), ensure_ascii=False, indent=1)
for u in units: print(f"{u['id']:3s} {u['start']:6.1f}-{u['end']:6.1f}s  ours {len(u['ours']):4d} chars | gemini {len(u['gemini']):4d} chars")
print(len(units), "units;", sum(len(u['ours']) for u in units), "chars ours,", sum(len(u['gemini']) for u in units), "chars gemini")
