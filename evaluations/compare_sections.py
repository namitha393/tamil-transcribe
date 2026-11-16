"""Compare all systems on the 3 class sections. No human reference yet, so each
system is scored against Gemini (section A) and against Sarvam v3 (all sections)."""
import json, re, jiwer
from normalize import normalize
secs = {s["label"][0]: s for s in json.load(open("results/astro_class/sections.json"))}
man = [json.loads(l) for l in open("manifests/astro_class.jsonl")]
files = {"A": "section_A_38m46s", "B": "section_B_59m34s", "C": "section_C_29m11s"}
def stitched(model, S):
    h = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in open(f"results/astro_class/{model}.jsonl")}
    return " ".join(h[p["id"]] for p in man if S["start"] <= p["start"] < S["end"])
def piece_text(model, tag):
    r = sorted((json.loads(l) for l in open(f"results/sec_{tag}/{model}.jsonl")), key=lambda x: x["id"])
    return " ".join(x["hyp"] for x in r)
def pipe(tag):
    return " ".join(re.sub(r"^\[[^\]]*\]\s*", "", l) for l in open(f"../sravaani_eval/class_transcripts/{files[tag]}/tamil.txt"))
out = {}
for tag in "ABC":
    sys_ = {"SraVaani pipeline": pipe(tag), "Indic-Transcribe Core": stitched("indic-transcribe-core", secs[tag]),
            "Indic-Transcribe Flex": stitched("indic-transcribe-flex", secs[tag]), "hima-asr": stitched("hima-asr", secs[tag]),
            "Sarvam v3": piece_text("sarvam-saaras-v3", tag), "Sarvam v4": piece_text("sarvam-saaras-v4", tag)}
    if tag == "A":
        sys_["Gemini 3.1 Pro"] = open("../sravaani_eval/references/gemini_section_A_tamil.txt").read()
    out[tag] = {k: normalize(v) for k, v in sys_.items()}
cer = lambda r, h: 100 * jiwer.cer(r.replace(" ", ""), h.replace(" ", ""))
print("Character disagreement % with each reference (lower = closer). Words in brackets.\n")
for ref_name in ["Gemini 3.1 Pro", "Sarvam v3"]:
    tags = "A" if ref_name.startswith("Gemini") else "ABC"
    print(f"vs {ref_name}:  " + "  |  ".join(f"section {t}" for t in tags))
    for name in out["A"]:
        if name == ref_name: continue
        cells = [f"{cer(out[t][ref_name], out[t][name]):5.1f} ({len(out[t][name].split())}w)" for t in tags if name in out[t]]
        print(f"  {name:22s} " + "  |  ".join(cells))
    print()
