"""Lay out every system's translations as results/<system>/:
     <input>.json        {unit id: English}
     translations.md     readable: Tamil source and English, unit by unit
Inputs: 'ours' = our pipeline's Tamil transcript, 'gemini' = Gemini's cleaner Tamil
transcript of the same stretch, 'audio' = translated straight from the audio."""
import json
from pathlib import Path

HERE = Path(__file__).parent
U = {u["id"]: u for u in json.load(open(HERE / "units.json"))}
ABOUT = {
    "sarvam-translate": "Sarvam-Translate text API (`sarvam-translate:v1`), cloud, paid (₹2 per 1,000 characters).",
    "sarvam-speech-translate": "Sarvam Saaras v3 speech → English (`mode=translate`), cloud, paid; works on ≤28 s pieces.",
    "claude-opus-5.5": "Claude Opus 5.5 translating whole paragraphs (stand-in for a cloud LLM API). Written in this "
                       "session from our transcript only; garbled words marked [unclear]. NOTE: the same model also "
                       "judges, and had seen Gemini's transcript earlier — treat as possibly optimistic.",
    "indictrans2-200M-sentences": "IndicTrans2 distilled 200M (AI4Bharat, MIT), offline; one sentence at a time.",
    "indictrans2-200M-paragraph": "IndicTrans2 distilled 200M, offline; whole paragraph in one go (max 256 tokens).",
    "indictrans2-1B-sentences": "IndicTrans2 1B (AI4Bharat, MIT), offline; one sentence at a time.",
    "indictrans2-1B-paragraph": "IndicTrans2 1B, offline; whole paragraph in one go (max 256 tokens).",
    "vaani-t2t-sentences": "gemma4-e2b-vaani-t2t (ARTPARK/IISc, Gemma licence), offline; one sentence at a time.",
    "vaani-t2t-paragraph": "gemma4-e2b-vaani-t2t, offline; whole paragraph in one go.",
}
by_sys = {}
for f in sorted((HERE / "out").glob("*__*.json")):
    sysname, inp = f.stem.split("__")
    by_sys.setdefault(sysname, {})[inp] = json.loads(f.read_text())
for sysname, inputs in by_sys.items():
    d = HERE / "results" / sysname
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"# {sysname}", "", ABOUT.get(sysname, ""), ""]
    for inp, tr in inputs.items():
        (d / f"{inp}.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1))
    for uid, u in U.items():
        lines += [f"## {uid} — section {u['section']}, {u['start']:.0f}–{u['end']:.0f} s", ""]
        for inp, tr in inputs.items():
            src = {"ours": u["ours"], "gemini": u["gemini"], "audio": "(audio)"}[inp]
            lines += [f"**Tamil ({inp}):** {src}", "", f"**English:** {tr.get(uid, '—')}", ""]
    (d / "translations.md").write_text("\n".join(lines))
    print(d, list(inputs))
