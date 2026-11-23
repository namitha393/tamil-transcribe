"""Sarvam: text translation (sarvam-translate:v1) of each unit, and direct
speech->English (Saaras v3, mode=translate) on the unit's audio in <=28 s pieces.
Needs SARVAM_API_KEY. Writes out/<system>__<input>.json"""
import json, os, sys, time
from pathlib import Path
import requests, soundfile as sf, io
sys.path.insert(0, "../.."); from pipeline.audio import cut, SR
KEY = os.environ["SARVAM_API_KEY"]; H = {"api-subscription-key": KEY}
U = json.load(open("units.json"))
AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../../tamil_asr_eval/for_customer/sections")
SECS = {"A": "section_A_38m46s", "B": "section_B_59m34s", "C": "section_C_29m11s"}
def post(url, **kw):
    for a in range(7):
        r = requests.post(url, headers=H, timeout=120, **kw)
        if r.status_code == 429 or r.status_code >= 500: time.sleep(3 * 2 ** a); continue
        r.raise_for_status(); time.sleep(0.5); return r.json()
    r.raise_for_status()
def text(inp):
    out = Path(f"out/sarvam-translate__{inp}.json"); res = json.loads(out.read_text()) if out.exists() else {}
    for u in U:
        if u["id"] in res: continue
        j = post("https://api.sarvam.ai/translate", json={"input": u[inp], "source_language_code": "ta-IN",
                 "target_language_code": "en-IN", "model": "sarvam-translate:v1"})
        res[u["id"]] = j["translated_text"]; out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(out, len(res))
def speech():
    out = Path("out/sarvam-speech-translate__audio.json"); res = json.loads(out.read_text()) if out.exists() else {}
    cache = {}
    for u in U:
        if u["id"] in res: continue
        if u["section"] not in cache: cache[u["section"]] = sf.read(AUDIO / f"{SECS[u['section']]}.wav", dtype="float32")[0]
        y = cache[u["section"]][int(u["start"] * SR): int(u["end"] * SR) + SR // 4]
        parts = []
        for a, b in cut(y):
            buf = io.BytesIO(); sf.write(buf, y[a:b], SR, format="WAV"); buf.seek(0)
            j = post("https://api.sarvam.ai/speech-to-text", data={"model": "saaras:v3", "mode": "translate", "language_code": "ta-IN"},
                     files={"file": ("p.wav", buf, "audio/wav")})
            parts.append(j.get("transcript", ""))
        res[u["id"]] = " ".join(parts); out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(out, len(res))
text("ours"); text("gemini"); speech()
