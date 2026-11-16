"""Pipeline (cutter + SraVaani Tamil-only + empty-retry) on the 6 long lectures with known text."""
import json, sys
import soundfile as sf
from pipeline.audio import cut
from pipeline.sravaani import SraVaaniTamil as Transcriber
from normalize import score
from data import DATA, SR
lects = json.loads((DATA / "lf_lectures" / "lectures.json").read_text())
tr = Transcriber(); hyps = []
for L in lects:
    y, _ = sf.read(L["audio"], dtype="float32")
    hyps.append(" ".join(w["w"] for a, b in cut(y) for w in tr.words(y[a:b], a / SR)))
s = score([L["text"] for L in lects], hyps)
print(f"pipeline cutter 15-28s: WER {100*s['wer']:.1f}  CER {100*s['cer']:.1f}  pieces {sum(len(cut(sf.read(L['audio'],dtype='float32')[0])) for L in lects)}")
