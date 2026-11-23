"""Fix test: limit each frame's self-attention to a local window (run-time mask, no
weight change), so a reply can't attend to the dominant voice seconds earlier.
Measures (1) reply words found in the 60 teacher+reply mixes and (2) accuracy on
class section A (pipeline pieces) vs Gemini, for several window sizes."""
import json, random, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch, jiwer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from normalize import normalize
from pipeline.audio import cut
exec(open(Path(__file__).with_name("suppression_probe.py")).read().split("ctc = getattr")[0].replace('if __name__','#'))

@torch.no_grad()
def transcribe(y, win):
    wav = torch.from_numpy(y).unsqueeze(0)
    feats, flen = m.extract_features(wav, torch.tensor([wav.shape[1]]))
    _, x = enc.pre_encode.forward(flen.to(torch.int64), feats.transpose(1, 2).to(m._io_dtype))
    x, pos = enc.pos_enc.forward(x); T = x.shape[1]
    idx = torch.arange(T)
    mask = torch.zeros(1, T, T, dtype=torch.bool) if win is None else ((idx[None, :] - idx[:, None]).abs() > win)[None]
    pad = torch.zeros(1, T, dtype=torch.bool)
    return decode(run_from(x, 0, pos, mask, pad))

secA = sf.read(Path(__file__).resolve().parents[3] / "tamil_asr_eval/for_customer/sections/section_A_38m46s.wav", dtype="float32")[0]
gem = normalize(open(Path(__file__).resolve().parents[2] / "sravaani_eval/references/gemini_section_A_tamil.txt").read())
gap = np.zeros(8000, np.float32)
print("window | reply found in mix | reply alone | section A CER vs Gemini | words")
for win in [int(w) for w in (sys.argv[1:] or ["0","125","62","31"])]:
    win = None if win == 0 else win
    hit = tot = ha = ta = 0
    for t, a in pairs:
        yt, ya = load(t), load(a)
        ya = ya / (np.sqrt((ya ** 2).mean()) + 1e-9) * np.sqrt((yt ** 2).mean())
        h, n = found(transcribe(np.concatenate([yt, gap, ya]), win), a["text"]); hit += h; tot += n
        h, n = found(transcribe(ya, win), a["text"]); ha += h; ta += n
    hyp = normalize(" ".join(transcribe(secA[s:e], win) for s, e in cut(secA)))
    c = jiwer.cer(gem.replace(" ", ""), hyp.replace(" ", ""))
    name = "full (original)" if win is None else f"±{win*0.08:.1f} s"
    print(f"{name:16s} {100*hit/tot:6.1f}%   {100*ha/ta:6.1f}%   {100*c:6.1f}%   {len(hyp.split())}", flush=True)
