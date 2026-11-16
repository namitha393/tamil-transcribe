"""Root-cause SraVaani's dropped speech.
1. Find speech (Silero VAD) where SraVaani wrote no words for >= 1.2 s.
2. Re-transcribe each such region in different ways:
   E1 alone (only that region, so the model's normalisation and attention see only it)
   E2 inside its original piece, but decoding every frame (no TDT frame skipping)
3. Compare the region's sound to the teacher's: level and high-frequency content
   (call audio is band-limited)."""
import json, types, numpy as np, soundfile as sf, torch
from silero_vad import load_silero_vad, get_speech_timestamps
from pipeline.audio import cut
from pipeline.sravaani import SraVaaniTamil as Transcriber
from data import SR
torch.set_num_threads(8)
vad = load_silero_vad(); tr = Transcriber(); m = tr.modelodel; cfg = m.config
def words_of(y, off=0.0):
    h = m.transcribe([y], return_hypotheses=True, timestamps=True)[0]
    return [w for w in h.timestamp["word"] if w["word"].strip()]
def hf_ratio(y):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))**2; f = np.fft.rfftfreq(len(y), 1/SR)
    return float(S[f > 3800].sum() / (S[f > 300].sum() + 1e-12))
orig_greedy = m._greedy_one
def no_skip(self_, enc, T):  # same Tamil-only decoder but never jump more than 1 frame
    durs = cfg.durations; cfg.durations = [min(d, 1) if d > 0 else 0 for d in durs]
    try: return orig_greedy(enc, T)
    finally: cfg.durations = durs
res = []
for f in ["section_A_38m46s", "section_B_59m34s", "section_C_29m11s"]:
    y, _ = sf.read(f"for_customer/sections/{f}.wav", dtype="float32")
    words = json.load(open(f"../sravaani_eval/class_transcripts/{f}/words.json"))
    speech = get_speech_timestamps(torch.from_numpy(y), vad, sampling_rate=SR, min_silence_duration_ms=300)
    covered = np.zeros(len(y), bool)
    for w in words: covered[int(w["start"]*SR):int(w["end"]*SR)+int(0.3*SR)] = True
    teacher = np.concatenate([y[int(w["start"]*SR):int(w["end"]*SR)] for w in words])
    t_db, t_hf = 20*np.log10(np.sqrt((teacher**2).mean())+1e-9), hf_ratio(teacher[:SR*60])
    for s in speech:
        a, b = s["start"], s["end"]
        if b - a < 0.5*SR or covered[a:b].mean() > 0.2: continue
        seg = y[max(0, a-int(0.2*SR)):b+int(0.2*SR)]
        e1 = " ".join(w["word"] for w in words_of(seg))
        pa, pb = next((pa, pb) for pa, pb in cut(y) if pa <= a < pb)
        m._greedy_one = types.MethodType(no_skip, m)
        e2w = [w for w in words_of(y[pa:pb]) if a/SR - 0.2 <= pa/SR + w["start"] <= b/SR + 0.2]
        m._greedy_one = orig_greedy
        res.append(dict(sec=f[8], t=round(a/SR, 1), dur=round((b-a)/SR, 1),
                        db=round(float(20*np.log10(np.sqrt((seg**2).mean())+1e-9) - t_db), 1),
                        hf=round(hf_ratio(seg) / t_hf, 2), alone=e1, noskip=" ".join(w["word"] for w in e2w)))
json.dump(res, open("results/dropped_regions.json", "w"), ensure_ascii=False, indent=1)
n = len(res); tot = sum(r["dur"] for r in res)
print(f"dropped speech regions: {n}, total {tot:.0f} s of the 14.8 min")
print(f"recovered when transcribed ALONE: {sum(bool(r['alone']) for r in res)}/{n}")
print(f"recovered with NO FRAME SKIPPING in the original piece: {sum(bool(r['noskip']) for r in res)}/{n}")
print(f"level vs teacher (dB): median {np.median([r['db'] for r in res]):+.1f}; high-freq content vs teacher: median {np.median([r['hf'] for r in res]):.2f}x")
for r in res[:12]: print(f"  {r['sec']} @{r['t']:6.1f}s {r['dur']:4.1f}s  {r['db']:+5.1f}dB hf{r['hf']:.2f}x | alone: {r['alone'][:60]!r} | noskip: {r['noskip'][:40]!r}")
