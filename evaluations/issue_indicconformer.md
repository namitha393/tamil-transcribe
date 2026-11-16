# Draft bug report — ai4bharat/indic-conformer-600m-multilingual

Where to post: the model's Hugging Face "Community" tab
(https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual/discussions).
Do NOT attach the customer audio — offer to share privately or reproduce with public audio.

---

**Title:** RNNT returns empty string (and CTC returns garbage) on ~20% of 10–25 s clips of clean lecture audio

**Summary**
On a clean, single-speaker Tamil lecture recording (WhatsApp AAC, 48 kHz mono, resampled to
16 kHz float32), `model(wav, "ta", "rnnt")` returns `''` for 56 of 287 clips (10–25 s each).
CTC on the same clips returns a few garbled characters. The same clips cut in half usually
transcribe correctly. Public test sets (IndicVoices/Kathbath/FLEURS Tamil, 600 clips, plus
joined IndicVoices inputs up to 60 s) show no empty outputs.

**What we checked**
- Not length: joined IndicVoices clips of 5–60 s give 0 empty outputs and stable WER (~28–33%).
- Not silence/clipping/level: failing clips match working ones on RMS, peak, silence share, and
  zero-sample share.
- Not NaN/Inf: encoder outputs are finite; preprocessor is deterministic and finite.
- Not ONNX Runtime graph optimisation: same result with `ORT_DISABLE_ALL`.
- Tipping point: one clip transcribes correctly up to 15.0 s and returns `''` at 15.5 s.
  Appending that 0.5 s changes the encoder output for the *first 15 s* by ~68% of its typical
  magnitude. Muting the appended region does not fix it; adding Gaussian noise (σ=1e-3) or
  scaling by 0.25 fixes it for that clip but only for 5–7 of the 56 failing clips.
- Splitting failing clips into 10 s windows still leaves ≥1 blank window in 21 of 56 clips.

**Environment**
transformers 4.57.6, onnxruntime 1.20.1 (CPU), torch 2.x, macOS arm64 (Apple M5), Python 3.12.
Repo revision 2a77d305038dfc502fd9464e7d028b90381976cb.

**Ask**
Is this a known instability of the exported encoder (e.g. attention / normalisation in the ONNX
export vs. the original NeMo checkpoint)? Is there a NeMo checkpoint we can compare against?
We can share failing clips privately.
