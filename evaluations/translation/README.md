# Translation: Tamil class → English, paragraph by paragraph

**Customer ask:** translate Tamil *paragraphs* into English — sentence-by-sentence translation
loses the meaning and intent of a teaching point. **Preference: offline**, cloud only if offline
is not good enough.

**Status: in progress — not scored yet.** Systems run so far are below; Vaani's own translator
and IndicTrans2-1B are next (need ~15 GB free disk for Vaani's 10 GB model).

## Test set — `units.json`
16 whole teaching points (~1 min each, ≥45 s) from class sections A/B/C, built by joining the
pipeline's paragraphs (`build_units.py`). Each unit has two Tamil versions of the same stretch:
- **ours** — our pipeline's transcript (SraVaani; has recognition errors)
- **gemini** — Gemini 3.1 Pro's cleaner transcript of the same stretch (found by word alignment)

Comparing a system on both shows how much transcription errors hurt its translation.

## Systems — one folder each in [`results/`](results/)
Each folder: `<input>.json` (unit → English) and `translations.md` (Tamil and English side by side).

| System | Runs | Licence / cost | Inputs | First impression (not scored) |
|---|---|---|---|---|
| [IndicTrans2-200M, sentence by sentence](results/indictrans2-200M-sentences/) | **offline** | MIT, free | ours, gemini | From ours: carries mishearings ("Thailand", "fear of the screen"). From gemini: much better ("fear of jail", "mother") but loses the thread; astrology terms wrong |
| [IndicTrans2-200M, whole paragraph](results/indictrans2-200M-paragraph/) | offline | MIT, free | ours, gemini | **Breaks down** — loops ("please, please, please…"); trained on single sentences |
| [Sarvam-Translate](results/sarvam-translate/) | cloud | ₹2 / 1,000 chars | ours, gemini | From ours: nonsense ("Thailand", "sperm cells"). From gemini: readable but literal ("lose your virtue" for the womb-period deduction) |
| [Sarvam speech → English](results/sarvam-speech-translate/) | cloud | ₹0.49 / min | audio | Surprisingly decent; avoids our recognition errors; ≤28 s chunks; still mangles terms |
| [Claude Opus 5.5, whole paragraph](results/claude-opus-5.5/) | cloud | per-token | ours | Coherent, keeps the teacher's point, marks garbled words `[unclear]`. **Caveat:** same model will judge, and had seen Gemini's transcript earlier — possibly optimistic |
| Vaani translator (`gemma4-e2b-vaani-t2t`) | offline | Gemma licence, free | — | **next** (10 GB) |
| IndicTrans2-1B | offline | MIT, free | — | next (~4 GB) |

## Scoring plan
For each unit and system: **meaning** (does it convey the teacher's point? 1–5), **fluency** (1–5),
**terms** (astrology terms right?), judged blind against Gemini's Tamil; plus a customer
spot-check of 3–4 paragraphs as the fair final check.

## Scripts
`build_units.py` (test set) · `translate_sarvam.py` (needs `SARVAM_API_KEY`) ·
`translate_indictrans2.py [model]` · `organize.py` (builds `results/` from `out/`)
