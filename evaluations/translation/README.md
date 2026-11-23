# Translation: Tamil class → English, paragraph by paragraph

**Customer ask:** translate Tamil *paragraphs* into English — sentence-by-sentence translation
loses the meaning and intent of a teaching point. **Preference: offline**, cloud only if offline
is not good enough.

## Result so far

Blind scoring by LLM judges (1–5, higher is better) against what was actually said, on 16
one-minute teaching points from the class:

| System | Runs | Input | **Meaning** | Fluency | Terms | Invented |
|---|---|---|--:|--:|--:|--:|
| **LLM, whole paragraph (clean run)** | cloud LLM | **our** transcript | **4.00** | 4.00 | 4.53 | 0/16 |
| Sarvam-Translate | cloud | Gemini's clean transcript | 3.44–3.62 | 3.1–3.6 | 2.6–2.9 | 0–1/16 |
| Sarvam speech → English | cloud | audio | 3.06–3.50 | 2.9–3.3 | 2.8–3.1 | 1–3/16 |
| Sarvam-Translate | cloud | our transcript | 2.56 | 2.50 | 2.07 | 5/16 |
| IndicTrans2-200M, per sentence | **offline** | Gemini's clean transcript | 2.06–2.38 | 1.9–2.3 | 1.5–2.2 | 1–4/16 |
| IndicTrans2-200M, per sentence | offline | our transcript | 1.94 | 1.94 | 1.53 | 5/16 |
| IndicTrans2-200M, whole paragraph | offline | Gemini's / ours | 1.88 / 1.50 | 2.31 / 1.75 | 1.73 / 1.40 | |
| IndicTrans2-1B (half precision), per sentence *(5 units)* | offline | our transcript | 1.60 | 1.80 | 1.40 | 0/5 |
| IndicTrans2-1B (half precision), whole paragraph *(5 units)* | offline | our transcript | 1.00 | 1.40 | 1.00 | 0/5 |

Ranges = two independent judging rounds (gives a feel for judge variability, ~±0.3).

**Takeaways**
1. **Paragraph-level LLM translation wins clearly** — even from our *noisy* transcript (4.0) it beats
   Sarvam's translator given the *clean* transcript (3.4–3.6). It uses context to recover the
   teacher's point and marks what it can't recover as `[unclear]` instead of inventing.
2. **The offline translators tested are not usable here** (1.0–2.4): they carry recognition errors
   straight through ("Thailand", "fear of the screen"), mistranslate astrology terms, and on whole
   paragraphs they fall into repetition loops ("please, please, please…") — they were trained on
   single sentences. IndicTrans2-1B in half precision was no better than the 200M model.
3. **Transcription quality matters for every non-LLM system:** Sarvam-Translate drops from 3.4–3.6
   (clean transcript) to 2.6 (ours).

**Caveats**
- Judges are LLMs (same model family as the LLM translator) — a customer spot-check of 3–4
  paragraphs is the fair final check.
- The first LLM run (`claude-opus-5.5`, 4.62) is **excluded from conclusions**: it was written by a
  model that had seen Gemini's clean transcript earlier, and the blind judge flagged it as
  "near-reference". The clean re-run (`llm-fresh-paragraph`) is the honest number.
- Not yet tested offline (this Mac's 17 GB memory can't safely run them): Vaani's translator
  (5B), NLLB-200, SeamlessM4T, a local general LLM. A **4-bit quantised local LLM** (~2–3 GB) is the
  realistic offline candidate to test next.

## Test set — `units.json`
16 whole teaching points (~1 min each) from class sections A/B/C (`build_units.py`). Each has two
Tamil versions of the same stretch: **ours** (our pipeline's transcript) and **gemini** (Gemini 3.1
Pro's cleaner transcript, found by word alignment).

## Files
| Path | What |
|---|---|
| [`results/<system>/`](results/) | each system's translations: `<input>.json` + side-by-side `translations.md` |
| [`judging/`](judging/) | blind packets, judges' scores, un-blinding key, [`results.md`](judging/results.md) (round 1), [`round2_results.md`](judging/round2_results.md) |
| `translate_sarvam.py`, `translate_indictrans2.py`, `translate_vaani.py`, `translate_meta.py` | runners (Vaani/Meta not completed — memory) |
| `fresh_llm/` | the our-transcript-only input given to the clean LLM run |
