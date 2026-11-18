# Evaluations — every model we compared

Goal: transcribe Tamil astrology-class recordings (one teacher + attendees on a call, quiet
room, English mixed in) as well as possible, ideally free and on a Mac.

**How to read the scores.** WER = % of words wrong; CER = % of characters wrong. Lower is better.
Tamil words are long and spellings vary, so CER is usually the fairer number.
All text is normalised the same way for every system (Unicode, punctuation, Tamil digits).

## The models

| Model | Size | Made by / trained on | Licence & cost | Speed (1 h of audio) |
|---|---|---|---|---|
| **SraVaani-1.0** *(chosen)* | 430M | ARTPARK + IISc; trained from scratch on Vaani (65 Indian languages) | MIT, free | **~3 min** on an Apple M5 CPU (whole pipeline; 78.6-min class in 2.8 min) |
| Indic-Transcribe Core | 1.2B | AI4Bharat + Bodhan AI; fine-tune of NVIDIA Canary-1b-v2, 27 languages | Indic Open Model Licence; free, hosting for others needs permission | 80–110 min CPU; 6–9 min Mac GPU |
| Indic-Transcribe Flex | 1.2B | Same; adds English-in-Latin and Tanglish output modes | same | same |
| IndicConformer-600M | 600M | AI4Bharat, 22 languages | MIT | ~4 min CPU — **ruled out** (goes blank, see below) |
| hima-asr | 0.46B | Independent developer; Qwen3-ASR encoder + Gemma-3 decoder; training data not detailed | Apache 2.0 + Gemma terms | ~30 min CPU |
| Whisper-Tamil small / medium | 244M / 769M | IIT Madras SPRING lab fine-tunes of Whisper | Apache 2.0 | 1–1.5 h / ~3 h CPU |
| Whisper large-v3-turbo | 809M | OpenAI | MIT | 30–45 min CPU |
| Sarvam Saaras v3 / v4 | not published | Sarvam AI; training data not published | Paid API, ₹0.49/min (₹29/h) | ~1 s per 21 s piece (cloud) |
| Gemini 3.1 Pro | not published | Google | Paid; used as a stand-in reference | — |

## 1. Public test sets (human transcripts) — WER % (CER %)

200 random clips each. **IndicVoices** = natural conversational speech (closest to the class);
**FLEURS** and **Kathbath** = people reading sentences aloud.

| Model | FLEURS (0.69 h) | **IndicVoices (0.38 h)** | Kathbath (0.37 h) |
|---|--:|--:|--:|
| Indic-Transcribe Core | 23.2 (9.6) | **28.3 (8.5)** | 18.0 (2.7) |
| Indic-Transcribe Flex | 24.5 (9.9) | 29.0 (8.5) | 19.9 (3.1) |
| **SraVaani, Tamil-only (pipeline)** | 25.6 (10.3) | **29.9 (9.7)** | 18.4 (2.8) |
| SraVaani, original | 26.0 (10.9) | 30.1 (9.9) | 18.4 (2.8) |
| Sarvam Saaras v3 | — | 31.5 (10.0) | — |
| IndicConformer RNNT | 25.2 (9.5) | 31.8 (9.9) | 19.5 (3.1) |
| hima-asr | 21.6 (6.0) | 32.9 (10.2) | 18.3 (3.0) |
| IndicConformer CTC | 26.1 (9.6) | 36.2 (11.0) | 20.3 (3.4) |
| Whisper-Tamil-medium | 20.8 (5.2) | 59.6 (26.9) | 13.0 (1.9) |
| Whisper-Tamil-small | 23.6 (6.3) | 63.8 (30.4) | 13.6 (2.0) |
| Whisper large-v3-turbo | 71.5 (21.4) | 123.1 (87.7) | 64.3 (16.7) |

Whisper-Tamil and hima-asr look strong on *read* speech but fall back on natural speech —
likely "home advantage" (trained on similar read data). Sarvam writes numbers as digits while
the references use words, which costs it a little here.

## 2. The customer's class (3 test sections) — CER % vs Gemini 3.1 Pro

No human-corrected transcript exists yet, so Gemini 3.1 Pro is the stand-in reference
(not human-checked). Sections were picked by profiling the whole 78.6-min class.

| Section | Class time | Length | Why chosen |
|---|---|--:|---|
| A — typical teaching | 38:46–43:33 | 4.8 min | average on every measure |
| B — astrology-heavy | 59:34–64:44 | 5.2 min | most astrology terms |
| C — discussion | 29:11–34:01 | 4.8 min | most English, attendees talking |

| System | A | B | C | **Average** |
|---|--:|--:|--:|--:|
| Sarvam v3 | 15.9 | 12.1 | 21.4 | **16.5** |
| Sarvam v4 | 16.7 | 12.5 | 22.9 | 17.4 |
| Indic-Transcribe Core | 18.9 | 16.5 | 23.5 | 19.6 |
| hima-asr | 25.8 | 17.6 | 25.9 | 23.1 |
| **SraVaani pipeline** | 23.1 | 20.4 | 29.5 | **24.3** |
| Indic-Transcribe Flex | 19.0 | 17.4 | 36.6 | 24.3 |
| IndicConformer RNNT | 61.1 | — | — | — |

Sarvam and Gemini agree with each other most (both cloud models; they may share habits).
On *public* natural speech Sarvam was behind SraVaani, so only a human-corrected class
transcript can settle the real gap. Why SraVaani loses ~5 points to Core on the class:
[../sravaani_eval/README.md](../sravaani_eval/README.md).

## 3. Other findings

- **IndicConformer goes blank** on ~20% of the class pieces (56/287), unpredictably. Not
  length (fine up to 60 s), not silence/volume, not invalid numbers, not ONNX settings; adding
  0.5 s of audio can flip a piece from correct to empty. Draft bug report:
  [issue_indicconformer.md](issue_indicconformer.md).
- **Input length** (joined IndicVoices clips, 5–60 s): neither SraVaani nor IndicConformer
  degrades with length (WER 27–32% throughout, no empty outputs).
- **Phone-quality audio** (8 kHz band): no effect on SraVaani or hima-asr (CER ±0.2).
- **Speed on CPU varies ~50×** between models (table above); this decided SraVaani over Core.

The full chronological write-up with every experiment: [FULL_REPORT.md](FULL_REPORT.md).

## Files

| Path | What |
|---|---|
| `results/<test set>/<model>.jsonl` | every clip: reference, model output, duration, seconds taken |
| `manifests/` | the clips used in each test set (audio not included — rebuild with `data.py`) |
| `data.py` | builds the test sets (FLEURS, IndicVoices, Kathbath, long recordings, your own folder) |
| `models.py` | one wrapper per model (incl. Sarvam API) |
| `run.py` / `score.py` | run models over test sets / print the score tables |
| `compare_sections.py`, `weakspots.py` | class-section comparisons and error breakdown |
| `lengthtest.py`, `longform_eval.py`, `eval_pipeline_longform.py` | input-length and cutting tests |
| `dropped_speech.py`, `mixscore.py` | root-cause experiments for missed attendee speech |
| `eval_tanglish_dakshina.py` | Tanglish converter vs human romanisation |
| `eval_punctuation.py` | punctuation vs Gemini's and Sarvam's (class sections) |
| `choose_sections.py`, `profile_audio.py`, `transcript.py` | picking the test sections, profiling the class |

**Re-running** (from this folder):
```bash
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
PYTHONPATH=.. .venv/bin/python data.py indicvoices --n 200      # gated: accept terms on Hugging Face first
PYTHONPATH=.. .venv/bin/python run.py --models sravaani-tamil indic-transcribe-core --sets indicvoices_ta
PYTHONPATH=.. .venv/bin/python score.py indicvoices_ta
```
Sarvam needs `SARVAM_API_KEY` in the environment (never commit it).
