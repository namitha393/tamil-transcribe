# SraVaani — deep dive on the chosen model

Everything about how the pipeline performs on the customer's astrology class, how each
pipeline step was chosen, and where (and why) it still fails.

## What's here

| Folder | Contents |
|---|---|
| [`class_transcripts/full_class/`](class_transcripts/full_class/) | Pipeline output for the whole 78.6-min class: `tamil.txt`, `tanglish.txt`, `subtitles.srt`, `words.json` |
| [`class_transcripts/section_*`](class_transcripts/) | Same for the 3 test sections (A typical, B astrology-heavy, C discussion) |
| [`references/`](references/) | Gemini 3.1 Pro (Tamil + Tanglish) and Sarvam v3/v4 (Tamil, English-in-Latin, Tanglish) transcripts of the same 3 sections |
| [`customer_pack/`](customer_pack/) | What goes to the customer: output-style options, the 3 draft transcripts to correct, correction instructions |
| [`experiments/`](experiments/) | Raw outputs of the root-cause experiments |

No audio is stored in this repo (customer recording).

## Results on the class (CER % vs Gemini; lower is better)

| Section (length) | SraVaani pipeline | Indic-Transcribe Core | Sarvam v3 |
|---|--:|--:|--:|
| A typical (4.8 min) | 23.1 | 18.9 | 15.9 |
| B astrology-heavy (5.2 min) | 20.4 | 16.5 | 12.1 |
| C discussion (4.8 min) | 29.5 | 23.5 | 21.4 |

Full class: 78.6 min → 6,035 words in 9.5 min on a Mac CPU.

## How each pipeline step was chosen (comparisons)

| Step | Options compared | Result | Chosen |
|---|---|---|---|
| Model | 10 models (see [evaluations](../evaluations/README.md)) | within ~1.6 WER of the best on natural speech, ~20× faster on CPU | SraVaani |
| Cutting | blind 25 s, Silero ≤15/≤25 s, our pause cutter 8–25 / ≤15 / 15–28 s, every speech turn | longer pieces win; 15–28 s: WER 27.4 vs 28.3 (old) | pause cutter 15–28 s |
| Wrong-language output | none vs block non-Tamil letters | 7/9 wrong-script pieces fixed; public sets slightly better | Tamil letters only |
| Empty pieces | none vs split-and-retry | section A CER 24.5 → 23.1 (with new cutting) | split-and-retry |
| Quiet voices | as-is vs boosting quiet parts 4×/8× | worse (CER 23.1 → 24.5/25.0) | not used |
| Feature normalisation | whole piece vs sliding 3 s / 6 s | no gain on dropped speech; worse elsewhere | unchanged |
| Tanglish | letter-by-letter (Aksharamukha), our sound rules v1/v2, + word lists, + English detector, Sarvam translit, Indic-Transcribe-Flex romanised | ours: 64.4% match with human spellings (Aksharamukha 45.5%); 78–83% with Gemini's Tanglish | our rules + word lists |
| English words | word list only vs automatic detector | detector recovers 30/38 vs 23/38, but ~30% false alarms | list (detector optional) |

End to end (audio → Tanglish, CER vs Gemini's Tanglish, A/B/C): SraVaani + our converter
23.8 / 23.5 / 30.6; Sarvam translit 23.3 / 18.8 / 23.3; Indic-Transcribe-Flex romanised
28.0 / 28.0 / 33.3. The converter is no longer the bottleneck — recognition is.

## Weaknesses and root causes

On the 1,000 words where Gemini **and** Sarvam agree:

| | wrong | missing |
|---|--:|--:|
| SraVaani | 20.7% | **7.9%** |
| Indic-Transcribe Core | 21.3% | 2.4% |
| hima-asr | 14.1% | 8.3% |

SraVaani's wrong-word rate equals Core's; **its extra gap is missing words**.

### 1. Misses attendees' replies (biggest)
Lines like "வாய்ஸ் ரொம்ப பிரேக் ஆகுது சார்" from attendees on the call go missing.

**Root cause — learned suppression of a secondary voice.** Controlled tests with public clips
(% of the target clip's words found):

| Test | SraVaani | hima-asr |
|---|--:|--:|
| Conversational reply **alone** | 71.5 | 65.6 |
| Same reply **after a clean read clip** (equal loudness) | **15.4** | 66.8 |
| …reply as phone-quality audio | 14.2 | |
| …reply first, then the read clip | 31.3 | |
| …with a 2 s gap | 17.0 | |
| Read clip after a *different* read speaker | 73.8 | 82.8 |
| Read clip after a conversational clip | 83.7 | |
| Conversational reply after another conversational speaker | 57.5 | |

SraVaani ignores a short, noisier conversational voice that follows a clean dominant one —
exactly teacher-on-mic + attendee-on-call. Other models don't. Most likely learned from its
training data (single-speaker recordings where background speech is not transcribed —
not confirmed).

**Ruled out:** loudness (dropped stretches are as loud as the teacher; boosting doesn't help),
phone bandwidth (8 kHz copies: no change), the decoder skipping frames (0/7 recovered),
whole-piece normalisation (sliding-window version: no recovery), cutting at every speech
turn (works on synthetic mixes, 63%, but not on the real class — attendees reply without
clean pauses — and costs accuracy).

**Fix:** fine-tune on mixed teacher + attendee audio (both voices transcribed), or send the
affected stretches to another model.

### 2. Misspells rare words (astrology terms, English words)
ஜோசியம் → ஜோஷியம், ஜாதகம் → ஜாதவம், நீச்ச வீடு → நீச்சல் வீடு, மியூட் → மீட்.

**Root cause:** (a) its tokenizer has only **282 Tamil pieces** (of 5,000 shared by 65
languages) — 3.5 pieces per word, so it spells unfamiliar words by sound; (b) these words
hardly occur in public Tamil speech data (in 74k words: சந்திரன் 0, ஜாதகம் 0, லக்னம் 0,
ஜோசியம் 0). Astrology terms: SraVaani 28% wrong vs Sarvam 9%.

**Fix:** fine-tune on this teacher's audio; quick win: correct near-misses against an
astrology glossary.

### 3. Small differences in spelling and word endings
43% of its "wrong" words differ by ≤⅓ of letters; 16 of 207 turn spoken forms into written
ones (சொல்றேன் → சொல்கிறேன்) — a habit from its training transcripts. Minor.

### 4. Not done yet
No punctuation/paragraphs; no speaker labels (teacher vs attendee); no human-corrected
reference yet (all class numbers are vs Gemini).

## Next steps
1. Customer corrects the 3 sections in [`customer_pack/`](customer_pack/) → true accuracy numbers.
2. Customer picks output styles ([`customer_pack/output_style_options.md`](customer_pack/output_style_options.md)).
3. Fine-tune SraVaani on corrected class audio (+ mixed teacher/attendee audio) for weaknesses 1–2.
4. Punctuation and paragraphs from the saved word timings.
