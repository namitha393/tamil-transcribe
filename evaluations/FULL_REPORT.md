# Tamil speech-to-text: evaluation so far

Goal: transcribe Tamil astrology-class recordings (one teacher + attendees, quiet room,
English mixed in) into Tamil script and Tanglish, ideally free and on the customer's Mac.
Sarvam (paid API, ~₹35/hour) is the reference to beat — not yet run (needs API key).

## 1. Public test sets — word error rate % (character error rate %), lower is better

200 random clips each. IndicVoices = natural/conversational speech (closest to the class);
FLEURS and Kathbath = people reading sentences aloud.

| Model (size) | FLEURS | **IndicVoices** | Kathbath | CPU time per 1 h audio* |
|---|--:|--:|--:|--:|
| **Indic-Transcribe Core** (1.2B, AI4Bharat+Bodhan) | 23.2 (9.6) | **28.3 (8.5)** | 18.0 (2.7) | ~80–110 min |
| Indic-Transcribe Flex (1.2B) | 24.5 (9.9) | 29.0 (8.5) | 19.9 (3.1) | ~80–110 min |
| **SraVaani-1.0, Tamil-only fix** (430M, ARTPARK/IISc) | 25.6 (10.3) | 29.9 (9.7) | 18.4 (2.8) | **~3–5 min** |
| SraVaani-1.0 original | 26.0 (10.9) | 30.1 (9.9) | 18.4 (2.8) | ~3–5 min |
| IndicConformer-600M RNNT (AI4Bharat) | 25.2 (9.5) | 31.8 (9.9) | 19.5 (3.1) | ~4 min |
| hima-asr (0.46B, individual dev) | 21.6 (6.0) | 32.9 (10.2) | 18.3 (3.0) | ~30 min (PyTorch; ONNX may be ~2x faster) |
| IndicConformer-600M CTC | 26.1 (9.6) | 36.2 (11.0) | 20.3 (3.4) | ~4 min |
| Whisper-Tamil-medium (769M) | 20.8 (5.2) | 59.6 (26.9) | 13.0 (1.9) | ~3 h |
| Whisper-Tamil-small (244M) | 23.6 (6.3) | 63.8 (30.4) | 13.6 (2.0) | ~1–1.5 h |
| Whisper large-v3-turbo | 71.5 (21.4) | 123 (87.7) | 64.3 (16.7) | ~30–45 min |

\*Apple M5, 8 CPU threads. Indic-Transcribe was scored on the Mac GPU for speed; its CPU
figure comes from a separate CPU test.

Reading it: Whisper-Tamil and hima-asr look great on *read* speech but fall back on natural
speech — likely "home advantage" (trained on similar read data). On natural speech the
top 3 are within ~1.6 points: Indic-Transcribe Core > Flex > SraVaani.

## 2. Customer class (78.6 min) — no correct transcript yet

- **IndicConformer** returned *nothing* for 56/287 pieces (20%). Not length (works up to
  60 s on public audio), not silence/volume, not NaN, not ONNX settings; tiny audio changes
  flip it. Model instability → ruled out for this customer. Draft bug report:
  `issue_indicconformer.md`.
- **SraVaani** wrote 9/287 pieces in the wrong alphabet (Malayalam, Hindi, ...). It has no
  language setting, so we block non-Tamil letters while it decodes: 7/9 fixed, 2 now empty.
- With the bugs out, the 4 good models disagree with each other on ~23% of characters
  (Core vs Flex closest, 16%).
- Content profile: ~9% of words English, ~5% astrology terms; astrology terms are a clear
  weak spot (e.g. நீச்ச வீடு heard as நீச்சல் வீடு "swimming house").

## 3. Cutting long audio into pieces (30 min of long audio with known text, SraVaani)

| Method | Pieces | WER % | CER % |
|---|--:|--:|--:|
| Blind 25 s cuts | 78 | **26.5** | **9.4** |
| Silero speech detector, ≤25 s | 83 | 26.9 | 9.5 |
| Our pause cutter, ≤25 s | 156 | 28.3 | 10.0 |
| Our pause cutter, ≤15 s | 226 | 29.1 | 10.9 |
| Silero, ≤15 s | 134 | 30.4 | 13.5 |

Fewer, longer pieces win (more context, fewer edges). Caveat: test pauses are perfect
silence. Plan: cut at pauses but keep pieces near 25–30 s and never drop audio.

## 4. Tanglish

Options compared on class lines (see `for_customer/output_style_options.md`):
letter-by-letter tool (E1), our sound-rule converter + English/astrology word lists (E2),
Indic-Transcribe-Flex built-in Tanglish (E3), LLM polish (E4). Indic-Transcribe-Flex
"mixed" and hima-asr "mixed" give Tamil script with English words in English letters (T2);
Flex also adds punctuation and digits, but its number conversion can be wrong
("பன்னெண்டு" → "02:30").

## 5. Round 3 (SraVaani pipeline)

**Gemini 3.1 Pro transcript of section A as a stand-in reference** (not human-checked):

| Model | WER % | CER % | words dropped |
|---|--:|--:|--:|
| Indic-Transcribe Core | 49.7 | **18.9** | 23 |
| Indic-Transcribe Flex | 50.1 | 19.0 | 23 |
| **SraVaani pipeline (new)** | **45.7** | 23.1 | 69 |
| SraVaani Tamil-only (old cutting) | 47.6 | 24.5 | 72 |
| hima-asr | 47.2 | 25.8 | 78 |
| IndicConformer RNNT | 70.5 | 61.1 | 281 |

SraVaani's main weak spot: it **drops attendees' replies / remote call voices**
("வாய்ஸ் ரொம்ப பிரேக் ஆகுது சார்", "குறைச்சு வந்துருவாங்கய்யா"). Boosting quiet
parts does not help (deleted 69 → 65–68, accuracy slightly worse) → model limitation;
candidates: fine-tuning on this class audio, or sending those stretches to Core.

**Pipeline cutter** (15–28 s at pauses, keeps all audio) on the long-form test:
WER 27.4 / CER 9.7 (old 8–25 s cutter 28.3 / 10.0; blind 25 s 26.5 / 9.4).

**Speed:** full 78.6-min class → Tamil + Tanglish + subtitles + word timings in
**9.5 min** on the Mac CPU.

**Tanglish converter** (`tanglish.py`), word match with human / Gemini spellings:

| Method | Dakshina (10k human-romanised sentences) | Gemini section A |
|---|--:|--:|
| Letter-by-letter (Aksharamukha) | 45.5% | 44.9% |
| Sound rules v1 | 53.5% | 59.0% |
| **Sound rules v2** (data-driven: த→th, ற்ற→ttr, ன்ற→ndr) | **64.5%** | 73.4% |
| **v2 + English & astrology lists** | — | **78.0%** |
| v2 + automatic English detector | — | 75.9% |

Automatic English detector (`english_detect.py`, CMU pronunciations): recovers 30/38
English words in section A (list alone: 23/38), but ~30% of its firings on general
Tamil text are wrong (e.g. ஸ்டாலின் → "stealing") → kept optional, off by default.

## 6. Sarvam (Saaras v3 / v4, free credits)

IndicVoices (human reference, 200 clips): Sarvam v3 **31.5 WER / 10.0 CER** — behind
Core 28.3/8.5 and SraVaani 29.9/9.7. Part of the gap is style: Sarvam writes numbers as
digits and the references write them as words.

Class sections, character disagreement % (no human reference yet):

| System | vs Gemini, A | vs Sarvam v3: A | B | C |
|---|--:|--:|--:|--:|
| Sarvam v3 | **15.9** | — | — | — |
| Sarvam v4 | 16.7 | 10.8 | 8.8 | 13.0 |
| Indic-Transcribe Core | 18.9 | 19.4 | 13.9 | 23.3 |
| Indic-Transcribe Flex | 19.0 | 19.5 | 14.9 | 36.9 |
| SraVaani pipeline | 23.1 | 24.5 | 18.6 | 28.4 |
| hima-asr | 25.8 | 24.1 | 15.3 | 25.7 |

On the class, the two cloud systems (Sarvam, Gemini) agree with each other most; SraVaani
is ~5 points behind Core on every section and writes ~15% fewer words. Section C
(discussion) is hardest for everyone.

SraVaani weak spots, class sections:
1. Drops attendee/remote-call speech (biggest).
2. Mis-hears astrology terms (நீச்ச→நீச்சல், கற்ப செல் நீக்கி→"கற்ப செல்லி கீம்பாருங்க", தசை→திசை).
3. English words misheard (மியூட்→மீட், கான்ஸ்டபிள்→கான்செல், ஜெயிலுக்கு→செயலுக்கு).
4. Turns some spoken forms into written ones (சொல்றேன்→சொல்கிறேன், இருந்தா→இருந்தால்):
   2.5 per 100 words vs ~1 for Sarvam/Gemini.
5. Splits/merges words (சிறைதண்டனைக்கு→சிறைதன் அன்னைக்கு); occasional odd guesses
   (தாய் தானங்க→தாய்லாந்து).

## 7. Gemini 3.1 Pro on all three sections (stand-in reference, not human-checked)

Character error % vs Gemini (word error % in brackets):

| System | A typical | B astrology | C discussion | avg CER |
|---|--:|--:|--:|--:|
| **Sarvam v3** | **15.9** (38.9) | **12.1** (30.6) | **21.4** (41.2) | **16.5** |
| Sarvam v4 | 16.7 (43.5) | 12.5 (32.4) | 22.9 (45.6) | 17.4 |
| Indic-Transcribe Core | 18.9 (49.7) | 16.5 (41.7) | 23.5 (50.8) | 19.6 |
| hima-asr | 25.8 (47.2) | 17.6 (38.6) | 25.9 (46.0) | 23.1 |
| SraVaani pipeline | 23.1 (45.7) | 20.4 (47.7) | 29.5 (53.1) | 24.3 |
| Indic-Transcribe Flex | 19.0 (50.1) | 17.4 (43.5) | 36.6 (59.2) | 24.3 |

(Sarvam writes numbers as digits; Gemini as words — this slightly *penalises* Sarvam.)

**Tanglish** vs Gemini's Tanglish, char error % (word match %):

| | A | B | C |
|---|--:|--:|--:|
| Converter only, same Tamil input: letter-by-letter | 14.7 (44.9) | 17.9 (38.7) | 16.1 (45.2) |
| Converter only: **our rules + lists (v3)** | **2.8 (82.7)** | **3.8 (80.5)** | **4.4 (78.4)** |
| End to end: SraVaani + our converter (v2) | 23.8 (44.5) | 23.5 (39.6) | 30.6 (38.5) |
| End to end: Sarvam translit mode | 23.3 (43.9) | 18.8 (45.1) | 23.3 (44.4) |
| End to end: Indic-Transcribe Flex romanized | 28.0 (20.0) | 28.0 (25.9) | 33.3 (22.9) |

→ The converter is no longer the bottleneck; end-to-end Tanglish quality is set by recognition.
v3 fixes: list matches only accept real Tamil endings (சார்ந்த was becoming "sir-ntha"),
more English words and place names, ஞா→gnya, ன்ப→nb. Dakshina unchanged (64.4%).

## 8. SraVaani weak spots — root causes (experiments)

Errors on the 1,000 words where Gemini and Sarvam v3 agree ("confirmed"):
SraVaani 20.7% wrong + 7.9% missing; Core 21.3% + 2.4%; hima-asr 14.1% + 8.3%.
→ SraVaani's extra gap vs Core is mostly **missing words**.

**1. Dropped attendee speech — learned suppression of a secondary voice.**
| Test (SraVaani, % of target words found) | |
|---|--:|
| Conversational reply alone | 71.5 |
| Same reply after a clean read clip (equal loudness) | **15.4** |
| …phone-band reply | 14.2 |
| …reply placed first | 31.3 |
| …2 s gap | 17.0 |
| Read clip after a different read speaker | 73.8 |
| Read clip after a conversational clip | 83.7 |
| Conversational reply after another conversational speaker | 57.5 |
| hima-asr: reply after read clip / alone | 66.8 / 65.6 |
Ruled out: loudness (dropped regions are at teacher level; boosting doesn't help),
phone bandwidth (8 kHz copies: CER unchanged), frame skipping in the decoder (0/7 recovered),
whole-piece feature normalisation (sliding-window normalisation: no recovery, worse elsewhere).
Cutting at every speech turn recovers synthetic replies (63%) but not real ones, because
attendees reply without clean pauses; it also costs accuracy (section A CER 23.1→24.6–28.1).
→ Fix: fine-tune on mixed teacher+attendee audio, or fall back to another model.

**2. Domain vocabulary (astrology, English loans).** Tokenizer has only 282 Tamil pieces
(5,000 shared by 65 languages; 3.5 pieces per word) → spells unfamiliar words by sound
(ஜோசியம்→ஜோஷியம், ஜாதகம்→ஜாதவம்). Astrology words are ~absent from public Tamil speech
transcripts (74k words: சந்திரன் 0, ஜாதகம் 0, லக்னம் 0, ஜோசியம் 0).
→ Fix: fine-tune on this teacher's audio; cheap now: glossary-based post-correction.

**3. Near-miss spellings / written forms.** 43% of SraVaani's wrong words differ by ≤⅓ of
letters; 16/207 are spoken→written (சொல்றேன்→சொல்கிறேன்) — from its training transcripts.

## 9. Next

1. Customer: corrects the 3 sections in `for_customer/sections/` (A typical 38:46,
   B astrology-heavy 59:34, C discussion 29:11; ~15 min total) and picks output styles.
   Note: drafts come from Indic-Transcribe Core, which may slightly favour Core in scoring
   if corrections are incomplete.
2. Then: score every model + Sarvam on those 15 minutes, overall and on astrology terms.
3. Then decide: Core (accurate, slow) vs SraVaani (fast) vs Sarvam; and whether
   fine-tuning on this teacher's audio is worth it.
