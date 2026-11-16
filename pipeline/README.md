# The pipeline, step by step

```mermaid
flowchart TD
    A[Audio file] --> B["<b>1. Read audio</b> — audio.py<br/>any format → 16 kHz mono"]
    B --> C["<b>2. Cut at pauses</b> — audio.py<br/>15–28 s pieces, cut at the quietest point"]
    C --> D["<b>3. Transcribe</b> — sravaani.py<br/>SraVaani-1.0, Tamil letters only,<br/>words + timings"]
    D -->|piece came back empty| R[split in two, retry]
    R --> D
    D --> E["<b>4. Lines</b> — transcribe.py<br/>new line at a 0.6 s pause or after 12 s"]
    E --> F["<b>5. Tanglish</b> — tanglish.py<br/>sound rules + word lists"]
    E --> O1[tamil.txt · subtitles.srt · words.json]
    F --> O2[tanglish.txt]
```

| File | Step |
|---|---|
| `audio.py` | 1 Read audio, 2 Cut |
| `sravaani.py` | 3 Transcribe (model + fixes) |
| `transcribe.py` | 4 Lines, saving files, command-line entry point |
| `tanglish.py` | 5 Tanglish converter and word lists |
| `english_detect.py` | Optional automatic English-word detector (`--english-detector`) |

---

## 1. Read audio
Any audio/video file → 16 kHz mono (what the model expects). Uses PyAV, which bundles ffmpeg,
so nothing else needs installing.

## 2. Cut at pauses
The model works on pieces, not hours. We cut into **15–28 second** pieces, each cut placed at
the quietest 0.3 s inside that window, and **keep all audio**.

**Why these numbers** (tested on 30 min of long audio with known text — word error %):

| Cutting method | Pieces | WER | CER |
|---|--:|--:|--:|
| Blind 25 s cuts | 78 | 26.5 | 9.4 |
| Silero speech detector, ≤25 s | 83 | 26.9 | 9.5 |
| **Ours, 15–28 s (used)** | 98 | **27.4** | **9.7** |
| Ours, 8–25 s (first version) | 156 | 28.3 | 10.0 |
| Ours, ≤15 s | 226 | 29.1 | 10.9 |
| Silero, ≤15 s | 134 | 30.4 | 13.5 |
| Every speech turn its own piece | — | 28.9–29.8 | 10.9–11.7 |

Fewer, longer pieces win (more context, fewer edges). SraVaani itself stays accurate up to
60 s per piece; 28 s keeps pieces usable by other models (30 s limit) for comparison.
Blind cuts score well only because the test pauses were perfectly silent; on real audio they
cut words.

## 3. Transcribe — SraVaani-1.0
430M-parameter FastConformer by ARTPARK & IISc (Vaani project), MIT licence. Chosen because
it is nearly as accurate as the best open model on natural Tamil speech but **~20× faster on
CPU** (see [../evaluations/README.md](../evaluations/README.md)).

Two fixes on top of the plain model:

| Fix | Why | Effect |
|---|---|---|
| **Tamil letters only** — block non-Tamil tokens while decoding | The model has no language setting; on 9 of 287 class pieces it wrote Malayalam/Hindi/Bengali script (often the right Tamil word in the wrong alphabet) | 7 of 9 fixed (2 became empty); public sets slightly better (FLEURS 26.0→25.6, IndicVoices 30.1→29.9 WER) |
| **Retry empty pieces** — split at the quietest point and retry | A long piece occasionally comes back empty | Section A: CER 24.5→23.1 together with the new cutting |

We checked whether the model would ever write English words in English letters: its
probability for English-letter tokens at English words was ≈0, so the Tamil-only rule hides
nothing.

## 4. Lines
Words are grouped into lines, starting a new line at a 0.6 s pause or after 12 s. Each line
gets its start time. (Punctuation and paragraphs: planned — the word timings needed for them
are saved in `words.json`.)

## 5. Tanglish
Tanglish = the same Tamil words written in English letters, by sound (not translated):
வணக்கம் → *vanakkam*.

Our converter, for each word:
1. **English word written in Tamil letters?** (சார், கரெக்டா) → English spelling + Tamil ending: *sir*, *correct-a*
2. **Astrology term or place name?** (குரு, விருச்சிகம், சென்னை) → standard spelling: *guru*, *viruchiga*, *Chennai*
3. **Otherwise, Tamil sound rules** — a Tamil letter sounds different by position
   (ட between vowels → *d*, க after ங → *g*, த → *th*, ற்ற → *ttr*): பாருங்க → *paarunga*

**Tested** (% of words spelled the same way):

| Converter | 10,000 sentences romanised by people (Dakshina) | Gemini's Tanglish of the class (A / B / C) |
|---|--:|--:|
| Letter-by-letter (Aksharamukha library) | 45.5% | 45 / 39 / 45% |
| **Ours** | **64.4%** | **83 / 81 / 78%** |

The rest is mostly genuine variety (*k/g*, *zh/l*, *chandran/chandhiran*).
Your own spellings: add them to `LOANS` or `ASTRO` at the top of [`tanglish.py`](tanglish.py).

**Optional `--english-detector`** finds English words not in the list by comparing how the
word sounds with ~25,000 English pronunciations (CMU dictionary). It recovers 30/38 English
words in a test section (list alone: 23/38), but ~30% of its guesses on general Tamil text are
wrong (ஸ்டாலின் → "stealing"), so it is off by default.
