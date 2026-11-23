# Translation scores (blind LLM judges, 1–5, higher is better)

Candidates scored: 138; missing: 0

| System | Input | Meaning | Fluency | Terms | Invented | n |
|---|---|--:|--:|--:|--:|--:|
| LLM, whole paragraph (Claude) | our transcript | **4.62** | 4.81 | 4.80 | 0/16 | 16 |
| Sarvam-Translate text (cloud) | Gemini's clean transcript | **3.62** | 3.62 | 2.87 | 1/16 | 16 |
| Sarvam speech → English (cloud) | audio | **3.50** | 3.31 | 3.13 | 3/16 | 16 |
| Sarvam-Translate text (cloud) | our transcript | **2.56** | 2.50 | 2.07 | 5/16 | 16 |
| IndicTrans2-200M, per sentence (offline) | Gemini's clean transcript | **2.38** | 2.25 | 2.20 | 4/16 | 16 |
| IndicTrans2-200M, per sentence (offline) | our transcript | **1.94** | 1.94 | 1.53 | 5/16 | 16 |
| IndicTrans2-200M, whole paragraph (offline) | Gemini's clean transcript | **1.88** | 2.31 | 1.73 | 1/16 | 16 |
| IndicTrans2-1B half-precision, per sentence (offline) | our transcript | **1.60** | 1.80 | 1.40 | 0/5 | 5 |
| IndicTrans2-200M, whole paragraph (offline) | our transcript | **1.50** | 1.75 | 1.40 | 4/16 | 16 |
| IndicTrans2-1B half-precision, whole paragraph (offline) | our transcript | **1.00** | 1.40 | 1.00 | 0/5 | 5 |

## Same 5 units only (A1, A2, A3, A4, A5) — includes IndicTrans2-1B

| System | Input | Meaning | Fluency | Terms |
|---|---|--:|--:|--:|
| LLM, whole paragraph (Claude) | our transcript | 4.80 | 5.00 | 4.80 |
| Sarvam-Translate text (cloud) | Gemini's clean transcript | 3.40 | 3.80 | 2.20 |
| Sarvam speech → English (cloud) | audio | 3.40 | 3.20 | 2.60 |
| Sarvam-Translate text (cloud) | our transcript | 2.60 | 2.80 | 1.80 |
| IndicTrans2-200M, per sentence (offline) | Gemini's clean transcript | 2.40 | 2.20 | 2.20 |
| IndicTrans2-200M, per sentence (offline) | our transcript | 2.00 | 2.00 | 1.40 |
| IndicTrans2-200M, whole paragraph (offline) | Gemini's clean transcript | 1.80 | 2.00 | 1.20 |
| IndicTrans2-1B half-precision, per sentence (offline) | our transcript | 1.60 | 1.80 | 1.40 |
| IndicTrans2-200M, whole paragraph (offline) | our transcript | 1.40 | 1.80 | 1.40 |
| IndicTrans2-1B half-precision, whole paragraph (offline) | our transcript | 1.00 | 1.40 | 1.00 |

## Judges' notes (lowest meaning first, 2 per system)

- **claude-opus-5.5__ours**: A4: Clear and correct; omits 'only three years' exchange; B3: Clear and natural; 'cinema' instead of engagement
- **sarvam-translate__gemini**: A1: Fluent but drops planet and Mars; 'background' substitute; A2: Womb/mind right; garbha sel as 'lose your virtue'
- **sarvam-speech-translate__audio**: A1: Mostly right but 'paragraph' for planet, 'Tuesday' for Mars; A2: Womb/mind right; garbha sel as 'removes malefic effects'
- **sarvam-translate__ours**: C1: Mostly incoherent; invented medicine, bowl, Sukham; A1: 'Para Khan'; reverses 'our words should be good' advice
- **indictrans2-200M-sentences__gemini**: A1: 'machete', 'Kharekata', 'I won't go to jail' errors; A2: Womb point and garbha sel neekki garbled
- **indictrans2-200M-sentences__ours**: B2: Incoherent; 'moon muscle', Hindi, Kilpauk lost; C1: Votes, calorie peak, Lucknow; meaning lost
- **indictrans2-200M-paragraph__gemini**: A4: 'No, no' repetition loop; B3: 'yes yes yes' repetition loop only
- **indictrans2-1B-sentences__ours**: A2: 'Screen fear', Thailand, bottle gourd nonsense; A3: Reverses sentence point, then 'yes' loop
- **indictrans2-200M-paragraph__ours**: A2: 'Please' repetition loop; A4: Truncated, 'moon is half of the prison'
- **indictrans2-1B-paragraph__ours**: A1: Degenerates into 'yes, yes' repetition loop; A2: 'Yes' repetition loop
