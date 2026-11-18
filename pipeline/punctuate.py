"""Component 5: punctuation and paragraphs, from pauses in the audio + Tamil question forms.

SraVaani gives word start times but no real end times (each word "ends" where the
next starts), so pauses are measured from the audio: the longest quiet stretch
between one word's start and the next word's start.

  full stop  after a pause >= STOP_S
  ?          instead of a full stop when the sentence looks like a question
             (ends in a question form like ...ா / ...ங்களா, or a tag like
             இல்லையா / கரெக்டா / சரியா, or contains என்ன / எப்படி / ஏன் ...)
  comma      after a shorter pause >= COMMA_S
  paragraph  after a pause >= PARA_S, or once a paragraph passes PARA_MAX_S
             (then at the longest pause in its last part)
Thresholds were tuned against Gemini's and Sarvam's punctuation of the class (see
sravaani_eval/README.md).
"""
from __future__ import annotations

import re

import numpy as np

from .audio import SR

STOP_S = 0.20       # tuned: F1 64 vs Gemini (two cloud systems agree at 74)
TAG_STOP_S = 0.10   # a strong question tag ends a sentence after an even shorter pause
COMMA_S = 0.25      # commas are off by default: even Sarvam and Gemini agree on only 10%
PARA_S = 1.0       # tuned: ~20-30 s paragraphs, closest to Gemini's breaks
PARA_MAX_S = 60.0

QUESTION_WORDS = ("என்ன", "எப்படி", "ஏன்", "எங்க", "எங்கே", "எத்தனை", "எந்த", "யாரு", "யார்", "எது", "எவ்வளவு", "எப்போ", "எப்ப")
QUESTION_TAGS = ("இல்லையா", "இல்லியா", "கரெக்டா", "சரியா", "சரியாங்களா", "புரியுதா", "புரிஞ்சுதா", "புரிஞ்சதா", "தெரியுமா", "ரைட்டா")
NOT_QUESTION_ENDINGS = ("நல்லா", "ஆமா", "அப்பா", "ஐயா", "அய்யா", "சரியா", "மாதிரியா", "தானா", "அம்மா", "ஒழுங்கா", "வேகமா", "மெதுவா")


def pauses(words: list[dict], y: np.ndarray, hop_s: float = 0.02) -> list[float]:
    """Seconds of silence after each word (the last word gets a large value)."""
    hop = int(hop_s * SR)
    n = len(y) // hop
    if n == 0 or not words:
        return [9.9] * len(words)
    rms = np.sqrt((y[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    quiet = db < max(np.percentile(db, 90) - 30, np.percentile(db, 10) + 6)
    out = []
    for i, w in enumerate(words):
        if i + 1 == len(words):
            out.append(9.9); continue
        a = int((w["start"] + 0.12) / hop_s)          # skip the start of the word itself
        b = int(words[i + 1]["start"] / hop_s)
        best = run = 0
        for q in quiet[a:b]:
            run = run + 1 if q else 0
            best = max(best, run)
        out.append(best * hop_s)
    return out


STRONG_TAG = re.compile(r"(ாங்களா|ீங்களா|ங்களா|லையா|லாமா|ுதா|ிதா|ுமா|ானா|ியா)$")


def strong_tag(word: str) -> bool:
    if word.endswith("ய்யா") and not word.startswith(QUESTION_TAGS):
        return False
    return word.startswith(QUESTION_TAGS) or (bool(STRONG_TAG.search(word)) and not word.startswith(NOT_QUESTION_ENDINGS))


ADDRESS = ("ஐயா", "அய்யா", "சார்", "மேடம்", "டீச்சர்")


def is_question(sentence_words: list[str]) -> bool:
    words = list(sentence_words)
    while len(words) > 1 and words[-1].startswith(ADDRESS):  # "கரெக்டா ஐயா" -> judge "கரெக்டா"
        words.pop()
    if not words:
        return False
    sentence_words = words
    last = words[-1]
    if last.endswith("ய்யா") and not last.startswith(QUESTION_TAGS):  # சரிங்கய்யா = "okay sir"
        return any(w.startswith(QUESTION_WORDS) for w in words[-3:])
    if last.startswith(QUESTION_TAGS):
        return True
    if strong_tag(last) or any(w.startswith(QUESTION_WORDS) for w in sentence_words[-3:]):
        return True
    if re.search(r"(ா|ாங்களா|ீங்களா|ங்களா|ாங்கய்யா\?)$", last) and not last.startswith(NOT_QUESTION_ENDINGS):
        return True
    return False


def punctuate(words: list[dict], gaps: list[float], stop_s: float = STOP_S, comma_s: float = COMMA_S,
              questions: bool = True, commas: bool = False, tag_stop_s: float = TAG_STOP_S) -> list[dict]:
    """-> sentences: {start, end, text, mark, pause_after}"""
    sents, cur = [], []
    for w, g in zip(words, gaps):
        cur.append(w)
        if g >= stop_s or (questions and g >= tag_stop_s and strong_tag(w["w"])):
            toks = [x["w"] for x in cur]
            mark = "?" if questions and is_question(toks) else "."
            sents.append({"start": cur[0]["start"], "end": w["end"], "words": cur, "mark": mark, "pause_after": g})
            cur = []
    if cur:
        toks = [x["w"] for x in cur]
        sents.append({"start": cur[0]["start"], "end": cur[-1]["end"], "words": cur,
                      "mark": "?" if questions and is_question(toks) else ".", "pause_after": 9.9})
    gi = 0
    for s in sents:  # commas inside sentences
        parts = []
        for k, w in enumerate(s["words"]):
            g = gaps[gi]; gi += 1
            last = k == len(s["words"]) - 1
            parts.append(w["w"] + ("," if commas and not last and g >= comma_s else ""))
        s["text"] = " ".join(parts) + s["mark"]
    return sents


def paragraphs(sents: list[dict], para_s: float = PARA_S, max_s: float = PARA_MAX_S) -> list[list[dict]]:
    paras, cur = [], []
    for s in sents:
        cur.append(s)
        if s["pause_after"] >= para_s:
            paras.append(cur); cur = []
        elif cur[-1]["end"] - cur[0]["start"] >= max_s and len(cur) > 1:
            # too long: break at the longest pause in the second half
            half = cur[len(cur) // 2:]
            k = len(cur) // 2 + max(range(len(half)), key=lambda j: half[j]["pause_after"])
            paras.append(cur[: k + 1]); cur = cur[k + 1:]
    if cur:
        paras.append(cur)
    return paras
