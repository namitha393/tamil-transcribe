"""Find English words that a Tamil transcript wrote in Tamil letters, and
recover their English spelling - without a hand-made list.

  லாஜிக் -> logic     ரைட் -> right     பிரேக் -> break     கான்ஸ்டபிள் -> constable

How:
 1. Gate: only words that look borrowed. Native Tamil words almost never
    contain ஜ ஷ ஸ ஹ ஃ, end in a hard consonant (ட் க் ப் ...), or start with ட ற ண ள ழ.
 2. Strip a Tamil ending (ல, க்கு, ஆ, ...): ஜெயிலுக்கு -> ஜெயில் + க்கு.
 3. Turn the stem into sounds (Tamil spelling is phonetic) and compare with how
    the ~25k most common English words are pronounced (CMU Pronouncing
    Dictionary), allowing for Tamil's extra vowels (break -> "pi-rek").
    Accept the closest English word if it is close enough.
"""
from __future__ import annotations

import functools
import re

from .tanglish import rules

GRANTHA = set("ஜஷஸஹஃ")
HARD_END = ("ட்", "ஸ்", "ஷ்", "ஜ்")  # not க் ப் ச் த்: Tamil adds those when joining words
BAD_START = set("டறணளழ")
SUFFIXES = sorted(["ல", "ில", "லே", "ுல", "க்கு", "ுக்கு", "ஆ", "ா", "ை", "ோட", "ும்", "ங்க", "ங்கய்யா",
                   "ன்னு", "னு", "ாக", "ாங்க", "ிலே", "ில்", "யா", "ு", "ுங்க", "ப்பா"], key=len, reverse=True)
MAX_DIST = 0.28  # normalised sound distance to accept a match

# coarse sound classes shared by both sides
_C = {"P": "P", "B": "P", "F": "F", "T": "T", "D": "T", "TH": "T", "DH": "T", "K": "K", "G": "K",
      "CH": "S", "JH": "J", "SH": "S", "ZH": "S", "S": "S", "Z": "S", "M": "M", "N": "N", "NG": "N",
      "L": "L", "R": "R", "V": "V", "W": "V", "Y": "Y", "HH": ""}
_V = {"AA": "A", "AE": "A", "AH": "A", "AO": "O", "EH": "E", "EY": "E", "IH": "I", "IY": "I",
      "OW": "O", "UH": "U", "UW": "U", "AY": "AI", "OY": "OI", "AW": "AU", "ER": "AR"}


def _eng_sounds(phones):
    out = []
    for p in phones:
        b = re.sub(r"\d", "", p)
        out += list(_V[b]) if b in _V else ([_C[b]] if _C.get(b) else [])
    return "".join(out)


def _tam_sounds(roman: str) -> str:
    s = roman.lower()
    for a, b in [("aa", "A"), ("ee", "I"), ("oo", "U"), ("ai", "AI"), ("au", "AU"), ("th", "T"), ("dh", "T"),
                 ("ch", "S"), ("sh", "S"), ("zh", "L"), ("ng", "N"), ("nj", "N")]:
        s = s.replace(a, b)
    m = {"a": "A", "e": "E", "i": "I", "o": "O", "u": "U", "k": "K", "g": "K", "s": "S", "j": "J", "t": "T",
         "d": "T", "n": "N", "p": "P", "b": "P", "m": "M", "y": "Y", "r": "R", "l": "L", "v": "V", "h": "", "f": "F"}
    s = "".join(m.get(c, c) for c in s)
    s = re.sub(r"([PTKSJMNLRV])\1+", r"\1", s)  # kk -> k
    return re.sub(r"([AEIOU])\1+", r"\1", s)


def _dist(a: str, b: str) -> float:
    """Edit distance where vowels and Y are cheap (Tamil adds/changes them)."""
    soft = set("AEIOUY")
    cost = lambda c: 0.4 if c in soft else 1.0
    D = [[0.0] * (len(b) + 1) for _ in range(len(a) + 1)]
    def cost_a(i):  # deleting from the Tamil side
        c = a[i - 1]
        if c in "IU" and 1 < i < len(a) and a[i - 2] not in soft and a[i] not in soft:
            return 0.1  # epenthetic vowel between consonants
        return cost(c)
    for i in range(1, len(a) + 1):
        D[i][0] = D[i - 1][0] + cost_a(i)
    for j in range(1, len(b) + 1):
        D[0][j] = D[0][j - 1] + cost(b[j - 1])
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            x, y = a[i - 1], b[j - 1]
            sub = 0 if x == y else (0.2 if {x, y} <= set("AO") or {x, y} <= set("IE") or {x, y} <= set("UO")
                                    else 0.4 if x in soft and y in soft else 1.0)
            D[i][j] = min(D[i - 1][j] + cost_a(i), D[i][j - 1] + cost(b[j - 1]), D[i - 1][j - 1] + sub)
    cons = sum(1 for c in a + b if c not in soft) / 2 or 1
    return D[-1][-1] / cons


@functools.lru_cache(1)
def _english():
    import cmudict
    from wordfreq import top_n_list, zipf_frequency

    cmu = cmudict.dict()
    out = []
    for w in top_n_list("en", 40000):
        if len(w) >= 2 and w.isalpha() and w in cmu and zipf_frequency(w, "en") >= 2.8:
            snd = _eng_sounds(cmu[w][0])
            out.append((w, snd, re.sub(r"[AEIOUY]", "", snd), 0.08 * max(0.0, 5.0 - zipf_frequency(w, "en"))))
    return out


def looks_borrowed(word: str) -> bool:
    return any(c in GRANTHA for c in word) or word.endswith(HARD_END) or (bool(word) and word[0] in BAD_START)


@functools.lru_cache(maxsize=50000)
def detect(word: str) -> tuple[str, str] | None:
    """Return (english_word, tamil_suffix) or None."""
    if not looks_borrowed(word):
        return None
    best = None
    VSIGN = set("\u0bbe\u0bbf\u0bc0\u0bc1\u0bc2\u0bc6\u0bc7\u0bc8\u0bca\u0bcb\u0bcc")
    cands = [(word, "")]
    for x in SUFFIXES:
        if word.endswith(x) and len(word) - len(x) >= 2:
            stem = word[: -len(x)]
            if x[0] in VSIGN:
                stem += "\u0bcd"  # ஜெயிலுக்கு -> ஜெயில் + ுக்கு
            cands.append((stem, x))
    for stem, suf in cands:
        ts = _tam_sounds(rules(stem))
        tcons = re.sub(r"[AEIOUY]", "", ts)
        if len(tcons) < 2:
            continue
        for w, es, econs, pen in _english():
            if econs[:1] != tcons[:1] or abs(len(econs) - len(tcons)) > 1:
                continue
            d = _dist(ts, es) + pen
            if d <= MAX_DIST and (best is None or d < best[0]):
                best = (d, w, suf)
    return (best[1], best[2]) if best else None


if __name__ == "__main__":
    import sys

    for w in (sys.argv[1:] or "லாஜிக் ரைட் பிரேக் மியூட் கான்ஸ்டபிள் மிலிட்டரில ஜெயிலுக்கு வாய்ஸ் பாயிண்ட் லாக்கப் டீச்சர் சைலென்ட்ல கிளாஸ் போலீஸ் பத்து வருஷம் சந்திரன் நட்சத்திரம் சாப்பாடு விஷயம் ஜாதகம் ராசி".split()):
        print(w, "->", detect(w))
