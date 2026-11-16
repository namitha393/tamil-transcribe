"""Text normalisation and scoring for Tamil ASR.

The same normalisation is applied to references and to every model's output,
so no model is rewarded or penalised for punctuation, casing or digit style.
"""
from __future__ import annotations

import unicodedata

import jiwer

# Zero-width joiners / non-joiners and BOM show up inconsistently in Tamil text.
_DROP = {"​", "‌", "‍", "﻿", "­"}
_TAMIL_DIGITS = {chr(0x0BE6 + i): str(i) for i in range(10)}


def normalize(text: str) -> str:
    """NFC, drop zero-width chars, strip punctuation/symbols, lowercase Latin,
    map Tamil digits to ASCII, collapse whitespace. Keeps Tamil vowel signs
    and the pulli (virama), which are Unicode marks (Mn/Mc)."""
    text = unicodedata.normalize("NFC", text or "")
    out = []
    for ch in text:
        if ch in _DROP:
            continue
        ch = _TAMIL_DIGITS.get(ch, ch)
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S"):  # punctuation, symbols
            out.append(" ")
        elif cat[0] in ("L", "M", "N"):
            out.append(ch.lower())
        else:  # whitespace, control chars
            out.append(" ")
    return " ".join("".join(out).split())


def has_latin(text: str) -> bool:
    return any("a" <= c <= "z" for c in text.lower())


def score(refs: list[str], hyps: list[str]) -> dict:
    """Corpus-level WER and CER on normalised text.

    CER is computed with spaces removed, which is the usual choice for Indic
    scripts: it stops word-boundary disagreements (very common in Tamil
    agglutination) from dominating the score."""
    r = [normalize(x) for x in refs]
    h = [normalize(x) for x in hyps]
    # jiwer refuses empty references; keep pairs aligned and skip empties.
    pairs = [(a, b) for a, b in zip(r, h) if a]
    r, h = [a for a, _ in pairs], [b for _, b in pairs]
    wer = jiwer.wer(r, h)
    cer = jiwer.cer([a.replace(" ", "") for a in r], [b.replace(" ", "") for b in h])
    empty = sum(1 for b in h if not b)
    return {"n": len(r), "wer": wer, "cer": cer, "empty_outputs": empty}
