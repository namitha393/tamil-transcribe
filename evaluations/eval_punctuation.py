"""Score punctuation on the 3 class sections against Gemini's and Sarvam's punctuation.

For words that line up between our transcript and the reference, compare what
follows each word: sentence end (. or ?), question mark, comma.
Reports F1 (0-100, higher is better) for each, plus how well Sarvam and Gemini
agree with each other (a practical ceiling).

  PYTHONPATH=.. python eval_punctuation.py AUDIO_DIR      # folder with the section wavs
"""
import json, re, sys
from pathlib import Path
import jiwer, soundfile as sf
from normalize import normalize
from pipeline.punctuate import pauses, punctuate

AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../tamil_asr_eval/for_customer/sections")
SECS = {"A": "section_A_38m46s", "B": "section_B_59m34s", "C": "section_C_29m11s"}
REF = Path("../sravaani_eval/references")


def tokens_with_marks(text: str, line_end_is_stop: bool):
    """-> [(normalised word, mark)] where mark in {'.', '?', ',', ''}"""
    out = []
    for line in text.splitlines():
        toks = line.split()
        for k, t in enumerate(toks):
            w = normalize(t)
            if not w:
                continue
            mark = "?" if "?" in t else "." if re.search(r"[.!…।]", t) else "," if "," in t else ""
            if not mark and line_end_is_stop and k == len(toks) - 1:
                mark = "."
            for sub in w.split()[:-1]:
                out.append((sub, ""))
            out.append((w.split()[-1], mark))
    return out


def f1(ref, hyp):
    """ref/hyp: [(word, mark)]; score marks on aligned word pairs."""
    r = [w for w, _ in ref]; h = [w for w, _ in hyp]
    res = {}
    pairs = []
    for a in jiwer.process_words(" ".join(r), " ".join(h)).alignments[0]:
        if a.type in ("equal", "substitute"):
            pairs += [(ref[a.ref_start_idx + k][1], hyp[a.hyp_start_idx + k][1]) for k in range(a.ref_end_idx - a.ref_start_idx)]
    for name, cls in [("sentence end", {".", "?"}), ("question", {"?"}), ("comma", {","})]:
        tp = sum(1 for x, y in pairs if x in cls and y in cls)
        p = tp / max(1, sum(1 for _, y in pairs if y in cls)); rc = tp / max(1, sum(1 for x, _ in pairs if x in cls))
        res[name] = 100 * 2 * p * rc / max(1e-9, p + rc)
    return res


def ours(tag, **kw):
    words = json.load(open(f"../sravaani_eval/class_transcripts/{SECS[tag]}/words.json"))
    y, _ = sf.read(AUDIO / f"{SECS[tag]}.wav", dtype="float32")
    sents = punctuate(words, pauses(words, y), **kw)
    return tokens_with_marks(" ".join(s["text"] for s in sents), False)


def refs(tag):
    g = tokens_with_marks((REF / f"gemini_section_{tag}_tamil.txt").read_text(), line_end_is_stop=True)
    s = tokens_with_marks((REF / f"sarvam_section_{tag}_v3_tamil.txt").read_text(), line_end_is_stop=False)
    return g, s


def avg(rows):
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


if __name__ == "__main__":
    R = {t: refs(t) for t in "ABC"}
    print("F1 (higher is better), average of sections A/B/C:  vs Gemini  |  vs Sarvam")
    print(f"{'':38s} {'end':>5s} {'?':>5s} {',':>5s}  | {'end':>5s} {'?':>5s} {',':>5s}")
    def show(name, hyp_by_tag):
        g = avg([f1(R[t][0], hyp_by_tag[t]) for t in "ABC"]); s = avg([f1(R[t][1], hyp_by_tag[t]) for t in "ABC"])
        print(f"{name:38s} {g['sentence end']:5.1f} {g['question']:5.1f} {g['comma']:5.1f}  | {s['sentence end']:5.1f} {s['question']:5.1f} {s['comma']:5.1f}")
    show("Sarvam v3 (ceiling: two cloud systems)", {t: R[t][1] for t in "ABC"})
    for stop in [0.25, 0.35, 0.45, 0.6, 0.8, 1.0]:
        show(f"pause >= {stop}s = full stop", {t: ours(t, stop_s=stop, questions=False, commas=False) for t in "ABC"})
    for stop in [0.35, 0.45, 0.6]:
        show(f"pause >= {stop}s + question forms", {t: ours(t, stop_s=stop, questions=True, commas=False) for t in "ABC"})
    for stop, comma in [(0.45, 0.2), (0.45, 0.25), (0.6, 0.3)]:
        show(f"  + commas at pause >= {comma}s (stop {stop})", {t: ours(t, stop_s=stop, comma_s=comma) for t in "ABC"})
