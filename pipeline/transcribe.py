"""Tamil recording -> Tamil-script transcript + Tanglish transcript + subtitles.
Runs offline on a normal computer's CPU.

    python -m pipeline.transcribe class.mp3 [more files...] [--out folder]

For each input file it writes a folder <file>_transcript/ containing:
    tamil.txt       the transcript in Tamil letters, with punctuation and paragraphs;
                    each paragraph starts with its [h:mm:ss] time
    tanglish.txt    the same in Tanglish (Tamil written in English letters)
    subtitles.srt   Tamil subtitles, one sentence each (open with the recording in VLC etc.)
    words.json      every word with its start time and the pause after it

Components (see pipeline/README.md): read audio -> cut at pauses -> SraVaani
(Tamil only) -> punctuation + paragraphs -> Tanglish -> save files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .audio import SR, cut, decode
from .punctuate import paragraphs, pauses, punctuate
from .tanglish import convert

SUB_MAX_S = 7.0  # subtitles longer than this are split


def _ts(s: float, srt: bool = False) -> str:
    h, m, sec = int(s // 3600), int(s % 3600 // 60), s % 60
    return f"{h:02d}:{m:02d}:{int(sec):02d},{int((sec % 1) * 1000):03d}" if srt else f"{h}:{m:02d}:{int(sec):02d}"


def _tanglish_sentence(s: dict, method: str) -> str:
    text = convert(" ".join(w["w"] for w in s["words"]), method)
    return (text[:1].upper() + text[1:] if text else text) + s["mark"]


def _subtitles(sents: list[dict]) -> list[tuple[float, float, str]]:
    subs = []
    for s in sents:
        chunk = []
        for w in s["words"]:
            chunk.append(w)
            if w["end"] - chunk[0]["start"] >= SUB_MAX_S and w is not s["words"][-1]:
                subs.append((chunk[0]["start"], w["end"], " ".join(x["w"] for x in chunk)))
                chunk = []
        if chunk:
            subs.append((chunk[0]["start"], chunk[-1]["end"], " ".join(x["w"] for x in chunk) + s["mark"]))
    return subs


def save(words: list[dict], gaps: list[float], out: Path, tanglish_method: str) -> None:
    """Components 5 + 6: punctuation/paragraphs and Tanglish, then write the files."""
    sents = punctuate(words, gaps)
    paras = paragraphs(sents)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tamil.txt").write_text("\n".join(
        f"[{_ts(p[0]['start'])}] " + " ".join(s["text"] for s in p) + "\n" for p in paras), encoding="utf-8")
    (out / "tanglish.txt").write_text("\n".join(
        f"[{_ts(p[0]['start'])}] " + " ".join(_tanglish_sentence(s, tanglish_method) for s in p) + "\n"
        for p in paras), encoding="utf-8")
    (out / "subtitles.srt").write_text("".join(
        f"{i}\n{_ts(a, True)} --> {_ts(b, True)}\n{t}\n\n" for i, (a, b, t) in enumerate(_subtitles(sents), 1)),
        encoding="utf-8")
    words_out = [{**w, "pause_after": round(g, 2)} for w, g in zip(words, gaps)]
    (out / "words.json").write_text(json.dumps(words_out, ensure_ascii=False), encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Transcribe Tamil audio to Tamil + Tanglish text.")
    ap.add_argument("audio", nargs="+", type=Path, help="audio or video file(s)")
    ap.add_argument("--out", type=Path, help="put results here instead of next to each file")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (default 8)")
    ap.add_argument("--english-detector", action="store_true",
                    help="also guess English words not in the built-in list (experimental: more English "
                         "words recovered, but ~30%% of its guesses on general Tamil text are wrong)")
    a = ap.parse_args(argv)

    from .sravaani import SraVaaniTamil

    print("Loading the speech model (first run downloads ~0.9 GB)...", flush=True)
    model = SraVaaniTamil(threads=a.threads)
    method = "auto" if a.english_detector else "rules+en"
    for path in a.audio:
        if not path.exists():
            print(f"!! not found: {path}", file=sys.stderr)
            continue
        t0 = time.time()
        y = decode(path)
        pieces = cut(y)
        words = []
        for i, (s, e) in enumerate(pieces, 1):
            words += model.words(y[s:e], s / SR)
            print(f"\r{path.name}: {i}/{len(pieces)} pieces", end="", flush=True)
        out = (a.out / (path.stem + "_transcript")) if a.out else path.with_name(path.stem + "_transcript")
        save(words, pauses(words, y), out, method)
        print(f"\n{path.name}: {len(y)/SR/60:.1f} min of audio -> {len(words)} words in "
              f"{(time.time()-t0)/60:.1f} min. Saved to: {out}", flush=True)


if __name__ == "__main__":
    main()
