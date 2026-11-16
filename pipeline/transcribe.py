"""Tamil recording -> Tamil-script transcript + Tanglish transcript + subtitles.
Runs offline on a normal computer's CPU.

    python -m pipeline.transcribe class.mp3 [more files...] [--out folder]

For each input file it writes a folder <file>_transcript/ containing:
    tamil.txt       the transcript in Tamil letters, one line per spoken stretch, with [h:mm:ss] times
    tanglish.txt    the same lines in Tanglish (Tamil written in English letters)
    subtitles.srt   Tamil subtitles (open with the video/audio in VLC etc.)
    words.json      every word with its start and end time, for later processing

Components (see pipeline/README.md): read audio -> cut at pauses -> SraVaani
(Tamil only) -> group words into lines -> Tanglish -> save files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .audio import SR, cut, decode
from .tanglish import convert

LINE_GAP = 0.6     # a pause this long (seconds) starts a new line
LINE_MAX_S = 12.0  # ...and no line is longer than this


def group_lines(words: list[dict]) -> list[dict]:
    """Component 4: words -> readable lines, breaking at pauses."""
    lines = []
    for w in words:
        if lines and w["start"] - lines[-1]["end"] < LINE_GAP and w["end"] - lines[-1]["start"] <= LINE_MAX_S:
            lines[-1]["text"] += " " + w["w"]
            lines[-1]["end"] = w["end"]
        else:
            lines.append({"start": w["start"], "end": w["end"], "text": w["w"]})
    return lines


def _ts(s: float, srt: bool = False) -> str:
    h, m, sec = int(s // 3600), int(s % 3600 // 60), s % 60
    return f"{h:02d}:{m:02d}:{int(sec):02d},{int((sec % 1) * 1000):03d}" if srt else f"{h}:{m:02d}:{int(sec):02d}"


def save(lines: list[dict], words: list[dict], out: Path, tanglish_method: str) -> None:
    """Components 5 + 6: Tanglish, then write the files."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "tamil.txt").write_text("".join(f"[{_ts(l['start'])}] {l['text']}\n" for l in lines), encoding="utf-8")
    (out / "tanglish.txt").write_text(
        "".join(f"[{_ts(l['start'])}] {convert(l['text'], tanglish_method)}\n" for l in lines), encoding="utf-8")
    (out / "subtitles.srt").write_text("".join(
        f"{i}\n{_ts(l['start'], True)} --> {_ts(l['end'], True)}\n{l['text']}\n\n" for i, l in enumerate(lines, 1)),
        encoding="utf-8")
    (out / "words.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")


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
        lines = group_lines(words)
        out = (a.out / (path.stem + "_transcript")) if a.out else path.with_name(path.stem + "_transcript")
        save(lines, words, out, method)
        print(f"\n{path.name}: {len(y)/SR/60:.1f} min of audio -> {len(words)} words in "
              f"{(time.time()-t0)/60:.1f} min. Saved to: {out}", flush=True)


if __name__ == "__main__":
    main()
