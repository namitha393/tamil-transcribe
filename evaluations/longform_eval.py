"""Compare ways of cutting long recordings, using long audio whose correct
text we know: 'lectures' built from consecutive IndicVoices clips joined with
natural-length pauses (0.2-2.0 s).

  python longform_eval.py build                  # lectures + one test set per splitter
  python run.py --models sravaani-tamil --sets lf_fixed25 lf_pause25 lf_pause15 lf_silero15 lf_silero25
  python longform_eval.py report sravaani-tamil

Splitters:
  fixed25    blind 25 s cuts (baseline: cuts mid-word)
  pause25    ours: quietest point 8-25 s after the previous cut
  pause15    ours: quietest point 5-15 s after the previous cut
  silero15   Silero VAD speech regions, merged into pieces <= 15 s, cut inside pauses
  silero25   same, <= 25 s
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from data import DATA, SR, decode, split_at_pauses
from normalize import score

N_LECT, LECT_S = 6, 300
SPLITTERS = ["fixed25", "pause25", "pause15", "silero15", "silero25"]


def build_lectures():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    clips = []
    for fn in ["tamil/valid-00000-of-00002.parquet", "tamil/valid-00001-of-00002.parquet"]:
        pf = pq.ParquetFile(hf_hub_download("ai4bharat/IndicVoices", fn, repo_type="dataset"))
        for g in range(pf.num_row_groups):
            clips += [r for r in pf.read_row_group(g, columns=["audio_filepath", "text", "duration"]).to_pylist()
                      if r["text"].strip()]
    rng = random.Random(7)
    out = DATA / "lf_lectures"
    out.mkdir(parents=True, exist_ok=True)
    lects = []
    for k in range(N_LECT):
        i = rng.randrange(len(clips) - 400)
        parts, texts, dur = [], [], 0.0
        while dur < LECT_S:
            y = decode(clips[i]["audio_filepath"]["bytes"])
            pause = np.zeros(int(rng.uniform(0.2, 2.0) * SR), np.float32)
            parts += [y, pause]
            texts.append(clips[i]["text"])
            dur += (len(y) + len(pause)) / SR
            i += 1
        y = np.concatenate(parts)
        p = out / f"lect{k}.wav"
        sf.write(p, y, SR)
        lects.append({"id": f"lect{k}", "audio": str(p), "text": " ".join(texts), "duration": len(y) / SR})
    (out / "lectures.json").write_text(json.dumps(lects, ensure_ascii=False))
    print(f"{N_LECT} lectures, {sum(l['duration'] for l in lects)/60:.1f} min")
    return lects


_vad = None


def silero_split(y: np.ndarray, max_s: float) -> list[tuple[int, int]]:
    """Speech regions from Silero VAD, merged greedily into pieces <= max_s,
    each cut placed in the middle of a pause."""
    global _vad
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    _vad = _vad or load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(y), _vad, sampling_rate=SR, min_silence_duration_ms=150,
                               speech_pad_ms=100)
    pieces, cur = [], None
    for t in ts:
        s, e = t["start"], t["end"]
        if cur is None:
            cur = [s, e]
        elif e - cur[0] <= max_s * SR:
            cur[1] = e
        else:
            pieces.append(cur)
            cur = [s, e]
    if cur:
        pieces.append(cur)
    # a single speech region longer than max_s: fall back to our pause cutter inside it
    out = []
    for s, e in pieces:
        if e - s > max_s * SR:
            out += [(s + a, s + b) for a, b in split_at_pauses(y[s:e], min_s=max_s / 3, max_s=max_s)]
        else:
            out.append((s, e))
    return out


def cut(y: np.ndarray, how: str) -> list[tuple[int, int]]:
    if how == "fixed25":
        return [(a, min(a + 25 * SR, len(y))) for a in range(0, len(y), 25 * SR)]
    if how == "pause25":
        return split_at_pauses(y, 8.0, 25.0)
    if how == "pause15":
        return split_at_pauses(y, 5.0, 15.0)
    if how.startswith("silero"):
        return silero_split(y, float(how[6:]))
    raise ValueError(how)


def build():
    lects = build_lectures()
    for how in SPLITTERS:
        d = DATA / f"lf_{how}"
        (d / "wav").mkdir(parents=True, exist_ok=True)
        n = 0
        with open(d / "manifest.jsonl", "w", encoding="utf-8") as f:
            for L in lects:
                y, _ = sf.read(L["audio"], dtype="float32")
                for j, (a, b) in enumerate(cut(y, how)):
                    pid = f"{L['id']}_{j:04d}"
                    p = d / "wav" / f"{pid}.wav"
                    sf.write(p, y[a:b], SR)
                    f.write(json.dumps({"id": pid, "text": "", "audio": str(p), "start": a / SR,
                                        "duration": (b - a) / SR}) + "\n")
                    n += 1
        print(f"lf_{how}: {n} pieces")


def report(models):
    lects = json.loads((DATA / "lf_lectures" / "lectures.json").read_text())
    root = Path(__file__).resolve().parent / "results"
    print("splitter  | pieces | " + " | ".join(f"{m} WER% / CER%" for m in models))
    for how in SPLITTERS:
        cells, npieces = [], "-"
        for m in models:
            p = root / f"lf_{how}" / f"{m}.jsonl"
            if not p.exists():
                cells.append("-")
                continue
            recs = sorted((json.loads(l) for l in p.read_text().splitlines()), key=lambda r: r["id"])
            npieces = len(recs)
            hyp = {L["id"]: " ".join(r["hyp"] for r in recs if r["id"].startswith(L["id"] + "_")) for L in lects}
            s = score([L["text"] for L in lects], [hyp[L["id"]] for L in lects])
            cells.append(f"{100*s['wer']:5.1f} / {100*s['cer']:4.1f}")
        print(f"{how:9s} | {npieces:>6} | " + " | ".join(cells))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        report(sys.argv[2:])
