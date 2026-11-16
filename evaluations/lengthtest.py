"""How does accuracy change with input length? Build inputs of a fixed length
by joining consecutive IndicVoices clips (same recording session, so usually
the same speaker) with 0.3 s gaps; the reference is the joined transcripts.

  python lengthtest.py build                    # writes data/len_<L>/manifest.jsonl
  python run.py --models indicconformer-rnnt --sets len_5 len_10 ...
  python lengthtest.py report indicconformer-rnnt sravaani ...
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from data import DATA, SR, decode
from normalize import score

LENGTHS = [5, 10, 15, 20, 25, 30, 45, 60]
PER_LEN = 20
FILES = ["tamil/valid-00000-of-00002.parquet", "tamil/valid-00001-of-00002.parquet"]


def build():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    clips = []
    for fn in FILES:
        pf = pq.ParquetFile(hf_hub_download("ai4bharat/IndicVoices", fn, repo_type="dataset"))
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g, columns=["audio_filepath", "text", "duration", "speaker_id"]).to_pylist():
                if r["text"].strip() and r["duration"] >= 1.0:
                    clips.append(r)
    rng = random.Random(1)
    gap = np.zeros(int(0.3 * SR), np.float32)
    for L in LENGTHS:
        out = DATA / f"len_{L}"
        (out / "wav").mkdir(parents=True, exist_ok=True)
        with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
            made = 0
            while made < PER_LEN:
                i = rng.randrange(len(clips))
                parts, texts, dur = [], [], 0.0
                while dur < L and i < len(clips) and clips[i]["speaker_id"] == clips[min(i, len(clips) - 1)]["speaker_id"]:
                    if parts and clips[i]["speaker_id"] != spk:
                        break
                    spk = clips[i]["speaker_id"]
                    y = decode(clips[i]["audio_filepath"]["bytes"])
                    if dur + len(y) / SR > L * 1.15 and parts:
                        break
                    parts += [y, gap]
                    texts.append(clips[i]["text"])
                    dur += len(y) / SR + 0.3
                    i += 1
                if not (0.8 * L <= dur <= 1.15 * L):
                    continue
                y = np.concatenate(parts[:-1])
                cid = f"len{L}_{made:02d}"
                sf.write(out / "wav" / f"{cid}.wav", y, SR)
                f.write(json.dumps({"id": cid, "text": " ".join(texts), "audio": str(out / "wav" / f"{cid}.wav"),
                                    "duration": len(y) / SR}, ensure_ascii=False) + "\n")
                made += 1
        print(f"len_{L}: {PER_LEN} inputs")


def report(models):
    root = Path(__file__).resolve().parent / "results"
    print("length | " + " | ".join(f"{m} WER% (empty)" for m in models))
    for L in LENGTHS:
        cells = []
        for m in models:
            p = root / f"len_{L}" / f"{m}.jsonl"
            if not p.exists():
                cells.append("-")
                continue
            r = [json.loads(l) for l in p.read_text().splitlines()]
            s = score([x["ref"] for x in r], [x["hyp"] for x in r])
            cells.append(f"{100*s['wer']:5.1f} ({s['empty_outputs']})")
        print(f"{L:4d}s  | " + " | ".join(cells))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        report(sys.argv[2:])
