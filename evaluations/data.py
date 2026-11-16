"""Build test-set manifests: one JSON line per clip
{"id", "audio", "text", "duration", ...metadata} with audio as 16 kHz mono WAV.

  python data.py fleurs       --n 200   google/fleurs ta_in test (open, read speech)
  python data.py indicvoices  --n 200   ai4bharat/IndicVoices tamil valid (gated, mostly spontaneous)
  python data.py kathbath     --n 200   ai4bharat/Kathbath tamil valid (gated, read)
  python data.py folder --dir D --name customer
        your own audio: D/*.wav|mp3|m4a... plus D/<same name>.txt with the
        corrected transcript (for customer audio)

Clips are sampled at random with a fixed seed (same sample every run), limited
to 1-30 s so every model (incl. Sarvam's sync API) sees the same input.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import tarfile
from pathlib import Path

import av
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SR = 16000
SEED = 0
MIN_S, MAX_S = 1.0, 30.0


def decode(src) -> np.ndarray:
    """Decode any audio (path or bytes) to 16 kHz mono float32 using PyAV
    (bundled ffmpeg, so m4a/mp3/opus work without a system ffmpeg)."""
    f = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src)
    with av.open(f) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []
        for frame in c.decode(audio=0):
            for o in rs.resample(frame):
                chunks.append(o.to_ndarray().reshape(-1))
        for o in rs.resample(None):
            chunks.append(o.to_ndarray().reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def _write(rows: list[dict], name: str) -> Path:
    out_dir = DATA / name
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as out:
        for r in rows:
            y = decode(r.pop("_audio"))
            dst = out_dir / "wav" / f"{r['id']}.wav"
            sf.write(str(dst), y, SR)
            out.write(json.dumps({**r, "audio": str(dst), "duration": len(y) / SR}, ensure_ascii=False) + "\n")
    hours = sum(json.loads(l)["duration"] for l in manifest.read_text().splitlines()) / 3600
    print(f"{name}: {len(rows)} clips, {hours:.2f} h -> {manifest}")
    return manifest


def build_fleurs(n: int) -> Path:
    from huggingface_hub import hf_hub_download

    tsv = hf_hub_download("google/fleurs", "data/ta_in/test.tsv", repo_type="dataset")
    tar = hf_hub_download("google/fleurs", "data/ta_in/audio/test.tar.gz", repo_type="dataset")
    meta = {}
    with open(tsv, encoding="utf-8") as f:
        for cols in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            # id, file_name, raw_transcription, transcription, chars, num_samples, gender
            dur = int(cols[5]) / SR
            if MIN_S <= dur <= MAX_S:
                meta[cols[1]] = {"id": f"fleurs_{Path(cols[1]).stem}", "text": cols[2], "gender": cols[6]}
    pick = set(random.Random(SEED).sample(sorted(meta), min(n, len(meta))))
    rows = []
    with tarfile.open(tar) as t:
        for m in t:
            base = Path(m.name).name
            if m.isfile() and base in pick:
                rows.append({**meta[base], "_audio": t.extractfile(m).read()})
    rows.sort(key=lambda r: r["id"])
    return _write(rows, "fleurs_ta")


def _parquet_rows(repo: str, files: list[str]):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for fn in files:
        p = hf_hub_download(repo, fn, repo_type="dataset")
        pf = pq.ParquetFile(p)
        for i in range(pf.num_row_groups):
            yield from pf.read_row_group(i).to_pylist()


def build_parquet(name: str, repo: str, files: list[str], n: int, keep: list[str]) -> Path:
    # Pass 1: metadata only, to sample without holding all audio in memory.
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    cands = []
    for fn in files:
        pf = pq.ParquetFile(hf_hub_download(repo, fn, repo_type="dataset"))
        cols = [c for c in pf.schema_arrow.names if c != "audio_filepath"]
        t = pf.read(columns=cols).to_pylist()
        for j, r in enumerate(t):
            if MIN_S <= r["duration"] <= MAX_S and r["text"].strip():
                cands.append((fn, j))
    pick = set(random.Random(SEED).sample(cands, min(n, len(cands))))
    rows = []
    for fn in files:
        pf = pq.ParquetFile(hf_hub_download(repo, fn, repo_type="dataset"))
        j = 0
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g).to_pylist():
                if (fn, j) in pick:
                    rid = f"{name}_{Path(fn).stem.split('-')[1]}_{j:05d}"
                    rows.append({"id": rid, "text": r["text"], "_audio": r["audio_filepath"]["bytes"],
                                 **{k: r.get(k) for k in keep}})
                j += 1
    return _write(rows, name)


def split_at_pauses(y: np.ndarray, min_s: float = 8.0, max_s: float = 25.0) -> list[tuple[int, int]]:
    """Cut long audio into <= max_s pieces, each cut placed at the quietest
    0.3 s stretch between min_s and max_s after the previous cut, so cuts
    land in pauses rather than mid-word."""
    hop = int(0.01 * SR)
    n = len(y) // hop
    rms = np.sqrt(np.convolve(y[: n * hop].reshape(n, hop).astype(np.float64).__pow__(2).mean(1),
                              np.ones(30) / 30, mode="same") + 1e-10)
    segs, start = [], 0
    while start < n:
        if n - start <= max_s * 100:
            segs.append((start, n))
            break
        lo, hi = start + int(min_s * 100), start + int(max_s * 100)
        cut = lo + int(np.argmin(rms[lo:hi]))
        segs.append((start, cut))
        start = cut
    floor = np.percentile(rms, 10)
    # drop pieces that are essentially silence
    return [(a * hop, b * hop) for a, b in segs if np.percentile(rms[a:b], 90) > 3 * floor]


def build_longform(src: Path, name: str) -> Path:
    """One long recording -> many <= 25 s clips (no reference text)."""
    y = decode(src.read_bytes())
    out_dir = DATA / name
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as out:
        for k, (a, b) in enumerate(split_at_pauses(y)):
            dst = out_dir / "wav" / f"{name}_{k:04d}.wav"
            sf.write(str(dst), y[a:b], SR)
            out.write(json.dumps({"id": f"{name}_{k:04d}", "text": "", "audio": str(dst), "start": a / SR,
                                  "end": b / SR, "duration": (b - a) / SR}, ensure_ascii=False) + "\n")
    rows = manifest.read_text().splitlines()
    print(f"{name}: {len(y)/SR/60:.1f} min -> {len(rows)} pieces -> {manifest}")
    return manifest


def build_folder(src_dir: Path, name: str) -> Path:
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm", ".mp4"}
    rows = []
    for src in sorted(p for p in src_dir.iterdir() if p.suffix.lower() in exts):
        txt = src.with_suffix(".txt")
        rows.append({"id": src.stem, "text": txt.read_text(encoding="utf-8").strip() if txt.exists() else "",
                     "_audio": src.read_bytes()})
    return _write(rows, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["fleurs", "indicvoices", "kathbath", "folder", "longform"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--file", type=Path, help="for 'longform': one long recording")
    ap.add_argument("--name", default="custom")
    a = ap.parse_args()
    if a.dataset == "fleurs":
        build_fleurs(a.n)
    elif a.dataset == "indicvoices":
        build_parquet("indicvoices_ta", "ai4bharat/IndicVoices",
                      ["tamil/valid-00000-of-00002.parquet", "tamil/valid-00001-of-00002.parquet"], a.n,
                      ["scenario", "district", "gender", "area", "speaker_id"])
    elif a.dataset == "kathbath":
        build_parquet("kathbath_ta", "ai4bharat/Kathbath",
                      ["tamil/valid-00000-of-00002.parquet", "tamil/valid-00001-of-00002.parquet"], a.n,
                      ["gender", "speaker_id"])
    elif a.dataset == "longform":
        build_longform(a.file, a.name)
    else:
        build_folder(a.dir, a.name)
