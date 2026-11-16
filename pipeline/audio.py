"""Component 1 + 2: read any audio file, and cut it into pieces at pauses.

Why 15-28 s pieces: the model was tested on long audio with a known transcript;
fewer, longer pieces scored best (each cut costs a little accuracy at its edges),
and 28 s stays under the 30 s limit of other models we compare against.
Nothing is thrown away: every sample of the recording ends up in some piece.
"""
from __future__ import annotations

import io

import numpy as np

SR = 16000          # the model expects 16 kHz mono
MIN_S, MAX_S = 15.0, 28.0


def decode(path_or_bytes) -> np.ndarray:
    """Any audio/video file (mp3, m4a, mp4, wav, ogg, opus...) -> 16 kHz mono float32.
    Uses PyAV, which bundles ffmpeg, so nothing else needs installing."""
    import av

    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes)
    with av.open(src) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = [o.to_ndarray().reshape(-1) for frame in c.decode(audio=0) for o in rs.resample(frame)]
        chunks += [o.to_ndarray().reshape(-1) for o in rs.resample(None)]
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def cut(y: np.ndarray, min_s: float = MIN_S, max_s: float = MAX_S) -> list[tuple[int, int]]:
    """Split into pieces of min_s..max_s seconds. Each cut is placed at the
    quietest 0.3 s inside that window, so cuts land in pauses, not mid-word.
    Returns (start_sample, end_sample) pairs covering the whole recording."""
    hop = int(0.01 * SR)
    n = len(y) // hop
    if n == 0:
        return []
    energy = (y[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1)
    loudness = np.sqrt(np.convolve(energy, np.ones(30) / 30, mode="same") + 1e-10)
    pieces, start = [], 0
    while start < n:
        if n - start <= max_s * 100:
            pieces.append((start, n))
            break
        lo, hi = start + int(min_s * 100), start + int(max_s * 100)
        c = lo + int(np.argmin(loudness[lo:hi]))
        pieces.append((start, c))
        start = c
    return [(a * hop, len(y) if i == len(pieces) - 1 else b * hop) for i, (a, b) in enumerate(pieces)]
