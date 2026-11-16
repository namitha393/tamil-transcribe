"""Run models over one or more test sets and save predictions.

  python run.py --models whisper-tamil-small indicconformer-ctc --sets fleurs_ta indicvoices_ta kathbath_ta
  python run.py ... --limit 3            # quick smoke test
  python run.py ... --cleanup            # delete each model's download after it finishes (low disk)

Each model is loaded once and run over every set. Predictions go to
results/<set>/<model>.jsonl and are resumable: clips already transcribed are
skipped. Timing is wall-clock per clip on this machine's CPU.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Keep model downloads inside this project so --cleanup only ever touches them.
MODEL_CACHE = ROOT / ".hf_models"
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE)

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402

from models import REGISTRY  # noqa: E402


def run_model(name: str, sets: list[str], limit: int | None) -> None:
    jobs = []
    for s in sets:
        clips = [json.loads(l) for l in (ROOT / "data" / s / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        clips = clips[:limit] if limit else clips
        out = ROOT / "results" / s / f"{name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
        todo = [c for c in clips if c["id"] not in done]
        print(f"[{name}] {s}: {len(done)} done, {len(todo)} to go", flush=True)
        if todo:
            jobs.append((s, out, todo))
    if not jobs:
        return

    factory, is_api = REGISTRY[name]
    t0 = time.perf_counter()
    model = factory()
    load_s = time.perf_counter() - t0
    print(f"[{name}] loaded in {load_s:.1f}s", flush=True)

    for s, out, todo in jobs:
        with open(out, "a", encoding="utf-8") as f:
            for i, c in enumerate(todo, 1):
                t = time.perf_counter()
                err = None
                try:
                    hyp = model.transcribe(c["audio"])
                except Exception as e:  # keep going; record the failure
                    hyp, err = "", f"{type(e).__name__}: {e}"
                    traceback.print_exc()
                dt = time.perf_counter() - t
                f.write(json.dumps({"id": c["id"], "ref": c["text"], "hyp": hyp, "duration": c["duration"],
                                    "seconds": dt, "api": is_api, "error": err}, ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(todo) or i <= 2:
                    print(f"[{name}] {s} {i}/{len(todo)}  RTF {dt / max(c['duration'], 1e-6):.2f}  {hyp[:60]!r}",
                          flush=True)
        out.with_suffix(".meta.json").write_text(json.dumps(
            {"model": name, "device": os.environ.get("ASR_DEVICE", "cpu"), "load_seconds": load_s, "threads": torch.get_num_threads(),
             "machine": platform.platform(), "cpu": platform.processor()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--sets", nargs="+", default=["fleurs_ta", "indicvoices_ta", "kathbath_ta"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for local models")
    ap.add_argument("--cleanup", action="store_true", help="delete downloaded model files after each model")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for m in a.models:
        try:
            run_model(m, a.sets, a.limit)
        except Exception:
            print(f"[{m}] FAILED to load/run", flush=True)
            traceback.print_exc()
        finally:
            if a.cleanup and MODEL_CACHE.exists():
                shutil.rmtree(MODEL_CACHE)
                print(f"[{m}] deleted model download", flush=True)
