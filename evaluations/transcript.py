"""Turn per-piece predictions for a long recording into readable transcripts,
in Tamil script and in Tanglish (Tamil in English letters), and measure how
much the models agree with each other.

  python transcript.py astro_class

Writes transcripts/<set>/<model>.tamil.txt and <model>.tanglish.txt.

Tanglish here is rule-based transliteration (aksharamukha "RomanReadable",
apostrophes removed): free, offline, consistent. It spells English loanwords
phonetically (கிளாஸ் -> "kilaas", not "class"); an LLM pass can fix those.
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import jiwer

from normalize import normalize

warnings.filterwarnings("ignore")
from aksharamukha import transliterate as _T  # noqa: E402

ROOT = Path(__file__).resolve().parent


def tanglish(text: str) -> str:
    out = _T.process("Tamil", "RomanReadable", text)
    return out.replace("'", "").replace("_", "")


def ts(sec: float) -> str:
    return f"{int(sec // 3600):d}:{int(sec % 3600 // 60):02d}:{int(sec % 60):02d}"


def main(name: str) -> None:
    man = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data" / name / "manifest.jsonl").read_text().splitlines()}
    out_dir = ROOT / "transcripts" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    hyps = {}
    for p in sorted((ROOT / "results" / name).glob("*.jsonl")):
        recs = {json.loads(l)["id"]: json.loads(l) for l in p.read_text().splitlines()}
        hyps[p.stem] = recs
        ta, en = [], []
        for i in sorted(recs):
            line = recs[i]["hyp"]
            stamp = f"[{ts(man[i]['start'])}]"
            ta.append(f"{stamp} {line}")
            en.append(f"{stamp} {tanglish(line)}")
        (out_dir / f"{p.stem}.tamil.txt").write_text("\n".join(ta) + "\n", encoding="utf-8")
        (out_dir / f"{p.stem}.tanglish.txt").write_text("\n".join(en) + "\n", encoding="utf-8")
        secs = sum(r["seconds"] for r in recs.values())
        audio = sum(r["duration"] for r in recs.values())
        print(f"{p.stem}: {len(recs)} pieces, {audio/60:.1f} min audio in {secs/60:.1f} min CPU "
              f"(RTF {secs/audio:.3f}), {sum(len(r['hyp'].split()) for r in recs.values())} words")

    print("\nAgreement between models (WER of one against the other, %; lower = more agreement):")
    for a, b in itertools.combinations(sorted(hyps), 2):
        ids = sorted(set(hyps[a]) & set(hyps[b]))
        ra = [normalize(hyps[a][i]["hyp"]) or "-" for i in ids]
        rb = [normalize(hyps[b][i]["hyp"]) or "-" for i in ids]
        print(f"  {a} vs {b}: WER {100*jiwer.wer(ra, rb):.1f}  "
              f"CER {100*jiwer.cer([x.replace(' ', '') for x in ra], [x.replace(' ', '') for x in rb]):.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "astro_class")
