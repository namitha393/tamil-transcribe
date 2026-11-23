"""Vaani (ARTPARK/IISc) gemma4-e2b-vaani-asr-s2t-t2t, offline, Tamil -> English.
  text, whole paragraph   (t2t; trained on sentences, so this tests paragraph handling)
  text, sentence by sentence
  speech -> English       (s2t) on the unit's audio in <=28 s pieces (model caps clips at 30 s)
Uses the model's own prompts (prompt_contract.py). Needs transformers>=5.5 (Gemma 4).
  python translate_vaani.py AUDIO_DIR"""
import json, re, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch
from huggingface_hub import snapshot_download
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from pipeline.audio import cut, SR
REPO = "ARTPARK-IISc/gemma4-e2b-vaani-asr-s2t-t2t"
d = snapshot_download(REPO); sys.path.insert(0, d)
from prompt_contract import build_prompt_s2t, build_prompt_t2t
from transformers import AutoModelForMultimodalLM, AutoProcessor
dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoProcessor.from_pretrained(d)
if proc.tokenizer.pad_token_id is None: proc.tokenizer.pad_token = proc.tokenizer.eos_token
model = AutoModelForMultimodalLM.from_pretrained(d, dtype=torch.bfloat16, attn_implementation="eager").eval().to(dev)

@torch.no_grad()
def gen(content, max_new=256):
    inputs = proc.apply_chat_template([[{"role": "user", "content": content}]], tokenize=True, return_dict=True,
                                      return_tensors="pt", add_generation_prompt=True)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new, num_beams=1, do_sample=False)
    return proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

t2t = lambda text, n=256: gen([{"type": "text", "text": build_prompt_t2t("ta", "en", text), "audio": None}], n)
s2t = lambda y: gen([{"type": "audio", "audio": y, "text": None},
                     {"type": "text", "text": build_prompt_s2t("ta", "en"), "audio": None}])

U = json.load(open(HERE / "units.json"))
AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "tamil_asr_eval/for_customer/sections"
SECS = {"A": "section_A_38m46s", "B": "section_B_59m34s", "C": "section_C_29m11s"}
out = HERE / "out"; out.mkdir(exist_ok=True)

def run(name, fn):
    p = out / f"{name}.json"; res = json.loads(p.read_text()) if p.exists() else {}
    t0 = time.time()
    for u in U:
        if u["id"] not in res:
            res[u["id"]] = fn(u); p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(name, f"{time.time()-t0:.0f}s", flush=True)

sents = lambda s: [x.strip() for x in re.split(r"(?<=[.?!])\s+", s) if x.strip()]
for inp in ["ours", "gemini"]:
    run(f"vaani-t2t-paragraph__{inp}", lambda u: t2t(u[inp], 768))
    run(f"vaani-t2t-sentences__{inp}", lambda u: " ".join(t2t(s) for s in sents(u[inp])))
cache = {}
def speech(u):
    if u["section"] not in cache: cache[u["section"]] = sf.read(AUDIO / f"{SECS[u['section']]}.wav", dtype="float32")[0]
    y = cache[u["section"]][int(u["start"] * SR): int(u["end"] * SR) + SR // 4]
    return " ".join(s2t(y[a:b]) for a, b in cut(y))
run("vaani-speech-translate__audio", speech)
