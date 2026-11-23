"""Meta translators, offline, Tamil -> English (CC BY-NC licences: non-commercial use only).
  nllb      facebook/nllb-200-distilled-1.3B — text; whole paragraph and sentence by sentence
  seamless  facebook/seamless-m4t-v2-large   — text (paragraph + sentences) and speech -> English
  python translate_meta.py nllb|seamless [AUDIO_DIR]"""
import json, re, sys, time
from pathlib import Path
import soundfile as sf, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from pipeline.audio import cut, SR
which = sys.argv[1]
AUDIO = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parents[2] / "tamil_asr_eval/for_customer/sections"
SECS = {"A": "section_A_38m46s", "B": "section_B_59m34s", "C": "section_C_29m11s"}
dev = "mps" if torch.backends.mps.is_available() else "cpu"
U = json.load(open(HERE / "units.json"))
sents = lambda s: [x.strip() for x in re.split(r"(?<=[.?!])\s+", s) if x.strip()]

if which == "nllb":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    M = "facebook/nllb-200-distilled-1.3B"
    tok = AutoTokenizer.from_pretrained(M, src_lang="tam_Taml")
    model = AutoModelForSeq2SeqLM.from_pretrained(M, torch_dtype=torch.float16).eval().to(dev)
    @torch.no_grad()
    def t2t(text):
        x = tok(text, return_tensors="pt", truncation=True, max_length=512).to(dev)
        y = model.generate(**x, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"), max_new_tokens=512, num_beams=4)
        return tok.batch_decode(y, skip_special_tokens=True)[0]
    s2t = None
else:
    from transformers import AutoProcessor, SeamlessM4Tv2Model
    M = "facebook/seamless-m4t-v2-large"
    proc = AutoProcessor.from_pretrained(M)
    model = SeamlessM4Tv2Model.from_pretrained(M, torch_dtype=torch.float16).eval().to(dev)
    @torch.no_grad()
    def t2t(text):
        x = proc(text=text, src_lang="tam", return_tensors="pt").to(dev)
        y = model.generate(**x, tgt_lang="eng", generate_speech=False, max_new_tokens=512, num_beams=4)
        return proc.decode(y[0].tolist()[0], skip_special_tokens=True)
    @torch.no_grad()
    def s2t(y):
        x = proc(audios=y, sampling_rate=SR, return_tensors="pt").to(dev)
        x = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in x.items()}
        out = model.generate(**x, tgt_lang="eng", generate_speech=False, max_new_tokens=256)
        return proc.decode(out[0].tolist()[0], skip_special_tokens=True)
name = "nllb-1.3B" if which == "nllb" else "seamless-m4t-v2"
def run(tag, fn):
    p = HERE / "out" / f"{tag}.json"; res = json.loads(p.read_text()) if p.exists() else {}
    t0 = time.time()
    for u in U:
        if u["id"] not in res:
            res[u["id"]] = fn(u); p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(tag, f"{time.time()-t0:.0f}s", flush=True)
for inp in ["ours", "gemini"]:
    run(f"{name}-paragraph__{inp}", lambda u: t2t(u[inp]))
    run(f"{name}-sentences__{inp}", lambda u: " ".join(t2t(s) for s in sents(u[inp])))
if s2t:
    cache = {}
    def speech(u):
        if u["section"] not in cache: cache[u["section"]] = sf.read(AUDIO / f"{SECS[u['section']]}.wav", dtype="float32")[0]
        y = cache[u["section"]][int(u["start"] * SR): int(u["end"] * SR) + SR // 4]
        return " ".join(s2t(y[a:b]) for a, b in cut(y))
    run(f"{name}-speech__audio", speech)
