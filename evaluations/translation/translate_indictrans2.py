"""IndicTrans2 (AI4Bharat, MIT) Tamil->English, offline.
Two ways: sentence by sentence (what it was trained on) and the whole paragraph at once."""
import json, re, sys, time, torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
MODEL = sys.argv[1] if len(sys.argv) > 1 else "ai4bharat/indictrans2-indic-en-dist-200M"
tag = "indictrans2-" + ("200M" if "200M" in MODEL else "1B")
torch.set_num_threads(8)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True).eval().to(DEV)  # half precision on the Mac GPU: ~2 GB
ip = IndicProcessor(inference=True)
def tr(batch):
    b = ip.preprocess_batch(batch, src_lang="tam_Taml", tgt_lang="eng_Latn")
    x = tok(b, truncation=True, padding="longest", return_tensors="pt", max_length=256).to(DEV)
    with torch.no_grad():
        out = model.generate(**x, max_length=256, num_beams=5, num_return_sequences=1, use_cache=False)
    dec = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ip.postprocess_batch(dec, lang="eng_Latn")
U = json.load(open("units.json"))
for inp in ["ours", "gemini"]:
    t0 = time.time()
    pp, sp_ = Path(f"out/{tag}-paragraph__{inp}.json"), Path(f"out/{tag}-sentences__{inp}.json")
    para = json.loads(pp.read_text()) if pp.exists() else {}; sent = json.loads(sp_.read_text()) if sp_.exists() else {}
    for u in U:  # saved after every unit, resumable
        if u["id"] in para and u["id"] in sent: continue
        para[u["id"]] = tr([u[inp]])[0]
        ss = [s.strip() for s in re.split(r"(?<=[.?!])\s+", u[inp]) if s.strip()]
        sent[u["id"]] = " ".join(tr(ss))
        pp.write_text(json.dumps(para, ensure_ascii=False, indent=1)); sp_.write_text(json.dumps(sent, ensure_ascii=False, indent=1))
        print(tag, inp, u["id"], f"{time.time()-t0:.0f}s", flush=True)
    print(tag, inp, "done", f"{time.time()-t0:.0f}s", flush=True)
