"""IndicTrans2 (AI4Bharat, MIT) Tamil->English, offline.
Two ways: sentence by sentence (what it was trained on) and the whole paragraph at once."""
import json, re, sys, time, torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
MODEL = sys.argv[1] if len(sys.argv) > 1 else "ai4bharat/indictrans2-indic-en-dist-200M"
tag = "indictrans2-" + ("200M" if "200M" in MODEL else "1B")
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True).eval()
ip = IndicProcessor(inference=True)
def tr(batch):
    b = ip.preprocess_batch(batch, src_lang="tam_Taml", tgt_lang="eng_Latn")
    x = tok(b, truncation=True, padding="longest", return_tensors="pt", max_length=256)
    with torch.no_grad():
        out = model.generate(**x, max_length=256, num_beams=5, num_return_sequences=1, use_cache=False)
    dec = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ip.postprocess_batch(dec, lang="eng_Latn")
U = json.load(open("units.json"))
for inp in ["ours", "gemini"]:
    t0 = time.time(); para, sent = {}, {}
    for u in U:
        para[u["id"]] = tr([u[inp]])[0]
        ss = [s.strip() for s in re.split(r"(?<=[.?!])\s+", u[inp]) if s.strip()]
        sent[u["id"]] = " ".join(tr(ss))
    Path(f"out/{tag}-paragraph__{inp}.json").write_text(json.dumps(para, ensure_ascii=False, indent=1))
    Path(f"out/{tag}-sentences__{inp}.json").write_text(json.dumps(sent, ensure_ascii=False, indent=1))
    print(tag, inp, f"{time.time()-t0:.0f}s", flush=True)
