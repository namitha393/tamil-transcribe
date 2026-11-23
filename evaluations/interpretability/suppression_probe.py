"""Where inside SraVaani does an attendee reply get suppressed?

Same 60 conversational replies, (a) alone and (b) after a 8-15 s clean read clip
("teacher") + 0.5 s gap, at equal loudness. We run the 17 encoder layers by hand.

1. Similarity: per layer, cosine similarity between the reply's frames when alone
   and the same frames inside the mix.
2. Blank probability: how strongly the model's CTC head says "nothing here" on the
   reply frames, alone vs in the mix.
3. Activation patching (causal): run the mix, but at layer L replace the reply's
   frames with the 'alone' version, continue to the end, decode. If the reply comes
   back, the suppression happens AFTER layer L.
"""
import json, random, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
from transformers import AutoModel
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from normalize import normalize
torch.set_num_threads(8)
D = Path(__file__).resolve().parents[3] / "tamil_asr_eval" / "data"
m = AutoModel.from_pretrained("ARTPARK-IISc/SraVaani-1.0", trust_remote_code=True).eval(); m._ensure_loaded()
sp, enc = m._get_tokenizer(), m.encoder
L = [getattr(enc.layers, str(i)) for i in range(17)]
act = L[16].feed_forward2.activation
SUB = 8 * m._p["hop_length"]  # samples per encoder frame (80 ms)

# --- the same 60 pairs as the earlier 'mix_wide' experiment (same seed and filters)
rng = random.Random(3)
kb = [json.loads(l) for l in open(D / "kathbath_ta/manifest.jsonl") if json.loads(l)["duration"] >= 8]
iv = [json.loads(l) for l in open(D / "indicvoices_ta/manifest.jsonl")
      if 1.0 <= json.loads(l)["duration"] <= 4 and len(json.loads(l)["text"].split()) >= 2]
pairs = [(rng.choice(kb), x) for x in iv[:60]]

def load(r):
    return sf.read(D / r["audio"].split("/data/")[-1], dtype="float32")[0] if "/data/" in r["audio"] else sf.read(r["audio"], dtype="float32")[0]

@torch.no_grad()
def layers_out(y):
    wav = torch.from_numpy(y).unsqueeze(0)
    feats, flen = m.extract_features(wav, torch.tensor([wav.shape[1]]))
    plen, x = enc.pre_encode.forward(flen.to(torch.int64), feats.transpose(1, 2).to(m._io_dtype))
    x, pos = enc.pos_enc.forward(x)
    T = x.shape[1]
    mask = torch.zeros(1, T, T, dtype=torch.bool); pad = torch.zeros(1, T, dtype=torch.bool)
    outs = [x]
    for i, l in enumerate(L):
        x = l.forward(act, x, pos, mask, pad) if i < 16 else l.forward(x, pos, mask, pad)
        outs.append(x)
    return outs, pos, mask, pad   # outs[0] = input to layer 0, outs[k] = output of layer k-1

@torch.no_grad()
def run_from(x, k, pos, mask, pad):
    for i in range(k, 17):
        x = L[i].forward(act, x, pos, mask, pad) if i < 16 else L[i].forward(x, pos, mask, pad)
    return x

@torch.no_grad()
def decode(x):
    e = x.transpose(1, 2)
    toks, _ = m._greedy_one(e, e.shape[2])
    return sp.decode(toks)

def found(text, ref):
    hw = set(normalize(text).split()); aw = normalize(ref).split()
    return sum(w in hw for w in aw), len(aw)

ctc = getattr(m, "ctc_decoder", None) or getattr(m, "decoder_ctc", None)
sims = np.zeros((len(pairs), 18)); recov = np.zeros((18, 2)); base = np.zeros(2); alone_hits = np.zeros(2)
gap = np.zeros(8000, np.float32)
for n, (t, a) in enumerate(pairs):
    yt, ya = load(t), load(a)
    ya = ya / (np.sqrt((ya ** 2).mean()) + 1e-9) * np.sqrt((yt ** 2).mean())
    mix = np.concatenate([yt, gap, ya])
    oa, *_ = layers_out(ya)
    om, pos, mask, pad = layers_out(mix)
    off = (len(yt) + len(gap)) // SUB
    k = min(oa[0].shape[1], om[0].shape[1] - off)
    for j in range(18):
        A, M = oa[j][0, :k], om[j][0, off:off + k]
        sims[n, j] = torch.nn.functional.cosine_similarity(A, M, dim=-1).mean().item()
    h, tot = found(decode(om[-1]), a["text"]); base += (h, tot)
    h, tot = found(decode(oa[-1]), a["text"]); alone_hits += (h, tot)
    for j in range(18):   # patch reply frames at the input of layer j (j=17: after the last layer)
        x = om[j].clone(); x[0, off:off + k] = oa[j][0, :k]
        h, tot = found(decode(run_from(x, j, pos, mask, pad)), a["text"]); recov[j] += (h, tot)
    if n % 10 == 9: print(f"{n+1}/60", flush=True)
res = {"reply_alone_found": alone_hits[0] / alone_hits[1], "reply_in_mix_found": base[0] / base[1],
       "similarity_by_layer": sims.mean(0).tolist(), "patched_found_by_layer": (recov[:, 0] / recov[:, 1]).tolist()}
Path("suppression_probe.json").write_text(json.dumps(res, indent=1))
print(f"reply words found: alone {100*res['reply_alone_found']:.1f}%  in mix {100*res['reply_in_mix_found']:.1f}%")
print("layer | similarity alone vs mix | words found if reply patched in at this layer's input")
for j in range(18):
    name = "input(after subsampling)" if j == 0 else ("after layer 17 (final)" if j == 17 else f"before layer {j+1}")
    print(f"{name:26s} {res['similarity_by_layer'][j]:.3f}   {100*res['patched_found_by_layer'][j]:5.1f}%")
