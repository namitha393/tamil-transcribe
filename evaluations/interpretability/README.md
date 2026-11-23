# Why SraVaani drops attendee replies — inside the model

**The problem:** SraVaani transcribes a short conversational reply when it is on its own
(68% of its words found) but mostly drops the same reply when it follows a clean dominant
voice in the same piece (16%). Already ruled out: loudness, phone bandwidth, decoder frame
skipping, feature normalisation alone (see `../../sravaani_eval/README.md`).

## 1. Where it happens — `suppression_probe.py`
60 pairs: the same reply alone vs after an 8–15 s read clip ("teacher") + 0.5 s gap.
SraVaani's encoder is 17 Conformer layers, each attending over the **whole** piece.
We ran the layers by hand and compared/patched the reply's frames.

| Swap in the "reply alone" version at… | Similarity alone vs mix | Reply words found |
|---|--:|--:|
| (no swap — the mix as is) | | 15.6% |
| input (after subsampling) | 0.37 | 21.2% |
| before layer 3 | 0.45 | 33.0% |
| before layer 6 | 0.51 | 40.2% |
| before layer 9 | 0.60 | 52.0% |
| before layer 12 | 0.63 | 57.3% |
| before layer 17 (last) | 0.60 | 66.5% |
| (reply alone) | | 67.9% |

Reading it: fixing the input alone (which includes the whole-piece feature normalisation)
recovers little (21%). The later we swap in the clean version, the more comes back — so the
suppression is **built up gradually by every layer's self-attention mixing in the dominant
voice's context**, not by one "switch" layer or one direction. That makes a single-direction
weight edit unlikely to work, and points at the attention context itself.

## 2. A run-time fix — `local_attention.py`
Limit each frame's self-attention to a local window (an attention-mask change at run time;
**no weights changed**), so the reply cannot attend to the teacher seconds earlier.

| Attention window | Replies found in mix | Reply alone | Section A CER vs Gemini | Words |
|---|--:|--:|--:|--:|
| whole piece (original) | 15.6% | 67.9% | 23.1% | 423 |
| ±10 s | 30.4% | 67.9% | 22.1% | 444 |
| **±7.5 s** | **46.9%** | 67.9% | **21.9%** | 457 |
| ±5 s | 55.6% | 67.9% | 24.1% | 458 |
| ±2.5 s | 48.3% | 66.5% | 28.0% | 460 |

±7.5 s triples the recovered replies **and** improves real class accuracy (Core: 18.9%).
Not yet checked on sections B/C, public sets and long-form tests before adopting it.

## 3. How this relates to "weight-space programming"
- **Original:** Takehi, Teoh, Zhang, Khattab, *Sparks of Weight-Space Programming* (MIT, Nov 2026,
  https://rikiyat.github.io/blog/weight-space-programming/): a frontier LLM reads and hand-edits
  raw weights without gradients. Striking results (e.g. removing one MLP direction in Qwen3.5-0.8B
  fixed reasoning loops: MATH-500 7.9% → 48.1%), but the authors frame it as an emerging
  capability, not a method; edits remove obstacles rather than add missing abilities.
- **Follow-ups** (community, not peer-reviewed; numbers inconsistent): fact injection vs LoRA —
  edits are very local (little collateral damage) but cost far more and vary run to run; LoRA
  usually has higher recall; putting knowledge in context often wins. Hu et al. (arXiv 2611.09382)
  found big gains on 0.8B, small on 9B, and edits that failed on held-out data.
- **Speech:** an agent-edit study on Qwen3-ASR found the "suppression" it was patching was really a
  **missing attention mask** in the inference backend — restoring the mask fixed as much as the
  best edit. Our result rhymes with that: the lever here is the **attention context (mask)**,
  found by activation patching (the method of Glazer et al., arXiv 2508.15882).

**What transfers:** discovery by activation patching ✓ (done above); a single-direction edit ✗
(suppression is distributed); run-time attention masking ✓ (works now); targeted fine-tuning on
synthetic teacher+reply mixes — the robust next step if we want more than 47%. For rare
astrology words the established cheap fix is text-only adaptation of the prediction/joint network
(Pylkkönen et al. 2104.11127; ILMA 2110.05354), plus glossary biasing.
