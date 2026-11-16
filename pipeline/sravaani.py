"""Component 3: speech -> Tamil words, with SraVaani-1.0 (ARTPARK + IISc, MIT licence).

Two fixes on top of the plain model:
  * Tamil letters only. SraVaani covers 65 languages and has no language
    setting; on ~3% of class pieces it wrote Malayalam/Hindi/Bengali script.
    We block every non-Tamil-letter token while it decodes (fixed 7 of 9).
  * Empty-piece retry. If a non-trivial piece comes back empty, split it at its
    quietest point and transcribe the halves (up to 2 levels).
Returns every word with start/end times in seconds.
"""
from __future__ import annotations

import types

import numpy as np
import torch

from .audio import SR

MODEL_ID = "ARTPARK-IISc/SraVaani-1.0"


class SraVaaniTamil:
    def __init__(self, threads: int = 8):
        from transformers import AutoModel

        torch.set_num_threads(threads)
        self.model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).eval()
        self._tamil_only()

    def _tamil_only(self):
        m = self.model
        m._ensure_loaded()
        sp, cfg = m._get_tokenizer(), m.config
        ban = torch.zeros(cfg.blank_id + 1 + cfg.num_durations)
        for t in range(sp.get_piece_size()):
            piece = sp.id_to_piece(t).replace("▁", "")
            if any(c.isalpha() and not ("஀" <= c <= "௿") for c in piece):
                ban[t] = float("-inf")

        @torch.no_grad()
        def greedy(self_, enc_out, T):  # same as the model's own greedy decoder, plus the ban
            nd, blank, durs = cfg.num_durations, cfg.blank_id, cfg.durations
            dev, dt = self_._anchor.device, self_._io_dtype
            h = torch.zeros(cfg.pred_rnn_layers, 1, cfg.pred_hidden, device=dev, dtype=dt)
            c = torch.zeros(cfg.pred_rnn_layers, 1, cfg.pred_hidden, device=dev, dtype=dt)
            last, toks, frames = blank, [], []
            tlen = torch.ones(1, dtype=torch.int32, device=dev)
            b, t = ban.to(dev), 0
            while t < T:
                f = enc_out.narrow(2, t, 1)
                added, need = 0, True
                while need and added < cfg.max_symbols:
                    logits, _, h2, c2 = self_.decoder_joint(f, torch.tensor([[last]], dtype=torch.int32, device=dev),
                                                            tlen, h, c)
                    logits = logits[0, 0, 0].float()
                    k = int((logits[:-nd] + b[: logits.shape[0] - nd]).argmax().item())
                    skip = durs[int(logits[-nd:].argmax().item())]
                    if k != blank:
                        toks.append(k); frames.append(t); h, c, last = h2, c2, k
                    added += 1; t += skip; need = skip == 0
                if added == cfg.max_symbols:
                    t += 1
            return toks, frames

        m._greedy_one = types.MethodType(greedy, m)

    def words(self, y: np.ndarray, offset: float = 0.0, depth: int = 0) -> list[dict]:
        hyp = self.model.transcribe([y], return_hypotheses=True, timestamps=True)[0]
        ws = [{"w": w["word"].strip(), "start": round(offset + w["start"], 2), "end": round(offset + w["end"], 2)}
              for w in hyp.timestamp["word"] if w["word"].strip()]
        if ws or depth >= 2 or len(y) < 3 * SR:
            return ws
        h = len(y) // 2
        e = np.abs(y[h - SR: h + SR])
        mid = h - SR + int(np.argmin(np.convolve(e, np.ones(1600), "same")))
        return self.words(y[:mid], offset, depth + 1) + self.words(y[mid:], offset + mid / SR, depth + 1)
