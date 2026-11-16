"""Model runners. Each returns an object with .transcribe(wav_path) -> str.

All local models run on CPU (that's the deployment target), in float32.
"""
from __future__ import annotations

import os
import time

import numpy as np
import soundfile as sf
import torch

SR = 16000


def _load(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    assert sr == SR, f"{path}: expected 16 kHz, got {sr} (build manifests with data.py)"
    return y


class Whisper:
    """Any Whisper checkpoint on the HF hub, forced to Tamil transcription."""

    def __init__(self, repo: str):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.proc = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(repo, torch_dtype=torch.float32).eval()
        # Older fine-tunes ship forced_decoder_ids that clash with language=...
        gc = self.model.generation_config
        gc.forced_decoder_ids = None
        if getattr(gc, "lang_to_id", None) is None:  # very old configs: fall back to prompt ids
            self.prompt = self.proc.get_decoder_prompt_ids(language="tamil", task="transcribe")
        else:
            self.prompt = None

    @torch.inference_mode()
    def transcribe(self, path: str) -> str:
        y = _load(path)
        feats = self.proc(y, sampling_rate=SR, return_tensors="pt")
        kw = {"max_new_tokens": 440, "num_beams": 1}
        if self.prompt is None:
            kw.update(language="tamil", task="transcribe")
        else:
            kw["forced_decoder_ids"] = self.prompt
        ids = self.model.generate(feats.input_features, **kw)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


class IndicConformer:
    """ai4bharat/indic-conformer-600m-multilingual (gated: accept terms on HF)."""

    def __init__(self, decoder: str = "ctc"):
        from transformers import AutoModel

        self.decoder = decoder
        self.model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
        )

    @torch.inference_mode()
    def transcribe(self, path: str) -> str:
        wav = torch.from_numpy(_load(path)).unsqueeze(0)
        out = self.model(wav, "ta", self.decoder)
        return (out[0] if isinstance(out, (list, tuple)) else out).strip()


class SraVaani:
    """ARTPARK-IISc/SraVaani-1.0 (gated: accept terms on HF)."""

    def __init__(self, tamil_only: bool = False):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained("ARTPARK-IISc/SraVaani-1.0", trust_remote_code=True).eval()
        if tamil_only:
            self._restrict_to_tamil()

    def _restrict_to_tamil(self):
        """SraVaani has no language input; it guesses. Force Tamil by banning,
        at every decoding step, any token whose text contains a letter outside
        the Tamil Unicode block (digits/punctuation/Latin-free pieces stay)."""
        import types

        m = self.model
        m._ensure_loaded()
        sp = m._get_tokenizer()
        n = sp.get_piece_size()
        cfg = m.config
        ban = torch.zeros(cfg.blank_id + 1 + cfg.num_durations)
        for t in range(n):
            piece = sp.id_to_piece(t).replace("▁", "")
            if any(c.isalpha() and not ("஀" <= c <= "௿") for c in piece):
                ban[t] = float("-inf")
        self.banned = int(torch.isinf(ban).sum())

        @torch.no_grad()
        def _greedy_one(self_, enc_out, T):
            nd, blank, durs = cfg.num_durations, cfg.blank_id, cfg.durations
            dev, dt = self_._anchor.device, self_._io_dtype
            h = torch.zeros(cfg.pred_rnn_layers, 1, cfg.pred_hidden, device=dev, dtype=dt)
            c = torch.zeros(cfg.pred_rnn_layers, 1, cfg.pred_hidden, device=dev, dtype=dt)
            last, toks, frames = blank, [], []
            tlen = torch.ones(1, dtype=torch.int32, device=dev)
            b = ban.to(dev)
            t = 0
            while t < T:
                f = enc_out.narrow(2, t, 1)
                added, need = 0, True
                while need and added < cfg.max_symbols:
                    tgt = torch.tensor([[last]], dtype=torch.int32, device=dev)
                    logits, _, h2, c2 = self_.decoder_joint(f, tgt, tlen, h, c)
                    logits = logits[0, 0, 0].float()
                    tok_logits = logits[:-nd] + b[: logits.shape[0] - nd]
                    k = int(tok_logits.argmax().item())
                    skip = durs[int(logits[-nd:].argmax().item())]
                    if k != blank:
                        toks.append(k); frames.append(t); h, c, last = h2, c2, k
                    added += 1; t += skip; need = (skip == 0)
                if added == cfg.max_symbols:
                    t += 1
            return toks, frames

        m._greedy_one = types.MethodType(_greedy_one, m)

    @torch.inference_mode()
    def transcribe(self, path: str) -> str:
        h = self.model.transcribe([path], return_hypotheses=True)[0]
        return getattr(h, "text", str(h)).strip()


class IndicTranscribeModel:
    """bodhan-ai/indic-transcribe-core|flex (gated; AI4Bharat + Bodhan AI).
    mode: native (Tamil script), mixed (English words + numbers in Latin),
    romanized (everything in Latin = Tanglish). Clips must be <= 30 s."""

    def __init__(self, repo: str, mode: str = "native"):
        import sys

        from huggingface_hub import snapshot_download

        d = snapshot_download(repo, ignore_patterns=["nemo/*", "*.png"])
        for mod in [m for m in sys.modules if m in ("indic_transcribe", "long_form", "lid")
                    or m.startswith(("modeling_indic_canary", "tokenization_indic_canary",
                                     "feature_extraction_indic_canary", "configuration_indic_canary"))]:
            del sys.modules[mod]  # core and flex ship modules with the same names
        sys.path.insert(0, d)
        from indic_transcribe import IndicTranscribe

        self.asr = IndicTranscribe.from_pretrained(d, device=os.environ.get("ASR_DEVICE", "cpu"))
        self.mode = mode

    def transcribe(self, path: str) -> str:
        kw = {"lang": "ta"}
        if self.mode != "native":
            kw["mode"] = self.mode
        out = self.asr(path, **kw)
        return (out if isinstance(out, str) else getattr(out, "text", str(out))).strip()


class Hima:
    """Prahaladha/hima-asr (needs transformers>=5.13: run from .venv-hima)."""

    def __init__(self, style: str = "native"):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained("Prahaladha/hima-asr", trust_remote_code=True,
                                               dtype=torch.float32).eval()
        self.style = style

    @torch.inference_mode()
    def transcribe(self, path: str) -> str:
        out = self.model.transcribe(path, language="ta", style=self.style)
        return (out if isinstance(out, str) else str(out)).strip()


class Sarvam:
    """Sarvam speech-to-text REST API (sync endpoint, clips < 30 s).
    Needs SARVAM_API_KEY in the environment."""

    URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, model: str = "saaras:v3", mode: str | None = None):
        import requests

        self.s = requests.Session()
        self.key = os.environ["SARVAM_API_KEY"]
        self.model, self.mode = model, mode

    def transcribe(self, path: str) -> str:
        data = {"model": self.model, "language_code": "ta-IN"}
        if self.mode:
            data["mode"] = self.mode
        for attempt in range(7):
            with open(path, "rb") as f:
                r = self.s.post(self.URL, headers={"api-subscription-key": self.key},
                                data=data, files={"file": (os.path.basename(path), f, "audio/wav")},
                                timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(3 * 2 ** attempt)  # rate limited: back off up to ~3 min
                continue
            r.raise_for_status()
            time.sleep(0.5)  # stay under the free-tier rate limit
            return (r.json().get("transcript") or "").strip()
        r.raise_for_status()
        return ""


# name -> (factory, is_api)
REGISTRY = {
    "whisper-large-v3-turbo": (lambda: Whisper("openai/whisper-large-v3-turbo"), False),
    "whisper-tamil-small": (lambda: Whisper("vasista22/whisper-tamil-small"), False),
    "whisper-tamil-medium": (lambda: Whisper("vasista22/whisper-tamil-medium"), False),
    "indicconformer-ctc": (lambda: IndicConformer("ctc"), False),
    "indicconformer-rnnt": (lambda: IndicConformer("rnnt"), False),
    "sravaani": (lambda: SraVaani(), False),
    "sravaani-tamil": (lambda: SraVaani(tamil_only=True), False),
    "indic-transcribe-core": (lambda: IndicTranscribeModel("bodhan-ai/indic-transcribe-core"), False),
    "indic-transcribe-flex": (lambda: IndicTranscribeModel("bodhan-ai/indic-transcribe-flex"), False),
    "indic-transcribe-flex-mixed": (lambda: IndicTranscribeModel("bodhan-ai/indic-transcribe-flex", "mixed"), False),
    "indic-transcribe-flex-roman": (lambda: IndicTranscribeModel("bodhan-ai/indic-transcribe-flex", "romanized"), False),
    "hima-asr": (lambda: Hima(), False),
    "hima-asr-mixed": (lambda: Hima("mixed"), False),
    "sarvam-saaras-v3": (lambda: Sarvam("saaras:v3"), True),
    "sarvam-saaras-v3-codemix": (lambda: Sarvam("saaras:v3", "codemix"), True),
    "sarvam-saaras-v4": (lambda: Sarvam("saaras:v4"), True),
    "sarvam-saaras-v3-translit": (lambda: Sarvam("saaras:v3", "translit"), True),
}
