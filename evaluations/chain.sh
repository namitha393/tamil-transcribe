#!/bin/zsh
# Round 2 runs, in order. Logs to chain.log.
set -x
P=.venv/bin/python
SETS="fleurs_ta indicvoices_ta kathbath_ta astro_class"
$P run.py --models sravaani-tamil --sets lf_fixed25 lf_pause25 lf_pause15 lf_silero15 lf_silero25 --threads 8
ASR_DEVICE=mps $P run.py --models indic-transcribe-core --sets ${=SETS} --threads 8
rm -rf .hf_models
ASR_DEVICE=mps $P run.py --models indic-transcribe-flex --sets ${=SETS} --threads 8
ASR_DEVICE=mps $P run.py --models indic-transcribe-flex-mixed indic-transcribe-flex-roman --sets astro_class --threads 8
rm -rf .hf_models
.venv-hima/bin/python run.py --models hima-asr --sets ${=SETS} --threads 8
.venv-hima/bin/python run.py --models hima-asr-mixed --sets astro_class --threads 8
rm -rf .hf_models
echo CHAIN DONE
