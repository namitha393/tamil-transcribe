#!/usr/bin/env bash
# Tamil recording -> Tamil + Tanglish transcripts, in one command:
#
#     ./transcribe.sh "class recording.mp3"
#
# The first run sets everything up (a few minutes, one time). Results are saved in a
# folder next to the audio file: "class recording_transcript/".
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ $# -eq 0 ]; then
  echo "Usage: ./transcribe.sh <audio file> [more audio files...]"
  echo "Works with mp3, m4a, mp4, wav, ogg, opus, WhatsApp audio, ..."
  exit 1
fi

# 1. uv: a small tool that installs the right Python and libraries (one time)
if ! command -v uv >/dev/null 2>&1; then
  echo ">> Installing uv (one time)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Python environment with the tested library versions (one time, ~1.5 GB)
if [ ! -x "$HERE/.venv/bin/python" ]; then
  echo ">> Setting up (one time)..."
  uv venv --quiet --python 3.12 "$HERE/.venv"
  VIRTUAL_ENV="$HERE/.venv" uv pip install --quiet -r "$HERE/requirements.txt"
fi
PY="$HERE/.venv/bin/python"

# 3. Access to the SraVaani model on Hugging Face (free; one time)
if ! "$PY" -c "from huggingface_hub import hf_hub_download as d; d('ARTPARK-IISc/SraVaani-1.0','config.json')" >/dev/null 2>&1; then
  echo ""
  echo ">> One-time step: the SraVaani model needs a free Hugging Face account."
  echo "   1) Open https://huggingface.co/ARTPARK-IISc/SraVaani-1.0 , sign in, and accept the terms."
  echo "   2) Create a 'Read' token at https://huggingface.co/settings/tokens and paste it below."
  echo ""
  "$HERE/.venv/bin/hf" auth login
fi

# 4. Transcribe
PYTHONPATH="$HERE" exec "$PY" -m pipeline.transcribe "$@"
