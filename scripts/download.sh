#!/bin/bash
# Download the pinned NVFP4 checkpoint (102 GB) into the Hugging Face cache. export.sh would also fetch it.
set -euo pipefail
. "$(dirname "$0")/config.sh"
export PATH=$HOME/.local/bin:$PATH
uvx --from huggingface_hub hf download "$CHECKPOINT" --revision "$CHECKPOINT_REVISION" --max-workers 16
