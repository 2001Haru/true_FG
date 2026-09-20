#!/usr/bin/env bash
set -euo pipefail
ENV_ROOT="${ENV_ROOT:-/tmp/fg_oss_bbox_deps}"
GROUNDING_ROOT="${GROUNDING_ROOT:-}"
mkdir -p "$ENV_ROOT"
python -m pip install --disable-pip-version-check --target "$ENV_ROOT" \
  'hydra-core>=1.3.2' 'iopath>=0.1.10' addict yapf
PYTHONPATH="$ENV_ROOT${PYTHONPATH:+:$PYTHONPATH}" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python - <<'PY'
import torch,torchvision,transformers,timm,hydra,iopath,addict,yapf
print({'torch':torch.__version__,'torchvision':torchvision.__version__,
       'transformers':transformers.__version__,'timm':timm.__version__})
PY
if [[ -n "$GROUNDING_ROOT" && -x /usr/local/cuda/bin/nvcc ]]; then
  (cd "$GROUNDING_ROOT" && CUDA_HOME=/usr/local/cuda python setup.py build_ext --inplace)
fi
