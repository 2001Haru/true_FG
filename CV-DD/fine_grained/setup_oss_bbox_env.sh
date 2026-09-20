#!/usr/bin/env bash
set -euo pipefail
ENV_ROOT="${ENV_ROOT:-/tmp/fg_oss_bbox_env}"
python -m venv --system-site-packages "$ENV_ROOT"
"$ENV_ROOT/bin/python" -m pip install --disable-pip-version-check \
  'hydra-core>=1.3.2' 'iopath>=0.1.10' addict yapf pycocotools
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "$ENV_ROOT/bin/python" - <<'PY'
import torch,torchvision,transformers,timm,hydra,iopath,addict,yapf
print({'torch':torch.__version__,'torchvision':torchvision.__version__,
       'transformers':transformers.__version__,'timm':timm.__version__})
PY
