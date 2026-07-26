#!/bin/bash
set -e

PIP="${PIP:-python -m pip}"
FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-}"

echo "1. install inference frameworks and pytorch"
if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq python3-dev build-essential
else
  echo "Skip apt-get setup; install python3-dev/build-essential manually if a wheel needs local compilation."
fi

echo "1.0. Pin installer tooling"
$PIP install --upgrade pip "setuptools<81" wheel

echo "1.1. Check existing PyTorch and CUDA"
python - <<'PY' || true
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
except ImportError:
    print("PyTorch is not installed yet.")
PY

echo "1.1.1. Install PyTorch 2.6.0"
$PIP install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
$PIP install psutil pandas

echo "1.2. Install flash-attn"
if [ -f "$FLASH_ATTN_WHL" ]; then
  $PIP install --no-index --no-deps "$FLASH_ATTN_WHL"
else
  $PIP install flash-attn==2.7.4.post1 --no-build-isolation
fi

echo "2. install verl"
$PIP install accelerate

echo "3. install requirements and vllm"
$PIP install -r requirements.txt
$PIP install -e ".[math]"

echo "3.1. Install vLLM"
$PIP install vllm==0.8.2 "debugpy>=1.8"

echo "5. Verify installation"
python -c "
import torch
import vllm
import ray
import grpc
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ vLLM: {vllm.__version__}')
print(f'✓ Ray: {ray.__version__}')
print(f'✓ grpcio: {grpc.__version__}')
print('✓ Success!')
"
