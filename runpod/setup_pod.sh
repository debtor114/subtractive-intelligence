#!/usr/bin/env bash
# RunPod 팟에서 1 회 실행: 환경 세팅. /workspace/subtractive-intelligence 에 코드가 풀려 있다고 가정.
#   bash runpod/setup_pod.sh
# 함정 (memory runpod_vllm_gotchas): 팟 드라이버가 CUDA 12.8 이면 torch 는 cu128 인덱스로. 기본 pip 는 cu13 을 깐다.
set -e
cd /workspace/subtractive-intelligence
nvidia-smi | head -4
python3 -m venv .venv_pod
source .venv_pod/bin/activate
pip install --upgrade pip -q
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRV:-0}" -ge 580 ]; then
  pip install torch torchvision -q
else
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q
fi
pip install torch-geometric snntorch matplotlib scipy tqdm tensorboard pyyaml pandas -q
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 데이터 미리 받기 (MNIST, CIFAR-10) + 사전 학습 ResNet-18 가중치
python - <<'EOF'
from torchvision import datasets
from torchvision.models import ResNet18_Weights, resnet18
datasets.MNIST("data", train=True, download=True); datasets.MNIST("data", train=False, download=True)
datasets.CIFAR10("data", train=True, download=True); datasets.CIFAR10("data", train=False, download=True)
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print("data ready")
EOF
python scripts/smoke_test.py | tail -3
echo "SETUP_OK"
