# Reproducible environment setup (Windows, PowerShell 5.1).
# Usage:  powershell -ExecutionPolicy Bypass -File setup_env.ps1
# Requires Python 3.10 registered with the py launcher and an NVIDIA driver supporting CUDA 12.8.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".venv")) {
    py -3.10 -m venv .venv
}
$py = Join-Path $root ".venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements.txt

& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
& $py scripts\smoke_test.py
