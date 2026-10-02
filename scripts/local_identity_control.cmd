@echo off
rem 토큰 스킵 대조군: 뒤쪽 블록만 스킵 + identity 치환 + 같은 점진 미세조정 (thalamic 선택). MNIST 3 시드, CIFAR 1 시드.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
echo ==== identity control start %date% %time% > results\_local_identity.log
for %%s in (0 1 2) do .venv\Scripts\python.exe -u experiments\exp12_prefilter\run.py --ckpt "results/baseline_vit_mnist/seed%%s_*/model_final.pt" --dataset mnist --mode thalamic --late_attn full --substitute identity --smax 0.7 --epochs 4 --seed %%s >> results\_local_identity.log 2>&1
.venv\Scripts\python.exe -u experiments\exp12_prefilter\run.py --ckpt "results/baseline_vit_cifar10/seed0_*/model_final.pt" --dataset cifar10 --mode thalamic --late_attn full --substitute identity --smax 0.7 --epochs 6 --seed 0 >> results\_local_identity.log 2>&1
echo IDENTITY_CONTROL_DONE %date% %time% >> results\_local_identity.log
