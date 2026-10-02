@echo off
rem 9 차 리뷰 대조군 (로컬 3060 Ti, 순차). 우선순위 순: 학습 후 점진 가지치기 -> RigL 3 배 학습 -> late block 제거 -> CNN 대조군 -> 사전학습 RigL/스케줄.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\_review9.log
echo ==== review9 controls start %date% %time% > %LOG%
rem 1. MNIST: prune-after with gradual schedule (same total steps as prune-after), 5 budgets x 3 seeds
%PY% -u experiments\core_prune_during_learning\run_all.py --arms ttp_gradual --densities 0.1 0.05 0.02 0.01 0.005 --seeds 0 1 2 --epochs 15 --skip_dense_big --skip_existing >> %LOG% 2>&1
rem 2. MNIST: RigL with 3x training length at the two tightest budgets
%PY% -u experiments\core_prune_during_learning\run_all.py --arms rigl_x3 --densities 0.01 0.005 --seeds 0 1 2 --epochs 15 --skip_dense_big --skip_existing >> %LOG% 2>&1
rem 3. token skipping: late blocks removed + same fine-tuning (MNIST 3 seeds, CIFAR-10 1 seed)
for %%s in (0 1 2) do %PY% -u experiments\exp12_prefilter\run.py --ckpt "results/baseline_vit_mnist/seed%%s_*/model_final.pt" --dataset mnist --mode drop_late --smax 1.0 --epochs 4 --seed %%s >> %LOG% 2>&1
%PY% -u experiments\exp12_prefilter\run.py --ckpt "results/baseline_vit_cifar10/seed0_*/model_final.pt" --dataset cifar10 --mode drop_late --smax 1.0 --epochs 6 --seed 0 >> %LOG% 2>&1
rem 4. CNN: prune-after with gradual schedule, 3 budgets x 3 seeds
%PY% -u experiments\core_prune_during_learning\run_all_cifar.py --model cnn --arms ttp_gradual --densities 0.1 0.03 0.01 --seeds 0 1 2 --skip_dense_big --skip_existing >> %LOG% 2>&1
rem 5. CNN: RigL with 3x training length at 1% and 3%
%PY% -u experiments\core_prune_during_learning\run_all_cifar.py --model cnn --arms rigl_x3 --densities 0.01 0.03 --seeds 0 1 2 --skip_dense_big --skip_existing >> %LOG% 2>&1
rem 6. CNN: shallow-and-wide dense small (4 conv layers) at the same budgets
%PY% -u experiments\core_prune_during_learning\run_all_cifar.py --model cnn --arms dense_small_shallow --densities 0.1 0.03 0.01 --seeds 0 1 2 --skip_dense_big --skip_existing >> %LOG% 2>&1
rem 7. transfer: RigL from the pre-trained magnitude mask (keeps inherited structure), and pruning that ends at 50%% of adaptation at 0.5%%
%PY% -u experiments\core_prune_during_learning\run_pretrained.py --arm pt_rigl_mag --density 0.005 --seed 0 >> %LOG% 2>&1
%PY% -u experiments\core_prune_during_learning\run_pretrained.py --arm pt_pd --density 0.005 --seed 0 --set prune_end=0.5 tag=end50 >> %LOG% 2>&1
%PY% -u experiments\core_prune_during_learning\run_pretrained.py --arm pt_rigl_mag --density 0.05 --seed 0 >> %LOG% 2>&1
%PY% -u experiments\core_prune_during_learning\run_pretrained.py --arm pt_rigl_mag --density 0.02 --seed 0 >> %LOG% 2>&1
%PY% -u experiments\core_prune_during_learning\run_pretrained.py --arm pt_rigl_mag --density 0.005 --seed 1 >> %LOG% 2>&1
%PY% -u experiments\core_prune_during_learning\run_pretrained.py --arm pt_pd --density 0.005 --seed 1 --set prune_end=0.5 tag=end50 >> %LOG% 2>&1
echo REVIEW9_CONTROLS_DONE %date% %time% >> %LOG%
