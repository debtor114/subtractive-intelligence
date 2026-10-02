# Subtractive Intelligence

Code, per-run logs and paper source for

> **What Survives the Translation from Brain to Von Neumann Machine: Learning Is Pruning, and Four Mechanisms That Did Not Transfer**
> Dongin Kang, Independent Researcher, 2026 (draft; arXiv link to follow). Repository: https://github.com/debtor114/subtractive-intelligence

Five brain mechanisms were ported to ordinary GPU training under one protocol and compared with matched baselines:
gradual synaptic pruning during learning, hippocampus--cortex sleep consolidation, thalamic routing, predictive-coding
token skipping, and STDP, plus one follow-up (per-input pruning at inference time). One principle transferred:
at matched final connection budgets, removing connections gradually *while* training beats training small, pruning once
after training and dynamic sparse training at tight budgets, with a margin that grows with sparsity (MNIST MLP, CIFAR-10
CNN, CIFAR-10 ResNet-18, and an ImageNet-pre-trained ResNet-18 adapted to CIFAR-10; from the pre-trained start the
standard schedule loses to one-shot pruning at 0.5%, which an earlier-ending schedule or RigL from the inherited mask
repairs). Four structures did not transfer (sleep
consolidation, thalamic routing, predictive coding, per-input pruning); the local pruning rules were mixed (about a
point behind per-layer magnitude pruning on the MLP, collapse on the CNN) and STDP reproduced its known accuracy
ceiling while sparsifying itself. Everything in the paper is produced from the stored run files in `results/`.

The paper PDF is `paper/main.pdf`; the Korean working report with every table and figure is `docs/report/REPORT.md`.

## Layout

| Path | Contents |
|---|---|
| `core/` | masked layers, pruning criteria and schedules (`pruning.py`), dynamic (per-sample) layers, predictive-coding and thalamic-routing modules |
| `baselines/` | MLP, VGG-style CNN, CIFAR ResNet-18, small ViT, shared training loop |
| `experiments/core_prune_during_learning/` | the main comparison (`run.py` MNIST, `run_cifar.py` CNN/ResNet-18, `run_pretrained.py` transfer) and the sweeps (`run_all*.py`) |
| `experiments/exp3_dual_learning/` | sleep consolidation (CLS) and its sparse redesign |
| `experiments/exp1_prerouting_attention/`, `exp2_predictive_coding/`, `exp12_prefilter/` | pre-routed attention, token skipping, thalamic pre-decision |
| `experiments/exp4_stdp_temporal/` | Diehl--Cook STDP network in pure PyTorch |
| `experiments/exp5_dynamic_pruning/` | input-dependent pruning at inference time |
| `results/` | one JSON per run (config, curves, pruning log, calibration, FLOPs) and the ViT/MLP checkpoints the skipping experiments start from |
| `scripts/analyze.py`, `scripts/analyze_exp12.py` | regenerate every table (`docs/report/tables/*.md`) and figure from `results/` |
| `scripts/md_tables_to_tex.py`, `scripts/make_arxiv_bundle.py` | appendix tables and the arXiv source bundle |
| `docs/` | report, design decisions (`docs/decisions/`), notes and work log (Korean) |
| `paper/` | LaTeX source, figures, generated tables, PDF |
| `runpod/` | shell scripts used to run the larger sweeps on a rented A40 |

## Setup

Python 3.10 or later and a CUDA build of PyTorch (2.x). Everything else is in `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick the index for your CUDA
pip install -r requirements.txt
python scripts/smoke_test.py                              # one-epoch MNIST run, ~10 s on a GPU
```

MNIST and CIFAR-10 are downloaded into `data/` on first use; the transfer experiment downloads the torchvision
ImageNet weights for ResNet-18.

## Reproducing the paper

```bash
# main comparison (MNIST MLP, 3 seeds, all arms and budgets)
python experiments/core_prune_during_learning/run_all.py --densities 0.1 0.05 0.02 0.01 0.005 --epochs 15 --skip_existing
# CIFAR-10 CNN and ResNet-18
python experiments/core_prune_during_learning/run_all_cifar.py --model cnn --skip_existing
python experiments/core_prune_during_learning/run_all_cifar.py --model resnet18 --densities 0.05 0.02 0.005 --seeds 0 1 2 --arms dense_small pd_mag_global pd_mag_erk rigl ttp --skip_existing
# ResNet-18 controls used in Tables 6 and 9: the 20% budget and the uniform-per-layer criterion (two seeds)
python experiments/core_prune_during_learning/run_all_cifar.py --model resnet18 --densities 0.2 --seeds 0 1 --arms dense_small pd_mag_global pd_mag_layer --skip_existing
python experiments/core_prune_during_learning/run_all_cifar.py --model resnet18 --densities 0.05 0.005 --seeds 0 1 --arms pd_mag_layer --skip_existing
# transfer from an ImageNet-pre-trained ResNet-18
python experiments/core_prune_during_learning/run_all_pretrained.py --skip_existing
# negative results
python experiments/exp3_dual_learning/run_all.py --skip_existing
python experiments/exp1_prerouting_attention/run_all.py
# baseline ViTs the skipping experiments start from (the checkpoints are included in results/; this retrains them)
for s in 0 1 2; do python scripts/train_baseline.py --config configs/vit_mnist.yaml --set seed=$s; done
python scripts/train_baseline.py --config configs/vit_cifar10.yaml
# predictive-coding token skipping (Tables 13-14): skip curves, then fine-tuning with predicted vs random skip scores
for s in 0 1 2; do python experiments/exp2_predictive_coding/run.py --ckpt "results/baseline_vit_mnist/seed${s}_*/model_final.pt" --dataset mnist --seed $s; done
for s in 0 1; do for sc in predicted random; do python experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_mnist/seed${s}_*/model_final.pt" --dataset mnist --score $sc --skip 0.5 --epochs 2 --seed $s; done; done
python experiments/exp2_predictive_coding/run.py --ckpt "results/baseline_vit_cifar10/seed0_*/model_final.pt" --dataset cifar10
for sc in predicted random; do python experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_cifar10/seed0_*/model_final.pt" --dataset cifar10 --score $sc --skip 0.5 --epochs 3; done
# thalamic pre-decision (Tables 15-16): three selectors, pre-routed late attention, and the identity-substitution control
for s in 0 1 2; do CK="results/baseline_vit_mnist/seed${s}_*/model_final.pt"
  for m in thalamic layerwise random; do python experiments/exp12_prefilter/run.py --ckpt "$CK" --dataset mnist --mode $m --seed $s; done
  python experiments/exp12_prefilter/run.py --ckpt "$CK" --dataset mnist --mode thalamic --late_attn pre --seed $s
  python experiments/exp12_prefilter/run.py --ckpt "$CK" --dataset mnist --mode thalamic --substitute identity --seed $s
done
CK="results/baseline_vit_cifar10/seed0_*/model_final.pt"
for m in thalamic layerwise random; do python experiments/exp12_prefilter/run.py --ckpt "$CK" --dataset cifar10 --mode $m --epochs 6; done
python experiments/exp12_prefilter/run.py --ckpt "$CK" --dataset cifar10 --mode thalamic --late_attn pre --epochs 6
python experiments/exp12_prefilter/run.py --ckpt "$CK" --dataset cifar10 --mode thalamic --substitute identity --epochs 6
python experiments/exp5_dynamic_pruning/run_all.py --skip_existing
python experiments/exp4_stdp_temporal/run.py
# controls added in revision: gradual prune-after, 3x RigL, shallow-and-wide dense small, late-block removal, magnitude-mask RigL,
# early-ending schedule (scripts/review9_controls.cmd runs the same list on Windows)
python experiments/core_prune_during_learning/run_all.py --arms ttp_gradual --densities 0.1 0.05 0.02 0.01 0.005 --epochs 15 --skip_dense_big --skip_existing
python experiments/core_prune_during_learning/run_all.py --arms rigl_x3 --densities 0.01 0.005 --epochs 15 --skip_dense_big --skip_existing
python experiments/core_prune_during_learning/run_all_cifar.py --model cnn --arms ttp_gradual dense_small_shallow --skip_dense_big --skip_existing
python experiments/core_prune_during_learning/run_all_cifar.py --model cnn --arms rigl_x3 --densities 0.01 0.03 --skip_dense_big --skip_existing
for s in 0 1 2; do python experiments/exp12_prefilter/run.py --ckpt "results/baseline_vit_mnist/seed${s}_*/model_final.pt" --dataset mnist --mode drop_late --smax 1.0 --seed $s; done
python experiments/exp12_prefilter/run.py --ckpt "results/baseline_vit_cifar10/seed0_*/model_final.pt" --dataset cifar10 --mode drop_late --smax 1.0 --epochs 6
for d in 0.05 0.02 0.005; do python experiments/core_prune_during_learning/run_pretrained.py --arm pt_rigl_mag --density $d; done
python experiments/core_prune_during_learning/run_pretrained.py --arm pt_rigl_mag --density 0.005 --seed 1
for s in 0 1; do python experiments/core_prune_during_learning/run_pretrained.py --arm pt_pd --density 0.005 --seed $s --set prune_end=0.5 tag=end50; done
# tables and figures, then the paper
python scripts/analyze.py && python scripts/analyze_exp12.py
PAPER_FIGS=1 python scripts/analyze.py core core_cifar core_pretrained && PAPER_FIGS=1 python scripts/analyze_exp12.py
python scripts/md_tables_to_tex.py && (cd paper && latexmk -pdf main.tex)
```

A single run takes seconds (MNIST) to minutes (ResNet-18, transfer) on a consumer GPU; the full set is a few GPU-days.
Every run writes `results/<experiment>/.../seed<k>.json`, and `--skip_existing` resumes a sweep.

## License

Code is released under the MIT License (`LICENSE`). The paper text and figures are CC BY 4.0. See `CITATION.cff` for
how to cite.
