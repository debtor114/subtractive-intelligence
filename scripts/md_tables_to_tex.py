# -*- coding: utf-8 -*-
"""docs/report/tables/*.md (analyze.py 출력) -> paper/tables/*.tex (booktabs) 변환 + 부록 자리 채우기.

  python scripts/md_tables_to_tex.py            # 표 변환 + paper/tables/all.tex 생성
  python scripts/md_tables_to_tex.py --patch    # 추가로 main.tex 의 부록 TODO 줄을 \\input{tables/all} 로 교체

표 밑의 한국어 주석은 버리고 (pdflatex 는 한글을 못 찍음) 캡션은 여기서 영어로 단다. 셀에 한글이 남으면 경고를 낸다.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(REPO, "docs", "report", "tables")
OUT = os.path.join(REPO, "paper", "tables")
MAIN = os.path.join(REPO, "paper", "main.tex")

# (md 이름, label, 캡션, 헤더 덮어쓰기 {열 번호: 영어 헤더})
SKIP_HEADERS = {2: "no skip", 3: "warm-up only, skip $s_{\\max}$", 4: "skip 0", 5: "skip 0.3", 6: "skip 0.5", 7: "skip 0.7",
                8: "skip 0.8", 9: "skip 0.9", 10: "FLOPs ratio (by skip level)"}
SPECS = [
    ("core_summary", "tab:app-mnist",
     "MNIST MLP (784-1024-1024-10, 1.86M weights), all arms and budgets, 3 seeds. Density is the final active fraction of the "
     "over-parameterised network; dense small is a width-matched network at the same budget. Training FLOPs count three forward "
     "passes per step at the actual density of that step.", {8: "samples to 97\\%"}),
    ("core_summary_cifar", "tab:app-cifar",
     "CIFAR-10 CNN (2.2M weights), all arms and budgets, 3 seeds. Columns as in Table~\\ref{tab:app-mnist}; the data-efficiency "
     "column is the number of training samples until 85\\% test accuracy.", {8: "samples to 85\\%"}),
    ("core_summary_resnet", "tab:app-resnet",
     "CIFAR-10 ResNet-18 (11.2M weights), all arms and budgets. Columns as in Table~\\ref{tab:app-mnist}; samples to 85\\% test "
     "accuracy.", {8: "samples to 85\\%"}),
    ("core_pretrained_summary", "tab:app-pretrained",
     "ImageNet-pre-trained ResNet-18 adapted to CIFAR-10 (128$\\times$128 input, 10 epochs) while being pruned. Density is "
     "relative to the 11.17M convolutional weights; the classifier is kept dense in every arm. Init acc is the accuracy before "
     "adaptation (fresh classifier). Adaptation FLOPs exclude the ImageNet pre-training, which is common to all pre-trained arms.",
     {}),
    ("exp3_split_mnist", "tab:app-split",
     "Split MNIST, class-incremental (single 10-way head, 5 tasks of 2 classes), 3 seeds. Rows marked [predict: \\dots] use an "
     "alternative read-out of the same run. Unknown-class AUROC: how well the confidence separates test samples of classes not yet "
     "learned (0.5 = chance).", {}),
    ("exp3_permuted_mnist", "tab:app-permuted",
     "Permuted MNIST, domain-incremental (10 tasks), 3 seeds. Columns as in Table~\\ref{tab:app-split}.", {}),
    ("exp2_mnist", "tab:app-skip-mnist",
     "MNIST ViT: post-hoc token skipping without fine-tuning, 3 seeds. Score: which tokens are skipped (predicted change, random, "
     "or the oracle true residual norm); substitute: what replaces a skipped token (identity, identity in the late blocks only, or "
     "the predicted residual).", {}),
    ("exp2_cifar10", "tab:app-skip-cifar",
     "CIFAR-10 ViT: post-hoc token skipping without fine-tuning, one seed. Columns as in Table~\\ref{tab:app-skip-mnist}.", {}),
    ("exp12_mnist", "tab:app-exp12-mnist",
     "MNIST ViT: token skipping in the late blocks. Skipped tokens receive the predicted residual, except in the rows marked "
     "identity substitution, where they are left unchanged. Each cell is test accuracy at the given skip ratio after fine-tuning "
     "that ramps the skip ratio from 0 to 0.7; the warm-up-only column is measured after the one-epoch warm-up of predictor and "
     "router with the ViT frozen, at the maximum skip ratio and before fine-tuning. Rows marked (eval thalamic) evaluate the same model with the "
     "thalamic router choosing the skipped tokens. For the pre-routed variant (late attn pre) the no-skip and warm-up columns "
     "are measured right after the late-block attention is swapped for pre-routed attention and before fine-tuning, which is "
     "why they are low; the skip columns are after fine-tuning.", SKIP_HEADERS),
    ("exp12_cifar10", "tab:app-exp12-cifar",
     "CIFAR-10 ViT: token skipping in the late blocks, as in Table~\\ref{tab:app-exp12-mnist}. late attn pre replaces the "
     "late-block attention by pre-routed attention (25\\% of keys).", SKIP_HEADERS),
    ("exp1_mnist", "tab:app-attn-mnist",
     "MNIST ViT (50 tokens): pre-routed sparse attention (an 8-dimensional router selects $k$ keys per query before attention) "
     "against post-hoc top-$k$ masking and full attention, 3 seeds. Attention FLOPs per block are given for the dense computation "
     "and with the sparsity exploited (post-hoc masking still computes the full score matrix and saves only the value aggregation; "
     "pre-routing computes scores for $k$ keys only); the total ratio replaces the dense attention FLOPs of every block by the "
     "mode's attention FLOPs and divides by the full model's FLOPs.", {}),
    ("exp1_cifar10", "tab:app-attn-cifar",
     "CIFAR-10 ViT (65 tokens): pre-routed sparse attention, one seed. Columns as in Table~\\ref{tab:app-attn-mnist}.", {}),
    ("paper2_split_mnist", "tab:app-sleep-split",
     "Split MNIST: sleep-decay schedules and distillation locality on the sparse redesign (CLS2, $k$-WTA 10\\% fast network), "
     "3 seeds. Delta is the change in final average accuracy against the reference row of each group, in points.", {}),
    ("paper2_permuted_mnist", "tab:app-sleep-perm",
     "Permuted MNIST: sleep-decay schedules and distillation locality, 3 seeds. Columns as in Table~\\ref{tab:app-sleep-split}.", {}),
    ("exp5_dynamic", "tab:app-dynamic",
     "Input-dependent (per-sample) pruning of a CIFAR-10 ResNet-18 against fixed masks at the same per-sample budget. Active "
     "weights are the per-sample expectation; Jaccard values are overlaps of the last-stage masks of two test images of the same "
     "or of different classes (200 images; a random mask gives 0.026); union coverage is the fraction of that stage's connections "
     "used by at least one of the 200 images.",
     {3: "active/sample", 8: "Jaccard same", 9: "Jaccard diff", 10: "coverage"}),
    ("exp4", "tab:app-stdp",
     "Unsupervised STDP (Diehl \\& Cook) against a backprop MLP of the same width trained on the same samples. SOP = synaptic "
     "operations (event-driven, accumulated). No energy comparison is made: everything runs on a GPU.",
     {0: "$n_e$", 2: "STDP acc", 3: "MLP acc", 4: "SNN SOP/img", 5: "MLP FLOPs/img", 6: "SNN train ops", 7: "MLP train FLOPs",
      8: "SNN to 80\\%", 9: "MLP to 80\\%", 10: "$w<1\\%\\,w_{\\max}$", 11: "time (s)"}),
]

# 셀 안의 한국어 토큰 -> 영어
KO = [("원본 (스킵 없음)", "no skip"), ("워밍업만, 스킵 smax", "warm-up only, skip smax"), ("스킵", "skip"), ("시드", "seeds"),
      ("FLOPs 비율 (스킵 순서대로)", "FLOPs ratio (by skip level)"), ("기준 미세조정", "fine-tuned"), ("미세조정 전", "before fine-tuning")]
HANGUL = re.compile("[\u3131-\u318e\uac00-\ud7a3]")


def esc(cell: str) -> str:
    s = cell.strip()
    for k, v in KO:
        s = s.replace(k, v)
    s = s.replace("\\", "\\textbackslash{}")
    for ch in "&%#_":
        s = s.replace(ch, "\\" + ch)
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace(" +- ", " $\\pm$ ").replace("+-", "$\\pm$")
    s = re.sub(r"(\d+\.\d+)e\+?(-?\d+)", lambda m: f"${m.group(1)}\\times10^{{{int(m.group(2))}}}$", s)
    s = s.replace(">=", "$\\ge$").replace("<=", "$\\le$").replace("->", "$\\to$").replace(" x ", " $\\times$ ")
    s = re.sub(r"(\d)x(\d)", r"\1$\\times$\2", s)
    if s == "":
        s = "--"
    return s


def parse_md(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("|"):
                if rows:
                    break
                continue
            cells = [c for c in line.strip().strip("|").split("|")]
            if all(set(c.strip()) <= set("-: ") for c in cells):
                continue
            rows.append(cells)
    return rows[0], rows[1:]


def to_tex(name: str, label: str, caption: str, header_override: dict) -> str | None:
    path = os.path.join(TAB, name + ".md")
    if not os.path.exists(path):
        print(f"[skip] {name}.md not found")
        return None
    header, body = parse_md(path)
    # 덮어쓴 헤더는 이미 TeX 이므로 이스케이프하지 않는다
    hdr = [header_override[i] if i in header_override else esc(h) for i, h in enumerate(header)]
    ncol = len(header)
    col = "l" * min(2, ncol) + "r" * (ncol - min(2, ncol))
    env = "sidewaystable" if ncol >= 10 else "table"   # 열이 많은 표는 가로로 눕힌다 (rotating 패키지)
    lines = [f"% generated by scripts/md_tables_to_tex.py from docs/report/tables/{name}.md -- do not edit by hand",
             f"\\begin{{{env}}}[p]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{col}}}", "\\toprule",
             " & ".join(hdr) + " \\\\", "\\midrule"]
    for r in body:
        r = (r + [""] * ncol)[:ncol]
        lines.append(" & ".join(esc(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{env}}}", ""]
    tex = "\n".join(lines)
    bad = sorted(set(HANGUL.findall(tex)))
    if bad:
        print(f"[warn] {name}: Hangul left in table: {''.join(bad)[:40]}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[table] paper/tables/{name}.tex ({len(body)} rows x {ncol} cols)")
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="store_true", help="main.tex 의 부록 TODO 줄을 \\input{tables/all} 로 교체")
    a = ap.parse_args()
    made = [n for n in (to_tex(*s) for s in SPECS) if n]
    with open(os.path.join(OUT, "all.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by scripts/md_tables_to_tex.py -- lists every appendix table in order\n")
        f.write("Tables~\\ref{tab:app-mnist}--\\ref{tab:app-stdp} give every number behind the figures and tables of the main text, "
                "as mean $\\pm$ standard deviation over seeds. FLOPs are analytic counts for one sample (inference) or accumulated "
                "over training at the actual density of each step. Five statements in the text rest on run logs rather than on these "
                "tables: the two-epoch collapse of the unnormalised drive rule on the CNN, the accuracy trajectory of prune-after on "
                "CIFAR-10 at its pruning step, the lowest point of the prune-during trajectory in Figure 1, the initial weight "
                "statistics of the spiking network, and its final Gini coefficient.\n\n")
        for n in made:
            f.write(f"\\input{{tables/{n}}}\n")
    print(f"[index] paper/tables/all.tex ({len(made)} tables)")
    if a.patch:
        with open(MAIN, encoding="utf-8") as f:
            s = f.read()
        todo = "\\todo{paste core\\_summary.md, core\\_summary\\_cifar.md, exp3 and exp12 tables from docs/report/tables/}"
        if todo in s:
            s = s.replace(todo, "\\input{tables/all}")
            with open(MAIN, "w", encoding="utf-8") as f:
                f.write(s)
            print("[patch] main.tex appendix now inputs tables/all.tex")
        elif "\\input{tables/all}" in s:
            print("[patch] main.tex already patched")
        else:
            print("[patch] TODO line not found; nothing changed", file=sys.stderr)


if __name__ == "__main__":
    main()
