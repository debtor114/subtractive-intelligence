# -*- coding: utf-8 -*-
"""arXiv 제출용 소스 묶음을 만든다 (paper/arxiv/ + paper/arxiv_submission.zip).

  python scripts/make_arxiv_bundle.py

- main.tex 의 주석 (% 뒤) 을 제거한다. arXiv 는 TeX 소스를 공개하므로 한국어 작업 주석이 그대로 노출된다.
- 참고문헌은 main.bbl 을 넣는다 (arXiv 는 bibtex 를 돌리지 않는다). refs.bib 도 참고용으로 넣는다.
- 본문이 쓰는 그림과 tables/*.tex 를 넣는다 (표 파일의 생성 주석도 제거).
- 묶음을 임시로 다시 컴파일해서 깨지지 않았는지 확인한다 (TinyTeX 의 latexmk).
- 제출 폼에 붙일 메타데이터 (제목·초록·분류·코멘트) 를 arxiv_metadata.txt 로 뽑는다.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "paper")
OUT = os.path.join(PAPER, "arxiv")
LATEXMK = os.path.join(os.environ.get("APPDATA", ""), "TinyTeX", "bin", "windows", "latexmk.exe")


def strip_comments(tex: str) -> str:
    out = []
    for line in tex.split("\n"):
        if line.lstrip().startswith("%"):
            continue                                   # 주석만 있는 줄은 버린다
        # 백슬래시가 앞에 없는 첫 % 부터 잘라낸다 (\% 는 퍼센트 기호)
        m = re.search(r"(?<!\\)%", line)
        if m:
            line = line[:m.start()].rstrip()
            if not line:
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def main() -> None:
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "figures"))
    os.makedirs(os.path.join(OUT, "tables"))
    main_tex = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(strip_comments(main_tex))
    bbl = os.path.join(PAPER, "main.bbl")
    if not os.path.exists(bbl):
        sys.exit("main.bbl 이 없다. 먼저 paper/ 에서 latexmk -pdf main.tex 를 돌릴 것")
    shutil.copy(bbl, os.path.join(OUT, "main.bbl"))
    shutil.copy(os.path.join(PAPER, "refs.bib"), os.path.join(OUT, "refs.bib"))
    for fig in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", main_tex))):
        src = os.path.join(PAPER, "figures", fig + ".png")
        assert os.path.exists(src), src
        shutil.copy(src, os.path.join(OUT, "figures", fig + ".png"))
    for name in sorted(os.listdir(os.path.join(PAPER, "tables"))):
        if name.endswith(".tex"):
            t = open(os.path.join(PAPER, "tables", name), encoding="utf-8").read()
            open(os.path.join(OUT, "tables", name), "w", encoding="utf-8").write(strip_comments(t))
    # 메타데이터
    title = re.search(r"\\title\{(.*?)\}\s*\n\s*\n", main_tex, flags=re.S).group(1)
    title = " ".join(title.replace("\\\\", " ").split())
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, flags=re.S).group(1)
    abstract = " ".join(abstract.replace("\\emph{", "").replace("}", "").replace("\\%", "%").replace("$\\times$", "x")
                        .replace("--", "-").replace("~", " ").replace("``", '"').replace("''", '"').split())
    n_fig = len((re.findall(r"\\includegraphics", main_tex)))
    main_log = open(os.path.join(PAPER, "main.log"), encoding="utf-8", errors="replace").read()
    n_pages = re.search(r"Output written on main.pdf \((\d+) pages", main_log).group(1)
    with open(os.path.join(OUT, "arxiv_metadata.txt"), "w", encoding="utf-8") as f:
        f.write("Title:\n" + title + "\n\nAuthors:\nDongin Kang\n\nAbstract:\n" + abstract + "\n\n")
        f.write(f"Comments:\n{n_pages} pages, {n_fig} figures, 22 tables. Code and per-run logs: https://github.com/debtor114/subtractive-intelligence\n\n")
        f.write("Primary category: cs.LG (Machine Learning)\nCross-list: cs.NE (Neural and Evolutionary Computing)\n")
        f.write("License: CC BY 4.0 (권장) 또는 arXiv perpetual non-exclusive\n")
        f.write("MSC/ACM class: 없음\nJournal-ref / DOI: 없음 (초고)\n")
    # 검증 컴파일
    if os.path.exists(LATEXMK):
        r = subprocess.run([LATEXMK, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=OUT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        log = open(os.path.join(OUT, "main.log"), encoding="utf-8", errors="replace").read()
        pages = re.search(r"Output written on main.pdf \((\d+) pages", log)
        print("test compile:", "ok" if r.returncode == 0 else "FAILED", "| pages:", pages.group(1) if pages else "?",
              "| undefined:", log.count("undefined"), "| overfull:", log.count("Overfull \\hbox"))
        for ext in (".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".blg", ".pdf"):
            p = os.path.join(OUT, "main" + ext)
            if os.path.exists(p):
                os.remove(p)
    else:
        print("latexmk 없음: 검증 컴파일 생략")
    zpath = os.path.join(PAPER, "arxiv_submission.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            for fn in files:
                if fn == "arxiv_metadata.txt":
                    continue
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, OUT))
    names = zipfile.ZipFile(zpath).namelist()
    print(f"bundle: {zpath} ({os.path.getsize(zpath) // 1024} KB, {len(names)} files)")
    print("metadata:", os.path.join(OUT, "arxiv_metadata.txt"))


if __name__ == "__main__":
    main()
