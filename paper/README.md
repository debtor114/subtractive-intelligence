# 논문 초안 (paper/)

- `main.tex` : 워크숍 분량 초안 (영문). article 클래스 기반이라 투고 시 학회 스타일(neurips_2026.sty 등)로 바꾸면 된다.
- `refs.bib` : 참고문헌 28 건. **권/호/쪽수는 기억에 의존해 적은 것이 많으므로 투고 전 원문과 대조할 것.**
  2026-10-01 추가분 (vandeven2020, shi2025, dong2023 는 DOI 로 확인, eshraghian2023 의 권/호/쪽은 기억 -> 대조 필요).
- `figures/` : 논문용 그림 (아래 '그림' 절). 결과가 바뀌면 PAPER_FIGS=1 로 다시 생성.
- `tables/` : 부록 전체 결과표. `python scripts/md_tables_to_tex.py` 가 docs/report/tables/*.md 를 booktabs 표로 변환해 생성하고,
  `tables/all.tex` 가 순서대로 input 한다. **손으로 고치지 말고 md 를 고친 뒤 다시 생성.** 한국어 표 주석은 버리고 캡션을 영어로 단다.

## 컴파일

2026-10-02 TinyTeX 를 사용자 폴더 (%APPDATA%\TinyTeX) 에 설치했다. 새 터미널에서 PATH 에 잡히며, 안 잡히면
`%APPDATA%\TinyTeXin\windows\latexmk.exe` 를 직접 호출한다.

    cd paper && latexmk -pdf -interaction=nonstopmode main.tex

- 결과 main.pdf (24 쪽, 미해결 참조 0, 넘침 0). 페이지 그림 확인은 `rungs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r70 -sOutputFile=p%02d.png main.pdf` (TinyTeX 동봉 Ghostscript).
- 추가 설치한 패키지: booktabs multirow microtype natbib lm geometry xcolor latexmk (rotating 은 기본 포함).
- Overleaf 로 옮길 때는 `main.tex`, `refs.bib`, `figures/`, `tables/` 를 올리면 된다.

## 그림

논문용 그림은 `PAPER_FIGS=1 python scripts/analyze.py core core_cifar core_pretrained` 와 `PAPER_FIGS=1 python scripts/analyze_exp12.py` 로
생성한다 (한국어 제목 생략, 기본 글꼴, paper/figures 에 저장). 환경변수 없이 돌리면 보고서용 (한국어 제목, docs/report/figures).

## 채워야 할 곳 (`\todo{}` 표시)

1. 저자/소속: 기입 완료 (Dongin Kang, Independent Researcher, 연락 이메일 포함). 이메일을 빼려면 main.tex 26행.
2. 실험 A 결과: 반영 완료 (Table 4 + Discussion, 2026-10-02). 남은 것은 Overleaf 컴파일 확인과 서지 대조뿐.

## 2026-10-01 저녁 반영 사항

- 인용 추가: van de Ven 2020 (뇌 영감 생성 리플레이, 해마-피질 성공 사례), CH-HNN 2025 (ANN+SNN 하이브리드, 성공 사례),
  Eshraghian 2023 (STDP 한계 리뷰), Dong 2023 (최강 STDP 단독 결과). Related work 두 문단, 실험 3 부정적 결과 문단, STDP 결과 문단, Discussion.
- 표현 수정: "해마-피질 구조가 작동하지 않는다" 류를 전부 "우리의 단순한 두 망 구현은 작동하지 않았다; 생성 능력 또는 이중 표현을 가진
  정교한 구현은 성공했다" 로 좁힘 (초록, 기여 목록, 실험 3, Discussion '실패의 패턴').
- 우리만의 부정적 결과 명시: 토큰 스킵 선택 기준 = 무작위, 시냅스 구동 규칙의 CNN 채널 사망 붕괴, STDP 의 자연 희소화.

## 본문 수치의 출처

모든 수치는 docs/report/tables/ 의 표에서 가져왔다 (core_acc_matrix, core_summary, core_acc_matrix_cifar, core_summary_resnet, exp3_*, exp12_*, exp4).
