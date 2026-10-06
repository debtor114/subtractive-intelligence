# ADR-006: 선행연구를 계획보다 먼저 확인한다

날짜: 2026-10-06. 상태: 채택.

## 배경

논문 2 탐색 밤샘 실험(OVERNIGHT_P2.md, results/p2) 7건을 돌린 뒤에 선행연구를 찾았다. E1(희소 커널 속도)·
E3(두 물결, 진도 맞춤)·E4(블록 학습 중 가지치기)·X1(k-WTA + 희소 연결)·X2(빠른/느린 가중치)는 이미 있는
결과였고(Sputnik 2020, Cyclical Pruning 2022, Early-Bird 2020, Block-Sparse RNN 2017, Numenta 2019,
Hinton & Plaut 1987), 가장 새로운 E2(발달 순서)는 결과가 중립이었다. 정리는 docs/plans/p2.md.
하룻밤 GPU 시간의 대부분이 이미 알려진 것을 재확인하는 데 쓰였다.

## 결정

1. 실험 계획을 세우기 전에 실험마다 선행연구를 먼저 확인한다.
2. 계획서는 docs/plans/<experiments 디렉터리 이름>.md 에 두고 "관련 선행연구와 우리 차이" 칸을 둔다.
   표의 열이든 절이든, 실험마다 (가) 무엇이 이미 있는지(저자·연도), (나) 우리가 더하는 것이 정확히
   무엇인지 적는다. 못 찾았으면 검색어와 날짜를 적는다.
3. 후속 연구 추천도 같은 순서로 한다: 선행연구 확인 → 차이 → 추천.
4. 훅으로 강제한다. Downloads/.claude/hooks/check_prior_art.py (PreToolUse):
   - experiments/<name>/ 코드를 실행하는 명령은 docs/plans/<name>.md 의 그 칸이 채워져 있어야 통과.
   - docs/plans/*.md, OVERNIGHT*.md, PLAN*.md 를 쓸 때 그 칸이 없거나 비어 있으면 차단.
   - 논문 1 실험은 docs/plans/_legacy.md 로 면제.
   - 레포 루트에 docs/plans/ 가 있는 레포에만 적용(옵트인).

## 결과

- docs/plans/p2.md: 밤샘 실험의 선행연구 사후 기록(사용자 외부 검토 표).
- docs/plans/p3_candidates.md: 후속 후보 F1~F8, X1 확장의 선행연구 검토와 판정.
- 비용: 계획서 한 장에 선행연구 검색 시간이 더 든다. 실험 하룻밤을 헛돌리는 비용보다 싸다.
