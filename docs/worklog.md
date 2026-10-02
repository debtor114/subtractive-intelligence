# 작업 일지

## 2026-10-01 (밤샘 1 차 실행)

### 00:35 시작. 환경
- Downloads 최상위의 ARCHITECTURE.md / requirements.txt 를 Projects/subtractive-intelligence/ 로 이동.
- Python 3.10.11 venv, torch 2.11.0+cu128 (RTX 3060 Ti 8GB). brian2 빌드 실패 (C 컴파일러) -> 제외. norse 설치됨(미사용).
- 연기 테스트 통과: MLP / ViT 순전파, FLOPs 집계(손계산 일치: MLP 537,600), Split/Permuted 로더, MNIST 50 스텝 학습.

### 00:55 기준선 (백그라운드)
- MLP MNIST 10 에폭 x 3 시드: 98.33 / 98.43 / ... (DataLoader 병목으로 1 회 5 분).
- ViT MNIST 10 에폭 x 3 시드, ViT CIFAR-10 50 에폭 x 1 시드 (진행 중).

### 01:00 실험 3 (이중 학습률 + 수면). Split MNIST 3 시드 완료
- 결론: 해마->피질 증류가 망각을 키움 (78.7 vs 85.0). 감쇠/가지치기 무효. ER 85.2 가 같은 수준. ADR-004 참고.
- Permuted MNIST 진행 중. joint 가 GPU 메모리를 10GB(커밋) 까지 먹어 프로세스를 죽이고 fp16 + 상한으로 고쳐 재개.

### 01:10 핵심 실험 (학습 중 가지치기) 스윕 시작 (5 밀도 x 10 방식 x 3 시드, 15 에폭)
- d=0.1 결과: 학습 중 크기 가지치기 98.67 = dense big 98.71, dense small 98.36, RigL 98.53, 헤비안 활동 98.19 (= 무작위).
- 헤비안 상관이 시냅스 기여를 못 잡는다고 판단 -> "시냅스 구동" 규칙 추가 (별도 스윕 예정).

### 01:20 실험 1 (사전 라우팅 어텐션) MNIST 스윕 시작 (7 변형 x 3 시드, 8 에폭)
- gather+expand 역전파 버퍼로 OOM -> 고급 인덱싱으로 교체. evaluate 의 GPU 라벨 numpy 변환 버그 수정.

### 01:30 메모리 사고
- 16GB RAM, 4 개 torch 프로세스 (각 커밋 3.4~3.9GB) + exp3 10GB 로 커밋 한도 소진. CPU 할당 실패 발생.
- 대응: 프로세스 사이 empty_cache, joint 데이터 상한, CIFAR DataLoader 워커 0, 동시 프로세스 4 개 이하 유지.

### 01:45 실험 4 (STDP), 실험 2 (토큰 스킵) 코드 작성. 분석 스크립트 (scripts/analyze.py) 작성, 부분 결과로 검증.

### 다음
- exp3 permuted 완료 -> exp4 STDP (n_e 400 x 3 시드, 100 x 1) 실행.
- 기준선 ViT 체크포인트 -> exp2 (MNIST 3 시드, CIFAR 1).
- exp1 MNIST 완료 -> exp1 CIFAR (full / post 0.25 / pre 0.25, 30 에폭).
- 핵심 스윕 완료 -> drive 규칙 추가 스윕 -> analyze -> 보고서.

### 01:50 ~ 03:20 세션 일시 정지 (사용량 한도). 그 사이 메모리 보호기가 백그라운드 4 개를 전부 강제 종료
- 원인: torch 프로세스 4 개 동시 실행 (각 커밋 3.4~3.9GB) 으로 16GB 머신의 커밋 한도 소진.
- 중단 시점 상태: 핵심 d0.1 완료 + d0.05 27/30, exp3 permuted 26/30, exp1 mnist 12/21 (pre 미실행), CIFAR 기준선 미완.
- 03:25 결정: 커밋 여유 5.3GB 확인 후, 한 번에 한 프로세스만 도는 순차 체인 (scripts/overnight.sh) 으로 재개. --skip_existing 으로 완료분은 건너뜀.

### 03:20 STDP 디버그
- 배치 평균 STDP 는 학습이 32 배 느려 1,280 장 뒤 우연 수준 (11%). 합산으로 바꾸고 입력 강도 128 -> 32 Hz 로 낮추자
  3,200 장에 55% (100 뉴런). 가중치 1% 이하 비율이 0.05 -> 0.24 로 커지는 '학습 중 가지치기' 가 STDP 에서 자연 발생.

### 03:25 실험 2 사전 검증
- 사후 적용 시 예측 변화량 기반 스킵(92.7%) < 무작위 스킵(96.8%) < 원본(98.3%) at 30% 스킵. 오라클(실제 잔차) 도 무작위보다 나쁨.
  배경 토큰이 모든 층에서 일관되게 멈춰 CLS 어텐션이 왜곡되는 것으로 해석. 미세조정 변형 (finetune.py) 을 2 차 체인으로 추가.

### 03:28 ~ 05:00 순차 체인 (scripts/overnight.sh) 진행
- 03:58 핵심 스윕 완료 (5 밀도 x 10 방식 x 3 시드). 학습 중 전역 크기 가지치기가 전 밀도 1 위, 0.5% 에서 97.8 (dense small 94.1,
  학습 후 가지치기 94.5, RigL 96.3). 학습 후 한 번에 깎기는 깎는 순간 0.6 아래로 무너짐 (core_trajectory.png).
- 04:07 STDP 400 뉴런 60k: 81.5% (역전파 96.2%). 가중치 58% 가 wmax 1% 이하로 -> 자연 가지치기. 평가 간격 버그 수정 후 재실행 예약.
- 04:24 실험 1 MNIST 완료: 사전 라우팅 k=5 로 98.08 (full 97.94), 어텐션 FLOPs -77%, 총 FLOPs -8.5%.
- 04:28 실험 3 Permuted 완료: CLS+수면(증류 없음) 88.6 > ER 87.4. 감쇠/활동 가지치기는 여기서도 해로움.
- 04:28 실험 2 MNIST 3 시드: 균일 스킵은 무작위보다 나쁨, 예측 잔차 대체는 80% 스킵에 94.4, 뒤쪽 2 블록만 70% 스킵은 98.0 (-0.3%p).
- 04:50 CIFAR ViT 기준선 81.8% (50 에폭). 04:52 실험 2 CIFAR: 사후 스킵이 크게 무너짐 (뒤쪽 3 블록 30% 스킵 80.1). 미세조정 변형 예약.
- 04:58 drive 규칙 스윕 완료: 지역 규칙 중 최고 (0.5% 에서 96.0), 크기 규칙엔 못 미침.
- 진행 중: 실험 1 CIFAR (full / post 0.25 / pre 0.25, 30 에폭) -> STDP 추가 시드 -> overnight2 (실험 2 미세조정, STDP 재실행).

### 05:10 두 번째 강제 종료
- 실험 1 CIFAR pre 변형 실행 중 메모리 보호기가 체인 셸을 다시 종료. 원인: CIFAR DataLoader 워커 4 개 (train 2 + test 2,
  persistent) 가 각각 커밋 1.3~1.4GB 를 먹고 주 프로세스가 5.2GB -> 한 "실행" 이 커밋 10.7GB. 이 PC 는 다른 프로그램이
  커밋 약 20GB 를 쓰고 있어 여유가 5GB 남짓이다.
- 파이썬 자식은 고아로 살아남아 pre 실행을 계속 (ep 5/30). 끝나면 JSON 을 쓰므로 기다린다. 체인의 나머지 (STDP 추가 시드,
  overnight2) 는 두 번째 중단이라 임의로 재시작하지 않는다. 남은 단계는 scripts/resume.sh 로 사용자가 직접 실행.
- 재발 방지: exp1 CIFAR 로더 워커 0 으로 변경. 앞으로 이 PC 에서는 DataLoader 워커를 쓰지 않는다.

### 05:30 마무리
- 실험 1 CIFAR pre 완료 (고아 프로세스): full 79.4 / post 80.3 / pre 80.0, 어텐션 FLOPs -68%, 총 -6.5% (시드 1, 30 에폭).
- 체인 셸이 살아남아 STDP 추가 시드 (400 s1 -> 100 s0 -> 400 s2 -> 1600 s0) 를 계속 돌리는 중. 새로 띄운 것은 없음.
- 보고서 docs/report/REPORT.md 최종 정리. 남은 단계는 scripts/resume.sh (STDP 재실행 + 실험 2 미세조정).

### 06:04 체인 종료 (OVERNIGHT_DONE)
- STDP 400 뉴런 시드 3: 81.3 +- 0.2 (역전파 96.5). 100 뉴런 37.0, 1600 뉴런 72.9 (시드 1, 400 기준 하이퍼파라미터 그대로라 미튜닝).
- 실행 프로세스 없음, 커밋 여유 5.5GB. overnight2 (실험 2 미세조정, STDP 5k 간격 재실행) 는 실행하지 않음 -> scripts/resume.sh.
- 보고서 docs/report/REPORT.md 최종본. 총 실행: 핵심 168 런, 실험 1 24, 실험 2 4, 실험 3 60, 실험 4 5, 기준선 7.

### 06:40 ~ 07:00 남은 단계 (사용자 지시로 하나씩 순차 실행)
- 실험 2 미세조정 (스킵 0.5 로 ViT + 예측기 학습): MNIST 2 시드 x {predicted, random}, CIFAR 1 시드 x {predicted, random}.
  MNIST 50% 스킵 손실 4.4%p -> 0.7%p, CIFAR 30%p -> 9~11%p. 선택 기준(predicted/random) 은 학습 때 쓴 기준으로 평가하면 동률.
  -> 이득은 적응에서 나오고 예측 변화량 자체는 정보가 없다. analyze.py 에 exp2_finetune 추가.
- STDP 400 seed 0 재실행 (5k 간격) 백그라운드 (results/_stdp_rerun.log).
- 07:10 STDP seed 0 재실행 완료 (80.8%, 5k 간격). 400 뉴런 3 시드 81.1 +- 0.3. 계획한 실행 전부 종료. 보고서/공유 문서/메모리 갱신.

### 07:50 ~ 13:50 핵심 실험 CIFAR-10 확장 (사용자 지시)
- 소형 CNN (2.2M), 밀도 0.1/0.03/0.01, 6 방식 x 3 시드 + dense big = 57 런, 런당 2~6 분, 총 약 6 시간 (예상 2 시간은 dense 런 기준 오산).
- 연기 테스트에서 3 가지 수정: 층별 균일 밀도 붕괴 -> ERK 배분, 구동 규칙 붕괴 -> 후뉴런별 정규화, 임계값 동점 부풀림 -> 정확한 top-k.
- 결과: 학습 중 전역 크기 가지치기 1% 에서 85.7 (dense small 77.1, 학습 후 80.5, RigL 75.9/1 시드 붕괴). FLOPs 를 ERK 로 묶어도 +4.2%p.
  구동 정규화 규칙은 1% 에서 60.0 으로 CNN 고희소에서는 실패. RigL seed 2 는 AMP non-finite grad 로 연결이 새어 붕괴 ->
  regrow 가드 추가 후 재실행했으나 학습 자체가 발산 (ERK 중간층 밀도 0.3~0.6%). 그대로 보고.
- 사전 검증 중 MNIST pd_mag_global d0.05 seed0 결과를 덮어써 재실행으로 복구 (98.77).

### 14:00 ~ 15:00 실험 3 재설계 (사용자 설계: 희소 해마, 증류 제거, 균형 리플레이, 크기 가지치기만)
- Split: ER 균형 87.8 (ER 85.2), CLS2 87.0, 희소 해마 무영향 (증류 없으면 경로 없음), 해마 단독 19.7, 확신도 결합 66~68.
- Permuted: 클래스 기준 균형이 버퍼를 안 써 52% 붕괴 -> (태스크, 클래스) 그룹 기준으로 수정 + 인덱스 캐시. CLS2 89.0, ER 균형 88.3 (ER 87.4).
- 결론: 이득은 균형 리플레이뿐. 감산 요소의 망각 감소 근거 없음. 보고서 4.5 절.

### 15:00 ~ 16:00 실험 1+2 통합 (사전 필터링, experiments/exp12_prefilter/)
- 시상 라우터(앞쪽 끝에서 뒤쪽 블록별 변화량 사전 예측) + 예측 잔차 대체 + 점진 스킵 미세조정 (+ 뒤쪽 사전 라우팅 어텐션).
- MNIST 3 시드: 워밍업만으로 70% 스킵 98.10 (원본 98.27). 미세조정은 오히려 -0.2%p. 선택 기준 3 종 동률.
- CIFAR 1 시드: 미세조정 후 70% 스킵 80.8 (원본 81.8, FLOPs 0.72). 선택 기준 동률. 사전 라우팅 어텐션은 -0.6%p 에 FLOPs -1%p.
- 결론: 효과는 '뒤쪽 블록만 + 예측 잔차 대체' 에서 나오고 어느 토큰을 고를지는 무관. 보고서 7b 절.

### 16:10 ~ 16:40 길 1 착수: ResNet-18 스윕 + 논문 초안
- baselines/resnet.py (CIFAR ResNet-18, 11.2M 가중치), run_cifar.py --model resnet18. 밀도 0.05/0.02/0.005 x 5 방식 x 2 시드 + dense big = 32 런,
  런당 6~10 분 (스텝 45 ms), 약 4 시간. results/core_resnet/.
- paper/main.tex (영문 워크숍 초안, article 클래스) + refs.bib + figures/. LaTeX 미설치 -> Overleaf 로 컴파일. TODO: 저자, ResNet 결과, 부록 표.

### 17:10 실험 A 준비 (사전 학습 ResNet-18 -> CIFAR-10 적응하며 깎기, '공짜 대리석')
- 사용자 제안: 학습 비용 50 배 약점은 사전 학습 모델에서 출발하면 사라진다 (사전 학습 비용은 이미 지불됨). 유전자 = 사전 학습 모델 비유.
- experiments/core_prune_during_learning/run_pretrained.py + run_all_pretrained.py. torchvision ImageNet 가중치, 128x128 업샘플, 10 에폭.
  arm: pt_dense / pt_pd / pt_pd_erk / pt_oneshot / pt_rigl / scratch_pd / scratch_small. 밀도 0.05/0.02/0.005, 시드 2 = 38 런, 약 2.5 시간.
- ResNet 스윕이 끝나면 자동으로 연기 테스트 후 스윕 시작 (results/_core_pretrained.log). 논문 Discussion 에 상각/전이 문단 추가.
- 선행 연구 경계: 사전 학습 모델 가지치기는 Han 2015, Zhu & Gupta 2017 의 고전 설정. 신규성은 비교 틀과 극단 희소, 비용 보고.
- 17:30 두 번째 논문 씨앗 노트: docs/notes/brain_to_von_neumann_translation.md (뇌 구조 -> 폰 노이만 번역 사전, 검증 순서: 수면 스케줄 -> 분산 증류 -> 인덱스 로드 -> 순서 인코딩).
- 17:45 논문 2 토대 코드: CLS2 에 decay_mode (boundary / periodic / continuous) 와 kd_mode (global / local 층별 특징 / local_logits) 추가,
  CPU 가짜 데이터로 전 경로 통과. 스윕 8 변형 x 2 데이터셋 x 3 시드를 실험 A 종료 후 자동 시작하도록 체인 (results/_paper2.log).
- ResNet-18 중간 (시드 0): 0.5% 에서 학습 중 전역 88.9 vs dense small 81.1 (+7.8%p), ERK 판 87.9 (FLOPs 40%). 2% 에서 학습 후 가지치기와 동률.
- 19:12 ResNet-18 스윕 완료 (32 런, 오류 0). 0.5%: 학습 중 전역 89.0 vs dense small 81.4 / 학습 후 83.6 / RigL 84.7, ERK 88.0 (FLOPs 40%).
  논문 1 Table 3 + Scale 문단 채움, 보고서 3.5b. 실험 A 자동 시작 (19:12).
- 19:35 실험 A 버그: 새로 붙인 fc 가 초기 가중치가 작아 전역 크기 원샷 가지치기에 통째로 잘림 (fc 밀도 0, 10% 고정). 분류층을 가지치기에서
  제외 (convert_to_masked skip=fc, 예산/활성 수도 conv 만) 하고 12 런 폐기 후 재시작. 체인: 실험 A -> 논문 2 스윕 (results/_core_pretrained.log).
- RunPod 이관 준비: runpod/setup_pod.sh, runpod/run_remaining.sh, Projects/subtractive-intelligence-code.tgz (2.9MB). 팟 SSH 정보 대기.

### 19:50 ~ 20:20 RunPod A40 이관 (사용자 지시: 로컬 체인 중단, 팟에서 새로)
- 팟: 드라이버 580 / CUDA 13 -> torch 2.14.1+cu130, Python 3.12. 할당 RAM 46GB, VRAM 48GB (free/nproc 은 호스트 값 503GB/96코어를 보임).
- cs.toronto.edu CIFAR 다운로드가 146kB/s 라 로컬 data/cifar-10-python.tar.gz 와 ~/.cache/torch 의 ResNet-18 가중치를 scp 로 올림 (30초/8초).
- 연기 테스트 통과 후 runpod/run_parallel.sh 로 5 갈래 동시 실행: 실험 A 시드 0 / 시드 1, 논문 2 스윕 split / permuted, ResNet-18 시드 2.
  로그 results/_pod_<stream>.log, 완료 표식 _pod_<stream>.done, _pod_all.done.
- 함정 1: 코드 tgz 가 실험 A fc 보호 수정 전에 만들어져 팟의 run_pretrained.py 가 구버전 (active 11,172,032 = conv + fc 로 들통). md5 대조로 그 파일만
  다른 것을 확인, 수정본 업로드 후 실험 A 두 갈래만 중단-폐기-재시작 (runpod/relaunch_expA.sh, 완료 표식 _pod_expA_all.done).
- 함정 2: ssh 원격에서 pkill -f '<스크립트명>' 은 ssh 가 띄운 bash -c 명령줄 자체에도 매칭되어 세션이 255 로 끊김 -> 대괄호 패턴 (setup_po[d]).
- 로컬: scripts/analyze.py 에 core_pretrained (실험 A 표 2 + 그림 3) 와 paper2 (수면 감쇠 / 증류 국소성 비교표 + 그림) 추가, 로컬 샘플로 통과.
  회수 스크립트 runpod/pull_results.sh (tar 스트림으로 core_pretrained / exp3 / core_resnet 를 로컬 results/ 에 덮어씀).
- 예상 소요: 실험 A 갈래당 19 런 x 6 분, ResNet 시드 2 16 런 x 9 분, permuted 24 런 x 5 분 -> 약 2.5 시간.
- 20:12 팟 함정 3: 팟 numpy 2.5 에는 np.trapz 가 없음 (로컬 2.2 는 별칭 유지). 실험 A / ResNet 갈래가 학습을 다 마치고 결과 저장 직전
  data_efficiency 에서 AttributeError 로 죽어 첫 런(6~9 분)이 세 갈래 모두 날아감. utils/metrics.py 에 _trapz = trapezoid 또는 trapz 호환 추가,
  세 갈래 재시작 (runpod/relaunch_fix.sh, 완료 표식 _pod_fix_all.done). 연기 테스트(MLP run.py)는 data_efficiency 를 안 거쳐 못 잡았음.
- 20:15 논문 부록 "Full result tables" 채움: scripts/md_tables_to_tex.py 가 docs/report/tables/*.md 9 개를 paper/tables/*.tex (booktabs,
  resizebox) 로 변환하고 main.tex 부록이 tables/all.tex 를 input. 한국어 헤더는 영어로 치환, 표 주석은 캡션으로 대체.
- 20:30 ~ 20:55 실험 5 (추론 중 가지치기, 로드맵 후보 1) 구현 + 팟 대기열. core/dynamic_layers.py (DynamicConv2d: dyn_local / dyn_random /
  kwta_in, unfold+bmm 표본별 가중치, 마스크 통계), experiments/exp5_dynamic_pruning/run.py + run_all.py, analyze.py 에 exp5. ADR-005.
  로컬 연기 테스트는 사용자 지시 ("내꺼에서 돌리지말고 runpod에서") 로 중단 -> 팟에서 연기 테스트 3 팔 통과.
  함정: (1) 평가 배치 1000 이면 표본별 가중치가 층 하나에 4.7GB -> 250 + 학습 중 2000 장 부분 평가. (2) 마스크 교집합 fp16 내적 inf -> fp32 조각 누적.
  팟 측정 (5 갈래와 공유): dyn_local B=128 스텝 0.3 s. runpod/queue_exp5.sh 가 _pod_all.done + _pod_fix_all.done 을 기다려 자동 시작
  (시드별 2 갈래 + 정적 0.2), 완료 표식 _pod_exp5.done.
- 21:00 ~ 21:50 실험 5 팟 실행 중 메모리/속도 수정 (상세 ADR-005 수정 이력): 체크포인트 + kthvalue 임계값 마스크로 26GB -> 2.7GB,
  dyn_random 베르누이, 임계값은 체크포인트 밖에서 1 회. 사용자 요청으로 대기열 대신 즉시 실행, 결정적 비교 (d=0.05 국소 vs 무작위) 가 먼저.
  현재 팟: 실험 A (10/19 x 2), ResNet 시드 2 (5/16), kwta_in (1/6) -> 정적 0.2, dyn_local / dyn_random (각 6 런) 동시 실행. 완료 표식 4 개
  (_pod_all, _pod_fix_all, _pod_exp5, _pod_exp5_dyn) 를 로컬 대기자가 감시.
- 23:07 실험 5 첫 결정적 쌍 (d=0.05, 시드 0, CIFAR-10 ResNet-18): dyn_local 89.94 (best 90.05) / dyn_random 52.3 (시드 1 도 52.2) /
  정적 학습 중 전역 크기 92.1 / dense small 88.7. 마스크 통계 (테스트 200 장): dyn_local 4 단계 J_same 0.436 vs J_diff 0.277 (무작위 기준 0.026)
  -> 뒤쪽 층에 클래스별 서브망이 실재. 앞쪽 층은 J 0.77/0.76 으로 입력 무관 (사실상 정적 뉴런별 크기 마스크). union coverage 4 단계 0.205, 1 단계 0.114.
  해석 (1 시드): 연결 단위 동적 선택은 같은 FLOPs 의 정적 마스크를 못 이김 (-2.2%p). 무작위 동적은 붕괴 -> 토큰 스킵의 "선택 기준 무의미" 는
  연결 단위로 확장되지 않음 (선택이 전부). 혼동 요인: dyn 팔은 층별 균일 5% (전역 배분 불가) -> 정적 층별 균일 대조군 pd_mag_layer 추가 실행
  (results/_pod_static_layer.log, 밀도 0.05/0.005/0.2 x 시드 0 1). 멈춰 둔 갈래 자동 재개됨 (14:06 UTC).

## 2026-10-02

### 01:30 ~ 02:10 실험 A 완료 (38 런, 오류 0) -> 집계 -> 논문/보고서 반영
- 실험 A (사전 학습 ResNet-18, 128x128, 10 에폭, 시드 2): 5% 93.7 / 2% 91.0 / 0.5% 80.9 (적응 중 전역). 원샷 93.1 / 90.3 / 81.7, RigL 86.9 / 83.6 / 76.8,
  처음부터 가지치기 88.4 / 87.5 / 83.8, dense small 84.8 / 82.4 / 76.5, 밀집 미세조정 95.9. **0.5% 에서 순위 역전** (처음부터 > 원샷 > 사전 학습 적응 중):
  사전 학습 팔은 중간에 93.1 까지 가다 마지막 4.5% 를 떼며 12%p 손실 (최종 밀도에서 3 에폭뿐). 논문에 두 결과 모두 적음.
- 논문 2 토대 스윕 (48 런): 수면 감쇠는 Split 오차 범위 (+0.1~+0.5), Permuted 손해 (-0.3~-4.0); 증류는 전역/국소 모두 손해 (국소 -4.2 양쪽).
- 실험 5 중간 (5%): dyn_local 89.9 (1 시드) < 정적 층별 균일 91.1 < 정적 전역 92.1; dyn_random 52.3, kwta_in 44.9 (20% 에서도 73.4).
  마스크 자카드 같은/다른 클래스 0.44/0.28 (무작위 0.026) -> 입력별 서브망 실재하나 이득 없음.
- main.tex: 초록·기여·결과(전이 문단 + Table 4 + Figure)·부정적 결과(수면 보강 + 추론 중 가지치기 문단 + Table 5)·토론·결론 12 곳 수정,
  남은 TODO 는 저자명 하나. 부록 표 10 개 재생성 (exp5 추가). REPORT.md 3.5c / 4.6 / 7c 추가. 그림 3 개 paper/figures 복사.
- 남은 팟 작업: ResNet 시드 2 (8 런), 정적 층별 (3), 정적 0.2 (3), dyn_random (3), dyn_local (5). 끝나면 표 3 을 3 시드로, Table 5 의 빈칸 보충.
- 02:48 사용자 지시로 로컬 3060 Ti 도 투입 (scripts/local_remaining.cmd, Start-Process 로 분리 실행해 메모리 보호기 영향 없음, 로그
  results/_local_remaining.log): ResNet 시드 2 d=0.005 5 팔 -> dyn_local d=0.2 시드 0 1. 팟은 runpod/rebalance.sh 로 ResNet 시드 2 를 ttp d0.02 뒤에
  멈추고 dyn_local 은 d=0.05 s1 뒤 d=0.005 만 다시 띄움 (완료 표식 _pod_dyn_local_0005.done). 로컬 커밋 여유 3.5GB 로 빡빡함 (torch 1 개 한계).
- 04:41 모든 실행 종료 (팟 14 표식 + 로컬 LOCAL_REMAINING_DONE). 실험 5 최종 (시드 2): dyn_local 20% 91.8 / 5% 89.8 / 0.5% 77.7,
  dyn_random 76.1 / 52.3 / 18.2, kwta_in 73.4 / 44.9 / 34.0, 정적 층별 균일 91.8 / 91.1 / 83.1, 정적 전역 91.9 / 92.0 / 89.0.
  -> 동적 벌점은 20% 에서 0, 5% 에서 1.3%p, 0.5% 에서 5.4%p (같은 배분 대비). 자카드 같은/다른 클래스 5% 0.44/0.27, 20% 0.61/0.46.
  ResNet-18 시드 3 완성 (로컬이 d=0.005 5 팔). 표 3 수치 갱신 (81.4->81.3, 84.7->84.9, 88.0->87.8, 86.7->86.5, 92.1->92.0).
- 06:xx main.tex 최종값 반영 (표 3 시드 3, 표 5 20%/5%/0.5% 완성, 실험 5 문단, 초록), REPORT.md 3.5b / 7c 최종값, 부록 표 10 개 재생성.
  analyze.py 라벨의 '|w|' 가 마크다운 표 열을 깨서 'abs(w)' 로 수정. 남은 TODO: 저자명, Overleaf 컴파일 확인, 공유 문서 갱신, 팟 종료.
- 06:30 공유 문서 (Claude Docs 59c036f3) 갱신: 요약 추가 문단, 판정표 2 행 (실험 A 조건부 / 실험 5 기각), 핵심 행 ResNet 3 시드 수치,
  3.5c 실험 A 절 + 그림, 4.6 논문 2 토대 절, 실험 5 절 + 그림, 남은 단계 (저자명·Overleaf·서지 대조·투고처·논문 2 축). 팟은 비어 있음 -> 사용자가 종료.
- 06:40 저자 기입 (사용자 지정: Dongin Kang, 소속 없음 -> Independent Researcher, 연락 이메일 추가). main.tex 의 \todo 0 개. 남은 것은 Overleaf 컴파일 확인·서지 대조·투고처.
- 06:50 ~ 07:20 로컬 컴파일 (사용자 요청). TinyTeX 를 %APPDATA%\TinyTeX 에 설치 (install-bin-windows.ps1), 패키지 8 개 추가, latexmk 로 main.pdf 생성.
  1 차: 19 쪽, 미해결 참조 0, 넘침 1 (표 5 -> tabcolsep 4pt). 페이지를 Ghostscript (rungs) 로 PNG 로 떠서 확인하니 그림 제목이 한국어 ->
  analyze.py 에 PAPER_FIGS=1 모드 (제목 생략, DejaVu Sans, paper/figures 저장) 추가 후 4 개 그림 재생성. 부록의 열 10 개 이상 표는 sidewaystable 로
  (rotating 패키지), exp4/exp5 헤더는 짧은 TeX 로 덮어씀. 2 차: 24 쪽, 넘침 0. 함정: Bash heredoc 파이썬에서 \u 가 또 깨짐 (Write 도구로).
- 07:40 arXiv 계정 확인 (사용자 요청, 크롬): 로그인 상태, 이달 제출 가능 2 건, 제출 이력 없음, 기본 분류 cs.LG. TeX 소스 업로드 필수.
  scripts/make_arxiv_bundle.py: 주석 제거한 main.tex + main.bbl + refs.bib + 그림 4 + 표 10 -> paper/arxiv_submission.zip (450KB, 검증 컴파일 24 쪽),
  제출 폼용 arxiv_metadata.txt (제목·초록·분류·코멘트). 초록/재현성 문구를 "코드는 공개 예정" 과 "3060 Ti + A40" 으로 정정. 제출은 사용자 승인 대기.
- 08:10 사용자 결정: 제목·초록을 "뇌 -> 폰 노이만 번역에서 무엇이 살아남는가" 틀로 (점진 가지치기는 알려진 결과라 이목을 못 끈다는 지적).
  제목 "What Survives the Translation from Brain to Von Neumann Machine: Learning Is Pruning, and Four Mechanisms That Did Not Transfer",
  초록 1,875 자 (arXiv 1,920 자 제한 안; 이전 3,007 자는 초과), 서론 첫 문단 + 프로그램 문단에 다섯 메커니즘 명시. 재컴파일·묶음 재생성.
- 09:00 ~ 09:40 사용자가 가져온 상세 리뷰 (24 항목: 초록·기여·결론 vs 본문·부록 표 불일치) 반영. 표 수치로 전부 검증 후 41 곳 수정:
  (1) 성공/조건부/실패 분류 통일 + Section 5 머리에 메커니즘 매핑 표 (tab:map); (2) "사전학습이면 공짜" -> 적응 FLOPs 는 처음부터와 같음
  (7.3e14 vs 7.7e14), 매몰비용은 사전학습뿐; (3) "모든 예산 최고" -> 2% 이하 최고, 느슨한 예산은 prune-after 와 동률; (4) RigL 평가 통일
  (0.5% 에서만 2 위, CIFAR 1% 는 dense small 아래, ResNet 비용 2.3~2.5x); (5) 비용 범위 3~59x, ERK 동일 FLOPs 는 ResNet 에서 RigL 대비만;
  (6) 98% -> 97% (최저 97.3); (7) Hebbian 은 random 보다 낫고 dense small 보다 못함; (8) 무작위 두 경로가 MNIST 최하위; (9) ECE 범위 재기술;
  (10) 표 1 캡션 표준편차 예외 명시; (11) ResNet 문단 재기술; (12) 전이 문단 90% 도달·ECE 범위 한정; (13) RigL 1% 두 값 설명 + 가드 켠 채 발산 명시;
  (14) 표본별 예산 1%/14% 초과·FLOPs 명시, class-specific -> input-specific, 구동 규칙 후뉴런 항 없음; (15) 수면 문단 수치 전부 재계산
  (85.0->78.7, forgetting 0.18->0.25, downscale 77.4/82.7, activity prune -6.3, Permuted 에서는 CLS 가 ER 이김); (16) 어텐션 k=5/50 vs k=17/65,
  "총 연산 절감 없음" -> 6~9%; (17~19) 다섯 메커니즘 + 후속 1, 지표 목록에 forgetting, 시드 수 정정; (21) 부록에 exp1/exp2/논문 2 표 6 개 추가 (16 개),
  로그 근거 3 건 명시; (22) 이름 통일 (activity (Hebbian)); (23) perceptron -> MLP, "gradient 없이" 는 기준에 한정; (24) STDP 5.0%, 1600 뉴런 28.7%.
  초록 재작성 1,901 자 (코드 공개 문장은 코멘트 칸으로). 30 쪽, 넘침 0.
- 10:00 2 차 리뷰 반영. A1 (예측 잔차 치환의 근거) 은 대조군 실험으로 답함: core/thalamic_skip.py 에 substitute=identity 옵션, exp12 run.py --substitute,
  로컬 3060 Ti 로 MNIST 3 시드 + CIFAR 1 시드 (총 6 분). 결과 identity 치환 + 미세조정 = 예측 치환 + 미세조정 (MNIST 97.89 vs 97.86,
  CIFAR 80.79 vs 80.79) -> 치환 효과 0. 예측 코딩을 "안 옮겨짐" 으로 통일 (초록·서론·표 5·5 절·6 절·결론), 지역 규칙은 가지치기 변형으로 제외.
  A2 데이터 효율: MNIST 97% 와 전이 90% 는 가지치기 시작 전 도달 -> 밀집 출발의 속성으로 적고, ResNet 5%/0.5% (385k/402k, 가지치기 도중) 로 주장 이동.
  A3 어텐션: 비중 5% -> 약 10%, post-hoc 는 최대 2.7% 절감, 부록 캡션에 총 비율 계산법. A4 수면 Permuted: no-distill +1.2 (같은 연산), CLS2 +0.8 (5 배).
  B/C 전부 반영 (ordering, RigL 관련 연구 문장, 그림 1 캡션 색 구분, accuracy gap -> +3.5~5.3, 다섯 + 변형, 공유 출력층 문구, 표 1 표준편차,
  ECE MNIST 한정 + static random, 표 참조 순서 (exp2 를 exp12 앞으로), 0.01/5.3/1.0 수치, one-shot 0.5% 미도달, 시드 목록, 로그 근거 5 건).
- 11:00 3 차 리뷰 반영. A1 어텐션 FLOPs 이중 계산 발견·수정: exp1 run.py 의 flops_total_exploited 가 pre 모드에서 라우팅된 모델의 측정값을
  기준으로 삼아 절감이 두 번 들어감 -> analyze_exp1 에서 full 모델 측정값 기준으로 재계산. 어텐션 비중 6.1%/5.3% (MNIST/CIFAR), pre 총 절감
  4.7%/3.6% (이전 8.5%/6.5%). 본문·초록 (4~5%)·6 절 수정. 기여 6 옛 문장, 구조/원리 기준을 결과 앞에 명시 + "pattern" -> "observation"
  + 예측기 항목 추가, ResNet 데이터 효율 비교 대상 (dense big 437k, RigL 1/3 시드), 전이 "약 99% 남은 시점", 0.7 점은 비영 비율만,
  5 절 제목 "Negative results", 수면 문구 3 곳 통일 (matches balanced replay; +1.2 over plain replay at equal compute), 인용 15~16,
  ECE 0.008~, late-only 는 같은 FLOPs 비교 (30% 전 블록 대비), 기여 3 fewer, exp12 캡션 (identity 행·warm-up 정의), 결론 명칭, 부록 로그 목록.
  29 쪽, 넘침 0. git 저장소 초기화 (main), 영문 README·MIT LICENSE·CITATION.cff·.gitignore, 팟 IP 를 <pod-ip> 로 치환, 첫 커밋. 원격은 사용자가 제공 예정.
