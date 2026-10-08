# 비폰노이만 기계 에뮬레이터로 실험하기: 조사 노트

작성 2026-10-08 (21:12 저장). 워크플로 emulator-prior-art (wf_6a416e83-3ae, 에이전트 100, 오류 0). 논문 84편을 한 편씩 검증했다(원문 63, 초록 21; confirmed 76, corrected 8). 근거 자료는 docs/notes/emulator_survey_20261008_evidence/ 에 있다(papers.json, tools.json, judges.json, gaps_critic.json, 판정 단계 탐침 스크립트 drn_probe.py·drn1h_time.py — 스크래치패드 경로 기준). 근거는 검증 워크플로 자료다. 논문은 판정이 confirmed 또는 corrected인 것만 썼고, 도구 사실에는 재확인에서 나온 정정을 반영했다. 7–8절에는 세 판정자(성공·새로움·맞춤)의 평가와 이 PC 실측을 반영했다.

- 표기 † : 검증 표 밖에서 2차로 언급된 문헌이다. 위협이나 확인 대상으로만 쓰고, 주장의 근거로는 쓰지 않는다.
- 에너지 수치는 모두 모델 추정이다. GPU 실측값은 쓰지 않는다.

## 1. 한 줄 답

할 수 있다. 다만 에뮬레이터가 바꾸는 것은 물리가 아니다. 바뀌는 것은 점수판, 즉 그 기계의 비용 장부와 제약이다. 크로스바·뉴로모픽·광자 칩에서도 역전파는 그 기계에 맞는 방식으로 홈경기를 한다. 정직한 양의 결과에 가장 가까운 자리는 다음 조건의 실험이다.

- 소자 하나를 덜면 그 소자의 전력과 갱신 회로가 함께 사라지는 기판이다.
- 국소 신호로 학습 중에 덜어낸다.

1순위는 이 PC에서 바로 도는 자기학습 저항망 시뮬레이터(Scellier 2024 코드)로 하는 학습 중 가지치기다. 실험 설계와 사전 등록 예측은 7절에, 계획서를 쓰기 전에 확인할 것은 8절에 있다.

## 2. 에뮬레이터가 바꿔 주는 것과 못 바꾸는 것

### 바꿔 주는 것: 점수판

**비용 단위.** GPU의 FLOP·바이트가 기계 고유 단위로 바뀐다.

| 기계 | 비용 단위 |
|---|---|
| 저항망 | 소산 전력 Σg·ΔV², 소자 수, 이완 시간 |
| 크로스바 | 쓰기 펄스, 쓰기 에너지, ADC 변환, 전치 읽기 |
| 뉴로모픽 칩 | 시냅스 이벤트, 스파이크 메시지, 갱신 횟수, 코어 메모리 |

**제약.**
- 국소성: Loihi 학습 엔진은 pre/post 트레이스의 곱의 합과 제3 인자만 받는다. 가중치는 8비트다.
- 낮은 정밀도: BrainScaleS-2 가중치는 6비트다. aihwkit은 dw_min, 증감 비대칭, D2D/C2C 변동을 모델링한다.
- 쓰기 비용과 내구성.
- 코어 메모리: SpiNNaker 2는 코어당 64 KB다. SANA-FE의 Loihi 프리셋은 코어당 최대 1024 뉴런이다.
- 역방향이 공짜가 아니다. 물리망에는 별도의 역방향 경로가 없다. 크로스바는 역방향을 전치 읽기로 하지만 ADC 비용이 붙는다.

**점수의 부호가 바뀌는 자리.**
- GPU에서 무구조 희소성은 이득이 없다. 자기학습 저항망에서는 엣지 하나를 없애면 그 엣지의 소산과 갱신 회로가 같이 사라진다.
- 크로스바 학습 비용 연구(Peng et al. 2021)는 갱신 비용을 배치 200으로 나눠 무시할 만하다고 봤다. 배치 1 온라인 학습에서는 이 상각이 사라진다.

### 못 바꾸는 것

- **에뮬레이터도 결국 폰노이만 PC에서 돈다.** 시뮬레이션이 GPU에서 몇 시간 돌았는지는 점수가 아니다.
- **에너지는 모델 추정이다.** 이는 프로젝트 규칙이다. 실측은 실제 칩(EBRAINS의 BrainScaleS-2, SpiNNaker)에서만 가능하다. 이 실측을 쓸지는 규칙 적용 범위를 먼저 정해야 한다.
- **비용 모델을 우리가 쓰면 승리를 미리 구워 넣을 수 있다.** 그래서 다음을 지킨다.
  - 제3자가 보정한 시뮬레이터와 공개 상수를 쓴다.
  - 비용 항은 실행 전에 등록한다.
  - 같은 정확도에서 비교한다.
- **에뮬레이터는 이상화되어 있다.**
  - Scellier 시뮬레이터에는 잡음과 편향이 없다. 실제 저항망 학습기에서 나타나는 편향성 망각(Dillavou et al. 2025a)을 보지 못한다.
  - SANA-FE에는 가소성 유닛이 없다.
  - aihwkit에는 에너지 모델이 없다.
  - NeuroBench는 학습 연산을 세지 않는다.
- **역전파도 비폰노이만 기계에서 홈경기를 한다.**
  - 크로스바: 전치 MVM과 외적 갱신이 O(1)이다(Gokmen & Vlasov 2016). 현장 BP가 소프트웨어 정확도를 낸다(Ambrogio et al. 2018). 123M Transformer까지 확장됐다(Wu et al. 2026).
  - 뉴로모픽: Loihi에서 정확한 BP가 전부 온칩으로 돈다(Renner et al. 2024).
  - 광자: 칩 위에서 BP를 한다(Ashtiani et al. 2026).
  - 물리망: 해밀토니안 에코 방식의 자율 학습이 있다(Lopez-Pastor & Marquardt 2023).
  - 그러므로 비교 상대는 GPU BP가 아니라 그 기계에 맞는 BP여야 한다.
- **하드웨어 몫과 규칙 몫을 섞기 쉽다.**
  - Renner et al. 2024: GPU 대비 이득은 배치 1에서 하드웨어가 낸 것이다.
  - Hajizada et al. 2025: 6,600배 에너지 이득을 알고리즘 22.6배와 설계 295배로 나눴다.
  - 규칙의 기여를 보려면 같은 기판에서 BP와 비교해야 한다.
- **현재 수준.**
  - 뉴로모픽 시스템은 여전히 뇌보다 최소 4자릿수 비효율이다(Ostrau et al. 2022).
  - 물리 신경망(PNN) 학습법 가운데 BP의 규모와 성능에 도달한 것은 아직 없다(Momeni et al. 2025).

## 3. 기계별 지형 표

| 기계 | 도구 | 무엇을 흉내 내나 | 학습 지원 | 비용 모델 | 이 PC(윈도우, WSL 없음) | 판정 |
|---|---|---|---|---|---|---|
| 자기학습 저항망 | energy-based-learning (Rain, Scellier 2024 코드, MIT, 마지막 커밋 2025-04-06) | 이상적 비선형 저항망(저항·다이오드·전원)의 정상상태를 QP로 정확히 푼다. 층형 DRN, 합성곱 Hopfield망 | CL, EP·CpL(양·음·중심 넛지), TBP·RBP 기준선. 추정기 클래스로 규칙을 추가할 수 있다. g=0은 저항 제거와 물리적으로 같다 | 에너지 함수 = 소산 전력(½Σg·ΔV²), 소자 수. 이완 시간·주변회로·갱신 회로는 없다 | 가능. 순수 파이썬, 버전 고정 없음, 기존 venv. README는 bash만 다루므로 `$env:PYTHONPATH`를 설정한다. torch 2.11은 미검증 | 1순위(후보 A) |
| 자기학습 저항망(RKM) | pyrkm 0.1.0 + rkm_energy (Apache-2.0) | 쌍둥이 저항망으로 만든 RBM 유사 모델, 이진 MNIST | 대조 국소 규칙 | 연산당 시간·전력·에너지 해석 모델(노드 RC 이완, 비교기, 게이트 전압 갱신) | 가능(pip, py3-none-any) | 후보 A의 비용 틀 |
| 무질서 유동·저항망 | maguzj/coupled_learning, learning_circuit | 임의 그래프의 선형 저항(=유동)망 | CL. 가지치기는 TODO 상태 | 전력만 | CPU에서 가능. 라이선스 파일 없음 | 저자 허락 전 보류 |
| 회로 수준 | ngspice-47, Xyce + PySpice/spicelib | 소자 모델(순방향 강하, 누설, 정착) | 없음 | 소산 전력, 정착 시간 | 가능 | 학습에는 너무 느리다(Kendall et al. 2020: 0.16M 망이 에폭당 18 h). 셀 하나 보정용 |
| 아날로그 크로스바(소자 물리) | IBM aihwkit (PyPI 1.1.0, master 1.2.0, MIT; Le Gallo et al. 2023) | MVM 비이상성(ADC/DAC, 잡음, IR 강하, 드리프트), 펄스 갱신(dw_min, 비대칭, 경계, D2D/C2C) | 펄스열 SGD, 혼합정밀, TT v1/v2, c-TTv2, AGAD. `tile.update(x,d)`로 외적형 규칙(Hebbian·NP·DFA)을 넣을 수 있다 | 없음. 펄스 카운터는 GPU 빌드에만 있다 | 불가. 윈도우 휠과 윈도우 conda가 없다. 윈도우 검증은 AMD ROCm 포크뿐이라 우리 GPU와 무관하다 → RunPod | 후보 B 2단계(사용자 승인 필요) |
| 아날로그 크로스바(학습+비용) | MLP+NeuroSim V3.0 (CC BY-NC 4.0, 2021-11, 브랜치 2025-01) | 400-100-10 MLP, eNVM·SRAM·FeFET 셀, LTP/LTD 비선형·변동 | 온칩 BP(SGD부터 Adam까지). 모든 갱신이 WriteCell을 거친다. Train.cpp를 고쳐 규칙과 마스크를 넣는다 | 읽기·쓰기 에너지, 지연, 면적, 누설, 펄스 수 | 아마 가능. MSYS2 g++ 11.2와 OpenMP가 있다. README는 리눅스만 다루고 빌드는 미검증 | 후보 B 1단계 |
| 아날로그 크로스바(CNN 학습) | DNN+NeuroSim V2.1 | VGG-8, CIFAR-10 | BP+WAGE. DFA 비교 선례가 있다(Wolters et al. 2023) | 순방향·활성 기울기·가중치 기울기·갱신으로 분해 | 불가(리눅스, CUDA). 1회 약 12 h | MLP 결과가 좋을 때만 |
| 아날로그 크로스바(정확도) | CrossSim 3.2.1 (BSD-3) | 프로그래밍 오차, 드리프트, 읽기 잡음, ADC/DAC, 기생 저항, 비선형 I-V | 현장 학습 없음(3.0에서 제거). Torch 하드웨어 인지 학습만 | 없음(공식 명시) | 가능(윈도우 10, Python 3.11.6에서 시험됨) | 후보 D의 정확도 |
| CIM 추론 비용 | NeuroSim V1.4/V1.5 (V1.5는 2026-01 릴리스) | 추론 PPA, 130 nm~1 nm | 없음 | 면적·지연·에너지·누설 | 불가 → RunPod | 후보 D의 비용 |
| 가속기 비용 | AccelForge + HWComponents | 소자·회로·아키텍처 비용 | 없음 | 면적·에너지·지연 | 불가(의존성이 Python 3.12 이상, 윈도우 휠 없음) | 선택적 비용 백엔드 |
| 크로스바 학습(경량) | XBTorch (2026) | 해석식·표 기반 소자 모델 | 소자 모델을 거치는 갱신 | 없음 | qtorch JIT 컴파일에 MSVC가 필요한데 없다 | 보류 |
| 뉴로모픽(Loihi 규칙) | Lava 0.10.0 (2026-05 보관 처리) | Loihi 1/2 고정소수점 연산과 학습 엔진 비트폭 | 2·3인자 규칙(트레이스 곱의 합 형태) | CPU에는 없다(프로파일러는 독점 하드웨어 전용) | 가능. 단 별도 Python 3.10 venv가 필요하다(numpy<2, networkx≤2.8.7) | 후보 C의 제약 검사기 |
| 뉴로모픽(비용) | SANA-FE 2.2.9 (GPL-3.0) | 칩 아키텍처 수준 이벤트 실행, Loihi·TrueNorth 프리셋 | 가소성 유닛 없음 | 이벤트당 에너지·지연. Loihi 실측 대비 에너지 12%, 지연 25% 이내(응용 3개, 추론) | 가능(cp310 win_amd64 휠) | 후보 C의 비용 |
| 뉴로모픽(지표) | NeuroBench 2.3.0 (Apache-2.0; Yik et al. 2025) | 하드웨어 무관 지표, FSCIL 과제 | 없음. 학습 연산을 세지 않는다 | 연산 수, 희소도 | 가능(torchaudio, tonic, numba 추가) | 후보 C의 점수판 |
| 뉴로모픽(Loihi 1 에뮬레이터) | Brian2Loihi 0.5.2 (Michaelis et al. 2022) | Loihi 1 | Loihi식 트레이스 규칙 | 없음 | 가능(brian2 2.8.0.4) | Lava 대체용 |
| GPU SNN 구조 가소성 | GeNN 5.5 / mlGeNN 2.40 | CUDA 코드를 생성하는 SNN 시뮬레이터 | add/remove_synapse, e-prop+Deep R | 없음 | 소스 빌드. VS Build Tools와 CUDA toolkit 설치가 필요하다 | 알고리즘 대조용 |
| 실제 칩(원격) | BrainScaleS-2 (EBRAINS, hxtorch) | 아날로그 칩 512 뉴런, 6비트 가중치, 내장 가소성 프로세서(PPU) | PPU에서 C++ 국소 규칙과 구조 가소성 | 칩 실측(논문) | 브라우저로 사용(EBRAINS, 무료 평가 할당) | 측정 규칙을 정한 뒤 확인용 |
| 실제 칩(원격) | sPyNNaker 7.4 (SpiNNaker 1) | 실제 칩 | STDP, 신경조절, 구조 가소성 | 소프트웨어 비용 모델 없음 | EBRAINS | 낮은 우선순위 |
| 섭동 학습(하드웨어 무관) | mgd_scaling (GPL-2.0, 2025-01-16). 정정: 코드가 있다 | WP·NP 시간 스케일링(Oripov et al. 2025) | WP, NP | 시간 분석 | 미확인 | 후보 B의 참조 구현 |
| 광자 | L2ight(+FLOPS·MixedTrain), SimPhony, torchonn, FFzero, neuroptica | MZI 메시 등 | L2ight는 현장 부분공간 학습과 ZO 기준선, FFzero는 순방향 국소 학습, neuroptica는 현장 adjoint BP | L2ight는 PTC 호출 수(정규화), SimPhony는 추론 에너지·면적 | L2ight·SimPhony는 리눅스 쪽이다. FFzero의 torch 2.11+cu128 버전 고정은 우리 venv와 같다 | 후보에서 제외(선행연구 밀집) |
| 이징·발진기·확률 | OIM-EP, Ising-Machine-EqProp, THRML | 발진기, 이진 스핀, Gibbs 샘플링 | EP | 없음 | OIM-EP는 아마 가능, THRML은 JAX CPU만 | 관찰 대상 |

그 밖에 제외한 도구와 이유는 부록 D에 있다.

## 4. 기계별로 이미 알려진 것

### 4.1 아날로그 인메모리 크로스바

**역전파가 원래 싼 곳.**
- Gokmen & Vlasov 2016
  - 전치 MVM과 펄스 일치 외적 갱신이 모두 O(1)이다.
  - 증감 비대칭은 5% 이내여야 하고, ADC가 타일의 면적과 전력을 지배한다.
- Marinella et al. 2018
  - 커널 수준에서 MAC당 11 fJ이다.
  - 에너지는 ADC와 적분기가 지배한다.
- Ambrogio et al. 2018: 204,900개 시냅스에서 현장 BP로 MNIST부터 CIFAR-100까지 소프트웨어 동등 정확도를 냈다. 계산값은 28,065 GOP/s/W로 GPU 대비 약 100배다.
- Nandakumar et al. 2020
  - 혼합정밀로 97.73%를 냈다.
  - 임계값 갱신으로 갱신 수가 1,000배 넘게 줄었다.
  - 디지털 32비트 구현 대비 172배(시스템 추정).
- Wu et al. 2026
  - 123M Transformer까지 학습했다. 손실 스케일링은 L∝N^-0.231로 디지털의 N^-0.238과 비슷하다.
  - 단, 기울기는 디지털에서 계산한다.
- Rasch et al. 2024: c-TTv2와 AGAD가 기준 전도도를 정밀하게 맞출 필요를 없앴다.

**역전파가 비싼 곳.**
- Peng et al. 2021: 병목은 버퍼 지연, DRAM 에너지, 가중치 기울기 계산이다. 갱신 비용은 배치 200으로 나눠져 무시된다.
- Wolters et al. 2023: 에너지의 95%가 칩 밖 버퍼 접근이다.
- 정리하면 활성값·오차의 이동, ADC가 붙는 역방향 패스, 비대칭 민감도, 배치 1에서 상각되지 않는 쓰기가 BP의 비용이다.

**BP 없는 학습의 성적.**

진 곳:
- Wolters et al. 2023
  - DFA는 C2C 변동 1%에서만 수렴했다. BP는 5%에서도 수렴했다.
  - DFA는 ADC 3비트 이상이 필요하고, 폭 8 망에서 0.08 대 BP 0.44였다.
  - 에너지는 BP와 같았다. 이득은 층수 N배의 지연뿐이었다.
- McCaughan et al. 2023
  - CIFAR-10에서 60.7%로 BP 68%보다 낮았다.
  - 시간 이득은 섭동 시간이 1 µs보다 훨씬 짧을 때만 생긴다. 1 ms면 5.6 h로 GPU BP의 480 s보다 느리다. 10 ns면 200 ms, 200 ps면 4 ms다.
- Demirag et al. 2021: PCM 위 e-prop은 다섯 구성 가운데 디지털 누적을 쓰는 혼합정밀만 통과했다.

비긴 곳:
- Renaudineau et al. 2026
  - 8,064개 소자 하드웨어 인 더 루프에서 FF 89.5%, CF 89.6%, BP 90.0%로 유의차가 없었다. 소자당 펄스 수도 비슷했다.
  - 리셋 펄스 0.84 pJ 대 프로그램-검증 387 pJ의 이득은 프로그래밍 방식에서 왔고, BP도 이 방식을 썼다.
- Oripov et al. 2025: WP·NP가 Fashion-MNIST에서 BP와 같은 91.6%를 냈다(하드웨어 무관 시뮬레이션).

이긴 곳:
- Ren et al. 2025: 크로스바에 올린 스파이킹 DFA가 ReRAM BP 가속기(PipeLayer)보다 학습 시간 1.1–10.5배, 에너지 1.37–2.1배 적었다. 정확도 손실은 2% 미만이었다.
- Yi et al. 2023: 규모를 키우면 디지털 대비 4자릿수 이상 이득이라는 모델 추정을 냈다.

쓰기 절약은 이미 있다:
- Payvand et al. 2020: 오류 트리거 국소 규칙으로 갱신이 약 100배 줄었다.
- Wang, Y. et al. 2019: 펄스를 90% 줄였다. 단 최고 정확도와 동시에 얻은 결과는 아니다.
- Cai et al. 2018: 수명을 356배 늘렸다.

**덜어냄의 성적.**
- Li et al. 2026
  - 무작위 RRAM 가중치는 그대로 두고 연결만 깎아서 학습했다. Fashion-MNIST에서 87.4%로 가중치 학습의 79.7%보다 높았다.
  - 갱신은 99.94% 줄었다. 같은 쓰기 예산이면 가중치 학습이 17.3%p 낮았다.
  - 점수 계산은 디지털 BP와 STE로 했다.
- Wang, B. et al. 2025(PRIME): 같은 계열의 SNN 연구다.
- 구조가 맞아야 이득이 난다.
  - 무구조 희소는 크로스바와 어긋난다(Chu et al. 2020, Ankit et al. 2017).
  - 공유 ADC의 정밀도는 가장 덜 희소한 열이 정한다. 열 사이 균형 희소성이 ADC 에너지를 최대 7.13배 줄였다(Ibrayev et al. 2024).
  - 가지친 망은 기생 저항에 더 약하다. 64x64 크로스바에서 C/F 가지치기 망은 정확도가 약 39% 떨어졌고 밀집 망은 약 21% 떨어졌다(Bhattacharjee et al. 2022).
- Rathi et al. 2019: STDP 기반 학습 중 가지치기로 에너지 3.1배, 면적 4배를 얻었다.

**열린 것(못 찾음).**
- NP·WP·FG·DFA를 TT·c-TTv2·AGAD·혼합정밀과 같은 보정 소자 모델과 쓰기 장부로 맞붙인 비교.
- 배치 1 온라인 클래스 증분 학습에서의 규칙 비교. 쓰기 비중이 정량되지 않았다.
- 활동에 비례한 열 예산.
- 활성값을 저장하지 않고 배열 안에서 즉시 갱신할 때도 Wolters의 결과가 유지되는지.

### 4.2 뉴로모픽 칩

**역전파.**
- Renner et al. 2024
  - Loihi에서 정확한 BP를 전부 온칩으로 돌렸다. 400-400-10 망으로 MNIST 95.7%를 냈다.
  - 표본당 0.6 mJ, 1.5 ms를 썼다. 추론만 하면 0.0025 mJ다.
  - 전력은 학습 엔진이 지배한다.
- Rostami et al. 2022
  - SpiNNaker 2에서 e-prop으로 GSC 91.12%를 냈다.
  - 에너지 추정은 54.7 kJ로 V100의 646.4 kJ보다 적다. 하지만 시간은 586 h로 V100의 1 h 58 min보다 훨씬 길다.
  - 메모리는 600 스텝을 넘는 긴 시퀀스에서만 e-prop이 유리했다.
- Davies et al. 2021
  - 일반 순방향 DNN은 Loihi에서 이득이 거의 없다.
  - 재귀, 스파이크 타이밍, 가소성, 희소성을 쓰는 망에서만 자릿수 이득이 난다.

**국소 학습.**
- Shrestha et al. 2021
  - Loihi에서 FA/DFA를 돌렸다. 합성곱 층은 오프라인에서 사전학습했다.
  - CIFAR-10에서 62.2%로 파이썬 64.4%보다 낮았다. 8비트 가중치로 2–8%p를 잃었다.
  - 학습 에너지는 8.4 mJ/img로 RTX 5000 배치 1의 77 mJ보다 적었다.
- Hajizada et al. 2025
  - Loihi 2에서 3인자 국소 규칙으로 재현 없이 재현 수준의 정확도를 냈다.
  - 특징 추출은 고정 EfficientNet-B0이 했고, 호스트 I/O는 계산에서 뺐다.
- Stewart et al. 2024: 마지막 층만 온칩에서 학습했다. 동적 전력은 1 mW 미만이었다.
- Wunderlich et al. 2019: BrainScaleS-2가 같은 스파이킹 망의 소프트웨어 시뮬레이션보다 10배 빠르고 1000배 효율적이었다.
- den Blanken et al. 2026: NeuroBench 키워드 FSCIL의 첫 하드웨어 기준을 냈다(71.8%, 9.5 µW).

**덜어냄과 예산.**
- Liu et al. 2018
  - DEEP R로 64 KB 코어 하나 안에서 학습했다. 활성 연결은 1.3%로 고정했다.
  - MNIST 96.6%를 냈고, x86 대비 전력이 약 2자릿수 낮았다.
- 뉴런당 팬인을 고정했다(Billaudelle et al. 2021, Bogdan et al. 2018).
- Nguyen et al. 2021: 온칩 STDP 중 가지치기로 학습 시간 2.1배, 에너지 64% 절감, 연결 92.83% 감소를 얻었다.
- George et al. 2017: 아날로그 VLSI에서 제한된 시냅스를 활동 기준으로 할당했다.
- Knight et al. 2026: e-prop과 DEEP R로 학습이 최대 10배 빨라졌다.

**비용 모델 경고.**
- Dampfhoffer et al. 2023: 메모리 접근까지 세면 SNN은 추론당 시냅스당 스파이크 0.15–1.38개 이하에서만 경쟁력이 있다.
- Ostrau et al. 2022: SpiNNaker는 CPU와 비슷하고 GPU보다 비효율적이었다.
- Dennler et al. 2024: 온칩 학습 주장을 해시 테이블 하나가 이겼다.
- 공개된 갱신 단가:
  - Tang et al. 2023: SENeCA에서 Hebbian 15.5 pJ, e-prop 22.9 pJ.
  - Davies et al. 2018: Loihi 시냅스 갱신 약 120 pJ. 원문은 비공개이고 2차 인용 둘로 확인한 설계값이다.

**열린 것(못 찾음).**
- 칩 위에서 BP와 DFA·FG·NP를 같은 메모리·에너지 조건으로 비교한 연구.
- 활동 비례 뉴런별 시냅스 예산(허브 배분)의 하드웨어 구현.
- Loihi 2 학습 엔진의 공개 연산당 에너지.
- NeuroBench의 학습 비용 지표.
- 전체 망을 온칩 학습한 CIFAR-10 결과.

### 4.3 물리 학습망(저항·유동·기계)

**하드웨어.**
- Dillavou et al. 2022
  - 엣지 16개 쌍둥이 망으로 아이리스 95% 초과를 냈다.
  - 100스텝마다 엣지를 하나씩 자르면 오차가 튀었다가 회복됐다.
  - 저항 드리프트를 일부러 위쪽으로 치우쳐 고전력 해를 피했다.
  - 과제를 바꾸면 이전 과제를 빨리 잊었다.
- Dillavou et al. 2024
  - 클록 없는 트랜지스터 망이 XOR와 비선형 회귀를 배웠다.
  - 엣지당 5–10 µW, 2 µs 가정 시 엣지당 10–20 pJ이다.
- Dillavou et al. 2025a: 측정·갱신 편향이 망각을 만든다. 과클램핑으로 완화된다.
- Dillavou et al. 2025b: 이중 하강이 시험 hinge 손실에서만 나타났다. 데이터 수 M을 바꾸며 측정했다.

**시뮬레이션.**
- Scellier 2024
  - DRN-XL(51.7M)을 SPICE보다 160배 빠르게 학습했다.
  - MNIST 시험 오차는 EP 1.33%, BP 1.30%다.
- Scellier et al. 2023
  - CIFAR-10 시험 오차는 C-EP 11.1%, C-CpL 14.9%, P-CpL 46.9%다. TBP는 10.1%다.
  - CpL은 어려운 과제에서 뒤처진다.
- Guzman et al. 2026
  - 이진 MNIST에서 약 92%로 RBM과 같았다.
  - 에너지가 은닉 수에 선형으로 늘어난다. 이는 스케일링 논증일 뿐, 같은 정확도에서 잰 에너지가 아니다.

**시간과 동시성.**
- Stern et al. 2022: 학습률 대 이완률 비가 임계값 아래면 동작 변화 없이 빨라진다. 임계값을 넘어도 학습은 된다.
- Anisetti et al. 2024: 주파수 전파는 활성과 오차를 다른 주파수로 동시에 흘린다.
- Tammali & Olin-Ammentorp 2026: 순방향·역방향 스윕을 따로 나누지 않는다.
- Lopez-Pastor & Marquardt 2023: 기울기 학습도 물리에 네이티브일 수 있다.

**덜어냄.**
- Stern et al. 2024
  - 국소 규칙에 λ·전력 항을 넣으면 오차는 λ²에 비례해 늘고 전력은 λ에 선형으로 줄어든다. 공짜가 아니다.
  - λ=1이면 directed aging이 된다.
- 목표 반응을 설계하는 가지치기 연구는 데이터 학습이 아니다(Pashine 2021, Goodrich et al. 2015, Rocks et al. 2017·2019).
- Farinha et al. 2020: 추상 방향망에서 EP와 L1 가지치기를 썼다.
- Wang, Q. et al. 2026: 희소 국소 격자가 밀집과 비슷했다. 희소성은 설계로 고정했다.
- Guzman et al. 2025: 감수율이 측정 가능하고 엣지 중요도와 양의 상관이 있다. 가지치기 실험은 없다.
- Straszak & Vishnoi 2022: Physarum 동역학은 IRLS와 같고 ℓ1 최소해를 찾는다. 볼록 문제에 한정된다.
- Ronellenfitsch & Katifori 2016: 깎기만 하면 지역 최소에 갇힌다. 담금질 같은 일정이 필요하다.

**외부 기준.**
- Momeni et al. 2023: 틀린 모델을 쓴 BP보다 견고했다. 이상적 BP를 상대로 한 결과는 아니다.
- Wright et al. 2022: 물리 순방향과 디지털 역방향을 섞은 BP가 작동한다.
- Nakajima et al. 2022
  - CIFAR-10 47.83%(실험)를 냈다.
  - 시간의 약 92%가 FPGA 전송과 DAC/ADC였다.

**열린 것(못 찾음).**
- 물리 학습망 안에서 중요도 기반 학습 중 가지치기.
- 학습기 전체를 한 에뮬레이터·한 에너지 모델·같은 정확도로 비교한 연구.
- 장난감 수준을 넘는 연속 학습.
- 물리 하드웨어에서의 CIFAR-10.

### 4.4 광자

- Gu et al. 2021a
  - 희소 ZO로 소자 2,500개 넘게 최적화했다.
  - 소자 5–15%만 활성으로 효율 3.7–7.6배, 전력 90% 넘게 절감했다.
- Gu et al. 2021b(L2ight)
  - ZO 프로토콜은 약 2,000 매개변수 규모만 다뤘다.
  - 사전학습 매핑과 1차 부분공간 학습으로 10M까지 늘렸다.
- Xu et al. 2024
  - 가지치기로 실험 정확도를 67.0%에서 95.0%로 올렸다.
  - 튜닝 전력이 10배 줄었다.
- Ashtiani et al. 2026: 칩 위 BP로 정확도 90%를 넘겼다.
- Guo et al. 2025: 칩에서 실시간 온라인 학습을 했다.

판단: 가지치기의 전력 이득, ZO와 가지치기의 결합, 현장 BP가 모두 이미 있다. 후보에서 뺀다.

### 4.5 확률·이징

- Niazi et al. 2024
  - 약 3만 매개변수의 희소 p-bit 망이 3.25M RBM과 같은 90%를 냈다.
  - 가중치 갱신은 CPU가 한다.
- Laydevant et al. 2024: D-Wave에서 EP를 돌렸다. 첫 층의 곱은 실리콘에서 계산했다.
- OIM-EP: 784-500-10 MLP로 97.2%를 냈고, 시뮬레이션에 약 40 h가 걸렸다.

비용 모델이 없어 관찰만 한다.

### 4.6 공통 교훈

1. BP 없는 규칙이 이겼다는 보고는 대부분 상대가 셋 중 하나다.
   - 틀린 모델을 쓴 BP(PhyLL, aDFA).
   - 다른 하드웨어(Renner, Shrestha, Hajizada의 설계 몫).
   - 시간만 센 비교(MGD).
2. 덜어냄은 기판 비용이 소자 수에 비례할 때 값을 한다.
   - 저항망 엣지의 소산과 갱신 회로.
   - RRAM 쓰기.
   - 광자의 정적 튜닝 전력.
   - 이벤트 구동 칩의 시냅스 이벤트.
   - 크로스바는 ADC 공유 때문에 구조가 맞아야 한다.
3. 이상적 에뮬레이터는 낙관적이다(Dillavou et al. 2025a).

## 5. 후보 실험

| 실험 | 관련 선행연구와 우리 차이 | 점수판 | 이 PC 가능 여부·예상 시간 | 성공 가능성·위험 |
|---|---|---|---|---|
| **A. 자기학습 저항망 학습 중 가지치기** (1순위). DRN-XS/1H, MNIST 다음 Fashion-MNIST. 학습기는 C-EP, CpL, 같은 엣지 수 BP | (가) 이미 있는 것: Dillavou 2022는 엣지 16개 망에서 무작위 절단 후 회복을 봤다. Pashine 2021, Goodrich 2015, Rocks 2019는 목표 반응 설계용 가지치기다. Stern 2024는 국소 전력 감쇠 규칙(오차∝λ², 전력은 선형 감소)을 냈다. Farinha 2020은 추상 망 EP+L1이다. Guzman 2025는 감수율이 엣지 중요도와 상관한다고 보였지만 가지치기는 없다. Wang, Q. 2026은 고정 희소 격자다. Scellier 2024는 에뮬레이터에서 EP≈BP를 보였으므로 "국소 규칙이 BP와 비긴다"는 새롭지 않다. (나) 우리가 더하는 것: Kirchhoff 망에서 EP·CpL로 지도 학습하는 동안 국소 물리 신호로 엣지를 영구 제거한다. 신호는 엣지 소산 g·ΔV², 노드 전류 예산(허브 배분의 물리판), 감수율이다. 같은 정확도에서 Stern 감쇠 규칙, 무작위 제거, 크기 가지치기, 같은 엣지 수 BP와 소산 전력·소자 수로 맞붙인다. 이 조합은 선행연구 못 찾음(2026-10-08). 검색어: arXiv abs:pruning AND (physical learning, coupled learning, equilibrium propagation, physical neural network(s), flow network(s), resistor network(s)); abs:'equilibrium propagation' AND (sparse, sparsity, pruned); DataCite c3p1~c3allo; 'tuning by pruning' | 물리 점수: 자유 상태 소산 전력 Σg·ΔV²(시뮬레이터의 에너지 함수)와 남은 전도도 수. 같은 시험 정확도에서 비교하고, 밀도 축의 시험 손실 곡선도 전부 보고한다. 추론 에너지는 엣지 수 × 엣지당 10–20 pJ(Dillavou 2024, 2 µs 가정)로 추정한다. 학습 시간·에너지는 RKM 해석 모델의 항(Guzman 2026, pyrkm)을 실행 전에 고정해 쓴다 | 가능. 기존 venv를 쓰고 torch 프로세스는 한 번에 하나만 돌린다. A100 기준 DRN-XS 10에폭 0:30, DRN-1H EP 50에폭 2 h 36 min. 3060 Ti도 같은 자릿수로 추정한다(미측정). 6개 팔 × 시드 3을 DRN-XS로 돌리면 1–2밤(추정)이고, 이후 DRN-1H 일부를 돌린다 | 중간(판단). 덜어낸 엣지는 소산과 갱신 회로를 함께 없앤다. 위험: (1) 같은 오차에서 Stern 2024 감쇠 규칙을 못 이기면 새 결과가 아니다. (2) 이중 하강(Dillavou 2025b) 때문에 밀도 축에서 시험 손실이 튈 수 있다. (3) 시뮬레이터가 이상적이라 편향 망각을 보지 못한다(Dillavou 2025a). 갱신 편향 모델과 과클램핑 팔을 넣는다. (4) 엣지 제거가 수렴 조건을 깰 수 있다(†arXiv:2606.15443). (5) MNIST급 MLP만 가능하고 입력 이득 A가 최대 4000이다. (6) 깎기만으로 갇히면 담금질 일정 팔을 쓴다(Ronellenfitsch & Katifori 2016) |
| **B. 크로스바 배치 1 온라인 학습의 쓰기 장부**. 1단계 MLP+NeuroSim V3.0, 2단계 aihwkit | (가) 이미 있는 것: Peng 2021은 갱신 비용을 배치 200으로 나눠 무시했다. Wolters 2023에서 BP와 DFA의 에너지는 같았고(버퍼가 95%), DFA는 잡음에 졌다. Renaudineau 2026에서 FF는 하드웨어에서 BP와 비겼고 펄스 수도 비슷했다. Ren 2025에서 DFA는 에너지 1.37–2.1배 이득을 냈다. Payvand 2020의 오류 트리거 규칙은 갱신을 약 100배 줄였다. Nandakumar 2020의 임계 혼합정밀은 갱신을 1,000배 넘게 줄였다. Wang, Y. 2019, Cai 2018도 쓰기를 줄였다. Li 2026은 연결을 깎아 갱신을 99.94% 줄였으므로, "프로그래밍 대신 가지치기"라는 일반 논문은 이미 나왔다. Rasch 2024는 c-TTv2·AGAD를 냈다. Demirag 2021은 국소 규칙도 디지털 누적이 필요함을 보였다. †Jabri & Flower 1992는 아날로그 VLSI 가중치 섭동이다. (나) 우리가 더하는 것: 같은 보정 소자·회로 모델에서 배치 1 온라인 클래스 증분 스트림을 돌린다. 이 구간에서는 쓰기 비용이 상각되지 않는다. BP, 임계 BP, 오류 트리거 국소 규칙, DFA, 탈상관 NP(Dalm 2026)를 맞붙인다. 그 위에 학습 중 점진 가지치기와 허브 예산이 정확도당 쓰기를 더 줄이는지 본다. 이 비교는 선행연구 못 찾음(2026-10-08). 검색어: arXiv 'perturbation memristor training', '"node perturbation" hardware', 'zeroth-order analog in-memory training', 'feedback alignment analog devices asymmetric', 'backpropagation-free analog hardware training'; Crossref 'local learning rules analog in-memory computing benchmark backpropagation' | NeuroSim이 내는 값을 그대로 쓴다: 쓰기·읽기 에너지, 쓰기 펄스 수, 지연, 면적. 여기에 스트림 정확도(A_AUC)를 더한다. 같은 정확도에서 비교하고 소자당 최대 쓰기 수(내구성)도 본다. 2단계는 aihwkit 펄스 카운터에 NeuroSim·HWComponents의 연산당 에너지를 곱한다. 기준선은 TT·c-TTv2·AGAD·혼합정밀이다 | 1단계는 아마 가능하다. MSYS2 빌드는 미검증이고, `make run`은 stdbuf/tee를 쓰므로 바이너리를 직접 실행한다. 포팅·빌드에 0.5–1일(추정). 실행 시간은 자료에 없으니 첫 실행으로 잰다. 2단계는 RunPod A40이 필요하다(비용 발생, 사용자 승인 필요) | "국소 규칙이 에너지로 BP를 이긴다"는 낮음(판단). 근거는 Wolters, Nandakumar, Payvand다. "가지치기가 임계 BP 위에서 정확도당 쓰기를 더 줄인다"는 중간. 위험: (1) 400-100-10 시그모이드 MLP, 옛 기술 모델, 비상업 라이선스다. (2) 가지친 소자도 이웃 갱신에 교란될 수 있어 "절대 쓰지 않는다"는 과장이다(†arXiv:2608.25781). (3) †arXiv:2111.09272와 †doi:10.1002/aisy.202500150을 확인하기 전에는 새로움을 주장하지 않는다. (4) 섭동 학습은 사전학습 모델 적응 구간에서만 기대할 수 있다(Katti 2025, Gu 2021b) |
| **C. 뉴로모픽 FSCIL: 코어 시냅스 예산 아래 국소 학습**. NeuroBench 키워드 FSCIL(MSWC) | (가) 이미 있는 것: Yik 2025의 FSCIL 기준선은 학습하지 않는 프로토타입 판독이다(세션 평균 ANN 89.27%, SNN 75.27%). den Blanken 2026은 하드웨어 71.8%, 9.5 µW를 냈다. Hajizada 2025는 Loihi 2에서 3인자 규칙으로 머리만 학습했다. Liu 2018은 층별 밀도를 고정했다. Billaudelle 2021과 Bogdan 2018은 뉴런당 팬인을 균일하게 잡았다. George 2017은 활동 기준으로 시냅스를 할당했다. Knight 2026, Han 2023도 있다. Meirovitch 2025는 활동 의존 단위 예산을 냈지만 하드웨어 비용은 없다. Butz & van Ooyen 2013에 따르면 뇌의 항상성 규칙은 반대 방향이다. (나) 우리가 더하는 것: 학습하는 머리(3인자 국소 규칙)에 뉴런별 시냅스 예산을 건다. 활동 비례(허브), 균일, 항상성(역비례) 예산을 맞붙인다. Loihi 학습 엔진 제약을 통과하는 규칙만 쓴다. 활동 비례 뉴런별 예산의 하드웨어 구현은 선행연구 못 찾음(2026-10-08). 검색어: arXiv '"structural plasticity" neuromorphic hardware', '"homeostatic structural plasticity"', '"fan-in" "structural plasticity"', '"deep rewiring"', '"on-chip" pruning spiking learning'; DataCite c4sp, c4cl, c4fscil | NeuroBench 지표: 기본 정확도, 세션 평균 정확도, footprint, 연결 희소도, SynOps. 여기에 SANA-FE Loihi 프리셋의 이벤트 에너지·지연을 더한다. 학습 갱신 항은 공개 설계값 두 가지로 모두 보고한다: Loihi 약 120 pJ/갱신(Davies 2018)과 SENeCA Hebbian 15.5 pJ, e-prop 22.9 pJ(Tang 2023). 기준선은 NCM, kNN, 해시(Dennler 2024) | 가능. NeuroBench는 본 venv를 복제해 쓰고, SANA-FE는 cp310 휠, Lava는 별도 Python 3.10 venv를 쓴다. 실행 시간은 자료에 없다. 설치·연결에 2–3일(추정) | 낮음~중간. 위험: (1) 순방향망은 Loihi 이득이 작다(Davies 2021). (2) 머리만 학습하면 판독 적응 수준에 그친다. (3) 갱신 비용 항은 공개값이지만 가정이다. (4) George 2017이 핵심 발상을 앞선다. (5) 허브 방향이 생물학적으로 맞는지 정당화해야 한다 |
| **D. 크로스바 추론 매핑에서 허브 배분이 살아남는지 검사** (빠른 반증). MNIST MLP 1/5/10% 밀도 | (가) 이미 있는 것: Ibrayev 2024에 따르면 공유 ADC의 정밀도는 가장 덜 희소한 열이 정하고, 열 사이 균형 희소성이 ADC 에너지를 최대 7.13배 줄인다(VGG11 64x64에서 4배, 무구조 2.6배). Chu 2020과 Ankit 2017은 무구조 희소가 크로스바와 어긋남을 보였다. Bhattacharjee 2022에서 가지친 망은 비이상성에 더 약했다. Rathi 2019는 STDP 기반 가지치기로 에너지 3.1배, 면적 4배를 냈다. Meirovitch 2025는 활동 의존 예산을 냈다. †TinyADC 2021, †Group Scissor 2017, †OWL 2023도 확인해야 한다. (나) 우리가 더하는 것: 수신 뉴런 활동에 비례한 열 예산을 비이상성 아래 정확도와 타일·ADC 장부로 채점한다. 허브 열을 별도 타일(고정밀 ADC)에 모으는 배치가 균형 희소성을 이기는지 본다. 열별 활동 비례 예산은 선행연구 못 찾음(2026-10-08). 검색어: arXiv 'column sparsity crossbar ADC', 'activity-based pruning neuromorphic hardware', 'unstructured sparsity in-memory computing crossbar'; DataCite c1rc, c1act, c1bal, c1hub | CrossSim 비이상성(기생 저항, ADC/DAC, 프로그래밍 오차, 드리프트, 읽기 잡음) 아래 정확도. 매핑 후 점유 타일 수와 타일별로 필요한 ADC 비트(Ibrayev 규칙). 에너지·면적은 NeuroSim V1.5 추론 PPA | 정확도는 가능하다(CrossSim, Python 3.11 venv 권장). 에너지는 RunPod가 필요하다. 추론만이라 짧다(실행 시간 자료 없음) | 낮음(판단). 허브 열이 ADC 정밀도를 끌어올리고, 극희소에서 IR 강하에 약하다. 학습 실험이 아니라 사후 매핑이다. 가치는 크로스바 이야기에서 허브 배분을 버릴지 싸게 정하는 데 있다 |

### 뺀 후보와 이유

- **C2 단독(섭동·ZO가 아날로그 BP를 이긴다).** 근거가 반대 방향이다(Wolters 2023, McCaughan 2023, Gu 2021b, Demirag 2021). 맞대결 비교는 B의 팔로 흡수한다. 섭동이 이길 여지는 두 곳뿐이다.
  - 사전학습 모델 적응(Katti 2025).
  - 섭동 시간이 1 µs보다 훨씬 짧은 기판(McCaughan 2023, Oripov 2025).
  - †Lei 2026(arXiv:2608.21223)과 †arXiv:2601.10037을 확인하기 전에는 따로 내지 않는다.
- **C5 단독(학습과 추론을 동시에).** 다음 연구들이 이미 덮는다. 남는 새로움은 가지치기뿐이라 A의 2단계로 넣는다.
  - Anisetti 2024, Tammali 2026, Stern 2022, Guo 2025.
  - Zhang 2026(BSD): 이름까지 "learning while inferring"이다.
  - Wang, S. 2025: RRAM 디지털 CIM에서 학습 중 가지치기를 했다.
  - Scellier 시뮬레이터는 정상상태 해법이라 동시 학습을 흉내 내지 못한다. 학습률/이완률 비를 스윕하는 동역학 시뮬레이터가 따로 필요하다.
- **C1 원안(열·행·타일이 통째로 빈다).** 활동 비례 예산은 열을 비우지 않고 열 사이 불균형을 만든다(Ibrayev 2024). 그래서 D로 좁혔다.
- **광자.** 4.4절 참고. 선행연구가 밀집해 있고 도구가 리눅스 쪽이다.

### 공통 사전 등록 규칙

1. 비용 항과 상수는 실행 전에 고정한다. 출처는 시뮬레이터 산출값과 논문 수치다. 우리가 만든 항이 있으면 따로 표기한다.
2. 같은 정확도 또는 같은 비용에서 비교하고, 정확도-비용 파레토 전선을 그린다.
3. BP 기준선은 그 기계에 맞는 것을 쓴다.
   - 크로스바: TT, c-TTv2, AGAD, 혼합정밀, 임계 BP.
   - 물리망: 같은 엣지 수 BP와 Stern 전력 감쇠 규칙.
   - 뉴로모픽: 프로토타입 판독.
4. 사소한 기준선(NCM, kNN, 해시)을 함께 돌린다.
5. 시드는 3개 이상이다. 검증을 축소하지 않고, 음성 결과도 보고한다.
6. 에너지는 "모델 추정"으로만 표기한다. EBRAINS 칩 실측을 쓸지는 먼저 결정한다.

## 6. 참고문헌

**크로스바**
- Gokmen & Vlasov 2016 (Front. Neurosci.). doi:10.3389/fnins.2016.00333; arXiv:1603.07341
- Marinella et al. 2018 (IEEE JETCAS). doi:10.1109/JETCAS.2018.2796379; arXiv:1707.09952
- Ambrogio et al. 2018 (Nature). doi:10.1038/s41586-018-0180-5
- Nandakumar et al. 2020 (Front. Neurosci.). doi:10.3389/fnins.2020.00406; arXiv:2001.11773
- Wu et al. 2026 (arXiv). arXiv:2609.36584
- Rasch et al. 2024 (Nat. Commun.). doi:10.1038/s41467-024-51221-z; arXiv:2303.04721
- Peng et al. 2021 (IEEE TCAD). doi:10.1109/tcad.2020.3043731; arXiv:2003.06471
- Wolters et al. 2023 (MWSCAS). doi:10.1109/mwscas57524.2023.10405905; arXiv:2212.14337
- Demirag et al. 2021 (arXiv). arXiv:2108.01804
- Renaudineau et al. 2026 (arXiv). arXiv:2601.09903
- Ren et al. 2025 (ICCAD). doi:10.1109/iccad66269.2025.11240745; arXiv:2507.15603
- Yi et al. 2023 (Nat. Electron.). doi:10.1038/s41928-022-00869-w
- Payvand et al. 2020 (IEEE JETCAS). doi:10.1109/JETCAS.2020.3040248; arXiv:2011.10852 (선행: arXiv:1910.06152)
- Wang, Y. et al. 2019 (arXiv). arXiv:1906.02393
- Cai et al. 2018 (DAC). doi:10.1109/dac.2018.8465850
- Li et al. 2026 (Nat. Commun.). doi:10.1038/s41467-025-67960-6; arXiv:2311.07164
- Wang, B. et al. 2025 (Sci. Adv.). doi:10.1126/sciadv.ads5340; arXiv:2407.18625
- Chu et al. 2020 (DAC). doi:10.1109/DAC18072.2020.9218523
- Ankit et al. 2017 (ICCAD). doi:10.1109/iccad.2017.8203823; arXiv:1708.07949
- Ibrayev et al. 2024 (arXiv). arXiv:2403.13082
- Bhattacharjee et al. 2022 (DATE). doi:10.23919/DATE54114.2022.9774736; arXiv:2201.05229
- Rathi et al. 2019 (IEEE TCAD). doi:10.1109/TCAD.2018.2819366; arXiv:1710.04734
- Le Gallo et al. 2023 (APL Mach. Learn.). doi:10.1063/5.0168089; arXiv:2307.09357
- Meirovitch et al. 2025 (arXiv). arXiv:2510.01263

**섭동·국소 학습 일반**
- McCaughan et al. 2023 (APL Mach. Learn.). doi:10.1063/5.0157645; arXiv:2303.03986
- Oripov et al. 2025 (APL Mach. Learn.). doi:10.1063/5.0258271; arXiv:2501.15403
- Dalm et al. 2026 (Neuromorph. Comput. Eng.). doi:10.1088/2634-4386/ae9be4; arXiv:2310.00965
- Katti et al. 2025 (arXiv). arXiv:2511.11362

**뉴로모픽**
- Renner et al. 2024 (Nat. Commun.). doi:10.1038/s41467-024-53827-9; arXiv:2106.07030
- Rostami et al. 2022 (Front. Neurosci.). doi:10.3389/fnins.2022.1018006
- Davies et al. 2021 (Proc. IEEE). doi:10.1109/JPROC.2021.3067593
- Davies et al. 2018 (IEEE Micro). doi:10.1109/MM.2018.112130359
- Shrestha et al. 2021 (DAC). doi:10.1109/DAC18074.2021.9586323; arXiv:2105.03649
- Hajizada et al. 2025 (arXiv). arXiv:2511.01553
- Stewart et al. 2024 (arXiv/Research Square). arXiv:2408.15800; doi:10.21203/rs.3.rs-3982214/v1
- Wunderlich et al. 2019 (Front. Neurosci.). doi:10.3389/fnins.2019.00260; arXiv:1811.03618
- den Blanken et al. 2026 (arXiv). arXiv:2607.29353
- Liu et al. 2018 (Front. Neurosci.). doi:10.3389/fnins.2018.00840
- Billaudelle et al. 2021 (Neural Networks). doi:10.1016/j.neunet.2020.09.024; arXiv:1912.12047
- Bogdan et al. 2018 (Front. Neurosci.). doi:10.3389/fnins.2018.00434
- Nguyen et al. 2021 (ICONS). doi:10.1145/3477145.3477157; arXiv:2010.04351
- Yan et al. 2019 (IEEE TBioCAS). doi:10.1109/TBCAS.2019.2906401; arXiv:1903.08500
- George et al. 2017 (BioCAS). doi:10.1109/biocas.2017.8325074
- Knight et al. 2026 (Neuromorph. Comput. Eng.). doi:10.1088/2634-4386/ae4535; arXiv:2510.19764
- Han et al. 2023 (IJCAI). doi:10.24963/ijcai.2023/334; arXiv:2308.04749
- Yik et al. 2025 (Nat. Commun.). doi:10.1038/s41467-025-56739-4; arXiv:2304.04640
- Dampfhoffer et al. 2023 (IEEE TETCI). doi:10.1109/TETCI.2022.3214509
- Ostrau et al. 2022 (Front. Neurosci.). doi:10.3389/fnins.2022.873935
- Dennler et al. 2024 (Nat. Mach. Intell.). doi:10.1038/s42256-024-00952-1; arXiv:2309.11555
- Tang et al. 2023 (ISCAS). doi:10.1109/ISCAS46773.2023.10181505; arXiv:2303.15224
- Michaelis et al. 2022 (Front. Neuroinform.). doi:10.3389/fninf.2022.1015624; arXiv:2109.12308
- Butz & van Ooyen 2013 (PLoS Comput. Biol.). doi:10.1371/journal.pcbi.1003259
- Zhang et al. 2026 (arXiv). arXiv:2610.03149
- Wang, S. et al. 2025 (Research Square 프리프린트). doi:10.21203/rs.3.rs-6896769/v1; arXiv:2506.13151

**물리 학습망**
- Scellier 2024 (ICML). arXiv:2402.11674
- Scellier et al. 2023 (NeurIPS). arXiv:2312.15103
- Kendall et al. 2020 (arXiv). arXiv:2006.01981
- Guzman et al. 2026 (PNAS). doi:10.1073/pnas.2525792123; arXiv:2509.15842
- Guzman et al. 2025 (PRX). doi:10.1103/f2hb-c9s1; arXiv:2412.19356
- Dillavou et al. 2022 (Phys. Rev. Applied). doi:10.1103/PhysRevApplied.18.014040; arXiv:2108.00275
- Dillavou et al. 2024 (PNAS). doi:10.1073/pnas.2319718121; arXiv:2311.00537
- Dillavou et al. 2025a (arXiv). arXiv:2505.22887
- Dillavou et al. 2025b (arXiv). arXiv:2511.17825
- Stern et al. 2022 (Phys. Rev. Research). doi:10.1103/PhysRevResearch.4.L022037; arXiv:2112.11399
- Stern et al. 2024 (APL Mach. Learn.). doi:10.1063/5.0181382; arXiv:2310.10437
- Anisetti et al. 2024 (Neural Comput.). doi:10.1162/neco_a_01648; arXiv:2208.08862
- Tammali & Olin-Ammentorp 2026 (arXiv). arXiv:2610.07283
- Lopez-Pastor & Marquardt 2023 (PRX). doi:10.1103/PhysRevX.13.031020; arXiv:2103.04992
- Pashine 2021 (Phys. Rev. Materials). doi:10.1103/PhysRevMaterials.5.065607; arXiv:2101.06793
- Goodrich et al. 2015 (PRL). doi:10.1103/physrevlett.114.225501
- Rocks et al. 2017 (PNAS). arXiv:1607.08562
- Rocks et al. 2019 (PNAS). doi:10.1073/pnas.1806790116; arXiv:1805.00504
- Farinha et al. 2020 (arXiv). arXiv:2006.08798
- Wang, Q. et al. 2026 (arXiv). arXiv:2601.21945
- Straszak & Vishnoi 2022 (Math. Program.). doi:10.1007/s10107-021-01644-z; arXiv:1601.02712
- Ronellenfitsch & Katifori 2016 (PRL). doi:10.1103/PhysRevLett.117.138301; arXiv:1606.00331
- Momeni et al. 2023 (Science). doi:10.1126/science.adi8474; arXiv:2304.11042
- Momeni et al. 2025 (Nature). doi:10.1038/s41586-025-09384-2; arXiv:2406.03372
- Wright et al. 2022 (Nature). doi:10.1038/s41586-021-04223-6; arXiv:2104.13386
- Nakajima et al. 2022 (Nat. Commun.). doi:10.1038/s41467-022-35216-2; arXiv:2204.13991

**광자·확률**
- Gu et al. 2021a (AAAI). doi:10.1609/aaai.v35i9.16928; arXiv:2012.11148
- Gu et al. 2021b (NeurIPS, L2ight). arXiv:2110.14807
- Xu et al. 2024 (Optica). doi:10.1364/OPTICA.523225; arXiv:2401.08180
- Ashtiani et al. 2026 (Nature). doi:10.1038/s41586-026-10262-8
- Guo et al. 2025 (arXiv). arXiv:2506.18041
- Niazi et al. 2024 (Nat. Electron.). doi:10.1038/s41928-024-01182-4; arXiv:2303.10728
- Laydevant et al. 2024 (Nat. Commun.). arXiv:2305.18321

## 7. 추천

### 7.1 판정 점수

세 판정자가 렌즈를 하나씩 맡아 점수를 매겼다. 렌즈마다 척도를 맞추지 않았으므로 점수를 합산하지 않는다.

| 후보 | 성공 | 새로움 | 맞춤 | 판정 요지 |
|---|---|---|---|---|
| A. 저항망 학습 중 가지치기 | 4 | 6 | 8 | 세 렌즈 모두 1위. 다만 원안의 머리 주장(국소 물리 신호가 크기를 이긴다)은 약하다 |
| B. 크로스바 쓰기 장부 | 3 | 4 | 4 | 세 렌즈 모두 2위(맞춤은 D와 공동). 임계 혼합정밀(Nandakumar 2020)과 오류 트리거(Payvand 2020)가 쓰기를 이미 거의 없앴다. 가지치기로 쓰기를 줄이는 방식은 Li 2026이 선점했다 |
| C. 뉴로모픽 FSCIL 예산 | 2 | 3 | 3 | 프로토타입 판독이 거의 최적이다(Yik 2025). George 2017이 발상을 앞서고, 설치에 2–3일이 든다 |
| D. 크로스바 허브 매핑 | 2 | 2 | 4 | Ibrayev 2024가 부정 결과를 예측한다. 반증용이다 |
| A 개정안 | A′ 약 6 | A*로 좁히기 권고(점수 없음) | A-local, 원안보다 낫다고 판단(점수 없음) | 판정자마다 1차 가설을 다르게 제안했다(7.3) |

### 7.2 고른 실험과 이유

- **주 실험: A′, 자기학습 저항망(DRN, Scellier 2024 코드)에서 학습 중 덜어냄.** 도구와 PC는 A와 같고, 1차 가설만 바꾼다.
- **백업: B 1단계(MLP+NeuroSim V3.0).** 조건부로만 연다(7.7).

세 렌즈 모두 A를 1위로 꼽았고, 다른 후보는 어느 렌즈에서도 4점을 넘지 못했다. A에서는 제3자 코드가 키르히호프 정상상태를 정확히 푼다. 에너지 함수가 곧 소산 전력이고 엣지 수가 곧 소자 수라서, 네 후보 가운데 점수판이 가장 덜 순환적이다. 판정 단계에서 이 PC(RTX 3060 Ti)로 EP 한 걸음(배치 4)을 재니 DRN-XS 13.9 ms, DRN-1H 14.1 ms였다. 두 망 모두 에폭당 약 3.5분이라 본 실행이 2밤 안에 끝나고, B·C·D처럼 빌드 미검증·RunPod·2–3일 설치가 먼저 걸리지도 않는다. 다만 원안의 머리 주장은 우리 결과가 반대를 예측한다(논문 1에서 국소 규칙은 전역 크기보다 0.1–0.9점 뒤졌고, p3에서 허브는 층 순위 크기와 거의 비겼다). 그래서 1차 가설을 '학습 중 덜어냄 자체'로 바꾸고, 국소성과 노드 예산은 2차 가설로 내린다.

5절 A행의 시간 추정('DRN-XS 먼저', '3060 Ti 미측정')은 이 실측으로 대체한다. DRN-1H로 바로 간다.

### 7.3 판정자 이견

**1차 가설이 갈린다.**
- 성공 판정자(A′): 기준을 전역 크기 g로 고정한 '덜어냄 자체'를 머리에 둔다.
- 새로움 판정자(A*): 활동 비례 노드별 전도도 예산을 머리에 둔다.
- 맞춤 판정자(A-local): 층 전체 top-k는 중앙 정렬, 곧 폰노이만 연산이라고 본다. 그래서 각 엣지가 자기 신호만 보게 바꾼다. 문턱은 방송 스칼라 하나(예: 전원이 공급하는 총전력)가 정하고, 쓰이지 않는 엣지는 전도도 감쇠로 바닥에 밀린 뒤 체류 시간이 지나면 지운다. 이 국소 규칙이 p3의 허브 배분을 스스로 만드는지 묻는다.

**채택.**
- A′는 1차 가설(P1–P3), A-local은 2차 가설(P4–P5)로 둔다.
- A*는 파일럿 5에서 허브가 크기를 앞서는 밀도가 있을 때만 팔로 넣는다.
- A*의 근거인 p3의 76.1 대 31.2는 재학습 없이 한 번 자른 뒤 행별 Wanda와 붙인 결과다. 층 순위 크기와는 거의 비겼다(1%에서 +4.7 불확실, 5%에서 크기가 1.2점 확실히 앞섬; results/p3/REPORT.md).
- 논문 1에서도 이득의 73–81%는 기준이 아니라 점진 일정에서 왔다(docs/worklog.md).
- 그래도 DRN-1H는 정규화 없는 MLP형이라 허브가 통한 영역에 있다. 허브가 실패한 BN CNN은 범위 밖이다.

**게이지 고정 방식이 갈린다.**
- 성공 판정자: 소자 창 [g_max/R, g_max](R = 10, 100). 창 아래 값은 바닥 유지와 제거 두 가지로 처리한다.
- 새로움 판정자: g_min 하한(예: 초기 중앙값의 1%) 또는 고정 출력 부하.
- 맞춤 판정자: 모든 팔에 공통 상한 g_max를 두고, 채점 때 max g = g_max로 재축척한다.
- 파일럿 3 뒤에 하나로 정한다.

**기준 팔을 접는 선.** 새로움 판정자는 ±0.5%p, 맞춤 판정자는 1점을 제안했다. 여기서는 1%p로 통일한다.

**사실 정리.** 새로움 판정자는 EP의 자연 0(초기 49.9%)을 위협으로 들었다. 맞춤 판정자 실측에서 이 비율은 0.2에폭 뒤 2.6%로 줄었고, 마스크 없이 0으로 만든 자리의 87.9%가 300걸음 안에 되살아났다. 자연 희소는 약한 기준선이고, 영구 마스크는 필수다.

### 7.4 설계·점수판·기준선

**설계.**
- 망: DRN-1H(입력 1568, 은닉 1024, 출력 10, 1,615,872 엣지).
- 데이터·학습기: MNIST, EP(centered). 모든 팔을 10에폭으로 맞춘다.
- 밀도: 9.77%(= DRN-XS 157,800 엣지)와 2%(≈ 은닉 20 밀집망).
- 참고 기준값: Scellier 2024 표 1에서 XS EP는 3.46 ± 0.07%(10에폭), 1H EP는 1.57 ± 0.07%(50에폭)다.
- Fashion-MNIST는 P1을 통과한 뒤에 한다.

| 팔 | 역할 |
|---|---|
| ① 밀집 1H | 상한 |
| ② 학습 중 전역 크기 점진 가지치기(3차 일정) | 1차 가설 |
| ③ 같은 엣지 수의 작은 밀집망, EP와 BP(TBP) | 소자 수 축 기준선 |
| ④ 정적 무작위 희소 1H(층별 밀도는 ②의 최종값) | 설계 희소 기준선 |
| ⑤ Stern 2024 λ·전력 규칙 + 0 클램프(λ 2값) | 가장 강한 공정 기준선 |
| ⑥ 국소 감쇠·체류 제거(Optimizer의 weight_decay로 시작, 방송 문턱 하나, 담금질 변형 포함) | 2차 가설 |

규모는 13개 설정 × 시드 3 = 39회다. 1H 10에폭 35분 기준 약 23 h(2밤, 추정)가 든다.

**점수판.** 실행 전에 PREDICTIONS에 고정한다.
- 1차 물리 축: 활성 엣지 수(= 갱신 회로 수). 영구 마스크로 강제하고, 부활 엣지가 0개인지 에폭마다 확인한다.
- 정확도: 시험 오차. 밀도 축 전 구간의 시험 hinge 손실도 함께 본다(Dillavou 2025b).
- 2차 물리 축: 게이지를 고정한 자유 상태 소산 전력 ½Σg·ΔV².
- 추정 추론 에너지: 활성 엣지 수 × 10–20 pJ(Dillavou 2024, 2 µs 가정). 모델 추정으로만 표기한다.
- 덜어냄에 불리한 비용도 보고한다: 제작 엣지 수(면적), 학습 에너지(Guzman 2026·pyrkm의 RKM 해석 모델 항을 옮긴 값이므로 우리 항으로 표기).
- 비교 방식: 같은 시험 오차에서 비교하고, 오차-엣지와 오차-전력의 파레토 전선을 그린다. 시드는 3개다.

**가장 강한 공정 기준선은 ⑤다.** 같은 제작망·같은 EP·같은 0 클램프·같은 게이지에서 돌린 Stern 2024 λ·전력 규칙이다. 세 판정자 모두 이것을 가장 가까운 선행으로 꼽았다. 클램프와 만나면 ΔV² 가중 L1처럼 정확한 0을 스스로 만들 수 있어서, 엣지 축과 전력 축 모두에서 맞붙는다. 소자 수 축에서 ②와 짝을 이루는 기준선은 ③(EP·BP 중 나은 쪽)이다.

### 7.5 사전 등록 예측

- **P1 (경로 판정).** 2% 밀도에서 ②의 시험 오차가 ③보다 낮다. 기대 폭은 1%p 안팎이다(논문 1 GPU MLP는 2%에서 +1.35점, 10%에서 +0.31점; docs/report/tables/core_acc_matrix.md). **시드 3 평균 차이가 0.5%p 미만이면 '이 기판에서 학습 중 덜어냄은 소자 수 점수판에서 이득이 없다'고 판정하고 이 경로를 접는다.**
- **P2 (설계 희소 대비).** 같은 2%에서 ②의 시험 오차가 ④보다 낮다. 논문 1에서는 +2.75점이었다. 하지만 ④는 층별 밀도를 맞춰 더 강하므로 방향만 등록한다. 차이가 0.5%p 미만이면, 학습 중 덜어냄의 몫이 고정 희소 설계(Wang, Q. 2026)와 구별되지 않는다고 적는다.
- **P3 (Stern 대비).** ⑤는 λ>0에서 정확한 0의 비율을 늘린다. 그래도 게이지를 고정한 오차-활성 엣지 전선에서는 ②가 ⑤보다 앞선다(방향). 뒤지거나 겹치면 엣지·전력 결과는 Stern 2024를 이 시뮬레이터에서 재현한 것으로 적고, 새 결과로 내지 않는다.
- **P4 (국소성).** 같은 활성 엣지 수에서 ⑥의 시험 오차는 ②보다 0–1%p 높다. 논문 1의 국소 규칙은 같은 밀도의 전역 크기보다 10%에서 0.14점, 2%에서 0.43점 뒤졌다. 차이가 1%p 안이면 '중앙 정렬 없이 같은 점에 닿는다'를 양의 결과로 쓴다. 1%p를 넘으면 이 기판에서도 덜어냄은 중앙 정렬에 기댄다고 적는다.
- **P5 (허브 재현, 판정자 이견).** ⑥이 남긴 은닉 노드별 입력 수는 그 노드가 받는 활동과 양의 스피어만 상관을 보인다(p3의 전역 기준에서는 0.80–0.99). A* 팔에 대해서는 판정자 예측이 갈린다. 성공·맞춤 판정자는 회복 뒤 전역 크기와 1%p 안에서 비긴다고 보고, 새로움 판정자는 A*가 앞선다고 본다. 파일럿 5에서 먼저 판독하고, 결과는 어느 쪽이든 그대로 보고한다.

### 7.6 먼저 할 싼 파일럿

GPU 파일럿은 torch 프로세스 하나씩 순차로 돌리며, 합계 약 4–5 h가 든다.

1. **선행연구 확인**(GPU 없음, 1–2 h, 다른 파일럿과 병행).
   - 8절 목록대로 확인한다.
   - 물리망에서 지도 EP/CL 학습 중 중요도 기준으로 엣지를 영구 제거한 결과가 하나라도 나오면, 그 논문을 비교 팔로 넣고 주장을 좁힌다.
2. **제거 규칙**(약 30분).
   - 매 clamp_ 직후 영구 마스크를 곱한다.
   - 빈 노드는 장치 목록에서 지우거나, 접지 누설 g_leak을 두고 그 전력을 장부에 넣는다.
   - DRN-XS에 1% 허브 마스크와 1% 무작위 마스크를 걸고 1에폭씩 돌린다.
   - 통과 조건: NaN 0건, 부활 엣지 0개, 실제 밀도와 목표 밀도의 일치.
   - 이 파일럿이 필요한 이유: 판정 단계 실측에서 은닉 노드 하나를 떼자 출력 전체가 NaN이 됐다(갱신 −b/2a에서 a = 0).
3. **게이지·Stern**(약 1 h).
   - 축척 퇴화는 이미 확인됐다(×0.1에서 출력 상대 차 약 4e-7, 전력 비 0.100).
   - XS 밀집과 10% 무작위 마스크를 3에폭씩 돌려 에폭별 Σg와 전력을 기록한다. 남은 엣지가 Σg를 키워 보상하면 원시 전력 비교는 무효다.
   - ⑤를 λ 두 값으로 1에폭씩 돌려 0 비율과 게이지 고정 전력-오차 점을 찍는다.
   - 게이지 규칙을 여기서 고른다.
   - ⑤가 같은 오차·같은 엣지 수에 이르면 전력 주장을 접고 엣지 축만 남긴다.
4. **재현과 덜어냄 자체**(약 1.5 h, P1 조기 판독).
   - XS 밀집 10에폭이 표 1의 3.46 ± 0.07%에 드는지 본다.
   - 이어서 XS 밀집, 1H 학습 중 9.77% 가지치기, 1H 정적 무작위 9.77%를 4에폭·시드 1로 돌린다.
   - 가지친 1H가 나머지 둘보다 0.5%p 이상 낫지 않으면 2%에서 한 번 더 본다. 거기서도 차이가 없으면 A를 접는다.
5. **기준 차이**(약 1.5 h).
   - 1H를 10에폭 학습한 뒤 5·2·1%에서 다섯 기준으로 한 번씩 자르고, 1에폭씩 회복시킨다.
   - 기준: 크기, 소산 g·ΔV², 허브(g·|V_pre|·|V_post| 층 순위), 감수율 대용(BP 추정기의 |∂출력/∂g|), 무작위.
   - 모든 밀도에서 크기 대비 1%p 안이면 기준 팔은 본 실행에서 뺀다.

### 7.7 백업: B 1단계(조건부)

다음 경우에 B 1단계(MLP+NeuroSim V3.0)를 연다.
- A가 P1에서 접힐 때.
- 파일럿 2–3에서 시뮬레이터 문제로 막힐 때.

B를 백업으로 고른 이유는 두 가지다.
- 세 렌즈 모두 2위다.
- 남은 후보 가운데 제3자 시뮬레이터가 학습 비용(쓰기·읽기 에너지, 펄스 수)을 내 주면서 이 PC에서 돌 가능성이 있는 것은 B 1단계뿐이다.

주장은 '임계 혼합정밀 BP 위에 학습 중 가지치기를 얹으면 정확도당 쓰기가 더 준다' 하나로 좁힌다. B를 여는 조건은 8절에 있다.

나머지 후보의 처리:
- D: 맞춤 판정자가 4점을 줬지만 백업으로 두지 않는다. A가 양으로 나온 뒤 크로스바로 넓힐지 반나절에 정하는 반증 시험으로 남긴다.
- C: 보류한다.

## 8. 열린 질문

계획서(docs/plans/, 가칭 p5)와 PREDICTIONS를 쓰기 전에 확인할 것이다.

**도구·환경**
- energy-based-learning 코드는 스크래치패드(세션 임시 폴더)에 있는 스냅숏뿐이다.
  - GitHub API로 받았고, .git과 LICENSE 파일이 없으며 커밋 해시도 기록돼 있지 않다.
  - 프로젝트 안으로 옮기면서 커밋 해시와 MIT LICENSE를 함께 고정한다.
- 판정 단계의 시간 실측은 저장소 학습 루프로 잰 값이 아니다.
  - 데이터를 GPU에 미리 올린 탐침 스크립트(스크래치패드의 drn_probe.py, drn1h_time.py)로 쟀다.
  - 1H의 35분은 300걸음에서 외삽한 값이므로, 첫 실제 에폭으로 다시 잰다.
  - 저장소 DataLoader(datasets.py, num_workers=1)를 쓰면 num_workers를 0으로 바꾼다.
  - 실행은 python -u로 하고, torch 프로세스는 하나만 돌린다.
- pyrkm은 venv에 없다(pip로 설치할 수 있다). 학습 에너지로 옮길 항(노드 RC 이완, 비교기, 게이트 전압 갱신)과 상수를 등록 전에 고정한다.
- aihwkit, CrossSim, NeuroBench, SANA-FE, Lava도 venv에 없음을 확인했다. A에는 필요 없다.

**등록 전에 정할 설계**
- 미리 고정할 항목:
  - 게이지 규칙(7.3의 세 안 중 하나).
  - 빈 노드 규칙.
  - ⑥의 감쇠 세기, 체류 길이 T, 방송 문턱 제어, 담금질 일정.
  - P5의 '받는 활동' 정의(예: 노드를 지나는 전류 절댓값 합).
- 층별 밀도 배분.
  - 2%에서 1H 출력층(10,240 엣지)은 예산의 약 32%를 차지한다. 출력층을 유지할지, 전역 순위와 층별 균일 중 무엇을 쓸지 정한다.
  - ④를 층별 균일 2%로 두면 은닉 노드당 출력 엣지가 평균 0.2개라, 대부분 막다른 노드가 된다. 그래서 ④는 ②의 최종 층별 밀도에 맞춘다.
- 갱신 편향. 5절은 갱신 편향 모델과 과클램핑 팔을 넣는다고 했지만, 노트에 편향 크기 수치가 없다. Dillavou 2025a 원문에서 가져오거나, 이번 범위에서 뺀다고 적는다.

**선행연구** (계획서 '관련 선행연구와 우리 차이' 칸의 전제)
- 원문으로 확인할 대상:
  - †Lin et al. 2026(arXiv:2602.03546)의 부분 갱신 절. 초록에 따르면 저항값 일부만 갱신해도 되는데, 이것이 갱신 회로 절감 논리와 겹친다.
  - Farinha 2020의 가지치기 절.
  - Stern 2024의 g_min 논의.
- 판정 단계에서 새로 나온 표 밖 문헌(초록만 확인, 부록 C에도 없음):
  - †Ji & Gross 2020(ISCAS, doi:10.1109/iscas45731.2020.9180548): 디지털 EP에 하드웨어 지향 가지치기를 적용했다.
  - †Tian et al. 2026(Natl. Sci. Rev., doi:10.1093/nsr/nwag547): 크로스바 현장 커널 가지치기. B의 위협이다.
- 2025-01 이후 문헌 훑기.
  - 대상: OpenAlex와 arXiv 목록(cond-mat.soft, cs.ET).
  - 검색 조합: coupled learning·equilibrium propagation·physical learning × prune·sparse·remove·dilute.
  - Stern 2024·Guzman 2025·Dillavou 2022를 낸 그룹의 새 프리프린트를 먼저 본다.
  - 방송 문턱·체류 제거 변형은 노트의 검색어가 일부만 덮는다.
  - WebSearch 한도가 소진돼 OpenAlex·arXiv·Crossref API를 쓴다.
- †arXiv:2606.15443·2606.15444·2606.09756의 초록에는 가지치기 내용이 없었다(판정 단계). 엣지 제거와 수렴 조건의 관계는 아직 원문으로 확인하지 않았고, 실무적으로는 파일럿 2의 NaN 검사로 대신한다.

**백업 B를 열 조건**
- 빌드: MSYS2 g++ 11.2 + OpenMP로 빌드되는지, 첫 실행 시간이 얼마인지 확인한다(둘 다 미검증). 라이선스는 CC BY-NC 4.0이다.
- 새로움: †arXiv:2111.09272, †doi:10.1002/aisy.202500150, †Tian et al. 2026을 확인하기 전에는 새로움을 주장하지 않는다.
- 전제 재작성: '배치 1에서 갱신 비용 상각이 사라진다'는 전제를 다시 쓴다. 판정 단계에서 README로 확인한 바로는 MLP+NeuroSim이 원래 온라인 학습 벤치마크이고, 배치 인자는 2021-08에 들어왔다.
- 스트림: 변별력 있는 스트림을 고른다. p4-A(배치 1, 과제당 2,000장, 단일 통과 클래스 증분)에서는 역전파도 마지막 정확도가 15–21%로 우연(10%) 근처였다(results/p4/REPORT_MORNING.md).
- 비용: 2단계(aihwkit)는 RunPod 비용이 들어 사용자 승인이 필요하다.

**측정 규칙**
- EBRAINS 칩 실측을 허용할지는 A와 무관하다. B·C나 칩 확인 단계로 갈 때 정한다.

## 부록: 제외 항목과 이유

**A. 판정이 unverifiable 또는 refuted인 것**
- LASANA(arXiv:2507.10748): unverifiable. 코드 링크가 없고, 재확인 단계에서 원문을 가져오지 못했다.
- "MGD 코드는 공개되지 않았다"는 스카우트 주장: refuted. 정정: github.com/bakhromtjk/mgd_scaling (GPL-2.0, 커밋 4개, 마지막 2025-01-16)에 있다.

**B. 수치가 재확인되지 않아 근거로 쓰지 않은 것**
- Penn CLLN 하드웨어의 학습 회로 전력(엣지당 약 100 mW), t_h = 100 µs, τ0 = 18 ms. 엣지당 10–20 pJ는 Dillavou 2024 항목에서 확인돼 썼다.
- THRML 논문(arXiv:2510.23972)의 에너지 수치.
- LightOn 광학 DFA 에너지(arXiv:2409.12965).
- SATA 논문(arXiv:2204.05422)의 1.27배 수치.
- Stewart 2024의 표본당 에너지 표. 자기 전력×시간 값과 맞지 않는다.
- PhyLL의 인라인 수치 두 개(60%, 55%). 판독할 수 없었다.
- NengoLoihi의 Nengo 3.2.0 상한.
- CrossSim IISWC2026 튜토리얼 내용.

**C. 검증 표 밖 2차 언급(본문 †, 위협·확인 대상으로만 사용)**
- 크로스바·IMC
  - Lei et al. 2026(arXiv:2608.21223, 논문 표 누락)
  - arXiv:2601.10037(아날로그-디지털 LoRA)
  - arXiv:2111.09272(크로스바 인지 온칩 가지치기)
  - doi:10.1002/aisy.202500150(선택 갱신)
  - arXiv:2411.18272(온도로 저장한 적격성 흔적)
  - arXiv:2304.11337
  - arXiv:2608.25781(갱신 교란)
  - Wu et al. 2024 NeurIPS(arXiv:2406.12774)
  - 1990년대 아날로그 VLSI 섭동 학습(Jabri & Flower 1992 등)
  - DNPU 연구
- 크로스바 가지치기: TinyADC 2021, Group Scissor 2017, Liang et al. 2018, OWL(arXiv:2310.05175)
- 물리망
  - arXiv:2606.15443(수렴 조건)
  - arXiv:2606.15444(보존 법칙)
  - Lin et al. 2026(arXiv:2602.03546, 코드 404)
  - Kim et al. 2026(arXiv:2606.09756)
  - Wycoff et al. 2022
  - Temporal Contrastive Learning(arXiv:2312.17723)
  - EBANA
- 뉴로모픽
  - EventProp(arXiv:2412.15021, arXiv:2302.07141)
  - doi:10.1038/s41467-026-70586-x
  - Cramer et al. 2022
  - arXiv:2609.32317
  - CLANE(arXiv:2605.28387)
  - Timcheck et al. 2026(arXiv:2601.10035)
  - Davidson & Furber 2021
  - ReckOn(arXiv:2208.09759)
  - EqSpike
  - 메타가소성 연구들
- 광자: CHAMP(OFC 2022)
- 원문을 충분히 읽지 못한 것
  - Hiratani et al. 2022: 이번 자료에서는 제목만 확인됐다. p4 계획서의 원문 확인 기록과는 별개다.
  - ANP-G: 짧은 초록만 있다.
  - Crafton 2019: 수치가 없다.
  - STELLAR: 프리프린트 초록만 있다.
  - Dalgaty 2021: 추적하지 않았다.
  - Imam & Cleland 답글: 읽지 않았다.

**D. 제외한 도구**

| 이유 | 도구 |
|---|---|
| 추론 전용 | aihwkit-lightning(torch==2.13 고정), MNSIM 2.0(라이선스 없음), MemTorch(2022 이후 방치), MemIntelli, ZigZag(Python 3.11 이상), SpikeSim |
| 비용 전용·지원 중단 | CiMLoop(MIT식 라이선스, AccelForge로 대체됨) |
| 옛 스택 | CrossSim 2.0(Python 3.7), lava-dl(torch<2.4), Brian2GeNN, NengoLoihi, holomorphic_eqprop |
| 윈도우 불가 | NEST, ANNarchy, Brian2CUDA, XyloSim |
| 온칩 학습 없음 | Rockpool/Xylo, Sinabs/Speck(samna 스텁이 import할 때 설치를 시도함), py-spinnaker2(비공개), Akida(1비트 마지막 층만) |
| 일반 상수·미성숙 | syops(45 nm, AC 0.9 pJ·MAC 4.6 pJ), spikeforge |
| 학습 기판 아님·방치 | AnalogVNN, photontorch, simphony/SAX, torchoptics/LightRidge, neurophox, Physics-Aware-Training(디지털 BP 중심), thermox, simulated-bifurcation, lightonml(OPU 필요), JAX-MD(계산 비용 모델 없음) |
| 코드 비공개 | LANL differentiable-circuits |

**E. 검색 한계**
- DBLP는 시간 초과 또는 봇 차단, Semantic Scholar와 arXiv API는 429로 막혔다. 대신 arXiv HTML, Crossref, DataCite, OpenAlex, Europe PMC를 썼다.
- 그래서 학회 전용 문헌은 덜 덮였을 수 있다. "선행연구 못 찾음"은 위 검색어와 날짜(2026-10-08)에 한정된 판정이다.