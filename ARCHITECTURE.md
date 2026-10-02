# Subtractive Intelligence: A Brain-Inspired Architecture Beyond Transformers

## Project Overview

현재 AI(Transformer)는 가산적(additive) 학습에 의존한다 — 랜덤 초기화에서 파라미터를 축적.
생물학적 뇌는 감산적(subtractive) 학습을 사용한다 — 과잉 연결에서 불필요한 것을 제거.
이 프로젝트는 뇌의 4대 원리를 통합한 아키텍처를 구현하고, 소규모 벤치마크에서 검증한다.

---

## 4대 핵심 원리

### 원리 1: 가지치기 (Subtractive Learning)
- 학습 = 연결을 추가하는 것이 아니라 불필요한 연결을 제거하는 것
- 과잉 초기 상태에서 시작 → 경험을 통해 깎아냄
- 뇌: 영아기 시냅스가 성인의 2배 → 사춘기까지 50% 제거 → 더 똑똑해짐
- AI 대응: Lottery Ticket Hypothesis (큰 네트워크 안에 "당첨 서브네트워크" 존재)
- 핵심 차이: AI는 학습 후 프루닝, 뇌는 학습 과정 자체가 프루닝

### 원리 2: HW=SW 통합 (Compute-Memory Unity)
- 폰 노이만: 프로세서(연산) ↔ 메모리(저장) 분리 → 데이터 이동 = 전력 소모 주범
- 뇌: 시냅스가 동시에 연산 + 기억 + 학습. 데이터 이동 없음
- 뉴런은 상태유지(stateful) 소자 — 이전 이력이 다음 반응에 영향
- 시냅스 가중치가 4중 시간 스케일로 변화:
  - 밀리초: 소포 고갈/회복 (단기 상태)
  - 분~시간: 수용체 수 변화 (LTP/LTD)
  - 일: 단백질 합성, 시냅스 크기 변화
  - 주~월: 시냅스 생성/소멸, 구조적 변화
- 소프트웨어 시뮬레이션 한계: 진짜 HW=SW 통합은 불가능하지만, 상태유지 그래프로 근사 가능

### 원리 3: 꺼내쓰기 구조 (Retrieval Architecture)
- AI는 이미 충분한 지식을 갖고 있다. 문제는 꺼내쓰는 구조.
- 뇌의 4대 꺼내쓰기 모듈:
  1. 시상 (Thalamus) = 동적 라우터. 입력을 사전 필터링. "뭘 처리할지" 결정. O(1)
  2. 억제 뉴런 = 관련 없는 것을 적극적으로 끔. 측면 억제 → 희소 활성화
  3. 전전두엽 = 메타인지. 출력의 확신도 판단. "이거 맞아?" 체크
  4. 해마 = 인덱스. 어디에 뭐가 저장되어 있는지 아는 시스템
- Transformer에 없는 것: 사전 필터링, 적극적 억제, 메타인지, 인덱스

### 원리 4: 정보 비균등 + 접지 (Information Concentration + Grounding)
- 정보는 균등하게 분포하지 않고 핵심 소수에 집중됨 (Zipf 법칙, 파레토)
- 뇌는 이걸 선험적으로 안다: 엣지 검출기, 움직임 검출기가 유전적으로 내장
- 물리적 사전 지식이 가장 밀도 높은 정보원
- 뇌는 신체를 통해 물리 법칙에 O(1) 접근
- AI는 이 사전 지식이 없어서 수조 토큰으로 비싸게 배움
- 르쿤의 JEPA가 이 방향을 시도 중

---

## 아키텍처 설계

### 전체 구조

```
입력
  │
  ▼
┌─────────────────────────────────────────────┐
│  시상 모듈 (Thalamic Router)                 │
│  - 입력 분석 → 관련 전문가 영역 결정          │
│  - 비관련 경로 억제 (희소 활성화)              │
│  - 피드백 기반 라우팅 조정                     │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  피질 컬럼 모듈 × N (Cortical Columns)       │
│  - 동일 구조의 반복 모듈 (Standard Cell)      │
│  - 각 컬럼: 상태유지 노드 + STDP 기반 학습    │
│  - 억제 뉴런이 지역적 희소성 강제 (20:80)     │
│  - Temporal Graph: 간선 가중치 다중 시간 스케일 │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Column 1 │←→│ Column 2 │←→│ Column N │   │
│  └──────────┘  └──────────┘  └──────────┘   │
│       ↕              ↕              ↕        │
│  [양방향 피드백: 예측 코딩 루프]               │
│  - 톱다운: 예측 생성                          │
│  - 바텀업: 실제 입력                          │
│  - 차이(prediction error)만 전파              │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  해마 모듈 (Hippocampal Module)              │
│  - 빠른 학습률: 원샷 기록                     │
│  - 패턴 분리: 유사 입력을 다른 패턴으로 저장    │
│  - 패턴 완성: 부분 입력 → 전체 복원            │
│  - 주기적 리플레이 → 피질 모듈로 전사          │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  메타인지 모듈 (Metacognition Module)         │
│  - 다른 모듈 출력을 모니터링                   │
│  - 확신도(confidence) 점수 산출               │
│  - 확신도 < 임계값 → 재처리 트리거             │
│  - 확신도 극히 낮음 → "모르겠다" 출력           │
└─────────────┬───────────────────────────────┘
              │
              ▼
출력 + 확신도 점수
```

### 핵심 메커니즘 상세

#### A. 시상 모듈 (Thalamic Router)
```python
# 개념적 의사코드
class ThalamicRouter:
    """
    트랜스포머 어텐션과의 핵심 차이:
    - 어텐션: 모든 토큰을 처리한 후 가중치 매김 (사후)
    - 시상: 처리하기 전에 관련 경로만 활성화 (사전)
    
    구현 방향:
    - 경량 분류기가 입력을 분석
    - Top-K 전문가 컬럼만 활성화 (K = 전체의 1~5%)
    - 나머지 95~99%는 억제 (연산 비용 0)
    - MoE의 게이팅과 유사하지만, 피드백 루프 추가
    """
    def route(self, input, feedback_from_columns=None):
        # 1차: 입력만으로 라우팅 결정
        relevance_scores = self.classifier(input)
        
        # 피드백이 있으면 라우팅 조정 (뇌의 피질→시상 피드백)
        if feedback_from_columns is not None:
            relevance_scores = self.adjust(relevance_scores, feedback)
        
        # 상위 K%만 활성화
        active_columns = top_k_percent(relevance_scores, k=5)
        return active_columns
```

#### B. 피질 컬럼 + STDP 학습
```python
# 개념적 의사코드
class CorticalColumn:
    """
    핵심 차이: 역전파 없이 지역 규칙으로 학습
    
    STDP (Spike-Timing-Dependent Plasticity):
    - A가 먼저 발화 → B 나중 발화: A→B 강화 (인과관계)
    - B가 먼저 발화 → A 나중 발화: A→B 약화
    - 시간 차이가 중요: 수십 ms 이내만 유효
    """
    def __init__(self, n_excitatory, n_inhibitory):
        self.neurons = StateNeurons(n_excitatory + n_inhibitory)
        self.synapses = TemporalGraph()  # 방향 그래프, 가중 간선
        
    def process(self, input, dt):
        # 1. 입력에 의해 일부 뉴런 활성화
        active = self.neurons.activate(input)
        
        # 2. 억제 뉴런이 주변을 끔 (측면 억제 → 희소성)
        active = self.lateral_inhibition(active)  # ~5%만 생존
        
        # 3. STDP로 시냅스 가중치 업데이트 (지역 규칙)
        for synapse in self.synapses.active_edges():
            pre_time = synapse.pre_neuron.last_spike_time
            post_time = synapse.post_neuron.last_spike_time
            delta_t = post_time - pre_time
            
            if delta_t > 0:  # pre가 먼저 → 인과관계 → 강화
                synapse.weight += A_plus * exp(-delta_t / tau_plus)
            else:  # post가 먼저 → 비인과 → 약화
                synapse.weight -= A_minus * exp(delta_t / tau_minus)
        
        # 4. 약한 시냅스 가지치기 (주기적)
        self.synapses.prune(threshold=min_weight)
        
        return active
```

#### C. 예측 코딩 루프
```python
# 개념적 의사코드
class PredictiveCodingLoop:
    """
    핵심: 예측과 다른 것만 처리
    
    트랜스포머: 매번 전체 입력을 처리
    예측 코딩: 변한 것만 처리 → 연산량 극적 감소
    """
    def process(self, actual_input):
        # 톱다운: 이전 상태로부터 예측 생성
        prediction = self.higher_level.predict()
        
        # 예측 오차 계산
        prediction_error = actual_input - prediction
        
        # 오차가 임계값 이하면 패스 (변화 없음 = 연산 불필요)
        if norm(prediction_error) < threshold:
            return None  # 연산 비용 0
        
        # 오차가 크면 위로 전파 → 모델 업데이트
        self.higher_level.update(prediction_error)
        return prediction_error
```

#### D. 해마 모듈 (이중 학습률)
```python
# 개념적 의사코드
class HippocampalModule:
    """
    이중 학습 시스템:
    - 해마: 높은 학습률 → 원샷 기록 → 불안정
    - 피질: 낮은 학습률 → 점진적 통합 → 안정적
    
    치명적 망각 해결: 새로운 것은 해마에 빠르게,
    수면(정리 단계)에서 피질로 서서히 전사
    """
    def __init__(self):
        self.fast_memory = FastNetwork(lr=0.1)   # 해마
        self.slow_memory = SlowNetwork(lr=0.001)  # 피질
        
    def learn(self, pattern):
        # 즉시 해마에 기록 (원샷)
        self.fast_memory.store(pattern)
    
    def consolidate(self):
        """수면 단계: 해마 → 피질 전사"""
        for memory in self.fast_memory.replay():
            self.slow_memory.integrate(memory)
        
        # 전역 가지치기: 약한 연결 제거 (시냅스 항상성)
        self.slow_memory.global_downscale(factor=0.9)
        self.slow_memory.prune_weak_connections()
    
    def recall(self, partial_input):
        """패턴 완성: 부분 → 전체"""
        # 해마에서 먼저 검색 (최근 기억)
        result = self.fast_memory.pattern_complete(partial_input)
        if result is None:
            # 피질에서 검색 (장기 기억)
            result = self.slow_memory.pattern_complete(partial_input)
        return result
```

#### E. 메타인지 모듈
```python
# 개념적 의사코드
class MetacognitionModule:
    """
    핵심: 1차 처리와 분리된 모니터링 시스템
    
    현재 AI에 없는 것:
    - GPT는 자기가 모른다는 걸 모름
    - 확률 높은 토큰 = 맞는 답이 아님
    - 메타인지 = "내 출력이 맞는지" 독립 판단
    """
    def evaluate(self, output, internal_states):
        # 내부 상태의 일관성 체크
        consistency = self.check_consistency(internal_states)
        
        # 활성화 패턴의 선명도 체크 (희소할수록 확신)
        sparsity = self.check_sparsity(internal_states)
        
        # 복수 경로의 합의도 체크
        agreement = self.check_agreement(internal_states)
        
        confidence = combine(consistency, sparsity, agreement)
        
        if confidence < LOW_THRESHOLD:
            return ReprocessSignal()  # 다시 처리하라
        elif confidence < MEDIUM_THRESHOLD:
            return output, confidence, "uncertainty_flag"
        else:
            return output, confidence
```

---

## 실험 계획

### 실험 1: 억제 기반 희소 어텐션 (Inhibitory Sparse Attention)
**가설**: 트랜스포머 어텐션에서 상위 K%만 남기고 나머지를 적극 억제하면,
성능 유지하면서 연산량이 감소한다.

**방법**:
1. 기본 Transformer (GPT-2 small 수준)를 MNIST/CIFAR-10에서 학습
2. 어텐션 레이어에 억제 메커니즘 추가:
   - 각 어텐션 헤드에서 상위 5% 어텐션 스코어만 유지
   - 나머지 95% 어텐션을 0으로 강제 (lateral inhibition 모방)
3. 성능 비교: 정확도, 연산량 (FLOPs), 에너지 (GPU 전력)

**기대 결과**: 연산량 대폭 감소, 정확도 소폭 변화

**구현**: PyTorch, 기존 Transformer에 마스킹 레이어 추가

### 실험 2: 예측 코딩 Transformer (Predictive Coding Transformer)
**가설**: 각 레이어가 다음 레이어 출력을 예측하고,
예측 오차만 전파하면 연산량이 줄면서 표현력은 유지된다.

**방법**:
1. 기본 Transformer 학습
2. 각 레이어에 예측 모듈 추가:
   - Layer N이 Layer N+1의 출력을 예측
   - 예측과 실제의 차이(residual)만 Layer N+1에 전달
   - 차이가 임계값 이하면 해당 토큰/패치 스킵
3. 비교: 정확도, 실제 연산된 토큰/패치 비율, 추론 속도

**기대 결과**: 입력의 상당 부분이 스킵됨, 속도 향상

**구현**: PyTorch, 기존 Transformer 레이어 사이에 예측 모듈 삽입

### 실험 3: 이중 학습률 시스템 (Complementary Learning System)
**가설**: 빠른 네트워크(해마) + 느린 네트워크(피질) + 주기적 전사(수면)로
치명적 망각이 감소한다.

**방법**:
1. 두 개의 네트워크: Fast (lr=0.1), Slow (lr=0.001)
2. 새 데이터 → Fast에 즉시 학습
3. 주기적으로 "수면 단계": Fast가 기억을 리플레이 → Slow에 전사
4. 수면 단계에서 Slow의 전역 가중치 감쇠 (synaptic homeostasis)
5. Split MNIST/CIFAR로 catastrophic forgetting 측정
   (Task A 학습 → Task B 학습 → Task A 성능 측정)

**기대 결과**: 기존 단일 네트워크 대비 이전 태스크 성능 유지율 향상

**구현**: PyTorch, 두 개 네트워크 + 리플레이 스케줄러

### 실험 4: STDP 기반 Temporal Graph 학습
**가설**: 역전파 없이 STDP(지역 규칙)만으로 패턴 인식이 가능하며,
에너지 효율에서 우위를 보인다.

**방법**:
1. Brian2 또는 Norse로 SNN 구성 (뉴런 1,000개 수준)
2. STDP 학습 규칙만으로 MNIST 학습
3. 네트워크를 Temporal Graph로 표현 (PyTorch Geometric 또는 DGL)
4. 학습 과정에서 간선 변화 추적 (가지치기 관찰)
5. 역전파 기반 동일 크기 네트워크와 비교: 정확도, 수렴 속도, 연산량

**기대 결과**: 정확도는 다소 낮을 수 있으나, 연산량과 에너지에서 우위

**구현**: Brian2/Norse + NetworkX/PyG

---

## 기술 스택

### 필수 라이브러리
- PyTorch (핵심 프레임워크)
- PyTorch Geometric 또는 DGL (Temporal Graph)
- Brian2 또는 Norse (SNN 시뮬레이션)
- NetworkX (그래프 분석/시각화)
- Matplotlib / Plotly (결과 시각화)
- Weights & Biases 또는 TensorBoard (실험 추적)

### 데이터셋
- MNIST (기본 검증)
- CIFAR-10 (시각 태스크)
- Split MNIST / Permuted MNIST (치명적 망각 테스트)
- Sequential MNIST (시간 패턴)

### 하드웨어
- 일반 GPU (RTX 3060 이상) 또는 CPU로 시작 가능
- 소규모 실험은 Colab으로도 가능

---

## 성공 지표

| 실험 | 1차 지표 | 2차 지표 | 목표 |
|------|---------|---------|------|
| 실험1 (희소 어텐션) | FLOPs 감소율 | 정확도 변화 | FLOPs 50%↓, 정확도 -2% 이내 |
| 실험2 (예측 코딩) | 스킵된 연산 비율 | 추론 속도 | 30%+ 스킵, 속도 1.5x+ |
| 실험3 (이중 학습) | 이전 태스크 유지율 | 새 태스크 정확도 | 유지율 80%+ (기존 50% 이하) |
| 실험4 (STDP) | 역전파 없이 수렴 여부 | 에너지 비교 | MNIST 90%+ 달성 |

---

## 프로젝트 구조

```
subtractive-intelligence/
├── ARCHITECTURE.md          ← 이 문서
├── experiments/
│   ├── exp1_sparse_attention/
│   │   ├── model.py         # 억제 기반 희소 어텐션 모델
│   │   ├── train.py         # 학습 스크립트
│   │   └── evaluate.py      # 평가 스크립트
│   ├── exp2_predictive_coding/
│   │   ├── model.py         # 예측 코딩 트랜스포머
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── exp3_dual_learning/
│   │   ├── fast_net.py      # 해마 네트워크
│   │   ├── slow_net.py      # 피질 네트워크
│   │   ├── consolidation.py # 수면 단계 (리플레이+전사)
│   │   ├── train.py
│   │   └── evaluate.py
│   └── exp4_stdp_temporal/
│       ├── snn_model.py     # SNN + STDP
│       ├── temporal_graph.py # Temporal Graph 표현
│       ├── train.py
│       └── evaluate.py
├── core/
│   ├── thalamic_router.py   # 시상 모듈
│   ├── cortical_column.py   # 피질 컬럼
│   ├── inhibitory.py        # 억제 메커니즘
│   ├── metacognition.py     # 메타인지 모듈
│   ├── hippocampus.py       # 해마 모듈
│   └── predictive_coding.py # 예측 코딩 루프
├── utils/
│   ├── visualization.py     # 그래프/결과 시각화
│   ├── metrics.py           # 성능 지표
│   └── data_loader.py       # 데이터셋 로더
├── notebooks/
│   ├── 01_sparse_attention_demo.ipynb
│   ├── 02_predictive_coding_demo.ipynb
│   ├── 03_dual_learning_demo.ipynb
│   └── 04_stdp_demo.ipynb
└── requirements.txt
```

---

## 참고 문헌 (핵심만)

### 가지치기 / Lottery Ticket
- Frankle & Carlin (2019). "The Lottery Ticket Hypothesis"
- Hoefler et al. (2021). "Sparsity in Deep Learning"

### 예측 코딩
- Friston (2010). "The free-energy principle"
- Millidge et al. (2022). "Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs"

### 상보적 학습 시스템
- McClelland et al. (1995). "Why There Are Complementary Learning Systems"
- Kumaran et al. (2016). "What Learning Systems do Intelligent Agents Need?"

### SNN / STDP
- Tavanaei et al. (2019). "Deep Learning in Spiking Neural Networks"
- Zenke & Ganguli (2018). "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks"

### 뉴로모픽 / HTM
- Hawkins et al. (2019). "A Framework for Intelligence and Cortical Function Based on Grid Cells"
- Davies et al. (2021). "Advancing Neuromorphic Computing With Loihi"

### JEPA / 월드 모델
- LeCun (2022). "A Path Towards Autonomous Machine Intelligence"

### 단일 뉴런 연산력
- Jones & Kording (2021). "Might a Single Neuron Solve Interesting Machine Learning Problems?"
- Beniaguev et al. (2021). "Single cortical neurons as deep artificial neural networks"

---

## 철학적 배경 (선택적 읽기)

이 프로젝트의 영감은 네 가지 관찰에서 나왔다:

1. **에너지 격차**: 뇌(20W)와 AI(수천W)의 10,000배 효율 차이는 
   알고리즘 최적화로 좁힐 수 없다. 아키텍처가 근본적으로 다르다.

2. **스케일링 한계**: 파라미터를 10배 늘려도 성능이 10배 올라가지 않는다.
   뇌는 시냅스를 줄이면서 더 똑똑해졌다. 방향이 반대다.

3. **정보의 비균등성**: 자연의 정보는 핵심 소수에 집중된다 (Zipf, 파레토).
   뇌는 이걸 선험적으로 알고 설계되어 있다. AI는 균등 처리한다.

4. **빼기의 힘**: 조합론적으로, N개 중 K%만 활성화하는 조합의 수는
   K가 작을수록 폭발적으로 크다. 적게 쓸수록 더 많이 표현한다.

이 프로젝트의 궁극적 질문:
**"지능은 더해서 만드는 것인가, 깎아서 드러내는 것인가?"**
