# ADR-002: 실행 환경

- 상태: 확정 (2026-10-01)

## 결정

- Python 3.10.11 가상환경 (`.venv`). 시스템 기본은 3.13 이지만 brian2 / norse / torch-geometric 의
  3.13 지원이 불확실해 3.10 을 택했다.
- PyTorch CUDA 빌드: `--index-url https://download.pytorch.org/whl/cu128`. 설치 결과 torch 2.11.0+cu128,
  torchvision 0.26.0+cu128. GPU 는 RTX 3060 Ti (8GB, Ampere sm_86). 드라이버 610.88 (CUDA 13.3 지원).
- 시스템 파이썬(3.13)의 torch 2.10 은 CPU 전용 빌드라 쓰지 않는다.
- SNN: 실험 4 는 순수 PyTorch 로 GPU 벡터화 구현한다 (Diehl & Cook 2015 구조). brian2 는 참고/검증용,
  norse 는 설치되면 선택 사용. snntorch 를 대체 후보로 설치.
- 실험 추적: TensorBoard (results/<run>/tb) + results.json. W&B 는 쓰지 않는다 (오프라인 재현성).

## FLOPs 집계 규약

- `torch.utils.flop_counter.FlopCounterMode` 로 1 샘플 순전파 dense FLOPs 를 센다. matmul 은 2*M*N*K.
- Linear / Conv 리프의 dense FLOPs 를 forward hook 으로 해석적으로 세고, `weight_mask` 밀도를 곱해
  effective FLOPs 를 만든다. 두 방식의 리프 합이 dense 총량을 넘지 않는지 연기 테스트에서 검증.
- 학습 FLOPs = 순전파 x 3 (역전파 2 배 근사). 프루닝 실험은 에폭마다 effective 를 다시 재서 누적한다.

## 재현

```
powershell -ExecutionPolicy Bypass -File setup_env.ps1
.venv\Scripts\python.exe scripts\smoke_test.py
.venv\Scripts\python.exe scripts\train_baseline.py --config configs\mlp_mnist.yaml
```

## 함정 기록

- Windows DataLoader 는 spawn 이라 transform 은 picklable 해야 한다 (lambda 금지). Permute 는 클래스.
- `.py` 파일은 cp949 인코딩 가능 문자만 쓴다 (Downloads 레벨 훅). 화살표/대시/이모지 금지.
