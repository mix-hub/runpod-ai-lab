# 검증 기록

2026-10-12 로컬 Windows, Python 3.12.14, PyTorch 2.14.1+cpu에서 검증했습니다.
테스트 전용 패키지는 프로젝트 바깥 작업 폴더에 설치했으며 저장소에 포함하지 않습니다.

```text
python -m unittest discover -s tests -v
test_checkpoint_prediction_roundtrip ... ok
test_independent_cli_and_learning ... ok
test_manual_gradients_with_finite_difference ... ok
Ran 3 tests in 18.773s
OK
```

- environment, tensors, backprop, train_mlp를 각각 별도 CPU 프로세스로 실행했습니다.
- 수동 역전파의 모든 파라미터 원소를 중앙 유한차분과 비교했습니다. backprop CLI 내부에서는 autograd와도 비교합니다.
- 기본 backprop 500 step 후 loss가 초기 loss의 20% 미만인지 검증했습니다.
- MLP 20 epoch 후 별도 검증 데이터 정확도가 90%를 넘는지 확인했습니다.
- 저장/복원한 MLP의 예측 일치를 확인했습니다.
- Python 문법 검사와 Git의 결과·데이터·모델·캐시·환경파일 제외 규칙을 확인했습니다.

Ubuntu/RunPod/CUDA 실측은 아직 수행하지 않았습니다. GitHub Actions CPU 테스트는 설정 파일만 추가했으며 원격 CI 실행은 아직 없습니다. Python/PyTorch의 선언된 전체 버전 범위를 모두 검증한 것은 아닙니다.

RunPod에서 README 설치 후 아래를 실행하여 GPU 검증을 완료합니다.

```bash
python -m runpod_ai_lab.environment --device cuda
python -m runpod_ai_lab.tensors --device cuda --size 1024
python -m runpod_ai_lab.backprop --device cuda
python -m runpod_ai_lab.train_mlp --device cuda
python -m unittest discover -s tests -v
```

GPU 이름, 이미지 태그, 드라이버, torch/CUDA 버전 및 각 실험 JSON을 기록하세요. 테스트 스위트는 CPU 검증이며 위 네 CUDA 명령을 대체하지 않습니다. GPU throughput 수치는 아직 제시하지 않습니다.
