# Tensor와 GPU 연산

상태: **구현됨**. 저장소를 `pip install -e .`로 설치한 후 다른 실습 없이 실행할 수 있습니다.

broadcasting, transpose의 stride 및 연속성을 관찰합니다. 행렬곱은 warmup 후 동기화하여 측정합니다.

```bash
python -m runpod_ai_lab.tensors --device cuda --size 1024 --repeats 20
python -m runpod_ai_lab.tensors --help
```

CPU에서는 `--device cpu`를 사용합니다. 결과는 `runs/`의 실험별 JSON에 저장됩니다. 구현은 [tensors.py](../../src/runpod_ai_lab/tensors.py)입니다.

실험 과제: size 256/1024/2048, float32/float64, CPU/GPU를 비교하세요. 전송 시간과 연산 시간은 분리됩니다. float16은 오차와 overflow도 관찰하세요.
