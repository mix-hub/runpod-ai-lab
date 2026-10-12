# GPU 환경 확인

상태: **구현됨**. 저장소를 `pip install -e .`로 설치한 후 다른 실습 없이 실행할 수 있습니다.

드라이버, PyTorch CUDA 빌드, GPU별 VRAM을 확인하고 실제 행렬곱 결과를 검증합니다.

```bash
python -m runpod_ai_lab.environment --device cuda
python -m runpod_ai_lab.environment --help
```

CPU에서는 `--device cpu`를 사용합니다. 결과는 `runs/`의 실험별 JSON에 저장됩니다. 구현은 [environment.py](../../src/runpod_ai_lab/environment.py)입니다.

실험 과제: CPU 실행과 비교하고 CUDA 빌드 버전과 nvidia-smi의 지원 버전 차이를 설명하세요.
