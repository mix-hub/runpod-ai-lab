# 수동 신경망과 역전파

상태: **구현됨**. 저장소를 `pip install -e .`로 설치한 후 다른 실습 없이 실행할 수 있습니다.

h=tanh(XW1+b1), prediction=hW2+b2, L=mean((prediction-y)^2). dL/dprediction=2(prediction-y)/N, tanh의 미분은 1-h²입니다. 행렬곱을 따라 chain rule을 적용하고 bias는 batch 축으로 합산합니다. autograd는 최초 기울기 검증에만 사용하며 SGD는 직접 수행합니다.

```bash
python -m runpod_ai_lab.backprop --device cuda --hidden 16 --steps 500 --lr 0.1
python -m runpod_ai_lab.backprop --help
```

CPU에서는 `--device cpu`를 사용합니다. 결과는 `runs/`의 실험별 JSON에 저장됩니다. 구현은 [backprop.py](../../src/runpod_ai_lab/backprop.py)입니다.

실험 과제: hidden과 lr을 바꾸어 손실 감소를 비교하세요. forward_backward에서 activation을 수정한다면 미분식도 함께 바꾸세요. float64는 미분 검증에 유용하나 일부 GPU에서 느립니다.
