# 기초 MLP 학습

상태: **구현됨**. 저장소를 `pip install -e .`로 설치한 후 다른 실습 없이 실행할 수 있습니다.

합성 XOR 분류를 수행합니다. 서로 다른 seed의 학습/검증 데이터를 만들고 cross entropy, backward, Adam, eval, inference_mode를 사용합니다. 이 검증셋은 학습 관찰용이며 하이퍼파라미터 선택 후 최종 평가에는 별도 테스트셋이 필요합니다.

```bash
python -m runpod_ai_lab.train_mlp --device cuda --epochs 30 --batch-size 256 --hidden 32 --depth 2
python -m runpod_ai_lab.train_mlp --help
```

CPU에서는 `--device cpu`를 사용합니다. 결과는 `runs/`의 실험별 JSON에 저장됩니다. 구현은 [train_mlp.py](../../src/runpod_ai_lab/train_mlp.py)입니다.

실험 과제: batch-size, hidden, depth, lr을 바꾸고 val_accuracy와 처리량을 비교하세요. build_model을 수정하여 ReLU를 Tanh로 바꿔보세요. 첫 학습에는 인터넷/데이터셋이 필요 없습니다.
