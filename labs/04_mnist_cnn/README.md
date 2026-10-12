# MNIST와 CNN

상태: **계획 / 미구현**. 실행 가능한 코드나 실행 명령은 아직 없습니다.

- 구현 순서: torchvision MNIST 다운로드와 정규화 → train/validation/test 분할 → MLP 기준선 → Conv2d/Pooling CNN → 오분류 관찰.
- 변경할 설정: batch-size, learning rate, channel 수, kernel 크기.
- 관찰 지표: 정확도, epoch 시간, VRAM, confusion matrix.
- 완료 조건: 실제 MNIST에서 학습·추론, 체크포인트 복원, 데이터 누수 없는 평가를 검증.

구현할 때 독립 실행 CLI, 작은 smoke 실험, 명령어, 필요 패키지와 검증 버전, 실측 GPU/VRAM 요구량을 추가합니다. 데이터·가중치·캐시는 Git에 포함하지 않습니다. 앞 실습의 결과 파일에 의존하지 않도록 만듭니다.
