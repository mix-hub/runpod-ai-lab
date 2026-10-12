# LoRA와 QLoRA Fine-tuning

상태: **계획 / 미구현**. 실행 가능한 코드나 실행 명령은 아직 없습니다.

- 구현 순서: 소규모 공개 데이터와 chat template → train/validation 분리 → LoRA → QLoRA → adapter 저장/복원.
- 변경할 설정: rank, alpha, dropout, target modules, learning rate, accumulation.
- 관찰 지표: 학습 가능 parameter 수, loss, VRAM, 전후 생성.
- 완료 조건: base model revision 및 adapter 재로드 일치, tokenizer padding/label masking 검증.

구현할 때 독립 실행 CLI, 작은 smoke 실험, 명령어, 필요 패키지와 검증 버전, 실측 GPU/VRAM 요구량을 추가합니다. 데이터·가중치·캐시는 Git에 포함하지 않습니다. 앞 실습의 결과 파일에 의존하지 않도록 만듭니다.
