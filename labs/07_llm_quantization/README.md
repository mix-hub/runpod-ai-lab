# LLM 추론과 양자화

상태: **계획 / 미구현**. 실행 가능한 코드나 실행 명령은 아직 없습니다.

- 구현 순서: 작은 causal LM 기준선 → KV cache 관찰 → 지원 backend로 8bit/4bit 비교.
- 변경할 설정: prompt 길이, max new tokens, temperature, quantization.
- 관찰 지표: TTFT, tokens/s, peak VRAM, 출력 품질.
- 완료 조건: EOS/길이 제한과 동일 입력·warmup 조건에서 비교, backend/CUDA 호환성 확인.

구현할 때 독립 실행 CLI, 작은 smoke 실험, 명령어, 필요 패키지와 검증 버전, 실측 GPU/VRAM 요구량을 추가합니다. 데이터·가중치·캐시는 Git에 포함하지 않습니다. 앞 실습의 결과 파일에 의존하지 않도록 만듭니다.
