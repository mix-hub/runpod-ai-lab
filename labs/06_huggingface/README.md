# Hugging Face 모델 실행

상태: **계획 / 미구현**. 실행 가능한 코드나 실행 명령은 아직 없습니다.

- 구현 순서: 작은 공개 모델 선정 → tokenizer 확인 → 모델 revision 고정 → 분류/생성 → 직접 forward와 pipeline 비교.
- 변경할 설정: model ID/revision, max tokens, device, dtype.
- 관찰 지표: 토큰 수, latency, VRAM.
- 완료 조건: 모델 라이선스·다운로드 크기·캐시·CPU/GPU 실행 확인.

구현할 때 독립 실행 CLI, 작은 smoke 실험, 명령어, 필요 패키지와 검증 버전, 실측 GPU/VRAM 요구량을 추가합니다. 데이터·가중치·캐시는 Git에 포함하지 않습니다. 앞 실습의 결과 파일에 의존하지 않도록 만듭니다.
