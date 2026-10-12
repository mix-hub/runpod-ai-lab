# FastAPI AI 추론 서버

상태: **계획 / 미구현**. 실행 가능한 코드나 실행 명령은 아직 없습니다.

- 구현 순서: lifespan에서 모델 1회 로드 → health/inference → 입력 제한 → bounded queue → 오류 처리.
- 변경할 설정: batch size, queue 한도, timeout.
- 관찰 지표: 요청 latency, queue 대기, 오류율, VRAM.
- 완료 조건: 먼저 127.0.0.1 바인딩 및 SSH 터널, 공개 노출 전 인증·요청 제한 구현. GPU별 중복 worker 모델 적재 방지.

구현할 때 독립 실행 CLI, 작은 smoke 실험, 명령어, 필요 패키지와 검증 버전, 실측 GPU/VRAM 요구량을 추가합니다. 데이터·가중치·캐시는 Git에 포함하지 않습니다. 앞 실습의 결과 파일에 의존하지 않도록 만듭니다.
