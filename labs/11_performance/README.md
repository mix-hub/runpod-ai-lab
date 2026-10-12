# 동시 요청과 Batching

상태: **계획 / 미구현**. 실행 가능한 코드나 실행 명령은 아직 없습니다.

- 구현 순서: 동기 기준선 → 클라이언트 동시 부하 → static/dynamic batching → profiler 분석.
- 변경할 설정: concurrency, batch size, max wait, sequence length.
- 관찰 지표: p50/p95/p99 latency, throughput, 오류율, peak VRAM.
- 완료 조건: warmup/측정 구간 분리, timeout 포함 부하 결과, 과부하 backpressure 검증.

구현할 때 독립 실행 CLI, 작은 smoke 실험, 명령어, 필요 패키지와 검증 버전, 실측 GPU/VRAM 요구량을 추가합니다. 데이터·가중치·캐시는 Git에 포함하지 않습니다. 앞 실습의 결과 파일에 의존하지 않도록 만듭니다.
