# Stable Diffusion 이미지 생성

상태: **계획 / 미구현**. 실행 가능한 코드나 실행 명령은 아직 없습니다.

- 구현 순서: 라이선스 확인 → 작은 diffusion pipeline → seed 고정 생성 → denoising 단계 관찰.
- 변경할 설정: scheduler, steps, guidance scale, 해상도, dtype.
- 관찰 지표: 이미지, 생성 시간, VRAM.
- 완료 조건: 동일 환경에서 seed 재현성, 출력 저장 및 GPU 메모리 측정.

구현할 때 독립 실행 CLI, 작은 smoke 실험, 명령어, 필요 패키지와 검증 버전, 실측 GPU/VRAM 요구량을 추가합니다. 데이터·가중치·캐시는 Git에 포함하지 않습니다. 앞 실습의 결과 파일에 의존하지 않도록 만듭니다.
