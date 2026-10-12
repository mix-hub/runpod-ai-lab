# Attention과 Transformer

상태: **계획 / 미구현**. 실행 가능한 코드나 실행 명령은 아직 없습니다.

- 구현 순서: 작은 텐서로 Q/K/V 및 softmax(QKᵀ/√d) 구현 → causal mask → multi-head attention → 작은 문자 언어모델.
- 변경할 설정: head 수, embedding 차원, context 길이, 층수.
- 관찰 지표: attention 행렬, loss/perplexity, context별 VRAM.
- 완료 조건: 미래 토큰 누출 차단과 attention shape 및 gradient 검증.

구현할 때 독립 실행 CLI, 작은 smoke 실험, 명령어, 필요 패키지와 검증 버전, 실측 GPU/VRAM 요구량을 추가합니다. 데이터·가중치·캐시는 Git에 포함하지 않습니다. 앞 실습의 결과 파일에 의존하지 않도록 만듭니다.
