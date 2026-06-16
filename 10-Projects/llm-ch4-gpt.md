# llm-ch4-gpt

[[llm-from-scratch]] 마스터 플랜의 **Phase 4 / 4장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 4장. 밑바닥부터 GPT 모델 구현하기.

## 개요

LayerNorm·GELU·FeedForward·residual을 조립해 transformer 블록을 만들고, N개 블록을 쌓아 GPT 아키텍처 전체를 구성하는 단계.

## 체크리스트

- [ ] 4.1 LayerNorm 직접 구현
- [ ] 4.2 GELU 활성화·FeedForward 블록
- [ ] 4.3 Residual connection 포함 transformer 블록
- [ ] 4.4 GPT 모델 아키텍처 조립 (임베딩 + N×블록 + 최종 norm + 출력 head)
- [ ] 4.5 초기화된 모델로 텍스트 생성 (greedy → top-k → temperature)
- [ ] 4.6 파라미터 수 계산·디바이스 메모리 점검

## 검증

책 코드와 동일 입력 → 동일/유사 출력 재현. 로컬(2.2.2)·Colab(2.6.0) 양쪽에서 코드 동작.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../30-References/pytorch-env-hybrid]] — 환경 정본
