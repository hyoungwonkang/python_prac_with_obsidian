# llm-ch3-attention

[[llm-from-scratch]] 마스터 플랜의 **Phase 3 / 3장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 3장. 어텐션 메커니즘 구현하기.

## 개요

단순 self-attention에서 시작해 학습 가능한 Q/K/V, causal 마스킹, multi-head까지 단계적으로 구현하는 핵심 장.

## 체크리스트

- [ ] 3.1 단순 self-attention (가중치 학습 없이 dot-product)
- [ ] 3.2 학습 가능한 가중치(Q, K, V) 도입
- [ ] 3.3 scaled dot-product attention
- [ ] 3.4 causal(마스킹) attention
- [ ] 3.5 multi-head attention 구현
- [ ] 3.6 텐서 shape 변화를 손으로 추적해 노트로 정리

## 검증

책 코드와 동일 입력 → 동일/유사 출력 재현. 로컬(2.2.2)·Colab(2.6.0) 양쪽에서 코드 동작.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../30-References/pytorch-env-hybrid]] — 환경 정본
