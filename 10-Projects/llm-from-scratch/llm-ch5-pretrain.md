# llm-ch5-pretrain

[[llm-from-scratch]] 마스터 플랜의 **Phase 5 / 5장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 5장. 레이블이 없는 데이터를 활용한 사전 훈련.

## 개요

cross-entropy/perplexity 기반 학습 루프를 구현해 소규모 코퍼스로 사전훈련하고, OpenAI 공개 GPT-2 가중치를 자체 구조에 매핑한다. 실제 학습은 Colab T4가 메인.

## 체크리스트

- [ ] 5.1 cross-entropy loss + perplexity 계산
- [ ] 5.2 학습 루프 (forward → loss → backward → optimizer.step)
- [ ] 5.3 작은 코퍼스로 1 epoch 학습 (로컬에서 검증)
- [ ] 5.4 학습률 워밍업·코사인 스케줄링
- [ ] 5.5 체크포인트 저장/로드 (`weights_only=True`)
- [ ] 5.6 OpenAI 공개 GPT-2 가중치 로드 → 자체 구조에 매핑
- [ ] 5.7 **Colab T4에서 소규모 사전훈련 실행** (로컬은 코드 디버그)

## 검증

사전훈련 loss 곡선이 우하향. 공개 GPT-2 가중치 로드 후 텍스트 생성이 의미 있는 출력.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../../30-References/pytorch-env-hybrid]] — 환경 정본
