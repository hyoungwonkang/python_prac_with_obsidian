# llm-ch6-classify

[[llm-from-scratch]] 마스터 플랜의 **Phase 6 / 6장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 6장. 분류를 위해 미세 튜닝하기.

## 개요

사전훈련된 모델에 분류 head를 붙이고 일부 레이어를 freeze해 분류 태스크(예: spam)로 미세 튜닝하는 단계.

## 체크리스트

- [ ] 6.1 분류 데이터셋 준비 (예: spam 분류)
- [ ] 6.2 모델에 분류 head 추가
- [ ] 6.3 일부 레이어 freeze 전략
- [ ] 6.4 fine-tuning 학습 루프 + 평가 지표 (accuracy)
- [ ] 6.5 결과 정리

## 검증

미세튜닝 전/후 지표 비교 — 분류 정확도 상승.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../30-References/pytorch-env-hybrid]] — 환경 정본
