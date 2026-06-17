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

## 별도 트랙 접점

이 장 완료 직후가 [[../30-References/bert_ocr_practice_plan]]의 **BERT 실습 진입 시점**(정본 결정). GPT에 분류 head를 직접 붙여 FT한 경험을 바탕으로 → "같은 분류 FT를 BERT+HF Transformers로는 이렇게" 비교 학습하면 효과가 가장 크다.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../30-References/bert_ocr_practice_plan]] — BERT·OCR 라이브러리 실습 트랙 (이 장 직후 진입)
- [[../30-References/pytorch-env-hybrid]] — 환경 정본
