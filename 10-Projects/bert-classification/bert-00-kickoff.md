# bert-00-kickoff — 교재→BERT 전환 정리

[[../bert-classification]] Phase 0 정본. **BERT 학습에 앞서 정리한 내용**(2026-07-03).
[[../llm-from-scratch]] 교재 완주 직후, "무엇이 이어지고 무엇이 새로운가"를 짚고 첫 과제를 확정하는 노트.

## 1. 선수 점검 — BERT 진입 준비 완료

| 확인 항목 | 상태 |
|---|---|
| 선행 트랙(교재 Raschka) | ✅ 완주 (부록 A + 본문 1~7장, 2026-07-03) |
| BERT 선수 기초 | ✅ SMS 스팸 2-class 분류 en/ko ~97% ([[../../30-References/rnd-bert-labeling-test-plan]]) |
| 첫 과제 정의 | ✅ **KLUE-TC 다중 클래스**(뉴스 7클래스) — `num_labels`만 확장 |
| 환경 | ✅ [[../../30-References/pytorch-env-hybrid]] (M4 Max 로컬 + Colab) |
| 검증 방식 | ✅ MLflow 한글 키 관례 재사용 |

결론: **바로 진입 가능.** 능력 주입(언어·트랜스포머 이해)은 교재에서 끝났고, 이제 "같은 트랜스포머를 인코더 방향으로, HF 추상화로" 다시 만나는 단계.

## 2. 세 학습 방식 비교 (교재 7.9 회고 요약) — 무엇을 손에 쥐고 넘어가나

셋 다 **다음-토큰/토큰 맞히기 = CE loss라는 같은 엔진**. 다른 건 세 손잡이뿐.

| | 사전훈련(5장) | 분류 FT(6장) | 지시 FT(7장) |
|---|---|---|---|
| 가르치는 것 | 언어 자체 | 하나의 판별 | 행동 양식 |
| head | 768→50257(원본) | 768→50257 **제거**, **768→2 신설** | 768→50257 유지 |
| 손실 위치 | 모든 위치 | **마지막 토큰 1개** | 응답 구간(패딩 -100 제외) |
| 결과물 | 자동완성 엔진 | 이진 분류기 | 지시 수행 비서 |

→ 상세 원본: [[../llm-from-scratch/llm-ch7-instruct]] "7.9 세 학습 방식 비교".

## 3. 교재 개념 → BERT 대응 (이어지는 것)

BERT 분류는 6장 분류 FT와 **같은 뼈대**다. 새로 배우는 건 "인코더 구조"와 "HF 추상화"뿐.

| 교재에서 배운 것 | BERT에서 어떻게 | 새로움 |
|---|---|---|
| head 교체(768→2, 6장) | `BertForSequenceClassification(num_labels=N)` — HF가 head 자동 부착 | 직접 짜던 걸 한 줄로 |
| 마지막 토큰 argmax(6장) | BERT는 **`[CLS]` 토큰**의 표현을 분류 head에 투입 | 문장 대표 토큰이 앞(`[CLS]`)에 |
| 자기회귀(디코더, causal mask) | BERT는 **양방향(인코더)** — 앞뒤 문맥 동시에 봄 | 생성 대신 이해에 특화 |
| CE loss, 정확도/precision/recall | 동일 | 그대로 재사용 |
| -100 ignore_index(6·7장) | 분류에선 거의 안 씀 → **NER(#2)에서 subword 정렬로 재등장** | 다음 프로젝트로 이월 |
| MLflow 한글 키 실험 추적 | 그대로 | — |

핵심 대비: **GPT-2(디코더)=왼→오 생성**, **BERT(인코더)=양방향 이해**. 분류처럼 "전체를 읽고 하나로 판단"하는 작업엔 인코더가 자연스럽다. 개념 정리: [[../../30-References/bert-vs-gpt2-classification]].

## 4. 첫 과제 (Phase 1) — 무엇부터

1. **재사용**: 기존 `finetune_bert_spam.py`(2-class)를 복사 → `num_labels` 2→7 변경.
2. **데이터**: KLUE-TC(뉴스 주제 7클래스) HF `datasets`로 로드. 처음엔 미니 서브셋으로 end-to-end "되는지" 확인(Alpaca OOM 교훈 — 작게 시작).
3. **기록**: MLflow run(한글 키), test 정확도 + 클래스별 precision/recall.
4. **검증 3종**: 실행 로그 · MLflow run · 학습 노트.

→ 진행은 [[../bert-classification/bert-01-klue-tc-multiclass]]에서.

## 관련 노트

- [[../bert-classification]] — 마스터 플랜
- [[../llm-from-scratch/llm-ch7-instruct]] — 7.9 세 학습 방식 회고 원본
- [[../llm-from-scratch/llm-ch6-classify]] — 6장 분류 FT (직계 선행)
- [[../../30-References/rnd-bert-labeling-test-plan]] — SMS 스팸 2-class 선수 산출물
- [[../../30-References/bert-vs-gpt2-classification]] — 인코더/디코더 개념
