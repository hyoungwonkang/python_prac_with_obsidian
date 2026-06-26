---
title: BERT R&D 산출물 계획 (스팸분류)
tags: [reference, bert, rnd, deliverable, plan]
---

# BERT R&D 산출물 계획 (스팸 분류)

> 업무 요청 대응 계획. BERT 개념·실습 정본은 [[bert_ocr_practice_plan]], 데이터/라벨링 실습은 [[../10-Projects/llm-from-scratch/llm-ch6-classify]].
> 학습 트랙: [[../10-Projects/llm-from-scratch]] 6장(분류 파인튜닝) 진행 중.

## 0. 요청 내용 (업무 메시지)

> "금일(연구문서/소스/사용법/가이드) 또는 화요일(시연)까지
> BERT 모델 및 라벨링 방법론 및 테스트 R&D 내용을 공유."

| 옵션 | 마감 | 산출물 형태 |
|---|---|---|
| **A** | **오늘 (2026-06-26 금)** | 문서 4종 — 연구문서 / 소스 / 사용법 / 가이드 |
| **B** | **화요일 (2026-06-30)** | 시연(데모) |

→ **선택: 3번(둘 다)** — 오늘 문서로 약속 이행, 화요일 데모로 완성.

## 1. 산출물 = 4종 (각각 독립 섹션)

| # | 요청 항목 | 형태 | 위치 |
|---|---|---|---|
| 1 | 연구문서 | 개념·접근 (**테스트 전략 포함**) | §1 |
| 2 | **소스코드** | 실제 `.py` (돌아가는 코드) | **§2 독립** |
| 3 | **사용법** | 설치·실행 명령 (How to run) | **§3 독립** |
| 4 | 가이드 | 재현 절차·주의점 | §4 |

> 핵심: **소스코드·사용법은 "소스 계획"에 묻지 않고 §2·§3로 분리.**
> 테스트 전략은 §1 문서에 두고, §2 평가 코드가 그것을 증명하는 구조.

## 2. 최종 문서 구조

```
BERT 스팸분류 R&D 문서
├─ §1. 연구문서 (개념·접근)
│    ├─ BERT란 / 왜 분류에 적합한가 ([CLS] + 분류층)
│    ├─ 라벨링 방법론 (ham/spam→0/1, 클래스 균형화, 70/10/20 분할)
│    └─ 테스트 전략 (test셋 분리, 정확도·혼동행렬 평가)
├─ §2. 소스코드  ← 실제 코드 (BertForSequenceClassification)
│    ├─ 데이터 준비 (dataset_finetuning.py 재활용)
│    └─ 학습·평가 스크립트
├─ §3. 사용법    ← 환경설치 → 실행 명령 → 결과 확인
└─ §4. 가이드    ← 재현 절차 + 주의점 (Colab GPU 등)
```

## 3. 기술 스택·자원 (재활용 가능 자산)

| 항목 | 사용할 것 | 출처 |
|---|---|---|
| 데이터셋 | SMS Spam Collection (균형화·라벨링·70/10/20 분할 완료) | `10-Projects/llm-from-scratch/dataset_finetuning.py` |
| 모델 | HF `BertForSequenceClassification` (`bert-base-uncased`, `num_labels=2`) | Hugging Face Transformers |
| 토크나이저 | `BertTokenizer.from_pretrained` | 〃 |
| 실행 환경 | **Colab GPU** (로컬 M4 Max는 TF·대형모델 부담 → BERT 학습은 Colab) | [[pytorch-env-hybrid]] |
| 평가 | accuracy + confusion matrix (sklearn) | — |

## 4. 진행 순서

1. **오늘 (A)**: §1~§4 문서 + **§2 소스코드 초안** 작성 → 오늘 약속 이행.
   - 작업 순서: **§2 소스코드 먼저** → 코드 기준으로 §1·§3·§4 채움 (코드가 있어야 사용법·가이드가 정확).
2. **화요일 (B)**: 소스코드를 **Colab GPU에서 실제 실행·데모**, §3 사용법·§4 가이드를 실행 결과로 확정.

## 5. 진행 가능성 점검 (feasibility) — 2026-06-26 실측 반영

| 요소 | 가능? | 실측 근거 / 조건 |
|---|---|---|
| 데이터 | ✅ | **확인됨** — `train.csv`(113KB)/`validation.csv`(18KB)/`test.csv`(32KB) 존재 |
| BERT 코드 작성 | ✅ | HF Transformers 표준 패턴, 분량 적음 |
| 오늘 문서+코드초안 | ✅ | vault 노트 자산으로 바로 작성 가능 |
| 로컬 실행 | ❌ | **`transformers` 미설치** (로컬 env). torch 2.8.0+MPS는 OK지만 BERT는 Colab로 |
| 화요일 데모(실행) | ✅ (조건부) | **Colab GPU 런타임 필요** — `pip install transformers` 후 실행 |
| 한국어 데이터 확장 | △ | `bert-base-uncased`는 영어. 한국어 필요 시 `klue/bert-base` 별도 (NSMC 등) |

> 결론: **3번 계획대로 진행 가능.** 단일 리스크는 **실행 환경이 Colab GPU 의존**이라는 점 하나 —
> 코드·문서는 오늘 로컬에서 완성, **실제 학습/데모만 Colab**에서 수행 (기존 LLM 트랙과 동일 패턴).

## 관련 노트
- [[bert_ocr_practice_plan]] — BERT/OCR 학습 로드맵 (개념·진입 시점 정본)
- [[../10-Projects/llm-from-scratch/llm-ch6-classify]] — 6장 분류 파인튜닝 (GPT 버전, 라벨링·분할 실습)
- [[pytorch-env-hybrid]] — 실행 환경 정본 (로컬 vs Colab)
