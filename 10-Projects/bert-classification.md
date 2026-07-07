# bert-classification

HuggingFace **BERT 계열로 텍스트 분류를 파인튜닝**하는 실습 프로젝트 마스터 플랜.
[[detection-ai-study]] 로드맵의 **1군 #1(최우선)** 을 구체 프로젝트로 분리한 것.
선행 트랙 [[llm-from-scratch]](교재 완주, 2026-07-03)에서 GPT-2 분류 FT를 바닥부터 짠 뒤 → 같은 문제를 BERT+HF 추상화로 비교 학습.

## Context

- 업무 프로젝트(아동·청소년 온라인 성착취 의심정황 탐지 AI 플랫폼)의 **1차 탐지 Text Classifier(KoBERT)** 계층을 코드로 재현하는 것이 목표.
- **왜 BERT를 교재(Raschka) 뒤에 배우나**: 상사 로드맵 — 트랜스포머·어텐션·파인튜닝을 바닥부터 손으로 짠 뒤 BERT(인코더 방향, HF 추상화)에 진입. GPT(디코더) 분류 FT 경험이 BERT 분류의 비교 기준.
- **선수 기초 이미 확보**: 업무 R&D로 SMS 스팸 2-class 분류 완료(HF `BertForSequenceClassification`, en/ko test acc ~97%) → [[../30-References/rnd-bert-labeling-test-plan]]. 이 프로젝트는 그걸 **다중 클래스로 확장**하는 데서 출발.
- 환경 정본: [[../30-References/pytorch-env-hybrid]] (M4 Max 로컬 MPS + Colab T4 하이브리드).

## 이 프로젝트의 위치

```
detection-ai-study (탐지 플랫폼 대응 학습 로드맵, 여러 기술)
  ├─ #1 BERT 텍스트 분류  ← ★ 이 프로젝트(bert-classification)
  ├─ #2 NER              (별도 프로젝트 예정, BertForTokenClassification)
  ├─ #3 YOLO / #4 OpenCV / #5 phi-2 ...
```

## 진행 현황

| Phase | 내용 | 노트 | 상태 |
|---|---|---|---|
| 0 | 교재→BERT 전환 정리 (선수 점검·이어지는 개념) | [[bert-classification/bert-00-kickoff]] | ✅ 완료 (2026-07-03) |
| 1 | KLUE-TC 다중 클래스 (뉴스 7클래스) — `num_labels` 확장 | [[bert-classification/bert-01-klue-tc-multiclass]] | ✅ 완료 (2026-07-07 — acc 0.8425 · macro F1 0.8313, 10k·2ep) |
| 2 | Korean HateSpeech (도메인 유사 — 탐지 플랫폼에 근접) | (예정) | [ ] |
| 3 | 저자 공식 비교 재현 (IMDb: GPT-2 vs BERT/RoBERTa/DeBERTa/ModernBERT) | (예정) | [ ] |

## 단계 (Phase)

> 이 노트는 **인덱스**다. 각 Phase의 상세는 링크된 정본 노트에.

- **Phase 0 — 교재→BERT 전환 정리** ✅ 완료 (2026-07-03) → [[bert-classification/bert-00-kickoff]]
  교재에서 넘어온 개념(head 교체·-100·CE 손실 위치)이 BERT에서 어떻게 대응되는지 + 선수 기초 점검 + 첫 과제 정의.
- **Phase 1 — KLUE-TC 다중 클래스** ✅ 완료 (2026-07-07) → [[bert-classification/bert-01-klue-tc-multiclass]]
  spam 골격 재사용 + `num_labels=7` 확장 실증. 10,000건·2에폭 → **accuracy 0.8425 · macro F1 0.8313** (공식 베이스라인 ~0.86 근접). 신개념: macro vs weighted 평균, mlflow 3.14 파일스토어 함정(→sqlite). 코드 `bert-classification/finetune_klue_tc.py`.
- **Phase 2 — Korean HateSpeech** [ ]
  도메인이 탐지 플랫폼과 가까운 데이터로 확장. 이진·다중 라벨 체험.
- **Phase 3 — 저자 공식 비교 재현** [ ]
  [rasbt ch06 bonus](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06/03_bonus_imdb-classification) — IMDb로 GPT-2 vs BERT·RoBERTa·DeBERTa·ModernBERT 성능·속도 비교. 교재 저자가 남긴 실질 BERT 학습 자료(부록 B엔 BERT 교재 추천 없음).

## 작업 분담 원칙

| 작업 | 환경 |
|---|---|
| 개념·소규모 코드·디버그 | 로컬 (`~/ml-env`, MPS). MPS 함정 주의 → [[llm-from-scratch/llm-ch7-failure-log]] |
| 실제 파인튜닝(무거우면) | Colab T4 |
| 실험 기록 | MLflow 한글 키 관례 유지 → [[../30-References/mlflow-practice/mlflow-terms-glossary]] |

## 검증 방법

각 Phase마다 ① 미니 데이터 end-to-end 실행 로그 ② MLflow run(한글 키) ③ 학습 노트(무엇이 플랫폼 어느 계층에 대응하는지) 3종이 남으면 완료 처리. — [[detection-ai-study]] 공통 검증 기준 그대로.

## Notion 미러

llm-from-scratch와 동일 구조 — **이 마스터 = 인덱스, Phase별 노트 = 자식 페이지 1:1 미러.**
매핑 완료: 마스터·bert-00·bert-01 (커밋 시 자동 동기). Phase 2·3 페이지는 선생성됨 — vault 노트를 만들면 config 주석만 해제. page_id 정본: [[../30-References/notion-mcp]].

## 관련 노트

- [[detection-ai-study]] — 상위 로드맵 (이 프로젝트의 부모)
- [[llm-from-scratch]] — 선행 트랙 (교재, GPT-2 분류 FT 비교 기준)
- [[../30-References/BERT_학습정리]] — **BERT 개념 정본(7섹션, 코드 없는 눈높이 노트).** 구조·어텐션·마스킹·[CLS]/토큰 출력·NER. 이 프로젝트의 개념 선행 자료.
- [[llm-from-scratch/llm-ch6-classify]] — 6장 분류 FT (head 교체·손실·BERT↔GPT-2 대비)
- [[../30-References/rnd-bert-labeling-test-plan]] — 선수 업무 산출물 (SMS 스팸 2-class)
- [[../30-References/bert-vs-gpt2-classification]] — 트랜스포머·어텐션·인코더/디코더 개념
- [[../30-References/pytorch-env-hybrid]] — 환경 정본
- [[../30-References/mlflow-practice/mlflow-terms-glossary]] — MLflow 용어·한글 키 관례
