# detection-ai-study — 탐지 플랫폼 대응 학습 트랙

## Context

- 업무 프로젝트: **아동·청소년 온라인 성착취 의심정황 탐지 AI 플랫폼** (Hybrid RAG + Graph RAG 중심 고도화 아키텍처, 2026-07-02 공유).
- 파이프라인: 수집 → 원본 보존 → 추출(OCR·임베딩) → **1차 탐지(Rule·KoBERT 분류·NER·이미지)** → 근거 검색(Hybrid RAG) → 관계 분석(Graph RAG) → LLM 판단·스코어링 → 검수 → 재학습(MLflow).
- [[llm-from-scratch]] (교재 트랙)가 마무리 단계에 들어서며, **플랫폼의 1차 탐지 계층을 코드로 재현할 수 있는 것**을 다음 학습 목표로 설정.
- **학습 순서의 배경**: 이 우선순위를 정해준 상사가 BERT 학습 전에 교재(Raschka)를 먼저 권함 — 트랜스포머 내부·파인튜닝 메커니즘을 바닥부터 다진 뒤 BERT 계열에 진입하는 의도. 우선순위와 교재 선행 모두 같은 상사의 로드맵.
- 아키텍처 원본 이미지는 업무 자료라 **vault에 커밋하지 않음** (로컬 보관).

## 목표

각 기술마다 **① 공개 데이터셋으로 ② 로컬에서 코드 테스트 가능한 파인튜닝/실습**을 완성한다.
검증 기준: 미니 데이터로 end-to-end 실행 + MLflow 기록 (llm-from-scratch 방식 그대로).

## 우선순위 (2026-07-02 확정 · 2026-07-08 갱신)

> **2026-07-08 갱신**: **phi-2 삭제** — PII와 혼동해 넣었던 항목이며 이제 하지 않음. 남은 순서를 **OpenCV → RULE → CLIP(VLM) → PaddleOCR**로 확정(OpenCV 우선).
>
> 아키텍처 탐지 흐름(§4): `① 수집 → ② OCR/추출 → ③ 1차 탐지(Rule·분류·NER·이미지) → ④ 근거검색(Hybrid RAG) → ⑤ 관계분석(Graph RAG) → ⑥ 정형판단 → ⑦ 위험도`. 아래 "흐름"은 이 단계 번호.

### 완료 — R&D 진행·보고됨 (2026-07-05 ~ 07)

| 기술 | 아키텍처 계층 (흐름) | 1차 학습 데이터셋 (공개) | 파인튜닝/실습 방법 | 상태 |
|---|---|---|---|---|
| **BERT** (텍스트 분류) | 1차 탐지 Text Classifier / KoBERT (③) | SMS Spam → **KLUE-TC**(뉴스 7클래스) | `finetune_bert_spam.py` 재사용, `num_labels` 확장 → 다중 클래스. **교재 저자 BERT 비교 코드**: [rasbt ch06 bonus](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06/03_bonus_imdb-classification) | ✅ 기초(en/ko ~97%) + **다중 클래스 KLUE-TC**(정확도 0.8425·매크로F1 0.8313) → [[bert-classification/bert-01-klue-tc-multiclass]] |
| **NER** (개체명 인식) | 1차 탐지 NER Agent (③, 인명·연령·연락처·계정) | **KLUE-NER**(13태그) · WikiAnn ko(7태그) | `BertForTokenClassification` — subword 정렬 -100 재사용 (7장 개념) | ✅ 기초(엔티티 F1 0.7057) + 도메인 확장(WikiAnn F1 0.8447, 전이 사슬 A/B) → [[../30-References/rnd-detection-models/00-학습메모|학습메모]] |
| **YOLO** (객체 탐지) | 1차 탐지 Image Analyzer (③) | **COCO128**(스모크) → Roboflow 공개셋 | `ultralytics` 전이학습, MPS(단 [[llm-from-scratch/llm-ch7-failure-log|MPS 함정]]) | ✅ 스모크(mAP50 0.606) / [ ] 커스텀 데이터 전이 |
| **PII** (개인정보 탐지·비식별화) | NER∩Rule (③, 인명·연락처) | 손수 라벨 샘플 | **ko-pii**(룰+사전+체크섬, MIT) — 학습 없음. 룰의 인명 미탐 한계 → NER과 하이브리드 | ✅ 완료(커버리지 90.9%, 마스킹·Vault 복원) → [[../30-References/rnd-detection-models/01-연구문서]] |

### 남은 순서 (2026-07-08 확정) — OpenCV → RULE → CLIP(VLM) → PaddleOCR

세 갈래 중 **CLIP만 실제 파인튜닝(linear probe)**이고, OpenCV·RULE·PaddleOCR은 라이브러리·룰 실습 성격(PII와 같은 방식). 데이터 흐름상 PaddleOCR·OpenCV(②)가 Rule·CLIP(③)보다 앞단이지만, 지금까지처럼 **각 모듈을 독립 실습**하므로 우선순위 순서로 진행한다.

| 순 | 기술 | 아키텍처 계층 (흐름) | 데이터셋 | 방법·성격 | 상태 |
|---|---|---|---|---|---|
| **1 (다음 착수)** | **OpenCV** (이미지 처리) | 추출·전처리 (②, 프레임 추출·비식별화) | 데이터셋 불필요 — 샘플 이미지/영상 | 파인튜닝 아님. 로드·리사이즈·블러(얼굴 비식별화)·영상 프레임 추출 → YOLO 전처리 파이프로 연결 | [ ] |
| 2 | **RULE** (룰 엔진) | 1차 탐지 Rule Engine (③, 1차 탐지 첫 관문) | 기존 스팸 데이터 재사용 | 학습 없음 — 정규식·키워드·패턴 직접 구현. **같은 test셋에서 Rule vs BERT의 precision/recall 비교**(트레이드오프 체감 + 하이브리드 근거) | [ ] |
| 3 | **CLIP** (VLM) | Image Analyzer (③, 이미지-텍스트 결합 판단) | 소규모 자체 이미지 + 텍스트 프롬프트 | **유일한 파인튜닝**: HF CLIP zero-shot 체험 → linear probe. 한국어는 KoCLIP 후보 | [ ] |
| 4 | **PaddleOCR** | 추출 계층 (②, 이미지→텍스트) | — (라이브러리) | 라이브러리 추론(한국어 인식 정확도 확인) → 출력이 BERT/NER 입력으로 이어지는 미니 파이프라인. 커스텀 학습은 무거워 최후순위. [[../30-References/bert_ocr_practice_plan]] 연결 | [ ] |

### 후순위 (플랫폼 2단계 도입 시점에)

- **Hybrid RAG** (SBERT 문장 임베딩 → 벡터 검색 → BM25 하이브리드 → Reranker) — 아키텍처 §2 MVP 1단계 핵심(④ 근거검색). 위 목록 완료 후 진입.
- Graph RAG (Neo4j·Entity Linking, ⑤) · **정형 판단 LLM/SLM(⑥, 이전 phi-2 자리 — 특정 모델 미정)** · MLOps 심화(Model Registry·Canary) · 인프라(K8s·Kafka·OpenSearch)

## 공통 원칙

- 로컬(M4 Max) 우선, 무거우면 Colab 우회 — 환경 정본 [[../30-References/pytorch-env-hybrid]]
- 모든 실습 MLflow 기록 (한글 키 관례 유지) — [[../30-References/mlflow-practice/mlflow-terms-glossary]]
- "미니 데이터로 우선 완주" — Alpaca OOM 교훈([[llm-from-scratch/llm-ch7-failure-log]]): 작게 시작해 실패 비용 절감
- 진행 순서: [[llm-from-scratch]] **교재 완주(2026-07-03) → 이 트랙 진입.** **BERT·NER·YOLO·PII R&D 완료·보고**(2026-07-05~07) → **다음 착수 = OpenCV**(2026-07-08) → RULE → CLIP(VLM) → PaddleOCR.

## 검증 방법

기술마다 ① 미니 데이터 end-to-end 실행 로그 ② MLflow run ③ 학습 노트(무엇이 플랫폼 어느 계층에 대응하는지) 3종이 남으면 완료 처리.

## 관련 노트

- [[../30-References/rnd-detection-models/01-연구문서|rnd-detection-models]] — **NER·YOLO·PII R&D 산출물** (2026-07-05, 상사 지시 2차 — 4종 문서+코드+학습메모)
- [[llm-from-scratch]] — 선행 트랙 (교재)
- [[../30-References/rnd-bert-labeling-test-plan]] — 이 트랙의 선행 업무 산출물 (BERT 스팸 R&D)
- [[../30-References/bert_ocr_practice_plan]] — BERT/OCR 기존 로드맵 (이 노트로 흡수·발전)
