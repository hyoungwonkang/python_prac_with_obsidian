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

## 우선순위 (2026-07-02 확정)

### 1군 — 최우선

| # | 기술 | 아키텍처 계층 | 1차 학습 데이터셋 (공개) | 파인튜닝/실습 방법 | 상태 |
|---|---|---|---|---|---|
| 1 | **BERT** (텍스트 분류) | 1차 탐지 Text Classifier (KoBERT) | SMS Spam(완료) → **KLUE-TC**(뉴스 7클래스) · Korean HateSpeech(도메인 유사) | `finetune_bert_spam.py` 재사용, `num_labels`만 변경 → 다중 클래스 확장. **교재 저자 공식 BERT 비교 코드**: [rasbt ch06 bonus](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06/03_bonus_imdb-classification) — IMDb로 GPT-2 vs BERT·RoBERTa·DeBERTa·ModernBERT 비교 (부록 B에 BERT 교재 추천은 없음, 이게 저자의 실질 자료) | ✅ 기초 완료(en/ko ~97%) / [ ] 다중 클래스 심화 |
| 2 | **NER** (개체명 인식) | 1차 탐지 NER Agent (인명·연령·연락처·계정 추출) | **KLUE-NER** (6개 태그, HF datasets) | `BertForTokenClassification` — 토큰 단위 라벨. **subword 정렬에 -100(ignore_index) 그대로 재등장** → 7장 개념 재사용 | ✅ **R&D 기초 완료** (2026-07-05, 상사 지시 2차 — 엔티티 F1 0.7057, MLflow `ner-klue`) → [[../30-References/rnd-detection-models/01-연구문서]] / ✅ **도메인 확장 실측** (2026-07-07 — WikiAnn ko F1 0.8447, 전이 사슬 A/B: 원본 출발 ≥ ner_klue 출발 → [[../30-References/rnd-detection-models/00-학습메모|학습메모]]) / [ ] 업무 도메인 심화 |
| 3 | **YOLO** (객체 탐지) | Image Analyzer (이미지 위험 분류) | **COCO128**(스모크 테스트) → Roboflow 공개셋(커스텀 라벨 형식 연습) | `ultralytics` 패키지, 사전훈련 가중치 전이학습. **MPS 지원** — 단 [[llm-from-scratch/llm-ch7-failure-log|MPS 함정]] 주의 | ✅ **스모크 완료** (2026-07-05 — mAP50 0.606, 사전학습 부분집합이라 파이프라인 검증용) / [ ] 커스텀 데이터 전이 |
| 4 | **OpenCV** (이미지 처리) | 추출·전처리 (프레임 추출, 비식별화) | 데이터셋 불필요 — 샘플 이미지/영상 | 파인튜닝 아님, 라이브러리 실습: 로드·리사이즈·블러(얼굴 비식별화)·영상 프레임 추출 → YOLO 전처리 파이프로 연결 | [ ] |
| 5 | **phi-2** (MS 경량 SLM, 2.7B) | 정형 판단 계층 LLM/SLM Reasoning (아키텍처의 "경량 모델" 대응) | instruction 데이터 재사용 (교재 1,100건 · Alpaca) | HF `microsoft/phi-2` 로드 → 추론 체험 → **LoRA/PEFT 파인튜닝** (2.7B 전체 튜닝은 로컬 무리 → 어댑터 방식 학습이 곧 PEFT 입문). 7장 instruction 튜닝의 직계 후속 | [ ] |
| 6 | **PII** (개인정보 탐지·비식별화) ※2026-07-05 상사 지시로 신설 (원안엔 없던 항목) | NER Agent와 겹침(인명·연락처가 곧 PII) + Rule 계층 | 학습 없음 — 손수 라벨 샘플로 커버리지 확인 | **ko-pii** 라이브러리(룰+사전+체크섬, MIT, Python 3.10+ 필수) — 사용자 지정. 룰의 인명 미탐 한계 → NER과 하이브리드가 실전 방향 | ✅ **R&D 완료** (2026-07-05 — 커버리지 90.9%, 마스킹·Vault 복원) → [[../30-References/rnd-detection-models/01-연구문서]] |

### 2군

| 기술 | 아키텍처 계층 | 데이터셋 | 실습 방법 | 상태 |
|---|---|---|---|---|
| **RULE** (룰 엔진) | 1차 탐지 Rule Engine | 기존 스팸 데이터 재사용 | 정규식·키워드·패턴 룰을 파이썬으로 직접 구현 → **같은 test셋에서 룰 vs BERT의 precision/recall 비교** (트레이드오프 체감 + 하이브리드 구성 근거) | [ ] |
| **CLIP (VLM)** | Image Analyzer (이미지-텍스트 결합 판단) | 소규모 자체 이미지 + 텍스트 프롬프트 | HF CLIP으로 zero-shot 분류 체험 → linear probe 파인튜닝. 한국어는 KoCLIP 후보 | [ ] |

### 3군

| 기술 | 아키텍처 계층 | 실습 방법 | 상태 |
|---|---|---|---|
| **PaddleOCR** | 추출 계층 (이미지 → 텍스트) | 라이브러리 사용 실습(한국어 인식 정확도 확인) → 출력이 BERT/NER 입력으로 이어지는 미니 파이프라인. 커스텀 파인튜닝은 무거워서 후순위. 기존 [[../30-References/bert_ocr_practice_plan]]과 연결 | [ ] |

### 후순위 (플랫폼 2단계 도입 시점에)

- **Hybrid RAG** (SBERT 문장 임베딩 → 벡터 검색 → BM25 하이브리드 → Reranker) — MVP 1단계 핵심이지만 사용자 우선순위 기준으로는 위 목록 이후
- Graph RAG (Neo4j·Entity Linking) · MLOps 심화(Model Registry·Canary) · 인프라(K8s·Kafka·OpenSearch)

## 공통 원칙

- 로컬(M4 Max) 우선, 무거우면 Colab 우회 — 환경 정본 [[../30-References/pytorch-env-hybrid]]
- 모든 실습 MLflow 기록 (한글 키 관례 유지) — [[../30-References/mlflow-practice/mlflow-terms-glossary]]
- "미니 데이터로 우선 완주" — Alpaca OOM 교훈([[llm-from-scratch/llm-ch7-failure-log]]): 작게 시작해 실패 비용 절감
- 진행 순서: [[llm-from-scratch]] **교재 완주(2026-07-03, 부록A+1~7장) → 이 트랙 진입(현재 활성).** 첫 과제 = 1군 #1 BERT **다중 클래스 심화**(KLUE-TC). 선수 기초는 완료(SMS 스팸 en/ko ~97%, [[../30-References/rnd-bert-labeling-test-plan]]).

## 검증 방법

기술마다 ① 미니 데이터 end-to-end 실행 로그 ② MLflow run ③ 학습 노트(무엇이 플랫폼 어느 계층에 대응하는지) 3종이 남으면 완료 처리.

## 관련 노트

- [[../30-References/rnd-detection-models/01-연구문서|rnd-detection-models]] — **NER·YOLO·PII R&D 산출물** (2026-07-05, 상사 지시 2차 — 4종 문서+코드+학습메모)
- [[llm-from-scratch]] — 선행 트랙 (교재)
- [[../30-References/rnd-bert-labeling-test-plan]] — 이 트랙의 선행 업무 산출물 (BERT 스팸 R&D)
- [[../30-References/bert_ocr_practice_plan]] — BERT/OCR 기존 로드맵 (이 노트로 흡수·발전)
