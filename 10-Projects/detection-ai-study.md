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
| **1** | **OpenCV** (이미지 처리) | 추출·전처리 (②, 프레임 추출·비식별화) | 데이터셋 불필요 — 샘플 이미지/영상 | 파인튜닝 아님. 로드·리사이즈·블러·영상 프레임 추출 + **YOLO 검출→OpenCV 블러 비식별화** | ✅ **R&D 완료** (2026-07-08 — 비식별 선명도 100.7→13.8, 파이프라인 사람 6명 비식별. cv2 5.0 objdetect 부재→YOLO 결합) → [[../30-References/rnd-detection-models-2/01-연구문서]] |
| 2 | **RULE** (룰 엔진) | 1차 탐지 Rule Engine (③, 1차 탐지 첫 관문) | 기존 스팸 데이터 재사용 | 학습 없음 — 정규식·키워드·패턴 직접 구현. **같은 test셋에서 Rule vs BERT의 precision/recall 비교**(트레이드오프 체감 + 하이브리드 근거) | ✅ **완료** (2026-07-09, 지시 2와 통합 수행 — RULE F1 0.8951·문턱 트레이드오프 실측·하이브리드 AND/OR 검증) → [[../30-References/rnd-rule-vs-bert/01-연구문서]] |
| 3 | **CLIP** (VLM) | Image Analyzer (③, 이미지-텍스트 결합 판단) | 소규모 자체 이미지 + 텍스트 프롬프트 | **유일한 파인튜닝**: HF CLIP zero-shot 체험 → linear probe. 한국어는 KoCLIP 후보 | [ ] |
| 4 | **PaddleOCR** | 추출 계층 (②, 이미지→텍스트) | — (라이브러리) | 라이브러리 추론(한국어 인식 정확도 확인) → 출력이 BERT/NER 입력으로 이어지는 미니 파이프라인. 커스텀 학습은 무거워 최후순위. [[../30-References/bert_ocr_practice_plan]] 연결 | [ ] |

### 새 지시 목록 (2026-07-08) — 기존 로드맵과의 매핑

상사 지시 6항목. 1번 완료, 나머지는 기존 항목과 겹치거나 신규.

| 지시 | 내용 | 기존 로드맵 대응 | 상태 |
|---|---|---|---|
| **1** | **학습 데이터 산출물 (확장성)** | 신규 — 범용 학습기·데이터 규약·산출물 3종 세트 규칙 | ✅ **환경 구성 완료** (2026-07-08 — 스팸 스모크 0.9367, YOLO 데모 mAP50 0.8718, PII 33라벨 스키마+검증) → [[../30-References/rnd-dataset-artifacts/01-연구문서]] |
| 2 | 분류 잘하는 법 | **RULE과 겹침** — 고정 test셋에서 Rule vs BERT 등 비교(MLflow) | ✅ **완료** (2026-07-09 — RULE F1 0.8951 / BERT-full F1 0.9517 / HYBRID-AND P 1.0·오탐 0 / HYBRID-OR R 0.9338. 결론: 단일 승자 없음, 업무 요구별 선택지 메뉴) → [[../30-References/rnd-rule-vs-bert/01-연구문서]] |
| 3 | 통합 UXUI | 신규 — **순서 3번째(2→4→3 확정, 2026-07-09)**: CLIP 완료 직후 착수. YOLO·CLIP·텍스트(분류/PII/NER) 모듈을 데모 UI로 통합. R&D 아닌 제작이므로 문서는 사용법·구성도 중심(연구문서 생략 가능) | [ ] |
| 3.1 | OCR — Paddle·EasyOCR 활용 | PaddleOCR 항목과 동일 (+EasyOCR 비교 추가) — **후순위(2026-07-09)**: UXUI까지 완료 후 | [ ] |
| 4 | CLIP 이미지 상황 판단 | CLIP(VLM) 항목과 동일 — **순서 2번째** (UXUI의 선행 요건: UI가 CLIP을 표현하려면 모듈이 먼저) | [ ] |
| 4′ | YOLO 라벨링 직접 → 등록 학습 | 신규 — [[../30-References/rnd-dataset-artifacts/03-사용법|YOLO 데이터 규약]] 위에서 진행 (뼈대 생성기 완비) | [ ] |

### 후순위 (플랫폼 2단계 도입 시점에)

- **Hybrid RAG** (SBERT 문장 임베딩 → 벡터 검색 → BM25 하이브리드 → Reranker) — 아키텍처 §2 MVP 1단계 핵심(④ 근거검색). 위 목록 완료 후 진입.
- Graph RAG (Neo4j·Entity Linking, ⑤) · **정형 판단 LLM/SLM(⑥, 이전 phi-2 자리 — 특정 모델 미정)** · MLOps 심화(Model Registry·Canary) · 인프라(K8s·Kafka·OpenSearch)

## 공통 원칙

- 로컬(M4 Max) 우선, 무거우면 Colab 우회 — 환경 정본 [[../30-References/pytorch-env-hybrid]]
- 모든 실습 MLflow 기록 (한글 키 관례 유지) — [[../30-References/mlflow-practice/mlflow-terms-glossary]]
- "미니 데이터로 우선 완주" — Alpaca OOM 교훈([[llm-from-scratch/llm-ch7-failure-log]]): 작게 시작해 실패 비용 절감
- 진행 순서: [[llm-from-scratch]] **교재 완주(2026-07-03) → 이 트랙 진입.** **BERT·NER·YOLO·PII R&D 완료·보고**(2026-07-05~07) → **OpenCV 완료**(2026-07-08, [[../30-References/rnd-detection-models-2/00-학습메모]]) → **학습 데이터 산출물 환경 완료**(2026-07-08, [[../30-References/rnd-dataset-artifacts/00-학습메모]]) → **RULE/지시 2 완료**(2026-07-09, [[../30-References/rnd-rule-vs-bert/01-연구문서|rnd-rule-vs-bert]]) → **다음 착수 = CLIP(VLM, 지시 4)** → **통합 UXUI(지시 3 — YOLO·CLIP·텍스트 모듈 데모 통합)** → PaddleOCR(+EasyOCR, 지시 3.1 — 후순위). *(순서 확정 2026-07-09: 지시 2→4→3. UXUI가 CLIP·YOLO 사용까지 표현하므로 CLIP이 UXUI에 선행, OCR은 UXUI 이후.)*

## 검증 방법

기술마다 ① 미니 데이터 end-to-end 실행 로그 ② MLflow run ③ 학습 노트(무엇이 플랫폼 어느 계층에 대응하는지) 3종이 남으면 완료 처리.

## 관련 노트

- [[../30-References/rnd-detection-models/01-연구문서|rnd-detection-models]] — **NER·YOLO·PII R&D 산출물** (2026-07-05, 상사 지시 2차 — 4종 문서+코드+학습메모)
- [[../30-References/rnd-dataset-artifacts/01-연구문서|rnd-dataset-artifacts]] — **학습 데이터 산출물 환경** (2026-07-08, 새 지시 1번 — 범용 학습기·데이터 규약·산출물 규칙)
- [[../30-References/rnd-rule-vs-bert/01-연구문서|rnd-rule-vs-bert]] — **분류 방법 비교: Rule vs BERT vs 하이브리드** (2026-07-09, 새 지시 2번 — 문서 4종+코드 3종+학습메모)
- [[llm-from-scratch]] — 선행 트랙 (교재)
- [[../30-References/rnd-bert-labeling-test-plan]] — 이 트랙의 선행 업무 산출물 (BERT 스팸 R&D)
- [[../30-References/bert_ocr_practice_plan]] — BERT/OCR 기존 로드맵 (이 노트로 흡수·발전)
