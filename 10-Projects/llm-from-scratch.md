# llm-from-scratch

밑바닥부터 LLM(GPT 계열)을 직접 구현·학습하는 마스터 학습 플랜.
교재: **Sebastian Raschka, 밑바닥부터 만들면서 배우는 LLM** (원서: *Build a Large Language Model (From Scratch)*).

## Context

- 사용자의 ML 학습 트랙 **최상위 목표**: LLM을 처음부터 구현해 보면서 동작 원리를 이해.
- 교재 본문은 1~7장으로 구성. **부록 A(PyTorch)**는 본문 진입 전 필수 기초 → 이미 [[pytorch-study]]로 분리 진행 중.
- 이 노트는 **마스터 인덱스**: 각 장(Phase)의 큰 흐름·체크리스트만 관리하고, 분량이 커지는 장은 별도 `10-Projects/llm-from-scratch/llm-ch{n}-*.md` 노트로 분리.
- 환경 정본은 [[../30-References/pytorch-env-hybrid]] (인텔 맥 로컬 2.2.2 + Colab 2.6.0 하이브리드)를 그대로 따른다.

## 교재 구성과 진행 현황

| 부 | 제목 | vault 노트 | 상태 |
|---|---|---|---|
| 부록 A | PyTorch 기초 | [[pytorch-study]] | ✅ 완료 (2026-06-17) |
| 1장 | 대규모 언어 모델 이해하기 | [[llm-ch1-overview]] | 미시작 |
| 2장 | 텍스트 데이터 다루기 | [[llm-ch2-text]] | 미시작 |
| 3장 | 어텐션 메커니즘 구현하기 | [[llm-ch3-attention]] | 미시작 |
| 4장 | 밑바닥부터 GPT 모델 구현하기 | [[llm-ch4-gpt]] | 미시작 |
| 5장 | 레이블이 없는 데이터를 활용한 사전 훈련 | [[llm-ch5-pretrain]] | 미시작 |
| 6장 | 분류를 위해 미세 튜닝하기 | [[llm-ch6-classify]] | 미시작 |
| 7장 | 지시를 따르도록 미세 튜닝하기 | [[llm-ch7-instruct]] | 미시작 |

## 작업 분담 원칙 (전체 공통)

| 작업 | 환경 |
|---|---|
| 개념·문법·소규모 코드 작성 | 로컬 (`~/ml-env`, PyTorch 2.2.2, MPS) |
| 교재 코드 정확 재현 | Colab (PyTorch 2.6.0) |
| GPT-2 가중치 로드·사전훈련·미세튜닝 | **Colab T4 GPU** (로컬은 코드 작성·디버그용) |
| 학습 메모·결과 기록 | 이 노트 + 각 장 노트 + [[../90-Daily]] |

## 단계 (Phase)

> 이 노트는 **인덱스**다. 각 Phase의 상세 체크리스트·검증 기준은 아래 링크된 장별 정본 노트에 있다.
> (Notion 미러에서도 각 장은 자식 페이지로 분리되어, 이 마스터 페이지에는 체크박스가 없다.)

- **Phase 0 — 부록 A: PyTorch 기초 (선행)** ✅ 완료 (2026-06-17) → [[pytorch-study]]
  교재 부록 A(A.1~A.10) 전 항목 완료. 회고: [[../30-References/python-basics#PyTorch 부록 A 회고]]. 다음: 본문 1장 진입.
- **Phase 1 — 1장. 대규모 언어 모델 이해하기** → [[llm-ch1-overview]]
  LLM·트랜스포머의 큰 그림. 코드 작성 거의 없음.
- **Phase 2 — 2장. 텍스트 데이터 다루기** → [[llm-ch2-text]]
  토큰화·임베딩·`DataLoader`로 입력 파이프라인 구축.
- **Phase 3 — 3장. 어텐션 메커니즘 구현하기** → [[llm-ch3-attention]]
  self-attention → Q/K/V → causal → multi-head 단계적 구현.
- **Phase 4 — 4장. 밑바닥부터 GPT 모델 구현하기** → [[llm-ch4-gpt]]
  LayerNorm·GELU·residual 블록 조립 → GPT 아키텍처 완성.
- **Phase 5 — 5장. 레이블이 없는 데이터를 활용한 사전 훈련** → [[llm-ch5-pretrain]]
  학습 루프·사전훈련, GPT-2 공개 가중치 로드. 실학습은 Colab T4 메인.
- **Phase 6 — 6장. 분류를 위해 미세 튜닝하기** → [[llm-ch6-classify]]
  분류 head 추가 + 레이어 freeze로 분류 FT.
  📌 **별도 트랙 접점:** 이 장 완료 직후가 [[../30-References/bert_ocr_practice_plan]] BERT 실습 진입 시점(정본 결정). GPT 분류 FT를 직접 짠 뒤 → 같은 걸 BERT+HF로 비교 학습. OCR은 7장 완주 후 최하 우선순위.
- **Phase 7 — 7장. 지시를 따르도록 미세 튜닝하기** → [[llm-ch7-instruct]]
  instruction 데이터셋·프롬프트 템플릿으로 지시 따르기 FT.

## 검증 방법

| 단계 | 검증 |
|---|---|
| Phase 0 | [[pytorch-study]]의 Phase별 검증 기준을 그대로 따름. |
| Phase 1 | 책의 그림·표를 본인 말로 1쪽 노트로 재작성 → 핵심 용어 정의 가능. |
| Phase 2~4 | 책 코드와 동일 입력 → 동일/유사 출력 재현. 로컬(2.2.2)·Colab(2.6.0) 양쪽에서 코드 동작. |
| Phase 5 | 사전훈련 loss 곡선이 우하향. 공개 GPT-2 가중치 로드 후 텍스트 생성이 의미 있는 출력. |
| Phase 6~7 | 미세튜닝 전/후 지표 비교 — 분류 정확도 상승, instruction 응답 품질 개선. |

각 Phase 종료 시 [[../90-Daily]] 데일리 노트에 "무엇을 했는지·다음 막힌 부분" 1~2단락 기록.

## Notion 미러

[[../30-References/notion-mcp]] 운영 원칙에 따라 이 노트는 **vault 정본 = 원본**.
구조: **이 마스터 노트 = 인덱스(체크박스 없음)**, 각 장(부록A·1~7장)은 **자식 페이지**로 분리되어 각 정본 노트(`pytorch-study.md`, `llm-from-scratch/llm-ch{n}-*.md`)와 1:1 미러. config 매핑은 `.notion-sync/config.yaml`, page_id는 [[../30-References/notion-mcp]] 참조.

## 관련 노트

- [[_Projects]] (인덱스)
- [[pytorch-study]] — 부록 A 정본
- [[llm-ch1-overview]] · [[llm-ch2-text]] · [[llm-ch3-attention]] · [[llm-ch4-gpt]] · [[llm-ch5-pretrain]] · [[llm-ch6-classify]] · [[llm-ch7-instruct]] — 1~7장 정본
- [[../30-References/pytorch-env-hybrid]] — 환경 정본
- [[../30-References/notion-mcp]] — Notion 미러 운영 규칙
- [[../30-References/bert_ocr_practice_plan]] — 별도 트랙(BERT·OCR 라이브러리 실습). 진입 시점은 6장 직후(BERT)·7장 후(OCR).
