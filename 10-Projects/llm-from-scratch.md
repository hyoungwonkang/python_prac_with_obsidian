# llm-from-scratch

밑바닥부터 LLM(GPT 계열)을 직접 구현·학습하는 마스터 학습 플랜.
교재: **Sebastian Raschka, 밑바닥부터 만들면서 배우는 LLM** (원서: *Build a Large Language Model (From Scratch)*).

## Context

- 사용자의 ML 학습 트랙 **최상위 목표**: LLM을 처음부터 구현해 보면서 동작 원리를 이해.
- 교재 본문은 1~7장으로 구성. **부록 A(PyTorch)**는 본문 진입 전 필수 기초 → 이미 [[pytorch-study]]로 분리 진행 중.
- 이 노트는 **마스터 인덱스**: 각 장(Phase)의 큰 흐름·체크리스트만 관리하고, 분량이 커지는 장은 별도 `10-Projects/llm-ch{n}-*.md` 노트로 분리.
- 환경 정본은 [[../30-References/pytorch-env-hybrid]] (인텔 맥 로컬 2.2.2 + Colab 2.6.0 하이브리드)를 그대로 따른다.

## 교재 구성과 진행 현황

| 부 | 제목 | vault 노트 | 상태 |
|---|---|---|---|
| 부록 A | PyTorch 기초 | [[pytorch-study]] | 진행 중 (Phase 1) |
| 1장 | 대규모 언어 모델 이해하기 | — | 미시작 |
| 2장 | 텍스트 데이터 다루기 | — | 미시작 |
| 3장 | 어텐션 메커니즘 구현하기 | — | 미시작 |
| 4장 | 밑바닥부터 GPT 모델 구현하기 | — | 미시작 |
| 5장 | 레이블이 없는 데이터를 활용한 사전 훈련 | — | 미시작 |
| 6장 | 분류를 위해 미세 튜닝하기 | — | 미시작 |
| 7장 | 지시를 따르도록 미세 튜닝하기 | — | 미시작 |

## 작업 분담 원칙 (전체 공통)

| 작업 | 환경 |
|---|---|
| 개념·문법·소규모 코드 작성 | 로컬 (`~/ml-env`, PyTorch 2.2.2, MPS) |
| 교재 코드 정확 재현 | Colab (PyTorch 2.6.0) |
| GPT-2 가중치 로드·사전훈련·미세튜닝 | **Colab T4 GPU** (로컬은 코드 작성·디버그용) |
| 학습 메모·결과 기록 | 이 노트 + 각 장 노트 + [[../90-Daily]] |

## 단계 (Phase)

### Phase 0 — 부록 A: PyTorch 기초 (선행)

정본: [[pytorch-study]]. 본문 1장 진입 전, 최소 Phase 2(첫 학습 루프)까지는 끝내는 게 권장.

- [x] Phase 0: 환경 셋업 ✅ (2026-06-16)
- [ ] Phase 1: 텐서 기초 (진행 중)
- [ ] Phase 2: autograd & 첫 학습 루프
- [ ] Phase 3: 첫 분류 모델 (MNIST)
- [ ] Phase 4: 검증·재현성·실험 관리
- [ ] Phase 5: 응용 (선택)

### Phase 1 — 1장. 대규모 언어 모델 이해하기

LLM이 무엇이고 왜 트랜스포머 기반인지, 책 전체에서 무엇을 만들지 큰 그림을 잡는 단계. 코드 작성은 거의 없음.

- [ ] 1.1 LLM 개념·역사·응용 사례 정리
- [ ] 1.2 트랜스포머 아키텍처 개요 (encoder/decoder, GPT는 decoder-only)
- [ ] 1.3 LLM 학습 단계 개요 (사전훈련 → 미세튜닝)
- [ ] 1.4 책 전체 구조·코드 레포·실습 환경 파악
- [ ] 1.5 핵심 용어 정리 노트 신설 (예: `30-References/llm-glossary.md`)

### Phase 2 — 2장. 텍스트 데이터 다루기

- [ ] 2.1 토크나이저 (단어 단위 → 서브워드 → BPE)
- [ ] 2.2 `tiktoken`으로 GPT-2 BPE 사용 실습
- [ ] 2.3 토큰 임베딩 + 위치(positional) 임베딩
- [ ] 2.4 슬라이딩 윈도우 데이터셋·`DataLoader` 구현
- [ ] 2.5 본인 텍스트로 동일 파이프라인 재현

### Phase 3 — 3장. 어텐션 메커니즘 구현하기

- [ ] 3.1 단순 self-attention (가중치 학습 없이 dot-product)
- [ ] 3.2 학습 가능한 가중치(Q, K, V) 도입
- [ ] 3.3 scaled dot-product attention
- [ ] 3.4 causal(마스킹) attention
- [ ] 3.5 multi-head attention 구현
- [ ] 3.6 텐서 shape 변화를 손으로 추적해 노트로 정리

### Phase 4 — 4장. 밑바닥부터 GPT 모델 구현하기

- [ ] 4.1 LayerNorm 직접 구현
- [ ] 4.2 GELU 활성화·FeedForward 블록
- [ ] 4.3 Residual connection 포함 transformer 블록
- [ ] 4.4 GPT 모델 아키텍처 조립 (임베딩 + N×블록 + 최종 norm + 출력 head)
- [ ] 4.5 초기화된 모델로 텍스트 생성 (greedy → top-k → temperature)
- [ ] 4.6 파라미터 수 계산·디바이스 메모리 점검

### Phase 5 — 5장. 레이블이 없는 데이터를 활용한 사전 훈련

- [ ] 5.1 cross-entropy loss + perplexity 계산
- [ ] 5.2 학습 루프 (forward → loss → backward → optimizer.step)
- [ ] 5.3 작은 코퍼스로 1 epoch 학습 (로컬에서 검증)
- [ ] 5.4 학습률 워밍업·코사인 스케줄링
- [ ] 5.5 체크포인트 저장/로드 (`weights_only=True`)
- [ ] 5.6 OpenAI 공개 GPT-2 가중치 로드 → 자체 구조에 매핑
- [ ] 5.7 **Colab T4에서 소규모 사전훈련 실행** (로컬은 코드 디버그)

### Phase 6 — 6장. 분류를 위해 미세 튜닝하기

- [ ] 6.1 분류 데이터셋 준비 (예: spam 분류)
- [ ] 6.2 모델에 분류 head 추가
- [ ] 6.3 일부 레이어 freeze 전략
- [ ] 6.4 fine-tuning 학습 루프 + 평가 지표 (accuracy)
- [ ] 6.5 결과 정리

### Phase 7 — 7장. 지시를 따르도록 미세 튜닝하기

- [ ] 7.1 instruction 데이터셋 구성 (Alpaca 등)
- [ ] 7.2 입력 포맷팅(프롬프트 템플릿) 적용
- [ ] 7.3 instruction fine-tuning 학습
- [ ] 7.4 정성 평가 + 간단한 자동 평가 루프
- [ ] 7.5 회고 — 사전훈련 vs 분류 FT vs 지시 FT의 차이 노트화

## 검증 방법

| 단계 | 검증 |
|---|---|
| Phase 0 | [[pytorch-study]]의 Phase별 검증 기준을 그대로 따름. |
| Phase 1 | 책의 그림·표를 본인 말로 1쪽 노트로 재작성 → 핵심 용어 정의 가능. |
| Phase 2~4 | 책 코드와 동일 입력 → 동일/유사 출력 재현. 로컬(2.2.2)·Colab(2.6.0) 양쪽에서 코드 동작. |
| Phase 5 | 사전훈련 loss 곡선이 우하향. 공개 GPT-2 가중치 로드 후 텍스트 생성이 의미 있는 출력. |
| Phase 6~7 | 미세튜닝 전/후 지표 비교 — 분류 정확도 상승, instruction 응답 품질 개선. |

각 Phase 종료 시 [[../90-Daily]] 데일리 노트에 "무엇을 했는지·다음 막힌 부분" 1~2단락 기록.

## Notion 미러 (예정)

[[../30-References/notion-mcp]] 운영 원칙에 따라 이 노트는 **vault 정본 = 원본**.
인증·부모 페이지 준비 후 Notion에 미러 페이지 생성 예정 — 1~7장 + 부록A를 to-do 블록으로, 각 장은 자식 페이지(또는 토글) 구조.

## 관련 노트

- [[_Projects]] (인덱스)
- [[pytorch-study]] — 부록 A 정본
- [[../30-References/pytorch-env-hybrid]] — 환경 정본
- [[../30-References/notion-mcp]] — Notion 미러 운영 규칙
