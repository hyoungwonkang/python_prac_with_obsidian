# llm-ch4-gpt

[[llm-from-scratch]] 마스터 플랜의 **Phase 4 / 4장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 4장. 밑바닥부터 GPT 모델 구현하기.

## 개요

LayerNorm·GELU·FeedForward·residual을 조립해 transformer 블록을 만들고, N개 블록을 쌓아 GPT 아키텍처 전체를 구성하는 단계.

## 체크리스트

- [x] 4.1 LayerNorm 직접 구현 — `layer_normalization.py`
- [x] 4.2 GELU 활성화·FeedForward 블록 — `feed_forward.py`
- [ ] 4.3 Residual connection 포함 transformer 블록
- [~] 4.4 GPT 모델 아키텍처 조립 — `dummy_gpt_model.py`로 **더미 골격** 완성(블록·Norm은 자리표시)
- [ ] 4.5 초기화된 모델로 텍스트 생성 (greedy → top-k → temperature)
- [ ] 4.6 파라미터 수 계산·디바이스 메모리 점검

## 학습 정리

### 4.4(선행) DummyGPTModel — 전체 골격 먼저 잡기

`dummy_gpt_model.py`: 실제 트랜스포머 블록·LayerNorm을 넣기 전, **데이터 흐름(shape)만** 검증하는 더미 버전.

- 구성: `tok_emb`(50257×768) + `pos_emb`(1024×768) → dropout → `DummyTransformerBlock×12`(통과만) → `DummyLayerNorm`(통과만) → `out_head`(768→50257).
- 입력 `(batch, seq_len)` 정수 → 출력 `(batch, seq_len, 50257)` logits. 마지막 50257 = 다음 토큰 후보 점수.
- `GPT_CONFIG_124M`: vocab 50257, context_length 1024, emb_dim 768, n_heads 12, n_layers 12, drop 0.1.
- `vocab_size`(토큰 종류) ↔ `context_length`(최대 토큰 수=위치 칸)는 각각 두 임베딩의 **행 개수**, `emb_dim`은 **벡터 길이**(둘이 같아야 더할 수 있음).

### 4.2 GELU + FeedForward — 확장→비선형→축소

`feed_forward.py`: GELU(tanh 근사) + `Linear→GELU→Linear` 블록.

- 구조 **768 → 3072(4배 확장) → GELU → 3072 → 768(축소)**. 입출력 차원 동일 → residual 연결·다음 블록과 모양 유지. 입력 `(2,3,768)` → 출력 `(2,3,768)`.
- **GELU vs ReLU**: ReLU는 음수를 0으로 딱 자르고 0에서 꺾임(미분 불연속). GELU는 부드러운 곡선이라 기울기가 매끄럽게 흐르고, 음수도 약간 통과해 dead neuron 완화. "더 정확"이 아니라 **학습이 더 안정적**이라 대형 트랜스포머에서 표준.
- **확장+GELU의 의미**: 선형만 쌓으면 `Linear·Linear=Linear`로 붕괴 → 확장(넓은 공간)은 **비선형(GELU)과 짝일 때만** 풍부한 표현 탐색이 됨. 확장=넓은 캔버스, GELU=직선 아닌 그림.
- 흔한 버그: 두 번째 Linear를 `(emb, 4*emb)`로 쓰면 shape 불일치 → `(4*emb, emb)`여야 함. `forward`(`return self.layers(x)`) 누락 시 `NotImplementedError`.

### 4.1 LayerNorm — 출력을 평균 0·분산 1로 재조정

`layer_normalization.py`: `(x - 평균) / sqrt(분산 + eps)` 후 학습 파라미터 `scale`·`shift` 적용.

- `dim=-1, keepdim=True`로 **특징축(마지막 차원)** 기준 정규화, keepdim으로 브로드캐스팅용 `(N,1)` 유지.
- 검증 출력 평균≈0(부동소수점 잔차 `1e-8`), 분산=1. `-0.0000`/`0.0000` 부호 차이는 행별 반올림 오차 방향일 뿐 의미 없음.
- `set_printoptions(sci_mode=False)`는 `1.19e-8`을 `0.0000`으로 **보기 좋게** 표시할 뿐 값은 동일.
- 정규화 대상은 **텐서 rank(축 개수)** 가 아니라 **특징 개수**. mean은 특징을 6→1로 줄여도 rank(2)는 유지.

## 검증

책 코드와 동일 입력 → 동일/유사 출력 재현. 로컬(2.2.2)·Colab(2.6.0) 양쪽에서 코드 동작.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../../30-References/pytorch-env-hybrid]] — 환경 정본
