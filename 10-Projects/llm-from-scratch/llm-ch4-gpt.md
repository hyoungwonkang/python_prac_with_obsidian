# llm-ch4-gpt

[[llm-from-scratch]] 마스터 플랜의 **Phase 4 / 4장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 4장. 밑바닥부터 GPT 모델 구현하기.

## 개요

LayerNorm·GELU·FeedForward·residual을 조립해 transformer 블록을 만들고, N개 블록을 쌓아 GPT 아키텍처 전체를 구성하는 단계.

## 체크리스트

- [x] 4.1 LayerNorm 직접 구현 — `layer_normalization.py`
- [x] 4.2 GELU 활성화·FeedForward 블록 — `feed_forward.py`
- [x] 4.3 Residual connection 포함 transformer 블록 — `transformer.py`(`TransformerBlock`)
- [x] 4.4 GPT 모델 아키텍처 조립 — `Gpt.py`(실제 `GPTModel`, 더미 골격 → 실블록 교체 완료)
- [x] 4.5 초기화된 모델로 텍스트 생성 — `Gpt.py` `generate_text_simple`(greedy). top-k·temperature는 5장
- [x] 4.6 파라미터 수 계산·메모리 점검 — `Gpt.py`(`numel` 합·weight tying·MB)
- [x] 연습 4.1 GPT-2 4종(124M/Medium/Large/XL) config 추가·크기별 파라미터 비교 — `Gpt.py`
- [ ] 연습: 드롭아웃 3곳(임베딩·숏컷·어텐션) 분리 설정 (`drop_rate`→3개 키)

> **4장 본문 완료 (2026-06-23).** 환경: Apple Silicon M4 Max 로컬(`~/ml-env`, torch 2.8.0). 다음: 5장 사전훈련.

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

### 4.3 TransformerBlock 조립 (`transformer.py`)

- 구조(Pre-LN): `norm1 → 어텐션 → drop → +shortcut` → `norm2 → FFN → drop → +shortcut`. residual로 그레이디언트 흐름 보존(아래 복습 참조).
- 입출력 모양 동일 `(2,4,768) → (2,4,768)` → 블록을 N개 쌓을 수 있는 근거.
- 🐛 버그: `x = self.norm2`(모듈 객체 대입) → `x = self.norm2(x)`(호출)로 수정. 미호출 시 `Linear`가 텐서 대신 모듈을 받아 `F.linear`에서 에러.

### 셀프 어텐션 vs FFN — 소통 vs 개별 소화

- **어텐션 = 수평(토큰↔토큰)**: 시퀀스 원소 *사이*의 관계 파악, 정보 섞음(communication).
- **FFN = 수직(토큰 내부)**: 각 위치를 독립 변환(다른 토큰 안 봄), 모든 위치에 같은 가중치 공유(computation).
- FFN = **F**eed-**F**orward **N**etwork(앞으로만 흐르는 신경망, = MLP). 트랜스포머 블록 = "섞기(어텐션)→혼자 소화(FFN)"를 한 세트로 N번 반복.

### 4.4 GPTModel 실제 조립 (`Gpt.py`)

- 흐름: `tok_emb`(단어→벡터) + `pos_emb`(위치→벡터) → `+`(둘 다 `emb_dim`이라 더할 수 있음) → `drop_emb` → `trf_blocks(×n_layers)` → `final_norm` → `out_head`(logits).
- `__init__`은 cfg로 부품을 *생성/준비*만, 실제 변환은 `forward`. 임베딩 가중치는 랜덤 시작 → 학습으로 채워짐.
- **`n_layers` = 트랜스포머 블록 개수**(12). `n_heads`(블록 내 어텐션 갈래 수)와 **다른 축**(둘 다 우연히 12).
- **`tok_emb`·`out_head`가 유독 큰 이유:** 둘 다 **어휘 크기 50257**에 닿음(각 50257×768 ≈ 3,860만). 내부 블록은 768차원이라 상대적으로 작음. 원조 GPT-2는 두 층 **weight tying**(공유)으로 124M.

### 4.5 텍스트 생성 — `generate_text_simple`(greedy)

- `logits → softmax → argmax`로 다음 토큰 선택, 자기회귀로 반복.
- **softmax는 단조(순서 보존) 함수** → 최댓값의 *위치*가 로짓과 동일 → greedy에선 **softmax 생략하고 `argmax(logits)`로 같은 결과**(계산 절약). 단 top-k·temperature 샘플링은 확률값이 필요해 생략 불가(5장).
- 미학습 모델이라 생성 텍스트는 의미 없는 토큰 나열(정상).

### 4.6 파라미터 수·메모리

- `sum(p.numel() for p in model.parameters())`로 집계. `out_head` 제외(weight tying 가정) 시 124M.
- 메모리 ≈ `total_params × 4바이트 / 1024²` MB(float32 기준).
- 연습 4.1: GPT-2 4종(124M/Medium emb1024·L24 / Large emb1280·L36 / XL emb1600·L48) config로 크기별 파라미터 비교.

### 그레이디언트 복습 — backward/step·흐름 보존 (2026-06-23)

residual(4.3) 진입 전, `feed_forward.py`의 `ExampleDeepNeuralNetwork`(shortcut 유무 비교 + `print_gradients`)를 계기로 그레이디언트 전반을 복습.

- **그레이디언트 = 각 파라미터로 손실을 편미분한 값들의 벡터/행렬.** 연쇄법칙은 그 *계산법*이고, PyTorch는 수식이 아니라 **현재 값에서 평가된 숫자**를 역방향 자동미분으로 채운다. 의미는 "손실이 가장 빨리 *증가*하는 방향" → 학습은 그 **반대(−)** 로 이동.
- **`backward()` ≠ 업데이트.** `loss.backward()`는 grad를 **계산해 각 파라미터 `.grad`에 저장만** 한다. 실제 파라미터 변경은 `optimizer.step()`(`w ← w − lr·w.grad`). `zero_grad()`는 누적되는 grad 리셋용(필수). → grad≈0이면 `step()`이 안 움직임 = 학습 정지.
- **용어:** 가중치 `w`(곱) + 편향 `b`(더함) = **파라미터**(학습 대상 전부). 업데이트·grad 모두 `w`·`b` 양쪽에 적용.
- **grad의 모양 = 파라미터의 모양.** `W`가 3×3이면 `W.grad`도 3×3. 칸마다 따로 계산·따로 업데이트(정답 칸은 grad=0 → 안 움직임, 많이 틀린 칸은 grad 큼).
- **`param.grad.abs().mean().item()` 해석:** `.abs()`(부호 버리고 크기만, +/− 상쇄 방지) → `.mean()`(전 원소 평균=스칼라) → `.item()`(float). **되돌릴 수 없는 요약 압축**(평균에서 개별값·부호 복원 불가)이며 **출력·층 비교 전용**. 정확한 grad는 `param.grad`(전체 행렬)에 보존돼 학습에 그대로 쓰인다.
- **흐름 보존이 중요한 이유:** 역전파는 층별 미분의 **곱셈 사슬** → 항<1이면 소실(앞층 grad≈0, 학습 정지), 항>1이면 폭주. 4장 장치가 전부 해법 — **Residual**(`x+F(x)`→미분에 +1 지름길, 가장 결정적)·**GELU**(음수도 미분 살아있음)·**LayerNorm**(값을 1 근처로 유지). `print_gradients`로 shortcut=False면 앞층 평균 grad가 0에 수렴, True면 건강하게 유지됨을 숫자로 확인 가능.

## 검증

책 코드와 동일 입력 → 동일/유사 출력 재현. 로컬(2.2.2)·Colab(2.6.0) 양쪽에서 코드 동작.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../../30-References/pytorch-env-hybrid]] — 환경 정본
