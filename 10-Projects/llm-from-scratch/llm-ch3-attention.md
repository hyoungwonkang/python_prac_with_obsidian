# llm-ch3-attention

[[llm-from-scratch]] 마스터 플랜의 **Phase 3 / 3장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 3장. 어텐션 메커니즘 구현하기.

## 개요

단순 self-attention에서 시작해 학습 가능한 Q/K/V, causal 마스킹, multi-head까지 단계적으로 구현하는 핵심 장.

## 체크리스트

- [x] 3.1 단순 self-attention (가중치 학습 없이 dot-product) — 개념 정리 + `attention.py` 실습 완료
- [x] 3.2 학습 가능한 가중치(Q, K, V) 도입 — `attention.py`(절차형) + `self-attention.py`(v1/v2 클래스)
- [x] 3.3 scaled dot-product attention — √d_k 스케일링 적용
- [x] 3.4 causal(마스킹) attention — `causal-attention.py`, 마스킹+dropout+배치
- [ ] 3.5 multi-head attention 구현
- [ ] 3.6 텐서 shape 변화를 손으로 추적해 노트로 정리

## 학습 정리

### 3.1 단순 self-attention — 입력 임베딩과 문맥 벡터

예제 텍스트: `"Your journey starts with one step"` (6 토큰), 각 토큰을 3차원으로 임베딩.

```python
inputs = torch.tensor(
  [[0.43, 0.15, 0.89],  # Your    (x^1)
   [0.55, 0.87, 0.66],  # journey (x^2)
   [0.57, 0.85, 0.64],  # starts  (x^3)
   [0.22, 0.58, 0.33],  # with    (x^4)
   [0.77, 0.25, 0.10],  # one     (x^5)
   [0.05, 0.80, 0.55]]  # step    (x^6)
)
```

- **이 벡터값은 랜덤이 아니라, 저자가 계산 과정을 보여주려 박아둔 고정 예시값.** 어느 PC에서도 같은 결과가 나오도록 고정. 3차원인 것도 손계산 가능하게 한 교육용 축소판.
- **개념 구분**: 실제 모델의 임베딩은 *초기엔 랜덤 초기화 → 역전파로 학습되는 파라미터*. 시작점만 랜덤이고 결과는 학습된 의미값. 이 예제는 "학습된 임베딩이 이미 있다고 치고" 어텐션 계산만 떼어 보여줌.

**계산 단계 (쿼리 = x², `journey` 기준)**
1. 쿼리와 모든 입력의 내적(dot product) → 어텐션 점수 ω
2. softmax 정규화 → 어텐션 가중치 α (합 = 1)
3. 가중치로 입력 벡터들을 가중합 → 문맥 벡터 z²

**"간소화(simplified)"인 이유**: 학습 가능한 가중치 행렬(W_q, W_k, W_v)이 없음. 입력끼리 바로 내적. 다음 단계(3.2~3.3)에서 W_q/W_k/W_v + 스케일링을 도입한 것이 진짜 트랜스포머의 scaled dot-product attention.

### 3.2 학습 가능한 가중치 W_q / W_k / W_v

- 입력 `xᵢ`에 세 행렬을 곱해 **query/key/value**로 변환: `q=x@W_q, k=x@W_k, v=x@W_v`.
- `W`의 모양은 `(d_in, d_out)` = (3, 2). 행렬곱으로 가운데 차원이 소거돼 **3차원 입력 → 2차원 q/k/v**. 출력 차원은 `W`의 열 수가 결정 (입력 차원과 독립).
- **가중치 행렬 W ≠ 어텐션 가중치 α**: `W`는 학습으로 고정되는 변환 규칙(모델 지식), `α`는 문장마다 새로 계산되는 주목 비율(softmax 결과, 합=1).
- 흐름: `W로 q,k,v 생성 → q·k = 어텐션 점수 ω → softmax = 가중치 α → α로 v 가중합 = 문맥 벡터`.
- 구현 2가지: **v1** `nn.Parameter`(직접 `inputs @ W`), **v2** `nn.Linear`(반드시 **호출** `W_query(inputs)`, 내부 저장은 `(d_out, d_in)`이라 v1로 복사 시 `.weight.T` 전치 필요).

### 3.3 scaled dot-product attention

- `softmax(점수 / √d_k)`. `d_k`=키 차원(=d_out, 여기선 2), **입력 임베딩 차원(3)이 아님**.
- 이유: 차원이 크면 내적값이 커져 softmax가 극단으로 쏠리고 기울기가 죽음 → √d_k로 눌러 학습 안정화.
- `/√d_k`는 **스케일링**, `softmax`는 **정규화(합=1)**. 둘은 별개 단계.

### 3.4 causal(마스킹) attention — 정보 누수 방지

- 미래 토큰을 못 보게 마스킹. 두 방법은 수학적으로 **동일 결과**:
  - 방법1: softmax → 미래를 0으로 → 행 합으로 재정규화. (분모에 미래가 잠깐 들어가도 **약분으로 제거**되어 누수 없음)
  - 방법2(권장): softmax **전** 미래 점수를 `-inf`로 → `exp(-inf)=0` → 재정규화 불필요.
- `causal-attention.py`: `torch.triu(diagonal=1)` 상삼각 마스크를 `register_buffer`로, `masked_fill_(-inf)`, `nn.Dropout`, 배치 입력 `(b, num_tokens, d_in)` 지원 (`keys.transpose(1,2)`).

## 검증

책 코드와 동일 입력 → 동일/유사 출력 재현. 로컬(2.2.2)·Colab(2.6.0) 양쪽에서 코드 동작.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../../30-References/pytorch-env-hybrid]] — 환경 정본
