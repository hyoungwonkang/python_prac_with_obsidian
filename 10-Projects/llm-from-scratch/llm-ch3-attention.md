# llm-ch3-attention

[[llm-from-scratch]] 마스터 플랜의 **Phase 3 / 3장** 정본 노트.
교재: Sebastian Raschka, *밑바닥부터 만들면서 배우는 LLM* — 3장. 어텐션 메커니즘 구현하기.

## 개요

단순 self-attention에서 시작해 학습 가능한 Q/K/V, causal 마스킹, multi-head까지 단계적으로 구현하는 핵심 장.

## 체크리스트

- [x] 3.1 단순 self-attention (가중치 학습 없이 dot-product) — 개념 정리 + `attention.py` 실습 완료
- [ ] 3.2 학습 가능한 가중치(Q, K, V) 도입
- [ ] 3.3 scaled dot-product attention
- [ ] 3.4 causal(마스킹) attention
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

## 검증

책 코드와 동일 입력 → 동일/유사 출력 재현. 로컬(2.2.2)·Colab(2.6.0) 양쪽에서 코드 동작.

## 관련 노트

- [[llm-from-scratch]] — 마스터 플랜 (인덱스)
- [[../../30-References/pytorch-env-hybrid]] — 환경 정본
