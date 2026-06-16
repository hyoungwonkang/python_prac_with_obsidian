# 역전파(backpropagation) 직관 정리

> 연쇄 법칙의 수학을 깊이 파지 않고, **PyTorch가 자동으로 풀어주는 기계장치**라는 추상화 수준에서 이해하는 것이 목적.

## 한 줄 정의

> **연쇄 법칙**은 loss라는 최종 숫자가 각 가중치의 변화에 얼마나 민감한지(= gradient)를, 계산 그래프를 거슬러 올라가며 정확히 계산해 준다.

이 한 줄만 알면 모델을 만들고 훈련하는 데 충분하다.

## 역방향이 필요한 이유

훈련의 목표는 **loss를 줄이는 방향으로 모든 가중치를 조금씩 돌리는 것**.
그러려면 각 가중치마다 이 질문에 답해야 한다:

> "내가 조금 돌아가면, loss가 얼마나 변하는가?" (= 그 가중치의 gradient)

이 정보가 없으면 옵티마이저는 어느 가중치를 어느 방향으로 얼마나 돌릴지 모른다.
**모든 파라미터에 대한 편미분값을 구하는 작업** = backward.

## 왜 이름이 "역방향"인가

데이터는 정방향으로 흐른다:

```
입력 x ─→ Linear1 ─→ ReLU ─→ Linear2 ─→ ReLU ─→ Linear3 ─→ 출력 ŷ ─→ Loss L
```

그런데 loss를 각 가중치로 미분할 때는 **출력 쪽에서부터 입력 쪽으로 거꾸로** 풀어내려가야 효율적이다. 그래서 이름이 backpropagation = 역전파.

```
정방향:   x ──────────────────────────→ L
역방향:   ∂L/∂x ←────────────────────── ∂L/∂L = 1
          (각 파라미터의 gradient가 도중에 떨어져 나옴)
```

## 회사 비유

- **정방향**: 신입 → 대리 → 과장 → 부장 → 결과물
- **역방향(책임 추궁)**: 결과물이 망함 → 부장 책임 N% → 과장 책임 M% → ... → 신입 책임 K%
- **옵티마이저**: 책임 비율에 따라 다음에는 각자 행동을 조금씩 조정

## 왜 굳이 역방향이 효율적인가

| | 비용 |
|---|---|
| 순방향 모드 미분 | 파라미터 N개당 망을 N번 통과 → ∝ N |
| 역방향 모드 미분 | **단 한 번의 backward 패스로 모든 파라미터 gradient 동시에** |

신경망은 파라미터가 수백만~수십억 개이고 출력(loss)은 보통 스칼라 1개.
출력에서 시작해 거꾸로 내려가는 reverse-mode가 압도적으로 빠르다.
이래서 backpropagation이 딥러닝의 표준이 됨.

## 사용자가 실제로 다루는 인터페이스

```python
y_hat = model(x)              # forward — 데이터 정방향 흐름
                              # autograd가 computation graph 자동 기록
loss = loss_fn(y_hat, y)      # 스칼라 loss 계산
loss.backward()               # ← 역전파 발동
                              #   computation graph를 거꾸로 타고 가며
                              #   모든 param.grad에 ∂L/∂param 채워 넣음
optimizer.step()              # 채워진 grad를 사용해 param ← param − lr·grad
optimizer.zero_grad()         # 다음 step 위해 grad 초기화
```

| 다루는 것 | 의미 |
|----------|------|
| `loss.backward()` | "PyTorch야, 모든 파라미터의 gradient를 계산해줘" |
| `param.grad` | 그 결과로 채워지는 숫자. "loss를 줄이려면 이 파라미터를 어느 방향·얼마만큼 돌려야 하는지의 힌트" |
| `optimizer.step()` | 그 힌트대로 파라미터를 실제로 돌림 |
| `optimizer.zero_grad()` | 다음 step 시작 전 힌트 메모지를 깨끗이 지움 |

- `requires_grad=True`로 만들어둔 텐서만 backward의 영향을 받음 → 이래서 "훈련 가능한 파라미터"라고 부른다
- backward가 끝나면 `model.layers[0].weight.grad`에 weight와 같은 모양의 gradient 텐서가 채워진다

## 정방향 vs 역방향 역할 분리

| | 정방향(forward) | 역방향(backward) |
|---|---|---|
| 흐르는 것 | 입력 데이터 | 오차에 대한 책임(gradient) |
| 방향 | 입력 → 출력 | 출력(loss) → 입력 |
| 목적 | "지금 모델이 뭐라고 예측하는가?" | "예측이 틀린 만큼 각 가중치를 어느 방향으로 돌려야 하는가?" |
| 무엇이 채워지는가 | activation 값들 | `.grad` 속성들 |

정방향이 만들어둔 자취가 곧 역방향이 거슬러 올라갈 길이다.

## 언제 연쇄 법칙 수학을 깊이 봐야 하는가

지금은 위 추상화로 충분하다. 다음과 같은 상황에서 deeper dive가 필요해진다:

- **gradient가 사라지거나 폭발하는 문제** (vanishing/exploding gradient) 디버깅
- 커스텀 autograd 함수(`torch.autograd.Function`) 구현
- 새로운 옵티마이저나 정규화 기법 설계

그 전에는 "PyTorch가 알아서 계산해주는 기계장치" 추상화로 모델을 만들고 훈련하는 데 막힘이 없다.
