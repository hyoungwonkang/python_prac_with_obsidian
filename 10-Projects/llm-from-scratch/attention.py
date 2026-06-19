import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your    (x^1)
     [0.55, 0.87, 0.66],  # journey (x^2)
     [0.57, 0.85, 0.64],  # starts  (x^3)
     [0.22, 0.58, 0.33],  # with    (x^4)
     [0.77, 0.25, 0.10],  # one     (x^5)
     [0.05, 0.80, 0.55]]  # step    (x^6)
)

# --- 1) 한 토큰(journey = x^2)에 대한 문맥 벡터 z^2 ---
query = inputs[1]  # journey 를 쿼리로 사용

# 1단계: 쿼리와 모든 입력의 내적(dot product) → 어텐션 점수 ω
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print("어텐션 점수 ω (journey):\n", attn_scores_2)

# 2단계: softmax 정규화 → 어텐션 가중치 α (합 = 1)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("어텐션 가중치 α (journey):\n", attn_weights_2)
print("가중치 합:", attn_weights_2.sum())

# 3단계: 가중치로 입력 벡터들을 가중합 → 문맥 벡터 z^2
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("문맥 벡터 z^2 (journey):\n", context_vec_2)

# --- 2) 모든 토큰에 대한 문맥 벡터 (행렬 연산으로 한 번에) ---
attn_scores = inputs @ inputs.T          # (6,6) 모든 쌍의 내적
attn_weights = torch.softmax(attn_scores, dim=-1)  # 행별 softmax
all_context_vecs = attn_weights @ inputs           # (6,3) 모든 문맥 벡터
print("\n모든 어텐션 가중치:\n", attn_weights)
print("\n모든 문맥 벡터:\n", all_context_vecs)

# 검증: 행렬 연산 결과의 2번째 행이 위 루프의 z^2와 같아야 함
print("\nz^2 일치 확인:", torch.allclose(all_context_vecs[1], context_vec_2))


d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

keys = inputs @ W_key
query_2 = inputs[1] @ W_query   # journey 도 W_query 로 2차원 투영해야 keys 와 내적 가능

attn_scores_2 = query_2 @ keys.T

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / (d_k ** 0.5), dim=-1)
print(attn_weights_2)