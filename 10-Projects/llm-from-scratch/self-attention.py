import torch
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your    (x^1)
     [0.55, 0.87, 0.66],  # journey (x^2)
     [0.57, 0.85, 0.64],  # starts  (x^3)
     [0.22, 0.58, 0.33],  # with    (x^4)
     [0.77, 0.25, 0.10],  # one     (x^5)
     [0.05, 0.80, 0.55]]  # step    (x^6)
)

d_in = inputs.shape[-1]
d_out = 2

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, inputs):
        queries = inputs @ self.W_query
        keys = inputs @ self.W_key
        values = inputs @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        context_vecs = attn_weights @ values
        return context_vecs
    


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, inputs):
        queries = self.W_query(inputs)   # nn.Linear는 호출해서 사용
        keys = self.W_key(inputs)
        values = self.W_value(inputs)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        context_vecs = attn_weights @ values
        return context_vecs
    

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
sa_v1 = SelfAttention_v1(d_in, d_out)

sa_v1.W_query.data = sa_v2.W_query.weight.T
sa_v1.W_key.data = sa_v2.W_key.weight.T
sa_v1.W_value.data = sa_v2.W_value.weight.T

print(sa_v2(inputs))
print(sa_v1(inputs))
print(torch.allclose(sa_v2(inputs), sa_v1(inputs)))