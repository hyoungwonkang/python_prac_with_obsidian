import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)   # nn.Linear는 호출해서 사용
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens] == 1, float("-inf"))
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vecs = attn_weights @ values
        return context_vecs


class MultiheadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out은 num_heads로 나누어 떨어져야 합니다"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)   # nn.Linear는 호출해서 사용
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        context_vec = self.out_proj(context_vec)
        return context_vec


if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your    (x^1)
         [0.55, 0.87, 0.66],  # journey (x^2)
         [0.57, 0.85, 0.64],  # starts  (x^3)
         [0.22, 0.58, 0.33],  # with    (x^4)
         [0.77, 0.25, 0.10],  # one     (x^5)
         [0.05, 0.80, 0.55]]  # step    (x^6)
    )

    # 같은 문장을 2번 복사해 배치 차원(2)을 만듦 → shape [2, 6, 3]
    batch = torch.stack((inputs, inputs), dim=0)
    print("batch.shape:", batch.shape)

    # --- 기본 예제 (d_in=3, d_out=2, num_heads=2) ---
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiheadAttention(
        d_in, d_out, context_length, 0.0, num_heads=2
    )
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    # --- 연습 3.3: GPT-2 small 사양 멀티헤드 어텐션 ---
    # 입력·출력 임베딩 768, 헤드 12개, 문맥 길이 1,024
    torch.manual_seed(123)
    gpt2_mha = MultiheadAttention(
        d_in=768, d_out=768, context_length=1024, dropout=0.0, num_heads=12
    )
    n_params = sum(p.numel() for p in gpt2_mha.parameters())
    x = torch.rand(1, 1024, 768)  # (배치1, 토큰1024, 임베딩768)
    out = gpt2_mha(x)
    print("\n[GPT-2 small 사양]")
    print("head_dim:", gpt2_mha.head_dim)       # 768 / 12 = 64
    print("파라미터 수:", f"{n_params:,}")
    print("출력 shape:", out.shape)              # [1, 1024, 768]
