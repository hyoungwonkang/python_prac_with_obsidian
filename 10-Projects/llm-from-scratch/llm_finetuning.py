from hf_weight_adapter import load_hf_weights_into_gpt   # TF 없이 HF에서 가중치 로드 (로컬 mutex 크래시 우회)
from previous_7 import GPTModel

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].strip("(").rstrip(")")

model = GPTModel(BASE_CONFIG)
load_hf_weights_into_gpt(model, model_size)   # "355M" → HF gpt2-medium (TF 불필요)
model.eval()