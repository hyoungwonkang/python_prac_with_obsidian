import torch

from hf_weight_adapter import load_hf_weights_into_gpt   # TF 없이 HF에서 가중치 로드 (로컬 mutex 크래시 우회)
from previous_7 import GPTModel
# 주의: 아래 import는 instruction_dataset_finetune.py 본문 전체를 실행함 (print·로더 확인 루프 포함)
from instruction_dataset_finetune import format_input, val_data, tokenizer

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

torch.manual_seed(123)
input_text = format_input(val_data[0])
print(input_text)

from previous_7 import generate, text_to_token_ids, token_ids_to_text

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)

response_text = generated_text[len(input_text):].strip()
print(response_text)