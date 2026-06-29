"""
로컬용 GPT-2 가중치 어댑터 — TensorFlow 없이 HuggingFace(PyTorch)에서 GPT-2 가중치를
받아 교재 GPTModel 구조로 로드한다. gpt_download.py(TF 의존)의 로컬 대체용.

원리: HF GPT-2의 Conv1D 가중치 방향((in, out))이 gpt_download(TF) params 형식과 동일.
      → HF state_dict를 같은 params 딕셔너리로 재구성 → 기존 load_weights_into_gpt 재사용.

사용:
    from previous_6 import GPTModel
    from hf_weight_adapter import load_hf_weights_into_gpt
    model = GPTModel(BASE_CONFIG)
    load_hf_weights_into_gpt(model, "124M")   # gpt2-small
"""
import os
os.environ["USE_TF"] = "0"        # transformers가 TF/Flax 임포트 안 하도록 (로컬 크래시 예방)
os.environ["USE_FLAX"] = "0"

from previous_6 import load_weights_into_gpt

# 교재 사이즈명 → HuggingFace 허브명
HF_NAMES = {
    "124M": "gpt2",
    "355M": "gpt2-medium",
    "774M": "gpt2-large",
    "1558M": "gpt2-xl",
}


def load_hf_weights_into_gpt(gpt, model_size="124M"):
    """HF에서 GPT-2 가중치를 받아 교재 GPTModel(gpt)에 로드 (TF 불필요)."""
    from transformers import GPT2LMHeadModel, logging
    logging.set_verbosity_error()

    hf_name = HF_NAMES.get(model_size, model_size)
    hf = GPT2LMHeadModel.from_pretrained(hf_name)
    sd = hf.state_dict()

    def g(key):                       # state_dict 텐서 → numpy (grad 분리)
        return sd[key].detach().cpu().numpy()

    n_layers = hf.config.n_layer
    params = {
        "wpe": g("transformer.wpe.weight"),
        "wte": g("transformer.wte.weight"),
        "g": g("transformer.ln_f.weight"),
        "b": g("transformer.ln_f.bias"),
        "blocks": [],
    }
    for b in range(n_layers):
        p = f"transformer.h.{b}."
        params["blocks"].append({
            "attn": {
                "c_attn": {"w": g(p + "attn.c_attn.weight"), "b": g(p + "attn.c_attn.bias")},
                "c_proj": {"w": g(p + "attn.c_proj.weight"), "b": g(p + "attn.c_proj.bias")},
            },
            "mlp": {
                "c_fc":   {"w": g(p + "mlp.c_fc.weight"),   "b": g(p + "mlp.c_fc.bias")},
                "c_proj": {"w": g(p + "mlp.c_proj.weight"), "b": g(p + "mlp.c_proj.bias")},
            },
            "ln_1": {"g": g(p + "ln_1.weight"), "b": g(p + "ln_1.bias")},
            "ln_2": {"g": g(p + "ln_2.weight"), "b": g(p + "ln_2.bias")},
        })

    load_weights_into_gpt(gpt, params)   # 기존 함수 재사용
    return gpt
