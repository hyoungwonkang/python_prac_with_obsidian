"""
로컬용 GPT-2 가중치 어댑터 — TensorFlow 없이 HuggingFace(PyTorch)에서 GPT-2 가중치를
받아 교재 GPTModel 구조로 로드한다. gpt_download.py(TF 의존)의 로컬 대체용.

원리: HF GPT-2의 Conv1D 가중치 방향((in, out))이 gpt_download(TF) params 형식과 동일.
      → HF state_dict를 같은 params 딕셔너리로 재구성 → load_weights_into_gpt로 주입.

자급자족(장 독립): assign/load_weights_into_gpt를 내장해 previous_N 어디에도 의존하지 않음.
교재 GPTModel 구조(pos_emb·tok_emb·trf_blocks·final_norm·out_head)면 5~7장 어느 스크립트든 사용 가능.

사용:
    from hf_weight_adapter import load_hf_weights_into_gpt
    model = GPTModel(BASE_CONFIG)                # 어느 장의 GPTModel이든 OK
    load_hf_weights_into_gpt(model, "124M")      # "124M"/"355M"/"774M"/"1558M"
"""
import os
os.environ["USE_TF"] = "0"        # transformers가 TF/Flax 임포트 안 하도록 (로컬 크래시 예방)
os.environ["USE_FLAX"] = "0"

import numpy as np
import torch


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """params 딕셔너리를 교재 GPTModel 구조에 주입 (교재 5장 함수를 내장한 사본)."""
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

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
