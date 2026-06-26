"""Colab: GPT-2 124M 파인튜닝 + MLflow 추적 (mlruns → 개인 Google Drive).

weight_load.py(가중치 로드) + train_with_mlflow.py(학습+MLflow)를 통합.
가중치 로드에 TF 필요 → **Colab 전용**. mlruns는 Drive에 영구 저장된다.

Colab 실행 순서:
  from google.colab import drive; drive.mount('/content/drive')   # ① Drive 마운트
  !pip install mlflow tiktoken -q                                  # ② 패키지
  # previous_5.py 를 cwd(/content)에 두기 (repo clone 또는 업로드)
  !python finetune_gpt2_mlflow.py                                  # ③ 실행
UI: Drive의 mlruns 를 로컬로 받아 `mlflow ui` 하거나 Colab 포트 연결.
"""
import os
# 최신 MLflow는 파일 저장소(file:./mlruns)를 기본 거부 → 허용 (DB 백엔드 대신 파일 사용)
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import urllib.request

import numpy as np
import torch
import tiktoken
import mlflow

from previous_5 import GPTModel, create_dataloader_v1, generate_text_simple

# ─────────────────────────────────────────────
# 설정 — 값 바꿔가며 실험 (전부 log_params로 기록)
# ─────────────────────────────────────────────
TRACKING_URI = "file:/content/drive/MyDrive/mlruns"   # 개인 Drive(영구). 로컬 테스트면 "file:./mlruns"
EXPERIMENT = "gpt2-finetune"
DATA_FILE = "tinyshakespeare.txt"        # the-verdict.txt 로 바꿔 비교 가능
LOG_MODEL = False                        # 124M 모델 ~500MB/run → 비교 실험 땐 False, 보관 필요 시 True
MODEL_SIZE = "124M"

# GPT-2 124M 구조 (가중치 로드용): context 1024 · qkv_bias True 필수
BASE_CONFIG = {
    "vocab_size": 50257, "context_length": 1024,
    "emb_dim": 768, "n_heads": 12, "n_layers": 12,
    "drop_rate": 0.1, "qkv_bias": True,
}

TRAIN = {
    "learning_rate": 5e-5,   # 파인튜닝은 작은 lr (사전지식 보존)
    "weight_decay": 0.1,
    "num_epochs": 1,
    "batch_size": 8,
    "max_length": 256,       # 데이터 자르는 길이 (≤ context_length)
    "train_ratio": 0.90,
    "eval_freq": 50,
    "eval_iter": 5,
}

DATA_URLS = {
    "tinyshakespeare.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "the-verdict.txt": "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
}
GPT_DOWNLOAD_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch05/01_main-chapter-code/gpt_download.py"


# ─────────────────────────────────────────────
# 헬퍼 / 가중치 매핑 (weight_load.py)
# ─────────────────────────────────────────────
def text_to_token_ids(text, tokenizer):
    return torch.tensor(tokenizer.encode(text, allowed_special={"<|endoftext|>"})).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"크기가 다릅니다. left: {left.shape}, right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight   = assign(gpt.trf_blocks[b].att.W_key.weight,   k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias   = assign(gpt.trf_blocks[b].att.W_key.bias,   k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# ─────────────────────────────────────────────
# 손실 / 학습 (train_with_mlflow.py)
# ─────────────────────────────────────────────
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total = 0.0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (x, y) in enumerate(data_loader):
        if i >= num_batches:
            break
        total += calc_loss_batch(x, y, model, device).item()
    return total / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        tr = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        vl = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return tr, vl


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter):
    train_losses, val_losses = [], []
    global_step = -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % eval_freq == 0:
                tr, vl = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(tr)
                val_losses.append(vl)
                print(f"Ep {epoch+1} (step {global_step:06d}): train {tr:.3f} | val {vl:.3f}")
                mlflow.log_metric("train_loss", tr, step=global_step)
                mlflow.log_metric("val_loss", vl, step=global_step)
    return train_losses, val_losses


def ensure_file(path, url):
    if not os.path.exists(path):
        print("다운로드:", path)
        urllib.request.urlretrieve(url, path)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu")
    print("device:", device, "| dataset:", DATA_FILE)

    # 파일 준비 (gpt_download + 데이터)
    ensure_file("gpt_download.py", GPT_DOWNLOAD_URL)
    ensure_file(DATA_FILE, DATA_URLS[DATA_FILE])
    from gpt_download import download_and_load_gpt2   # TF import는 여기서 (Colab만)

    # GPT-2 가중치 로드 → 내 모델에 매핑
    settings, params = download_and_load_gpt2(model_size=MODEL_SIZE, models_dir="gpt2")
    gpt = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    print(f"GPT-2 {MODEL_SIZE} 가중치 로드 완료 → 파인튜닝 시작")

    # 데이터로더 (train/val)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text_data = f.read()
    split = int(TRAIN["train_ratio"] * len(text_data))
    train_loader = create_dataloader_v1(
        text_data[:split], batch_size=TRAIN["batch_size"],
        max_length=TRAIN["max_length"], stride=TRAIN["max_length"],
        drop_last=True, shuffle=True, num_workers=0)
    val_loader = create_dataloader_v1(
        text_data[split:], batch_size=TRAIN["batch_size"],
        max_length=TRAIN["max_length"], stride=TRAIN["max_length"],
        drop_last=False, shuffle=False, num_workers=0)
    print(f"train 배치 {len(train_loader)} | val 배치 {len(val_loader)}")

    optimizer = torch.optim.AdamW(
        gpt.parameters(), lr=TRAIN["learning_rate"], weight_decay=TRAIN["weight_decay"])

    # ── MLflow (mlruns → Drive) ──
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params({**BASE_CONFIG, **TRAIN, "dataset": DATA_FILE,
                           "device": str(device), "init": "gpt2-pretrained"})  # ① 조절값

        train_losses, val_losses = train_model_simple(
            gpt, train_loader, val_loader, optimizer, device,
            TRAIN["num_epochs"], TRAIN["eval_freq"], TRAIN["eval_iter"])        # ② train/val loss

        if train_losses:
            mlflow.log_metric("final_train_loss", train_losses[-1])
            mlflow.log_metric("final_val_loss", val_losses[-1])
        if LOG_MODEL:
            mlflow.pytorch.log_model(gpt, name="gpt2_finetuned")               # ③ 모델(옵션)

    # 파인튜닝 후 샘플 생성
    tokenizer = tiktoken.get_encoding("gpt2")
    gpt.eval()
    ids = generate_text_simple(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=30, context_size=BASE_CONFIG["context_length"])
    print("생성:", token_ids_to_text(ids.to("cpu"), tokenizer))
    print(f"\n완료. mlruns → {TRACKING_URI}")


if __name__ == "__main__":
    main()
