"""5장 학습 루프 + MLflow 실험 추적 (로컬, from-scratch).

TF 불필요 → 로컬(M4 Max)에서 동작. mlruns는 실행 폴더에 영구 저장.
레퍼런스 패턴: ../../30-References/mlflow-practice/mlflow_quickstart.py

실행:  cd 10-Projects/llm-from-scratch && ~/ml-env/bin/python train_with_mlflow.py
확인:  같은 폴더에서  ~/ml-env/bin/mlflow ui   →  localhost:5000
값을 바꿔(아래 GPT_CONFIG/TRAIN) 여러 번 실행하면 run들이 쌓여 비교 가능.
"""
import torch
import tiktoken
import mlflow

from previous_5 import GPTModel, create_dataloader_v1, generate_text_simple

# ─────────────────────────────────────────────
# 설정 — 여기 값을 조절해 실험 (전부 MLflow log_params로 기록됨)
# ─────────────────────────────────────────────
DATA_FILE = "tinyshakespeare.txt"      # tinyshakespeare.txt 로 바꿔 비교 가능

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

TRAIN = {
    "learning_rate": 4e-4,
    "weight_decay": 0.1,
    "num_epochs": 3,
    "batch_size": 8,
    "train_ratio": 0.90,
    "eval_freq": 50,
    "eval_iter": 5,
}


# ─────────────────────────────────────────────
# 헬퍼 / 손실 (교재 5장)
# ─────────────────────────────────────────────
def text_to_token_ids(text, tokenizer):
    return torch.tensor(
        tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    ).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


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
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


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
                # ★ MLflow: step별 train/val loss → UI 곡선
                mlflow.log_metric("train_loss", tr, step=global_step)
                mlflow.log_metric("val_loss", vl, step=global_step)
    return train_losses, val_losses


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("device:", device, "| dataset:", DATA_FILE)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text_data = f.read()
    split = int(TRAIN["train_ratio"] * len(text_data))
    train_data, val_data = text_data[:split], text_data[split:]

    train_loader = create_dataloader_v1(
        train_data, batch_size=TRAIN["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True, shuffle=True, num_workers=0)
    val_loader = create_dataloader_v1(
        val_data, batch_size=TRAIN["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False, shuffle=False, num_workers=0)
    print(f"train 배치 {len(train_loader)} | val 배치 {len(val_loader)}")

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN["learning_rate"], weight_decay=TRAIN["weight_decay"])

    # ── MLflow ── (로컬 mlruns에 영구 저장)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("gpt2-from-scratch")
    with mlflow.start_run():
        mlflow.log_params({**GPT_CONFIG_124M, **TRAIN,
                           "dataset": DATA_FILE, "device": device})  # ① 조절값

        train_losses, val_losses = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=TRAIN["num_epochs"],
            eval_freq=TRAIN["eval_freq"], eval_iter=TRAIN["eval_iter"])  # ② train/val loss

        mlflow.log_metric("final_train_loss", train_losses[-1])
        mlflow.log_metric("final_val_loss", val_losses[-1])
        mlflow.pytorch.log_model(model, name="gpt2_model")             # ③ 모델

    # 학습 후 샘플 생성 (확인용)
    tokenizer = tiktoken.get_encoding("gpt2")
    model.eval()
    ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"])
    print("생성:", token_ids_to_text(ids.to("cpu"), tokenizer))
    print("\n완료. `mlflow ui`(같은 폴더)로 train/val 곡선·params 확인.")


if __name__ == "__main__":
    main()
