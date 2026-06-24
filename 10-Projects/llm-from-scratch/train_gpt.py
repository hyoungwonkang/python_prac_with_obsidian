"""5.2 LLM 사전훈련 루프 + MLflow 실험 추적 통합.

교재(Raschka 5.2)의 학습 함수에 MLflow 3동사를 붙였다:
  - log_params : GPT_CONFIG + 학습 하이퍼파라미터
  - log_metric : train_loss / val_loss (step별 → UI에서 곡선)
  - log_model  : 학습된 GPTModel

데이터셋 전환은 DATA_FILE 한 줄만 바꾸면 됨.
"""
import torch
import tiktoken
import mlflow

from previous_5 import GPTModel, create_dataloader_v1, generate_text_simple

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATA_FILE = "the-verdict.txt"      # ← tinyshakespeare.txt 로 바꾸면 더 큰 데이터

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,         # 랩톱 학습용 축소
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

TRAIN_SETTINGS = {
    "learning_rate": 4e-4,
    "weight_decay": 0.1,
    "num_epochs": 10,
    "batch_size": 2,
    "train_ratio": 0.90,
    "eval_freq": 5,
    "eval_iter": 5,
}


# ─────────────────────────────────────────────
# 토크나이저 헬퍼
# ─────────────────────────────────────────────
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


# ─────────────────────────────────────────────
# 5.2 손실 계산 (교재)
# ─────────────────────────────────────────────
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    print("  샘플:", token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()


# ─────────────────────────────────────────────
# 5.2 학습 루프 (교재) + MLflow log_metric 통합
# ─────────────────────────────────────────────
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"train {train_loss:.3f} | val {val_loss:.3f}")

                # ★ MLflow ② : step별 loss 기록 → UI에서 train/val 곡선
                mlflow.log_metric("train_loss", train_loss, step=global_step)
                mlflow.log_metric("val_loss", val_loss, step=global_step)

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens


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

    # 데이터 로드 + train/val 분할
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text_data = f.read()
    split = int(TRAIN_SETTINGS["train_ratio"] * len(text_data))
    train_data, val_data = text_data[:split], text_data[split:]

    train_loader = create_dataloader_v1(
        train_data, batch_size=TRAIN_SETTINGS["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True, shuffle=True, num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data, batch_size=TRAIN_SETTINGS["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False, shuffle=False, num_workers=0,
    )
    print(f"train 배치 수: {len(train_loader)} | val 배치 수: {len(val_loader)}")

    # 모델 + 옵티마이저
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_SETTINGS["learning_rate"],
        weight_decay=TRAIN_SETTINGS["weight_decay"],
    )

    # ★ MLflow : 추적 위치·실험 이름
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("gpt2-pretrain")

    with mlflow.start_run():                       # ★ 실험 1회 시작
        mlflow.log_params({**GPT_CONFIG_124M, **TRAIN_SETTINGS,  # ★ ① 설정 기록
                           "dataset": DATA_FILE, "device": device})

        train_losses, val_losses, _ = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=TRAIN_SETTINGS["num_epochs"],
            eval_freq=TRAIN_SETTINGS["eval_freq"],
            eval_iter=TRAIN_SETTINGS["eval_iter"],
            start_context="Every effort moves you",
            tokenizer=tiktoken.get_encoding("gpt2"),
        )

        mlflow.log_metric("final_train_loss", train_losses[-1])  # 최종 요약값
        mlflow.log_metric("final_val_loss", val_losses[-1])
        mlflow.pytorch.log_model(model, name="gpt_model")        # ★ ③ 모델 저장

    print("\n완료. mlflow ui 로 결과 확인 가능.")


if __name__ == "__main__":
    main()
