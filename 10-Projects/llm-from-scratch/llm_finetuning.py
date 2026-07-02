import torch

from hf_weight_adapter import load_hf_weights_into_gpt   # TF 없이 HF에서 가중치 로드 (로컬 mutex 크래시 우회)
from previous_7 import GPTModel, calc_loss_loader, train_model_simple, plot_losses
# 주의: 아래 import는 instruction_dataset_finetune.py 본문 전체를 실행함 (print·로더 확인 루프 포함)
from instruction_dataset_finetune import (
    format_input, val_data, tokenizer, device, train_loader, val_loader
)

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

model.to(device)
torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
    )

print("훈련 손실:", train_loss)
print("검증 손실:", val_loss)

import time
import mlflow

mlflow.set_tracking_uri("file:./mlruns")            # 5·6장과 같은 저장소 (cwd 주의)
mlflow.set_experiment("gpt2-instruct-finetune")

start_time = time.time()
torch.manual_seed(123)
lr, weight_decay = 0.00005, 0.1
optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, weight_decay=weight_decay
)
num_epochs = 2
eval_freq, eval_iter = 5, 5

with mlflow.start_run(run_name="355M-no-masking"):   # 연습문제 7.2에서 masking run과 비교 예정
    mlflow.log_params({
        **BASE_CONFIG, "model": CHOOSE_MODEL,
        "lr": lr, "weight_decay": weight_decay, "num_epochs": num_epochs,
        "batch_size": 8, "eval_freq": eval_freq, "eval_iter": eval_iter,
        "train_size": 935, "val_size": 55,
        "masking": "none",                            # 지시 마스킹 미적용 (baseline)
        "device": str(device),
    })

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"훈련 소요 시간: {execution_time_minutes:.2f}분")

    # train_model_simple이 돌려준 loss 곡선을 step 붙여 기록 (previous_7은 수정하지 않음)
    for i, (tr, vl, ts) in enumerate(zip(train_losses, val_losses, tokens_seen)):
        step = i * eval_freq                          # 평가 시점의 global step
        mlflow.log_metric("train_loss", tr, step=step)
        mlflow.log_metric("val_loss", vl, step=step)
        mlflow.log_metric("tokens_seen", ts, step=step)

    mlflow.log_metric("final_train_loss", train_losses[-1])
    mlflow.log_metric("final_val_loss", val_losses[-1])
    mlflow.log_metric("train_minutes", execution_time_minutes)
    # log_model은 생략 — 355M은 run당 ~1.4GB (finetune_gpt2_mlflow.py의 LOG_MODEL=False와 같은 이유)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)