import json
import os
import urllib.request

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r") as file:
        data = json.load(file)
    return data

# 데이터 선택: "book" = 교재 1,100건 / "alpaca" = Stanford Alpaca 52k (연습문제 7.3)
# alpaca는 MPS OOM으로 종결 — 실패 기록: llm-ch7-failure-log.md (버킷 패딩 미검증 가설로 남김)
DATASET = "book"

if DATASET == "alpaca":
    file_path = "alpaca_data.json"
    url = "https://mng.bz/NBnE"   # → tatsu-lab/stanford_alpaca (301 리다이렉트, urllib이 따라감)
    SUBSET = 5000                 # 전체 52,002건은 2에폭 ~1.5시간 → 우선 5천 건으로
else:
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    SUBSET = None

data = download_and_load_file(file_path, url)
if SUBSET is not None:
    data = data[:SUBSET]
dataset_name = f"{DATASET}-{len(data)}건"
print("샘플 개수:", len(data), f"({dataset_name})")

print("샘플 예시:\n", data[50])
print("다른 샘플:\n", data[999])

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)

model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion+test_portion]
val_data = data[train_portion+test_portion:]

print("훈련 세트 크기:", len(train_data))
print("검증 세트 크기:", len(val_data))
print("테스트 세트 크기:", len(test_data))

import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

def custom_collate_draft_1(
        batch,
        pad_token_id=50256,
        device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor
    
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
    inputs_1,
    inputs_2,
    inputs_3
)
# print(custom_collate_draft_1(batch))

def custom_collate_draft_2(
        batch,
        pad_token_id=50256,
        device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_draft_2(batch)
print(inputs)
print(targets)

def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        bucket_multiple=None,
        device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    if bucket_multiple is not None and allowed_max_length is not None:
        # 배치 폭을 bucket_multiple의 배수로 올림 → 배치 모양 종류를 소수(예: 512/64=8종)로 제한.
        # MPS는 모양마다 연산 그래프를 영구 캐시해서, 동적 패딩의 다양한 모양이 OOM을 유발
        # (empty_cache로도 못 비움 — 'other allocations' 폭증으로 실측 확인)
        width = -(-(batch_max_length - 1) // bucket_multiple) * bucket_multiple
        batch_max_length = min(width, allowed_max_length) + 1
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)

logits_1 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5]]
)
targets_1 = torch.tensor([0,1])
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print(loss_1)

logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]
)
targets_2 = torch.tensor([0, 1, 1])
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print(loss_2)

targets_3 = torch.tensor([0, 1, -100])
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print(loss_3)
print("loss_1 == loss_3:", loss_1 == loss_3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
print("장치:", device)

from functools import partial

# Alpaca는 긴 샘플이 있어 1024 상한이면 MPS OOM (355M, 2026-07-02 실측) → 512로 천장 낮춤
allowed_max_length = 512 if DATASET == "alpaca" else 1024

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=allowed_max_length,
    bucket_multiple=64 if DATASET == "alpaca" else None
)

from torch.utils.data import DataLoader

num_workers = 0
# Alpaca: 512 천장으로도 355M MPS OOM (step 105) → 배치 절반으로 활성값 메모리 절감
batch_size = 4 if DATASET == "alpaca" else 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)


class MPSCacheClearingLoader:
    """MPS 캐시 할당자는 배치 모양(shape)마다 버퍼를 따로 쌓는다 — 가변 길이 배치(동적 패딩)가
    다양할수록 캐시가 무한 증식해 OOM (Alpaca에서 step ~110 사망, batch 반감해도 동일 → 누적 문제 확증).
    N배치마다 캐시를 비워 메모리를 평탄하게 유지. previous_7(교재 모듈)은 수정하지 않기 위한 래퍼."""
    def __init__(self, loader, every=20):
        self.loader = loader
        self.every = every

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            yield batch
            if (i + 1) % self.every == 0:
                torch.mps.empty_cache()

    def __len__(self):
        return len(self.loader)


if DATASET == "alpaca" and device.type == "mps":
    train_loader = MPSCacheClearingLoader(train_loader)
    val_loader = MPSCacheClearingLoader(val_loader)
    test_loader = MPSCacheClearingLoader(test_loader)

print("훈련 데이터 로더:")
# for inputs, targets in train_loader:
#     print(inputs.shape, targets.shape)