import json

import torch

from previous_7 import GPTModel, generate, text_to_token_ids, token_ids_to_text
from instruction_dataset_finetune import format_input, test_data, tokenizer, device

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 1024, "n_layers": 24, "n_heads": 16,   # gpt2-medium (355M)
}

model = GPTModel(BASE_CONFIG)                          # ① 뼈대 세우고
model.load_state_dict(                                 # ② 파인튜닝된 눈금 채우기
    torch.load("gpt2-medium355M-sft.pth", map_location=device)
)
model.to(device)
model.eval()

torch.manual_seed(123)

for entry in test_data[:3]:
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    print(input_text)
    print(f"\n올바른 응답:\n>> {entry['output']}")
    print(f"\n모델 응답:\n>> {response_text}")
    print("------------------------------------")

from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text

with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)

print(test_data[0])

# 교재 7.7 말미의 모델 저장 코드(re.sub … -sft.pth)는 노트북 기준 — 파일 분리 구조에서는
# 모델이 살아 있는 llm_finetuning.py(7.6) 끝에서 수행. 이 파일은 로드만 하므로 재저장 불필요.
# (교재 셀 102의 주석 `# model.load_state_dict(torch.load(...))`가 곧 이 파일 상단의 로드 코드)