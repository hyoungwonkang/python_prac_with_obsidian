import psutil
import json
from tqdm import tqdm

from instruction_dataset_finetune import format_input   # 7.7과 동일한 프롬프트 포맷 재사용

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")

if not ollama_running:
    raise RuntimeError(
        "Ollama가 실행 중이 아닙니다. 먼저 Ollama를 실행하세요."
        )
print("Ollama 실행:", ollama_running)

import urllib.request

def query_model(
        prompt,
        model="llama3",
        url="http://localhost:11434/api/chat"
):
    data = {
        "model":model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )

    request.add_header("Content-Type", "application/json")

    response_data=""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data

model = "llama3"
result = query_model("What do Llamas eat?", model)
print(result)

with open("instruction-data-with-response.json", "r") as file:   # 7.7 save_response.py 산출물
    test_data = json.load(file)

for entry in test_data[:3]:
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}` "
        f"score the model response `{entry['model_response']}` "
        f"on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\n데이터셋 응답:")
    print(">>", entry['output'])
    print("\n모델 응답:")
    print(">>", entry['model_response'])
    print("\n점수:")
    print(">>", query_model(prompt))
    print("---------------------------")

def generate_model_scores(json_data, json_key, model="llama3"):
    scores=[]
    for entry in tqdm(json_data, desc="평가 항목"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}` "
            f"score the model response `{entry[json_key]}` "
            f"on a scale from 0 to 100, where 100 is the best score. "
            f"Response with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"점수로 변환할 수 없습니다: {score}")
            continue

    return scores

scores = generate_model_scores(test_data, "model_response")
print(f"평가 횟수: {len(test_data)}개 중 {len(scores)}개")
print(f"평가 점수: {sum(scores)/len(scores):.2f}\n")