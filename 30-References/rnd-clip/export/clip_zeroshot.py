"""
CLIP zero-shot 상황 판단 — 'CLIP 이미지 상황 판단' R&D 1단계.

핵심 체험: **학습 0줄로 이미지 분류가 된다.** 클래스가 가중치에 박혀 있지 않고
텍스트 프롬프트로 주어진다 — 후보 문장을 바꾸면 그 자리에서 분류 체계가 바뀐다.
(BERT 문장 분류기: 클래스 수가 머리에 고정, 바꾸려면 재학습 — 과의 결정적 차이)

동작: 이미지 1장 + 상황 후보 문장 N개 → CLIP이 이미지·문장을 같은 좌표계에 놓고
      가장 가까운 문장을 고름 (유사도 → softmax 확률).

사용:
  ~/rnd-env/bin/python clip_zeroshot.py                    # coco128 샘플 N장 데모
  N=12 ~/rnd-env/bin/python clip_zeroshot.py               # 샘플 수 변경
  IMAGES=a.jpg,b.jpg ~/rnd-env/bin/python clip_zeroshot.py # 원하는 이미지로
  PROMPTS="...,..." 로 상황 후보 교체 (재학습 없이 분류 체계 교체 — 이게 핵심)
  MLFLOW_URI=http://127.0.0.1:5000 을 붙이면 통합 서버에 기록
"""
import os
import time
from pathlib import Path

import mlflow
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

HERE = Path(__file__).resolve().parent
BASE = os.environ.get("BASE", "openai/clip-vit-base-patch32")  # 영어 학습 모델 — 한국어는 KoCLIP 단계
N = int(os.environ.get("N", 8))
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))  # 맥=MPS / 윈도우 GPU=CUDA / 그 외=CPU

# 상황 후보 (업무 시나리오: 탐지 대상 1개 + 일상 상황들). '|' 구분, 표시명=프롬프트 앞부분
PROMPTS = [p.strip() for p in os.environ.get(
    "PROMPTS",
    "a screenshot of an online gambling or casino website|"
    "a photo of people in daily life|"
    "a photo of food on a table|"
    "a photo of vehicles on a road|"
    "a photo of animals|"
    "a document or sign with text").split("|")]


def pick_images():
    env = os.environ.get("IMAGES")
    if env:
        return [Path(p.strip()) for p in env.split(",")]
    pool = sorted((HERE / "datasets/coco128/images/train2017").glob("*.jpg"))
    step = max(1, len(pool) // N)          # 골고루 N장 샘플
    return pool[::step][:N]


def main():
    paths = pick_images()
    print(f"모델: {BASE} / 이미지 {len(paths)}장 / 상황 후보 {len(PROMPTS)}개 ({DEVICE})")

    t0 = time.time()
    model = CLIPModel.from_pretrained(BASE).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(BASE)
    print(f"로드 {time.time()-t0:.1f}초 — 파라미터 {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(text=PROMPTS, images=images, return_tensors="pt",
                       padding=True).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image     # [이미지 수 × 후보 수] 유사도
        probs = logits.softmax(dim=-1)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_URI", f"sqlite:///{HERE / 'mlflow.db'}"))
    mlflow.set_experiment("CLIP-제로샷")
    lines = []
    with mlflow.start_run(run_name=f"zeroshot-{len(paths)}장"):
        mlflow.log_params({"베이스모델": BASE, "이미지수": len(paths),
                           "후보수": len(PROMPTS), "후보": " | ".join(PROMPTS)})
        for path, p in zip(paths, probs):
            top = p.argsort(descending=True)[:2]
            line = (f"{path.name}: [{PROMPTS[top[0]][:40]}] {p[top[0]]:.1%}"
                    f"  (2위: {PROMPTS[top[1]][:30]} {p[top[1]]:.1%})")
            print("  " + line)
            lines.append(line)
        mlflow.log_text("\n".join(lines), "predictions.txt")
    print(f"완료 ({time.time()-t0:.0f}초) — 프롬프트를 바꾸면 재학습 없이 분류 체계가 바뀝니다")


if __name__ == "__main__":
    main()
