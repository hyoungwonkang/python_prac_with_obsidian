"""
CLIP linear probe — 지시 4 'CLIP 이미지 상황 판단' R&D 2단계 (로드맵상 유일한 파인튜닝).

과제: 사진에 사람이 있는가 (person / no-person) — coco128의 YOLO 라벨에서 이미지 단위 라벨 유도.
설계: BERT에서 배운 "몸통 재사용 + 머리만 학습"의 이미지판 —
  CLIP 이미지 인코더(151M)는 **얼리고**, 임베딩(512차원) 위의 선형 머리(512→2, 약 1천 눈금)만 학습.
비교: zero-shot(프롬프트 2문장) vs linear probe(라벨 89장 학습)를 **같은 고정 test셋**으로 채점
  (지시 2에서 확립한 동일 잣대 원칙).
산출물: artifacts/clip-person-probe/ 에 3종 세트 (head.pt + label_map.json + meta.json — 지시 1 규약).

사용:
  ~/rnd-env/bin/python clip_linear_probe.py
  EPOCHS=300 LR=0.01 ~/rnd-env/bin/python clip_linear_probe.py
  MLFLOW_URI=http://127.0.0.1:5000 을 붙이면 통합 서버에 기록
"""
import json
import os
import time
from pathlib import Path

import mlflow
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import CLIPModel, CLIPProcessor

HERE = Path(__file__).resolve().parent
BASE = os.environ.get("BASE", "openai/clip-vit-base-patch32")
EPOCHS = int(os.environ.get("EPOCHS", 200))
LR = float(os.environ.get("LR", 1e-2))
SEED = 123                                   # 기존 R&D와 동일 (분할 재현)
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))  # 맥=MPS / 윈도우 GPU=CUDA / 그 외=CPU
CLASSES = ["no-person", "person"]            # 0/1 — label_map에 기록
# 클래스 순서와 일치. 부정형("no people")은 CLIP의 약점 — ZS_PROMPTS로 교체 실험 가능
# (실측: 부정형 0.4444 → 긍정형 대비쌍 0.7778, 같은 test셋)
ZS_PROMPTS = [p.strip() for p in os.environ.get(
    "ZS_PROMPTS", "a photo with no people in it|a photo with people in it").split("|")]
OUT_DIR = HERE / "artifacts" / "clip-person-probe"


def load_labeled_images():
    """coco128 YOLO 라벨 → 이미지 단위 2클래스 (person=클래스 0 박스 존재 여부)."""
    root = HERE / "datasets/coco128"
    items = []
    for img in sorted((root / "images/train2017").glob("*.jpg")):
        txt = root / "labels/train2017" / (img.stem + ".txt")
        has_person = txt.exists() and any(
            line.split()[0] == "0" for line in txt.read_text().splitlines() if line.strip())
        items.append((img, int(has_person)))
    return items


def split(items):
    """70:10:20 (seed 고정) — test는 최종 채점 전용."""
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(items), generator=g).tolist()
    t_end = int(len(items) * 0.7)
    v_end = t_end + int(len(items) * 0.1)
    pick = lambda ids: [items[i] for i in ids]
    return pick(idx[:t_end]), pick(idx[t_end:v_end]), pick(idx[v_end:])


def report(name, golds, preds, params):
    tp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 1)
    metrics = {"정확도": accuracy_score(golds, preds),
               "정밀도": precision_score(golds, preds, zero_division=0),
               "재현율": recall_score(golds, preds, zero_division=0),
               "F1": f1_score(golds, preds, zero_division=0),
               "정탐tp": tp, "오탐fp": fp, "미탐fn": fn}
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    print(f"{name:<14} 정확도 {metrics['정확도']:.4f} | P {metrics['정밀도']:.4f} "
          f"R {metrics['재현율']:.4f} F1 {metrics['F1']:.4f} | 정탐 {tp} 오탐 {fp} 미탐 {fn}")


@torch.no_grad()
def embed(model, processor, paths, batch=32):
    """얼린 인코더로 이미지 → 512차원 임베딩 (L2 정규화)."""
    feats = []
    for i in range(0, len(paths), batch):
        images = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        # transformers 5.x: get_image_features가 출력 객체를 반환 — 512차원은 pooler_output
        f = model.get_image_features(**inputs).pooler_output
        feats.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(feats)


def main():
    items = load_labeled_images()
    train, val, test = split(items)
    print(f"데이터: {len(items)}장 (사람 {sum(l for _, l in items)}) → "
          f"train {len(train)} / val {len(val)} / test {len(test)}")

    model = CLIPModel.from_pretrained(BASE).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(BASE)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_URI", f"sqlite:///{HERE / 'mlflow.db'}"))
    mlflow.set_experiment("CLIP-사람판별-비교")
    test_paths = [p for p, _ in test]
    test_golds = [l for _, l in test]

    # ---- 방법 1: zero-shot (학습 0) — 같은 test셋 ----
    t0 = time.time()
    inputs = processor(text=ZS_PROMPTS, images=[Image.open(p).convert("RGB") for p in test_paths],
                       return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        zs_preds = model(**inputs).logits_per_image.argmax(-1).cpu().tolist()
    report("zero-shot", test_golds, zs_preds,
           {"방법": "프롬프트 2문장, 학습 0", "프롬프트": " | ".join(ZS_PROMPTS)})
    print(f"  ({time.time()-t0:.1f}초)")

    # ---- 방법 2: linear probe — 얼린 임베딩 + 선형 머리 학습 ----
    t0 = time.time()
    X_train, X_val, X_test = (embed(model, processor, [p for p, _ in part])
                              for part in (train, val, test))
    y_train = torch.tensor([l for _, l in train], device=DEVICE)
    y_val = [l for _, l in val]

    torch.manual_seed(SEED)
    head = torch.nn.Linear(X_train.shape[1], len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR)
    best_acc, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        head.train()
        loss = torch.nn.functional.cross_entropy(head(X_train), y_train)  # 전체 89장 = 1배치
        opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            val_acc = accuracy_score(y_val, head(X_val).argmax(-1).cpu().tolist())
        if val_acc > best_acc:                        # 검증 최고 시점의 머리를 보관
            best_acc, best_state = val_acc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    with torch.no_grad():
        lp_preds = head(X_test).argmax(-1).cpu().tolist()
    report("linear-probe", test_golds, lp_preds,
           {"방법": "얼린 임베딩 + 선형 머리", "학습장수": len(train),
            "에폭": EPOCHS, "학습률": LR, "검증최고": round(best_acc, 4)})
    print(f"  (임베딩+학습 {time.time()-t0:.1f}초 / 머리 파라미터 "
          f"{sum(p.numel() for p in head.parameters()):,}개 vs 몸통 {sum(p.numel() for p in model.parameters())/1e6:.0f}M)")

    # ---- 산출물 3종 세트 (지시 1 규약) ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), OUT_DIR / "head.pt")
    (OUT_DIR / "label_map.json").write_text(json.dumps(
        {"label2id": {c: i for i, c in enumerate(CLASSES)},
         "id2label": {str(i): c for i, c in enumerate(CLASSES)}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "meta.json").write_text(json.dumps(
        {"name": "clip-person-probe", "task": "image-binary-classification",
         "base_model": BASE, "head": "linear(512->2)", "seed": SEED,
         "data": "coco128 (person 박스 존재 여부로 유도)",
         "metrics": {"테스트정확도": round(accuracy_score(test_golds, lp_preds), 4)}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"산출물 저장 → {OUT_DIR}/ (head.pt + label_map.json + meta.json)")


if __name__ == "__main__":
    main()
